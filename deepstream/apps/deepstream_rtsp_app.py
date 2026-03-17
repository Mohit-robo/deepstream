"""
DeepStream SUTrack Tracker Application - Phase 5 (Native OSD + RTSP Streaming)

Pipeline:
    filesrc / v4l2src
        -> decodebin
        -> nvvideoconvert (compute-hw=1, NVMM RGBA)
        -> nvstreammux
        -> [Pad Probe: SUTrack inference + NvDsObjectMeta attachment]
        -> nvdsosd (GPU bounding box drawing)
        -> tee
            Branch A: queue -> nveglglessink / fakesink  (local display)
            Branch B: queue -> nvvideoconvert -> nvv4l2h264enc
                           -> rtph264pay -> udpsink      (RTSP feed)
    GstRtspServer wraps the UDP stream:
        rtsp://<jetson-ip>:8554/sutrack

Changes from Phase 4 (appsink-based):
  - Drawing: OpenCV cv2.rectangle removed; nvdsosd draws in GPU memory (zero-copy)
  - Logic: pad probe instead of appsink callback; inline processing per frame
  - Streaming: hardware H.264 via nvv4l2h264enc + GstRtspServer
  - Headless: fakesink display branch; static_roi required (no GUI)

Usage (headless, static ROI, RTSP):
    python deepstream/apps/deepstream_rtsp_app.py \\
        --config deepstream/configs/tracker_config.yml \\
        --input /path/to/video.mp4 \\
        --headless

Usage (with local display):
    python deepstream/apps/deepstream_rtsp_app.py \\
        --config deepstream/configs/tracker_config.yml \\
        --input /path/to/video.mp4

View RTSP stream (on any machine on the same network):
    vlc rtsp://<jetson-ip>:8554/sutrack

Requirements:
    - DeepStream Python bindings (pyds)
    - gi.repository.GstRtspServer  (standard on JetPack)
    - sutrack_fp32.engine at repo root
    - static_roi set in tracker_config.yml OR --init_bbox CLI arg
"""

# ---------------------------------------------------------------------------
# NOTE: Only ASCII characters are used in this file (Lesson 19).
# ---------------------------------------------------------------------------

import os
import sys
import argparse
import logging
import time
import threading

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import cv2
import yaml

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

# --- Optional: GstRtspServer ---
try:
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import GstRtspServer
    RTSP_AVAILABLE = True
except Exception:
    GstRtspServer = None
    RTSP_AVAILABLE = False

# --- Optional: pyds (DeepStream Python bindings) ---
try:
    import pyds
    PYDS_AVAILABLE = True
except ImportError:
    pyds = None
    PYDS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project root: deepstream/apps/ -> deepstream/ (ROOT)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tracker.sutrack_engine import SUTrackEngine
from tracker.tracker_manager import TrackerManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(level_str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S')


# ---------------------------------------------------------------------------
# Application state (shared across probe callbacks)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, manager, cfg, headless):
        self.manager     = manager
        self.cfg         = cfg
        self.headless    = headless

        self.initialized = False
        self.init_frame  = None    # RGBA frame captured in probe
        self.frame_ready = threading.Event()  # Triggered when first frame is captured
        self.init_done   = threading.Event()  # Triggered when ROI selection is finished
        self.init_bbox   = None    # [x, y, w, h] from CLI or config
        self.frame_idx   = 0
        self.frame_count = 0
        self.total_time  = 0.0
        self.loop        = None    # GLib.MainLoop set by main()


# ---------------------------------------------------------------------------
# Pad probe: tracker inference + NvDsObjectMeta attachment
# ---------------------------------------------------------------------------

def tracker_probe(pad, info, state):
    """
    GStreamer BUFFER pad probe placed on the nvdsosd sink pad.

    Steps per frame:
      1. Extract batch metadata from the GstBuffer.
      2. Get frame pixels via pyds.get_nvds_buf_surface (returns RGBA NumPy array).
      3. Convert RGBA -> RGB for the SUTrack preprocessor.
      4. On frame 0: initialize the tracker from static_roi / init_bbox.
         On frame N: call tracker_manager.update() to get bounding boxes.
      5. Attach NvDsObjectMeta for each tracked box so nvdsosd draws it in GPU.
    """
    if not PYDS_AVAILABLE:
        return Gst.PadProbeReturn.OK

    log = logging.getLogger('probe')

    gst_buffer = info.get_buffer()
    if gst_buffer is None:
        return Gst.PadProbeReturn.OK

    try:
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    except Exception as e:
        log.warning('Failed to get batch meta: %s', e)
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # --- Extract frame pixels from NVMM via pyds ---
        try:
            n_frame    = pyds.get_nvds_buf_surface(hash(gst_buffer),
                                                    frame_meta.batch_id)
            frame_rgba = np.array(n_frame, copy=True, order='C')
        except Exception as e:
            log.warning('Frame extraction failed: %s', e)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        # RGBA -> RGB  (tracker trained on RGB; Lesson 12)
        frame_rgb = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)

        # --- Initialize tracker on first frame ---
        if not state.initialized:
            bbox = state.init_bbox

            if bbox is None:
                static_roi = state.cfg.get('tracker', {}).get('static_roi', '')
                if static_roi:
                    try:
                        bbox = [float(v.strip()) for v in static_roi.split(',')]
                    except Exception as e:
                        log.error('Invalid static_roi format: %s', e)

            if bbox is not None:
                # Static ROI: initialize directly in probe (thread-safe)
                x, y, w, h = [int(v) for v in bbox]
                state.manager.initialize(frame_rgb, [x, y, w, h], frame_idx=0)
                state.initialized = True
                log.info('Tracker initialized with static bbox=[%d, %d, %d, %d]', x, y, w, h)
            elif not state.headless:
                # GUI selection: catch frame, signal main, and WAIT for result
                if state.init_frame is None:
                    state.init_frame = frame_rgba.copy()
                    state.frame_ready.set()
                
                log.debug('Probe thread blocking for ROI selection...')
                # This blocks the GStreamer streaming thread until main thread sets init_done
                state.init_done.wait() 
                log.debug('Probe thread unblocked.')
                
                # After unblocking, the tracker should be initialized by main thread.
                # If selection failed, main thread will have set init_done but not initialized.
                if not state.initialized:
                    state.loop.quit()
                    return Gst.PadProbeReturn.DROP
            else:
                log.error('Headless mode requires static_roi or --init_bbox.')
                state.loop.quit()
                return Gst.PadProbeReturn.DROP

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        # --- Track ---
        t0      = time.perf_counter()
        results = state.manager.update(frame_rgb, state.frame_idx)
        dt      = time.perf_counter() - t0

        state.total_time  += dt
        state.frame_count += 1
        state.frame_idx   += 1
        fps = 1.0 / dt if dt > 0 else 0.0

        # --- Attach NvDsObjectMeta for each tracked box (nvdsosd draws them) ---
        for obj_id, bbox in results.items():
            x1, y1, bw, bh = [float(v) for v in bbox]

            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            obj_meta.unique_component_id = 1
            obj_meta.object_id           = int(obj_id)

            # Bounding box visual properties
            rect                  = obj_meta.rect_params
            rect.left             = x1
            rect.top              = y1
            rect.width            = bw
            rect.height           = bh
            rect.border_width     = 3
            rect.border_color.red   = 0.0
            rect.border_color.green = 1.0
            rect.border_color.blue  = 0.0
            rect.border_color.alpha = 1.0
            rect.has_bg_color     = 0

            # Label text properties
            txt                       = obj_meta.text_params
            txt.display_text          = 'ID %d | %.1f FPS' % (obj_id, fps)
            txt.x_offset              = int(x1)
            txt.y_offset              = max(int(y1) - 12, 0)
            txt.font_params.font_name = 'Serif'
            txt.font_params.font_size = 10
            txt.font_params.font_color.red   = 1.0
            txt.font_params.font_color.green = 1.0
            txt.font_params.font_color.blue  = 1.0
            txt.font_params.font_color.alpha = 1.0
            txt.set_bg_clr            = 1
            txt.text_bg_clr.red   = 0.0
            txt.text_bg_clr.green = 0.0
            txt.text_bg_clr.blue  = 0.0
            txt.text_bg_clr.alpha = 1.0

            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------
# Dynamic pad handler (decodebin / uridecodebin)
# ---------------------------------------------------------------------------

def on_decoder_pad_added(decoder, pad, next_element):
    """Link the newly created decoder src pad to the next pipeline element."""
    log = logging.getLogger('pipeline')
    caps = pad.get_current_caps() or pad.query_caps(None)
    name = caps.get_structure(0).get_name()

    if name.startswith('video/'):
        sinkpad = next_element.get_static_pad('sink')
        if sinkpad and not sinkpad.is_linked():
            ret = pad.link(sinkpad)
            if ret != Gst.PadLinkReturn.OK:
                log.error('Decoder pad link failed: %s', ret)
            else:
                log.debug('Decoder -> nvvideoconvert linked (%s)', name)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(state, pipe_cfg):
    """
    Construct the Phase 5 GStreamer pipeline element-by-element.

    Element chain:
        source -> [decoder] -> nvvideoconvert(NVMM RGBA) -> mux
        mux -> osd -> tee
        tee -> queue -> nvvideoconvert -> [nv3dsink | nveglglessink | fakesink]
        tee -> queue -> nvvideoconvert -> encoder -> rtp -> udpsink

    Pad probe is attached to the osd sink pad.
    """
    Gst.init(None)
    pipeline = Gst.Pipeline.new('sutrack-rtsp-pipeline')

    width        = int(pipe_cfg.get('width',  1280))
    height       = int(pipe_cfg.get('height',  720))
    input_source = pipe_cfg.get('input_source') or ''

    # ------------------------------------------------------------------ Source
    if input_source.startswith('rtsp://') or input_source.startswith('http://'):
        src          = Gst.ElementFactory.make('uridecodebin', 'src')
        src.set_property('uri', input_source)
        need_decoder = False          # uridecodebin handles decode internally
    elif input_source:
        src          = Gst.ElementFactory.make('filesrc', 'src')
        src.set_property('location', input_source)
        need_decoder = True
    else:
        src          = Gst.ElementFactory.make('v4l2src', 'src')
        src.set_property('device', '/dev/video0')
        need_decoder = True

    decoder = Gst.ElementFactory.make('decodebin', 'decoder') if need_decoder else None

    # ------------------------------------------------ NV color convert (NVMM)
    # compute-hw=1 uses the VIC engine and avoids EGL display requirement
    # (Lessons 20, 21 -- headless safe)
    nvconv1 = Gst.ElementFactory.make('nvvideoconvert', 'nvconv1')
    nvconv1.set_property('compute-hw', 1)
    caps1   = Gst.ElementFactory.make('capsfilter', 'caps1')
    caps1.set_property(
        'caps', Gst.Caps.from_string('video/x-raw(memory:NVMM),format=RGBA'))

    # ------------------------------------------------------- nvstreammux
    mux = Gst.ElementFactory.make('nvstreammux', 'mux')
    mux.set_property('batch-size', 1)
    mux.set_property('width',  width)
    mux.set_property('height', height)
    mux.set_property('batched-push-timeout', 4000000)
    mux.set_property('live-source', 0)

    # ------------------------------------------------------------ nvdsosd
    osd = Gst.ElementFactory.make('nvdsosd', 'osd')
    osd.set_property('process-mode', 0)   # 0=CPU (safe); 1=GPU (faster if EGL)
    osd.set_property('display-text', 1)

    # --------------------------------------------------------------- tee
    tee = Gst.ElementFactory.make('tee', 'tee')

    # -------------------------------------------- Branch A: local display
    #
    # nvdsosd outputs NVMM surface arrays that nveglglessink cannot consume
    # directly (error: gst_eglglessink_cuda_buffer_copy).
    # Fix: always insert nvvideoconvert(compute-hw=1) before the display sink
    # to convert the NVMM surface to a format the sink understands.
    #
    # Sink priority:
    #   nv3dsink      -- Jetson native; handles NVMM directly, no EGL issues
    #   nveglglessink -- EGL-based; needs nvvideoconvert bridge + DISPLAY set
    #   fakesink      -- headless / RTSP-only mode
    q_disp    = Gst.ElementFactory.make('queue',         'q_disp')
    conv_disp = None   # populated for non-headless paths

    if state.headless:
        sink_disp = Gst.ElementFactory.make('fakesink', 'sink_disp')
        sink_disp.set_property('sync', False)
    else:
        # nvvideoconvert bridge: converts NVMM RGBA -> format the sink expects
        conv_disp = Gst.ElementFactory.make('nvvideoconvert', 'conv_disp')
        conv_disp.set_property('compute-hw', 1)

        # Prefer nv3dsink (Jetson AGX / Orin) -- no EGL context needed
        sink_disp = Gst.ElementFactory.make('nv3dsink', 'sink_disp')
        if sink_disp is None:
            # Fall back to nveglglessink (requires DISPLAY=:0)
            sink_disp = Gst.ElementFactory.make('nveglglessink', 'sink_disp')
        sink_disp.set_property('sync', False)

    # ---------------------------------------------- Branch B: RTSP stream
    rtsp_enabled = (pipe_cfg.get('rtsp_enabled', True) and RTSP_AVAILABLE)
    rtsp_elements = []

    if rtsp_enabled:
        q_rtsp   = Gst.ElementFactory.make('queue',          'q_rtsp')
        nvconv2  = Gst.ElementFactory.make('nvvideoconvert',  'nvconv2')
        nvconv2.set_property('compute-hw', 1)
        encoder  = Gst.ElementFactory.make('nvv4l2h264enc',   'encoder')
        encoder.set_property('bitrate', int(pipe_cfg.get('rtsp_bitrate', 4000000)))
        pay      = Gst.ElementFactory.make('rtph264pay',      'pay')
        pay.set_property('config-interval', 1)
        pay.set_property('pt', 96)
        udpsink  = Gst.ElementFactory.make('udpsink',         'udpsink')
        udpsink.set_property('host', '127.0.0.1')
        udpsink.set_property('port', int(pipe_cfg.get('rtsp_udp_port', 5400)))
        udpsink.set_property('sync', False)
        rtsp_elements = [q_rtsp, nvconv2, encoder, pay, udpsink]

    # ------------------------------------------------------- Add to pipeline
    core_elements = [src, nvconv1, caps1, mux, osd, tee, q_disp, sink_disp]
    if decoder:
        core_elements.append(decoder)
    if conv_disp is not None:
        core_elements.append(conv_disp)

    for el in core_elements + rtsp_elements:
        if el is None:
            raise RuntimeError('Failed to create a required GStreamer element.')
        pipeline.add(el)

    # --------------------------------------------------- Link static chain
    # source -> decoder (file/camera only)
    if decoder:
        src.link(decoder)
        decoder.connect('pad-added', on_decoder_pad_added, nvconv1)
    else:
        src.connect('pad-added', on_decoder_pad_added, nvconv1)

    # nvconv1 -> caps1 -> mux.sink_0 (request pad)
    nvconv1.link(caps1)
    caps1.get_static_pad('src').link(mux.get_request_pad('sink_0'))

    # mux -> osd -> tee
    mux.link(osd)
    osd.link(tee)

    # tee -> display branch (queue -> [conv_disp ->] sink_disp)
    tee.get_request_pad('src_%u').link(q_disp.get_static_pad('sink'))
    if conv_disp is not None:
        q_disp.link(conv_disp)
        conv_disp.link(sink_disp)
    else:
        q_disp.link(sink_disp)

    # tee -> RTSP branch
    if rtsp_enabled:
        tee.get_request_pad('src_%u').link(q_rtsp.get_static_pad('sink'))
        q_rtsp.link(nvconv2)
        nvconv2.link(encoder)
        encoder.link(pay)
        pay.link(udpsink)

    # ----------------------------------------- Attach pad probe on osd sink
    osd_sink_pad = osd.get_static_pad('sink')
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_probe, state)

    return pipeline


# ---------------------------------------------------------------------------
# RTSP server
# ---------------------------------------------------------------------------

def setup_rtsp_server(rtsp_port, udp_port, mount_path='/sutrack'):
    """
    Create a GstRtspServer that wraps the udpsink output as an RTSP stream.

    Clients connect to:
        rtsp://<jetson-ip>:<rtsp_port><mount_path>
    e.g.: rtsp://192.168.1.100:8554/sutrack
    """
    if not RTSP_AVAILABLE:
        return None

    server  = GstRtspServer.RTSPServer.new()
    server.props.service = str(rtsp_port)

    factory = GstRtspServer.RTSPMediaFactory.new()
    # The factory re-reads RTP packets from the udpsink and serves them via RTSP
    factory.set_launch(
        '( udpsrc name=pay0 port=%d '
        'caps="application/x-rtp,media=video,clock-rate=90000,'
        'encoding-name=H264,payload=96" )' % udp_port
    )
    factory.set_shared(True)

    server.get_mount_points().add_factory(mount_path, factory)
    server.attach(None)   # attaches to the default GLib main context

    return server


# ---------------------------------------------------------------------------
# GStreamer bus callbacks
# ---------------------------------------------------------------------------

def on_eos(bus, msg, state):
    log = logging.getLogger('pipeline')
    avg = state.frame_count / state.total_time if state.total_time > 0 else 0.0
    log.info('EOS -- frames=%d  total=%.2fs  avg_fps=%.2f',
             state.frame_count, state.total_time, avg)
    state.loop.quit()


def on_error(bus, msg, state):
    err, debug = msg.parse_error()
    logging.getLogger('pipeline').error('GStreamer error: %s\n%s', err, debug)
    state.loop.quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SUTrack DeepStream Phase 5 -- Native OSD + RTSP Streaming')
    parser.add_argument('--config', default='deepstream/configs/tracker_config.yml',
                        help='Path to tracker_config.yml')
    parser.add_argument('--input', '-i', default=None,
                        help='Video file path or RTSP/HTTP URL. Omit for camera.')
    parser.add_argument('--init_bbox', type=float, nargs=4,
                        metavar=('X', 'Y', 'W', 'H'),
                        help='Initial bounding box; overrides static_roi in config.')
    parser.add_argument('--headless', action='store_true',
                        help='Disable local display (fakesink). '
                             'Requires static_roi or --init_bbox.')
    parser.add_argument('--no-rtsp', action='store_true',
                        help='Disable RTSP streaming (useful for debugging).')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging(cfg.get('pipeline', {}).get('log_level', 'INFO'))
    log = logging.getLogger('main')

    if not PYDS_AVAILABLE:
        log.error('pyds (DeepStream Python bindings) not found.')
        log.error('Install DeepStream SDK and ensure pyds is on PYTHONPATH.')
        sys.exit(1)

    if not RTSP_AVAILABLE:
        log.warning('GstRtspServer not available -- RTSP streaming disabled.')

    # ------------------------------------------------- Merge CLI into config
    pipe_cfg    = cfg.get('pipeline', {})
    model_cfg   = cfg.get('model',    {})
    tracker_cfg = cfg.get('tracker',  {})

    if args.input:
        pipe_cfg['input_source'] = args.input
    if args.headless:
        pipe_cfg['headless'] = True
    if args.no_rtsp:
        pipe_cfg['rtsp_enabled'] = False

    # --------------------------------------------------------- Load TRT engine
    engine_path = os.path.normpath(
        os.path.join(ROOT, model_cfg.get('engine_path', '../sutrack_fp32.engine')))
    log.info('Loading TRT engine: %s', engine_path)

    engine = SUTrackEngine(
        engine_path    = engine_path,
        template_size  = model_cfg.get('template_size',  112),
        search_size    = model_cfg.get('search_size',    224),
        encoder_stride = model_cfg.get('encoder_stride',  16),
    )

    # ---------------------------------------------------------- Tracker manager
    manager = TrackerManager(
        engine              = engine,
        max_age             = tracker_cfg.get('max_age',             30),
        min_confidence      = tracker_cfg.get('min_confidence',      0.25),
        iou_match_threshold = tracker_cfg.get('iou_match_threshold', 0.3),
        template_factor     = model_cfg.get('template_factor',       2.0),
        search_factor       = model_cfg.get('search_factor',         4.0),
        use_hanning         = model_cfg.get('use_hanning_window',    True),
    )

    state          = AppState(manager=manager, cfg=cfg,
                              headless=pipe_cfg.get('headless', False))
    state.init_bbox = args.init_bbox  # may be None; probe will fall back to static_roi

    # -------------------------------------------------------- Build GStreamer pipeline
    pipeline = build_pipeline(state, pipe_cfg)

    # -------------------------------------------------------- RTSP server
    rtsp_port  = int(pipe_cfg.get('rtsp_port',     8554))
    udp_port   = int(pipe_cfg.get('rtsp_udp_port', 5400))
    rtsp_path  = pipe_cfg.get('rtsp_path', '/sutrack')

    if pipe_cfg.get('rtsp_enabled', True) and not args.no_rtsp:
        server = setup_rtsp_server(rtsp_port, udp_port, rtsp_path)
        if server:
            import socket
            try:
                host = socket.gethostbyname(socket.gethostname())
            except Exception:
                host = '127.0.0.1'
            log.info('RTSP stream ready: rtsp://%s:%d%s', host, rtsp_port, rtsp_path)
            log.info('View with:  vlc rtsp://%s:%d%s', host, rtsp_port, rtsp_path)

    # ---------------------------------------- GLib main loop + bus callbacks
    loop       = GLib.MainLoop()
    state.loop = loop

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message::eos',   on_eos,   state)
    bus.connect('message::error', on_error, state)

    log.info('Starting Phase 5 pipeline...')
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        log.error('Pipeline failed to reach PLAYING state.')
        sys.exit(1)

    # --- Option A: Snapshot Hybrid (Main Thread GUI Selection) ---
    if not state.initialized and not state.headless:
        log.info('Waiting for first frame to open ROI selector...')
        if state.frame_ready.wait(timeout=10.0) and state.init_frame is not None:
            # We are in the MAIN THREAD here. Safe to use cv2 GUI.
            disp = cv2.cvtColor(state.init_frame, cv2.COLOR_RGBA2BGR)
            win_name = "ROI Selection (Snapshot)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(win_name, 960, 720)
            cv2.putText(disp, 'Select ROI + SPACE / ENTER. C to cancel.', (20, 40),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 200, 0), 2)
            cv2.imshow(win_name, disp)
            cv2.waitKey(1)
            
            roi = cv2.selectROI(win_name, disp, fromCenter=False)
            x, y, w, h = [int(v) for v in roi]
            cv2.destroyWindow(win_name)

            if w > 0 and h > 0:
                # Manager needs RGB
                frame_rgb = cv2.cvtColor(state.init_frame, cv2.COLOR_RGBA2RGB)
                state.manager.initialize(frame_rgb, [x, y, w, h], frame_idx=0)
                state.initialized = True
                log.info('Tracker initialized via GUI: bbox=[%d, %d, %d, %d]', x, y, w, h)
                
                # Signal the PROBE thread to continue
                state.init_done.set()
            else:
                log.error('ROI selection cancelled or invalid. Quitting.')
                state.init_done.set() # Unblock probe so it can quit
                pipeline.set_state(Gst.State.NULL)
                sys.exit(0)
        else:
            if not state.initialized:
                log.error('Timeout or error capturing first frame for ROI selection.')
                state.init_done.set()
                pipeline.set_state(Gst.State.NULL)
                sys.exit(1)

    try:
        loop.run()
    except KeyboardInterrupt:
        log.info('Interrupted by user.')
    finally:
        pipeline.set_state(Gst.State.NULL)
        if state.frame_count > 0:
            avg = state.frame_count / state.total_time if state.total_time > 0 else 0.0
            log.info('Done -- frames=%d  avg_fps=%.2f', state.frame_count, avg)


if __name__ == '__main__':
    main()

"""
DeepStream SUTrack Tracker Application - Phase 6 (PGIE Click-to-Select ROI)

Pipeline (with PGIE enabled):
    filesrc / v4l2src
        -> decodebin
        -> nvvideoconvert (compute-hw=1, NVMM RGBA)
        -> nvstreammux
        -> nvinfer (PGIE: ResNet-10 detector, first-frame only)
        -> nvvideoconvert (compute-hw=1)
        -> [Pad Probe: read detections / SUTrack inference + NvDsObjectMeta]
        -> nvdsosd (GPU bounding box drawing)
        -> tee
            Branch A: queue -> nvvideoconvert -> nv3dsink / fakesink  (local display)
            Branch B: queue -> nvvideoconvert -> nvv4l2h264enc
                           -> rtph264pay -> udpsink                   (RTSP feed)
    GstRtspServer wraps the UDP stream:
        rtsp://<jetson-ip>:8554/sutrack

ROI Initialization modes (in priority order):
  1. static_roi in tracker_config.yml  -- fully headless, no GUI
  2. --init_bbox X Y W H               -- headless via CLI
  3. PGIE click-to-select (pgie_config in config or --pgie-config CLI)
       - Detector runs on frame 0; detected boxes shown as colored overlays
       - User LEFT-CLICKS on the object to track; smallest enclosing box wins ties
       - Press Q/ESC to skip PGIE and fall back to mode 4
  4. Manual selectROI                  -- draw a box with the mouse

Changes from Phase 5:
  - PGIE (nvinfer) added between nvstreammux and nvdsosd
  - Detections from first frame captured in probe and stored in AppState
  - New select_bbox_click_to_select() GUI function
  - Main thread GUI flow: click-to-select -> manual-ROI fallback

Usage (click-to-select):
    python deepstream/apps/deepstream_rtsp_app.py \\
        --config deepstream/configs/tracker_config.yml \\
        --input /path/to/video.mp4

Usage (headless, static ROI, RTSP):
    python deepstream/apps/deepstream_rtsp_app.py \\
        --config deepstream/configs/tracker_config.yml \\
        --input /path/to/video.mp4 \\
        --headless

View RTSP stream (on any machine on the same network):
    vlc rtsp://<jetson-ip>:8554/sutrack

Requirements:
    - DeepStream Python bindings (pyds)
    - gi.repository.GstRtspServer  (standard on JetPack)
    - sutrack_fp32.engine at repo root
    - deepstream/configs/pgie_config.txt pointing to a valid detector engine
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
from app_utils import (
    load_yaml, setup_logging, setup_rtsp_server, get_local_ip,
    manual_roi_select, select_bbox_click_to_select, CLASS_COLORS
)


# ---------------------------------------------------------------------------
# Application state (shared across probe callbacks and main thread)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, manager, cfg, headless, pgie_enabled=False, pgie_config_path='', pgie_one_shot=True):
        self.manager     = manager
        self.cfg         = cfg
        self.headless    = headless

        # Phase 6: PGIE support
        self.pgie_enabled = pgie_enabled
        self.pgie_config_path = pgie_config_path
        self.pgie_one_shot = pgie_one_shot
        self.pgie_element = None # Reference to nvinfer element for optimization
        self.first_frame_detections = [] # list of dicts: [x,y,w,h,class_id,label,conf]

        self.initialized = False
        self.init_frame  = None    # RGBA frame captured in probe
        self.frame_ready = threading.Event()  # Triggered when first frame is captured
        self.init_done   = threading.Event()  # Triggered when ROI selection is finished
        self.init_bbox   = None    # [x, y, w, h] from CLI or config
        self.frame_idx   = 0
        self.frame_count = 0
        self.total_time  = 0.0
        self.loop        = None    # GLib.MainLoop set by main()


# GUI selection logic moved to app_utils.py


# ---------------------------------------------------------------------------
# Pad probe: read PGIE detections (frame 0) + tracker inference + metadata
# ---------------------------------------------------------------------------

def tracker_probe(pad, info, state):
    """
    GStreamer BUFFER pad probe placed on the nvdsosd sink pad.

    Steps per frame:
      1. Extract batch metadata from the GstBuffer.
      2. Get frame pixels via pyds.get_nvds_buf_surface (returns RGBA NumPy array).
      3. Convert RGBA -> RGB for the SUTrack preprocessor.
      4. Frame 0 (not yet initialized):
           a. If PGIE enabled: read NvDsObjectMeta detections placed by nvinfer.
           b. Capture frame + store detections in state.
           c. Signal main thread (frame_ready); BLOCK until init_done.
      5. Frame N (initialized): call tracker_manager.update(), attach result
         NvDsObjectMeta so nvdsosd draws the tracked box in GPU memory.
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

            # Priority 1: static_roi or CLI --init_bbox
            if bbox is None:
                static_roi = state.cfg.get('tracker', {}).get('static_roi', '')
                if static_roi:
                    try:
                        bbox = [float(v.strip()) for v in static_roi.split(',')]
                    except Exception as e:
                        log.error('Invalid static_roi format: %s', e)

            if bbox is not None:
                # Headless static init -- no GUI required
                x, y, w, h = [int(v) for v in bbox]
                state.manager.initialize(frame_rgb, [x, y, w, h], frame_idx=0)
                state.initialized = True
                log.info('Tracker initialized with static bbox=[%d, %d, %d, %d]',
                         x, y, w, h)
                
                # Phase 6.5: Optimize PGIE (one-shot mode)
                if state.pgie_element and state.pgie_one_shot:
                    log.info('PGIE optimization: Setting interval to 10000 for power-save.')
                    state.pgie_element.set_property('interval', 10000)

            elif not state.headless:
                # GUI path: capture first frame once, then block until main thread
                # completes ROI selection (Lesson 26 -- threading.Event sync).
                if state.init_frame is None:
                    state.init_frame = frame_rgba.copy()

                    # Priority 2: PGIE click-to-select
                    # Read detections that nvinfer attached to this frame's metadata.
                    if state.pgie_enabled:
                        dets = []
                        l_obj = frame_meta.obj_meta_list
                        while l_obj is not None:
                            try:
                                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                            except StopIteration:
                                break
                            rect = obj_meta.rect_params
                            dets.append({
                                'x':          int(rect.left),
                                'y':          int(rect.top),
                                'w':          int(rect.width),
                                'h':          int(rect.height),
                                'class_id':   int(obj_meta.class_id),
                                'label':      str(obj_meta.obj_label)
                                              if obj_meta.obj_label else '',
                                'confidence': float(obj_meta.confidence),
                            })
                            try:
                                l_obj = l_obj.next
                            except StopIteration:
                                break
                        state.first_frame_detections = dets
                        log.info('PGIE detections on first frame: %d', len(dets))

                    state.frame_ready.set()

                log.debug('Probe thread blocking for ROI selection...')
                state.init_done.wait()
                log.debug('Probe thread unblocked.')

                if not state.initialized:
                    # Selection was cancelled -- quit pipeline
                    state.loop.quit()
                    return Gst.PadProbeReturn.DROP
                
                # Phase 6.4: Clear detections for frame 0 too!
                if state.pgie_one_shot:
                    while frame_meta.obj_meta_list is not None:
                        try:
                            obj_meta = pyds.NvDsObjectMeta.cast(frame_meta.obj_meta_list.data)
                            pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                        except Exception:
                            break
                    while frame_meta.display_meta_list is not None:
                        try:
                            disp_meta = pyds.NvDsDisplayMeta.cast(frame_meta.display_meta_list.data)
                            pyds.nvds_remove_display_meta_from_frame(frame_meta, disp_meta)
                        except Exception:
                            break

            else:
                log.error('Headless mode requires static_roi or --init_bbox.')
                state.loop.quit()
                return Gst.PadProbeReturn.DROP

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        # --- Track (all frames after initialization) ---
        if state.initialized:
            # Phase 6.4: Clear PGIE artifacts (boxes, labels, display meta).
            # Do this for EVERY frame after init (including frame 0 unblocking).
            if state.pgie_one_shot:
                while frame_meta.obj_meta_list is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(frame_meta.obj_meta_list.data)
                        pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                    except Exception:
                        break

                while frame_meta.display_meta_list is not None:
                    try:
                        disp_meta = pyds.NvDsDisplayMeta.cast(frame_meta.display_meta_list.data)
                        pyds.nvds_remove_display_meta_from_frame(frame_meta, disp_meta)
                    except Exception:
                        break

        t0      = time.perf_counter()
        results = state.manager.update(frame_rgb, state.frame_idx)
        dt      = time.perf_counter() - t0

        state.total_time  += dt
        state.frame_count += 1
        state.frame_idx   += 1
        fps = 1.0 / dt if dt > 0 else 0.0

        # --- Attach NvDsObjectMeta for each tracked box (nvdsosd draws them) ---
        for obj_id, bbox_out in results.items():
            x1, y1, bw, bh = [float(v) for v in bbox_out]

            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            obj_meta.unique_component_id = 2 # Distinct from PGIE
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
    Construct the Phase 6 GStreamer pipeline element-by-element.

    Element chain (with PGIE):
        source -> [decoder] -> nvconv1(NVMM RGBA) -> mux
        mux -> pgie(nvinfer) -> nvconv_pgie -> osd -> tee
        tee -> queue -> [conv_disp ->] [nv3dsink | nveglglessink | fakesink]
        tee -> queue -> nvconv2 -> encoder -> rtp -> udpsink

    Element chain (without PGIE, e.g. headless/static_roi):
        source -> [decoder] -> nvconv1(NVMM RGBA) -> mux -> osd -> tee -> ...

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
        need_decoder = False
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
    # compute-hw=1 uses VIC engine -- no EGL display context required (Lesson 20)
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

    # ------------------------------------ Optional: nvinfer PGIE (Phase 6)
    # Added between mux and osd; runs the detector and attaches NvDsObjectMeta.
    # A nvvideoconvert bridge follows to satisfy format requirements for nvdsosd.
    pgie        = None
    nvconv_pgie = None
    if state.pgie_enabled and state.pgie_config_path:
        pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
        if pgie is None:
            logging.getLogger('pipeline').error(
                'Failed to create nvinfer element -- PGIE disabled.')
            state.pgie_enabled = False
        else:
            pgie.set_property('config-file-path', state.pgie_config_path)
            state.pgie_element = pgie # Store reference for optimization
            nvconv_pgie = Gst.ElementFactory.make('nvvideoconvert', 'nvconv_pgie')
            nvconv_pgie.set_property('compute-hw', 1)

    # ------------------------------------------------------------ nvdsosd
    osd = Gst.ElementFactory.make('nvdsosd', 'osd')
    osd.set_property('process-mode', 0)   # 0=CPU (safe without EGL; Lesson 23)
    osd.set_property('display-text', 1)

    # --------------------------------------------------------------- tee
    tee = Gst.ElementFactory.make('tee', 'tee')

    # -------------------------------------------- Branch A: local display
    # nvdsosd outputs NVMM surface arrays that nveglglessink cannot consume
    # directly.  Fix: insert nvvideoconvert(compute-hw=1) as a bridge first.
    # Prefer nv3dsink (Jetson-native NVMM) over nveglglessink (Lesson 24).
    q_disp    = Gst.ElementFactory.make('queue',     'q_disp')
    conv_disp = None

    if state.headless:
        sink_disp = Gst.ElementFactory.make('fakesink', 'sink_disp')
        sink_disp.set_property('sync', False)
    else:
        conv_disp = Gst.ElementFactory.make('nvvideoconvert', 'conv_disp')
        conv_disp.set_property('compute-hw', 1)
        sink_disp = Gst.ElementFactory.make('nv3dsink', 'sink_disp')
        if sink_disp is None:
            sink_disp = Gst.ElementFactory.make('nveglglessink', 'sink_disp')
        sink_disp.set_property('sync', False)

    # ---------------------------------------------- Branch B: RTSP stream
    rtsp_enabled  = (pipe_cfg.get('rtsp_enabled', True) and RTSP_AVAILABLE)
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
    if decoder      is not None: core_elements.append(decoder)
    if pgie         is not None: core_elements.append(pgie)
    if nvconv_pgie  is not None: core_elements.append(nvconv_pgie)
    if conv_disp    is not None: core_elements.append(conv_disp)

    for el in core_elements + rtsp_elements:
        if el is None:
            raise RuntimeError('Failed to create a required GStreamer element.')
        pipeline.add(el)

    # --------------------------------------------------- Link static chain
    if decoder:
        src.link(decoder)
        decoder.connect('pad-added', on_decoder_pad_added, nvconv1)
    else:
        src.connect('pad-added', on_decoder_pad_added, nvconv1)

    # nvconv1 -> caps1 -> mux.sink_0
    nvconv1.link(caps1)
    caps1.get_static_pad('src').link(mux.get_request_pad('sink_0'))

    # mux -> [pgie -> nvconv_pgie ->] osd -> tee
    if pgie is not None:
        mux.link(pgie)
        pgie.link(nvconv_pgie)
        nvconv_pgie.link(osd)
    else:
        mux.link(osd)
    osd.link(tee)

    # tee -> display branch
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

# RTSP server moved to app_utils.py


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
        description='SUTrack DeepStream Phase 6 -- PGIE Click-to-Select ROI')
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
    parser.add_argument('--pgie-config', default=None,
                        help='Path to pgie_config.txt. '
                             'Overrides pgie_config in tracker_config.yml. '
                             'Enables click-to-select ROI.')
    parser.add_argument('--no-pgie', action='store_true',
                        help='Disable PGIE even if pgie_config is set in config.')
    parser.add_argument('--no-rtsp', action='store_true',
                        help='Disable RTSP streaming (useful for debugging).')
    parser.add_argument('--no-one-shot', action='store_true',
                        help='Disable PGIE power-save optimization (keep detector running).')
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
    if args.no_one_shot:
        pipe_cfg['pgie_one_shot'] = False

    headless = pipe_cfg.get('headless', False)

    # ------------------------------------------------ Resolve PGIE config path
    # Priority: --pgie-config CLI > pgie_config in tracker_config.yml
    # Paths in config are relative to the deepstream/ directory (ROOT).
    pgie_config_rel = pipe_cfg.get('pgie_config', '') or ''
    if args.pgie_config:
        pgie_config_rel = args.pgie_config

    pgie_enabled = bool(pgie_config_rel) and not headless and not args.no_pgie
    log.info('PGIE enabled: %s (one-shot: %s)', pgie_enabled, pipe_cfg.get('pgie_one_shot', True))

    pgie_config_path = ''
    if pgie_enabled:
        if os.path.isabs(pgie_config_rel):
            pgie_config_path = pgie_config_rel
        else:
            pgie_config_path = os.path.normpath(
                os.path.join(ROOT, pgie_config_rel))
        if not os.path.exists(pgie_config_path):
            log.warning('pgie_config not found at %s -- PGIE disabled.',
                        pgie_config_path)
            pgie_enabled     = False
            pgie_config_path = ''
        else:
            log.info('PGIE config: %s', pgie_config_path)

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

    state            = AppState(manager       = manager,
                                cfg           = cfg,
                                headless      = headless,
                                pgie_one_shot = pipe_cfg.get('pgie_one_shot', True))
    state.init_bbox  = args.init_bbox   # may be None
    state.pgie_enabled = pgie_enabled
    state.pgie_config_path = pgie_config_path

    # -------------------------------------------------------- Build GStreamer pipeline
    pipeline = build_pipeline(state, pipe_cfg)

    # -------------------------------------------------------- RTSP server
    rtsp_port  = int(pipe_cfg.get('rtsp_port',     8554))
    udp_port   = int(pipe_cfg.get('rtsp_udp_port', 5400))
    rtsp_path  = pipe_cfg.get('rtsp_path', '/sutrack')

    if pipe_cfg.get('rtsp_enabled', True) and not args.no_rtsp:
        server = setup_rtsp_server(rtsp_port, udp_port, rtsp_path)
        if server:
            host = get_local_ip()
            log.info('RTSP stream ready: rtsp://%s:%d%s', host, rtsp_port, rtsp_path)
            log.info('View with:  vlc rtsp://%s:%d%s', host, rtsp_port, rtsp_path)

    # ---------------------------------------- GLib main loop + bus callbacks
    loop       = GLib.MainLoop()
    state.loop = loop

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message::eos',   on_eos,   state)
    bus.connect('message::error', on_error, state)

    log.info('Starting Phase 6 pipeline...')
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        log.error('Pipeline failed to reach PLAYING state.')
        sys.exit(1)

    # -------------------------------- GUI ROI selection (main thread -- Lesson 25)
    # Only when not already initialized via static_roi / --init_bbox.
    if not state.initialized and not headless:
        log.info('Waiting for first frame...')
        if not state.frame_ready.wait(timeout=15.0) or state.init_frame is None:
            log.error('Timeout waiting for first frame. Check input source.')
            state.init_done.set()
            pipeline.set_state(Gst.State.NULL)
            sys.exit(1)

        selected_bbox = None

        # --- Mode A: PGIE click-to-select ---
        if state.pgie_enabled:
            if state.first_frame_detections:
                log.info('Showing %d PGIE detections for click-to-select...',
                         len(state.first_frame_detections))
                selected_bbox = select_bbox_click_to_select(
                    state.init_frame, state.first_frame_detections)
                if selected_bbox is not None:
                    log.info('Click-to-select result: %s', selected_bbox)
                else:
                    log.info('No click selection -- falling back to manual ROI.')
            else:
                log.warning('PGIE found no detections on first frame. '
                            'Falling back to manual ROI.')

        # --- Mode B: Manual selectROI (fallback) ---
        if selected_bbox is None:
            log.info('Manual ROI selection...')
            selected_bbox = manual_roi_select(state.init_frame)

        if selected_bbox is None:
            log.error('ROI selection cancelled. Quitting.')
            state.init_done.set()
            pipeline.set_state(Gst.State.NULL)
            sys.exit(0)

        # Initialize tracker from the selected box (main thread -- safe)
        if isinstance(selected_bbox, dict):
            x, y, w, h = selected_bbox['x'], selected_bbox['y'], selected_bbox['w'], selected_bbox['h']
            # Optionally capture target_id if RTSP App state supports it (Phase 6 might not)
        else:
            x, y, w, h = selected_bbox
        
        frame_rgb   = cv2.cvtColor(state.init_frame, cv2.COLOR_RGBA2RGB)
        state.manager.initialize(frame_rgb, [int(x), int(y), int(w), int(h)],
                                 frame_idx=0)
        state.initialized = True
        log.info('Tracker initialized: bbox=[%d, %d, %d, %d]',
                 int(x), int(y), int(w), int(h))

        # Phase 6.5: Optimize PGIE (one-shot mode)
        if state.pgie_element and state.pgie_one_shot:
            log.info('PGIE optimization: Setting interval to 10000 for power-save.')
            state.pgie_element.set_property('interval', 10000)

        # Unblock the probe thread (Lesson 26)
        state.init_done.set()

    try:
        loop.run()
    except KeyboardInterrupt:
        log.info('Interrupted by user.')
    finally:
        pipeline.set_state(Gst.State.NULL)
        if state.frame_count > 0:
            avg = (state.frame_count / state.total_time
                   if state.total_time > 0 else 0.0)
            log.info('Done -- frames=%d  avg_fps=%.2f', state.frame_count, avg)


if __name__ == '__main__':
    main()

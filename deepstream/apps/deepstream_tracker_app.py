"""
DeepStream SUTrack Tracker Application - Phase 4.2 (Manual Init + Appsink)

Pipeline:
    filesrc / v4l2src
        -> decodebin / nvv4l2decoder
        -> nvvideoconvert
        -> appsink  <- Python callback: SUTrack tracking loop

Usage (file input, GUI ROI selection):
    cd ~/SUTrack
    python deepstream/apps/deepstream_tracker_app.py \
        --config deepstream/configs/tracker_config.yml \
        --input /path/to/video.mp4

Usage (file input, headless with saved output):
    python deepstream/apps/deepstream_tracker_app.py \
        --config deepstream/configs/tracker_config.yml \
        --input /path/to/video.mp4 --init_bbox 300 200 120 140 \
        --output out.mp4 --headless

Usage (camera):
    python deepstream/apps/deepstream_tracker_app.py \
        --config deepstream/configs/tracker_config.yml

Design:
- No PyTorch at runtime - pure NumPy + TRT (cuda.cudart)
- BGR -> RGB conversion before every tracker call (Lesson 12)
- Shared SUTrackEngine across all TrackerInstance objects
- Config loaded from YAML; no hardcoded paths
"""

import os
import sys
import argparse
import logging
import time
import ctypes

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import cv2 as cv
import yaml
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# ---------------------------------------------------------------------------
# Project root on sys.path so tracker package is importable
# deepstream/apps/ -> deepstream/ (ROOT)
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tracker.sutrack_engine import SUTrackEngine
from tracker.tracker_manager import TrackerManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S')


# ---------------------------------------------------------------------------
# NvBufSurface -> NumPy (RGBA -> BGR via OpenCV)
# ---------------------------------------------------------------------------

def nvbuf_to_bgr(sample) -> np.ndarray:
    """
    Convert a GStreamer buffer from appsink to a BGR NumPy array.

    Works for video/x-raw(memory:NVMM) and plain video/x-raw.
    Falls back to mapping the buffer directly when pyds is unavailable.
    """
    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)

    width  = structure.get_int('width')[1]
    height = structure.get_int('height')[1]
    fmt    = structure.get_string('format')

    result, mapinfo = buf.map(Gst.MapFlags.READ)
    if not result:
        raise RuntimeError('Could not map GStreamer buffer')

    try:
        data = np.frombuffer(mapinfo.data, dtype=np.uint8)
    finally:
        buf.unmap(mapinfo)

    # Handle common formats
    if fmt in ('RGBA', 'RGBx'):
        img = data.reshape(height, width, 4)
        return cv.cvtColor(img, cv.COLOR_RGBA2BGR)
    elif fmt == 'RGB':
        img = data.reshape(height, width, 3)
        return cv.cvtColor(img, cv.COLOR_RGB2BGR)
    elif fmt in ('BGRx', 'BGRA'):
        img = data.reshape(height, width, 4)
        return img[:, :, :3].copy()
    elif fmt == 'BGR':
        return data.reshape(height, width, 3).copy()
    elif fmt in ('I420', 'NV12'):
        img_yuv = data.reshape(height * 3 // 2, width)
        return cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR_NV12 if fmt == 'NV12'
                           else cv.COLOR_YUV2BGR_I420)
    else:
        raise ValueError(f'Unsupported pixel format from appsink: {fmt}')


# ---------------------------------------------------------------------------
# Application state (shared between GLib main loop and appsink callback)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, manager: TrackerManager, cfg: dict,
                 output_path: str, headless: bool):
        self.manager      = manager
        self.cfg          = cfg
        self.output_path  = output_path
        self.headless     = headless

        self.initialized  = False
        self.init_frame   = None   # first frame held until ROI is selected
        self.frame_idx    = 0
        self.frame_count  = 0
        self.total_time   = 0.0

        self.writer       = None
        self.loop         = None   # GLib.MainLoop set by caller
        self.win_name     = 'SUTrack DeepStream'

        if not headless:
            cv.namedWindow(self.win_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(self.win_name, 960, 720)


# ---------------------------------------------------------------------------
# Appsink callback
# ---------------------------------------------------------------------------

def on_new_sample(appsink, state: AppState):
    sample = appsink.emit('pull-sample')
    if sample is None:
        return Gst.FlowReturn.ERROR

    try:
        frame_bgr = nvbuf_to_bgr(sample)
    except Exception as e:
        logging.getLogger('appsink').warning('Frame decode error: %s', e)
        return Gst.FlowReturn.OK

    # Lazy-initialize video writer once we know the frame size
    if state.writer is None and state.output_path:
        H, W = frame_bgr.shape[:2]
        state.writer = cv.VideoWriter(
            state.output_path,
            cv.VideoWriter_fourcc(*'mp4v'),
            30.0, (W, H))
        logging.getLogger('appsink').info('Writer opened: %s (%dx%d)', state.output_path, W, H)

    # --- First frame: grab ROI if not yet initialized ---
    if not state.initialized:
        state.init_frame = frame_bgr.copy()
        
        # Priority: 1. CLI --init_bbox, 2. YAML tracker:static_roi, 3. GUI selection
        bbox = state.cfg.get('init_bbox')
        if bbox is None:
            static_roi = state.cfg.get('tracker', {}).get('static_roi')
            if static_roi:
                try:
                    bbox = [float(v.strip()) for v in static_roi.split(',')]
                except Exception as e:
                    logging.getLogger('appsink').error('Invalid static_roi format: %s', e)

        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            state.manager.initialize(frame_rgb, [x, y, w, h], frame_idx=0)
            state.initialized = True
        else:
            # Need GUI — pause processing until ROI is drawn
            if state.headless:
                logging.getLogger('appsink').error('Headless mode requires init_bbox or static_roi')
                state.loop.quit()
                return Gst.FlowReturn.ERROR
                
            disp = frame_bgr.copy()
            cv.putText(disp, 'Draw ROI then press SPACE / ENTER',
                       (20, 40), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 200, 0), 2)
            # imshow BEFORE selectROI to init Qt window handler (Lesson 9)
            cv.imshow(state.win_name, disp)
            cv.waitKey(1)
            x, y, w, h = cv.selectROI(state.win_name, disp, fromCenter=False)
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            state.manager.initialize(frame_rgb, [int(x), int(y), int(w), int(h)], frame_idx=0)
            state.initialized = True
        
        return Gst.FlowReturn.OK

    # --- Tracking frame ---
    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)   # BGR -> RGB (Lesson 12)

    t0 = time.perf_counter()
    results = state.manager.update(frame_rgb, state.frame_idx)
    dt = time.perf_counter() - t0

    state.total_time += dt
    state.frame_count += 1
    state.frame_idx   += 1
    fps = 1.0 / dt if dt > 0 else 0.0

    # Draw bounding boxes and IDs
    disp = frame_bgr.copy()
    for obj_id, bbox in results.items():
        x1, y1, bw, bh = [int(v) for v in bbox]
        cv.rectangle(disp, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
        cv.putText(disp, f'ID {obj_id}', (x1, max(y1 - 8, 12)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.putText(disp, f'{fps:.1f} FPS  |  tracks: {state.manager.num_active}',
               (20, 36), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 2)

    if state.writer:
        state.writer.write(disp)

    if not state.headless:
        cv.imshow(state.win_name, disp)
        key = cv.waitKey(1)
        if key == ord('q'):
            state.loop.quit()

    return Gst.FlowReturn.OK


def on_eos(bus, msg, state: AppState):
    avg_fps = state.frame_count / state.total_time if state.total_time > 0 else 0
    logging.getLogger('pipeline').info(
        'EOS — frames=%d  total=%.2fs  avg_fps=%.2f',
        state.frame_count, state.total_time, avg_fps)
    state.loop.quit()


def on_error(bus, msg, state: AppState):
    err, debug = msg.parse_error()
    logging.getLogger('pipeline').error('GStreamer error: %s\n%s', err, debug)
    state.loop.quit()


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(input_source: str, use_nvmm: bool = True) -> Gst.Pipeline:
    """
    Build a GStreamer pipeline: source -> decode -> convert -> appsink.

    HW decode (nvv4l2decoder) is auto-selected by decodebin on Jetson.
    Color conversion always uses software videoconvert because nvvideoconvert
    requires EGL which is unavailable over SSH / headless (Lesson 20).
    """
    Gst.init(None)

    if input_source == '' or input_source is None:
        # Camera
        src_desc = 'v4l2src device=/dev/video0'
        decode   = '! videoconvert'
    elif input_source.startswith('rtsp://'):
        src_desc = f'uridecodebin uri={input_source} ! queue'
        decode   = ''
    else:
        # File
        src_desc = f'filesrc location={input_source} ! decodebin'
        decode   = ''

    # On Jetson, decodebin usually selects nvv4l2decoder which outputs NVMM memory.
    # Standard 'videoconvert' cannot handle NVMM memory, leading to 'not-negotiated'.
    # We MUST use 'nvvideoconvert' to bridge from NVMM to system RAM.
    # User requested GPU conversion (compute-hw=0). Note: requires DISPLAY=:0.
    convert = (
        '! nvvideoconvert compute-hw=0 '
        '! video/x-raw,format=RGBA '
        '! videoconvert '
        '! video/x-raw,format=RGBA'
    )

    pipeline_str = (
        f'{src_desc} {decode} {convert} '
        f'! appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false'
    )
    logging.getLogger('pipeline').debug('Pipeline: %s', pipeline_str)
    pipeline = Gst.parse_launch(pipeline_str)
    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SUTrack DeepStream Tracker (Phase 4.2)')
    parser.add_argument('--config', default='deepstream/configs/tracker_config.yml',
                        help='Path to tracker_config.yml')
    parser.add_argument('--input', '-i', default=None,
                        help='Video file path or RTSP URL. Omit for camera.')
    parser.add_argument('--init_bbox', type=float, nargs=4,
                        metavar=('X', 'Y', 'W', 'H'),
                        help='Initial bounding box. Skips GUI selection if provided.')
    parser.add_argument('--output', default=None,
                        help='Output video path (optional).')
    parser.add_argument('--headless', action='store_true',
                        help='No display window. Requires --init_bbox.')
    parser.add_argument('--no-nvmm', action='store_true',
                        help='Use software decode/convert (non-Jetson fallback).')
    args = parser.parse_args()

    if args.headless and args.init_bbox is None:
        parser.error('--headless requires --init_bbox')

    # Load config
    cfg = load_yaml(args.config)
    setup_logging(cfg.get('pipeline', {}).get('log_level', 'INFO'))
    log = logging.getLogger('main')

    # Override config with CLI args
    if args.input:
        cfg.setdefault('pipeline', {})['input_source'] = args.input
    if args.init_bbox:
        cfg['init_bbox'] = args.init_bbox
    if args.output:
        cfg.setdefault('pipeline', {})['output_path'] = args.output
    if args.headless:
        cfg.setdefault('pipeline', {})['headless'] = True

    model_cfg   = cfg.get('model', {})
    tracker_cfg = cfg.get('tracker', {})
    pipe_cfg    = cfg.get('pipeline', {})

    # engine_path is relative to ROOT (deepstream/)
    engine_path = os.path.join(ROOT, model_cfg.get('engine_path', '../sutrack_fp32.engine'))
    engine_path = os.path.normpath(engine_path)
    log.info('Engine path: %s', engine_path)

    # Load shared TRT engine
    engine = SUTrackEngine(
        engine_path=engine_path,
        template_size=model_cfg.get('template_size', 112),
        search_size=model_cfg.get('search_size', 224),
        encoder_stride=model_cfg.get('encoder_stride', 16),
    )

    # Build tracker manager
    manager = TrackerManager(
        engine=engine,
        max_age=tracker_cfg.get('max_age', 30),
        min_confidence=tracker_cfg.get('min_confidence', 0.25),
        iou_match_threshold=tracker_cfg.get('iou_match_threshold', 0.3),
        template_factor=model_cfg.get('template_factor', 2.0),
        search_factor=model_cfg.get('search_factor', 4.0),
        use_hanning=model_cfg.get('use_hanning_window', True),
    )

    state = AppState(
        manager=manager,
        cfg=cfg,
        output_path=pipe_cfg.get('output_path') or '',
        headless=pipe_cfg.get('headless', False),
    )

    # Build GStreamer pipeline
    input_source = pipe_cfg.get('input_source') or ''
    use_nvmm = not args.no_nvmm
    pipeline = build_pipeline(input_source, use_nvmm=use_nvmm)

    appsink = pipeline.get_by_name('appsink')
    appsink.connect('new-sample', on_new_sample, state)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    loop = GLib.MainLoop()
    state.loop = loop
    bus.connect('message::eos',   on_eos,   state)
    bus.connect('message::error', on_error, state)

    log.info('Starting pipeline...')
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        log.info('Interrupted by user')
    finally:
        pipeline.set_state(Gst.State.NULL)
        if state.writer:
            state.writer.release()
        cv.destroyAllWindows()

        if state.frame_count > 0:
            avg = state.frame_count / state.total_time if state.total_time > 0 else 0
            log.info('Done — frames=%d  avg_fps=%.2f', state.frame_count, avg)


if __name__ == '__main__':
    main()

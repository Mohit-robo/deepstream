"""
DeepStream SUTrack Tracker Application - Phase 10
Live OSD Selection + ID Persistence + Dynamic Target Management

Selection Flow:
  1. Press 's'  → enter SELECTING mode (OSD highlights a candidate box)
  2. Press 'n'  → cycle to next candidate (left-to-right)
  3. Press 'p'  → cycle to previous candidate
  4. Press 'l' or SPACE → lock onto highlighted box (→ LOCKED)
  5. Press 'q'  → cancel selection / stop tracking (→ IDLE)
  6. Press 'x'  → exit application

States: IDLE → SELECTING → LOCKED → SEARCHING → STALE → IDLE
"""

import os
import sys
import argparse
import logging
import threading
import math
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import cv2
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

# ---------------------------------------------------------------------------
# Project Setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app_utils import (
    load_yaml, setup_logging, get_local_ip,
    get_iou, RTSP_AVAILABLE,
    IDHistory, TrackingState,
    KeyReader,
    COLOR_LOCKED, COLOR_CANDIDATE, COLOR_DETECTION,
    COLOR_SEARCHING, COLOR_STALE,
)
from tracker.sutrack_engine import SUTrackEngine
from tracker.tracker_manager import TrackerManager

try:
    import pyds
    PYDS_AVAILABLE = True
except ImportError:
    pyds = None
    PYDS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
LOW_CONF_THRESHOLD  = 0.20   # below this → increment lost_frames counter
REID_IOU_THRESHOLD  = 0.35   # IoU-based re-ID (object returned to same spatial area)
REID_HIST_THRESHOLD = 0.58   # histogram similarity for appearance re-ID

# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, manager, cfg, headless, pgie_config_path,
                 pgie_one_shot=True, debug_boxes=False):
        self.manager          = manager
        self.cfg              = cfg
        self.headless         = headless
        self.pgie_config_path = pgie_config_path
        self.pgie_one_shot    = pgie_one_shot
        self.pgie_element     = None
        self.loop             = None

        # State machine
        self.tracking_state   = TrackingState.IDLE
        self.target_id        = -1
        self.id_history       = IDHistory()

        # Counters
        self.frame_idx   = 0
        self.frame_count = 0

        # Selection state (OSD cycling)
        self.current_dets    = []     # sorted left→right each frame (thread-safe via GIL)
        self.candidate_index = 0      # which det is highlighted
        self._pending_lock   = None   # set by key handler, consumed by probe

        # Auto-loss
        self.lost_frames     = 0
        self.hibernate_after = int(cfg.get('tracker', {}).get('hibernate_after', 150))

        self.debug_boxes = debug_boxes

        # Exit flag (set by KeyReader handler)
        self.exit_requested = False


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _capture_dets(frame_meta):
    """Extract all NvDsObjectMeta from a frame into plain dicts, sorted by X."""
    dets = []
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            om = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break
        rect = om.rect_params
        if rect.width > 0 and rect.height > 0:
            dets.append({
                'x': int(rect.left), 'y': int(rect.top),
                'w': int(rect.width), 'h': int(rect.height),
                'class_id': int(om.class_id),
                'label': str(om.obj_label) if om.obj_label else '',
                'confidence': float(om.confidence) if hasattr(om, 'confidence') else 1.0,
                'tracker_id': int(om.object_id),
            })
        try:
            l_obj = l_obj.next
        except StopIteration:
            break
    # Sort left-to-right by centre X
    dets.sort(key=lambda d: d['x'] + d['w'] // 2)
    return dets


def _find_native_bbox(frame_meta, tracker_id):
    """Return [x,y,w,h] for a specific nvtracker ID in the current frame, or None."""
    if tracker_id == -1:
        return None
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            om = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break
        if int(om.object_id) == tracker_id:
            r = om.rect_params
            return [r.left, r.top, r.width, r.height]
        try:
            l_obj = l_obj.next
        except StopIteration:
            break
    return None


def _reid_scan(frame_rgb, frame_meta, id_history, exclude_id=-1):
    """
    Scan current frame detections for a Re-ID match.
    1. IoU match with last_bbox (spatial re-entry)
    2. Histogram similarity match (appearance re-ID)
    Returns the best matching detection dict, or None.
    """
    log  = logging.getLogger('reid')
    dets = _capture_dets(frame_meta)

    best_det   = None
    best_score = 0.0

    for det in dets:
        if det['tracker_id'] == exclude_id:
            continue
        bbox = [det['x'], det['y'], det['w'], det['h']]

        if id_history.last_bbox is not None:
            iou = get_iou(bbox, id_history.last_bbox)
            if iou >= REID_IOU_THRESHOLD:
                log.info('Re-ID IoU match: native_id=%d iou=%.2f', det['tracker_id'], iou)
                return det

        hist_score = id_history.match_score(frame_rgb, bbox)
        if hist_score > best_score:
            best_score = hist_score
            best_det   = det

    if best_score >= REID_HIST_THRESHOLD and best_det is not None:
        log.info('Re-ID histogram match: native_id=%d score=%.3f',
                 best_det['tracker_id'], best_score)
        return best_det

    return None


def _write_osd_box(batch_meta, frame_meta, bbox, label, color_rgba,
                   border_w=3, comp_id=10):
    """Write one NvDsObjectMeta overlay (box + label) into OSD metadata."""
    sx, sy, sw, sh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    om = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    om.unique_component_id = comp_id
    om.object_id = 0

    r = om.rect_params
    r.left, r.top, r.width, r.height = sx, sy, sw, sh
    r.border_width = border_w
    r.border_color.set(*color_rgba)
    r.has_bg_color = 0

    t = om.text_params
    t.display_text = label
    t.x_offset = int(sx)
    t.y_offset = max(int(sy) - 22, 0)
    t.font_params.font_name = 'Serif'
    t.font_params.font_size = 11
    t.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    t.set_bg_clr = 1
    t.text_bg_clr.set(0.0, 0.35, 0.15, 0.85)

    pyds.nvds_add_obj_meta_to_frame(frame_meta, om, None)


def _write_status_text(batch_meta, frame_meta, text, color_rgba=(1.0, 1.0, 1.0, 1.0)):
    """Write a status label in the top-left corner via NvDsDisplayMeta."""
    try:
        dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        dm.num_labels = 1
        lp = dm.text_params[0]
        lp.display_text = text
        lp.x_offset, lp.y_offset = 20, 20
        lp.font_params.font_name = 'Serif'
        lp.font_params.font_size = 14
        lp.font_params.font_color.set(*color_rgba)
        lp.set_bg_clr = 0
        pyds.nvds_add_display_meta_to_frame(frame_meta, dm)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Core Probe (Phase 10 — OSD Cycling)
# ---------------------------------------------------------------------------

def tracker_probe(pad, info, state):
    if not PYDS_AVAILABLE:
        return Gst.PadProbeReturn.OK

    log = logging.getLogger('probe')
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    try:
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    except Exception as e:
        log.warning('batch_meta: %s', e)
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Lazy pixel extraction
        _rgb_cache = [None]

        def get_rgb():
            if _rgb_cache[0] is not None:
                return _rgb_cache[0]
            try:
                n = pyds.get_nvds_buf_surface(hash(buf), fm.batch_id)
                rgba = np.array(n, copy=True, order='C')
                _rgb_cache[0] = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
                return _rgb_cache[0]
            except Exception as e:
                log.warning('pixel extract: %s', e)
                return None

        # ── A. Always refresh detection list for SELECTING ─────────────────
        dets = _capture_dets(fm)
        state.current_dets = dets     # exposed to key-listener thread (GIL safe)

        # ── B. LOCKED init from key handler (_pending_lock) ───────────────
        if state._pending_lock is not None:
            det = state._pending_lock
            state._pending_lock = None
            bbox = [det['x'], det['y'], det['w'], det['h']]
            rgb  = get_rgb()
            if rgb is not None:
                state.manager.active_trackers.clear()
                state.manager.initialize(rgb, [int(v) for v in bbox],
                                         frame_idx=state.frame_idx)
                state.target_id = det['tracker_id']
                state.id_history.clear()
                state.id_history.update(rgb, bbox, state.frame_idx)
                state.lost_frames    = 0
                state.tracking_state = TrackingState.LOCKED
                log.info('SUTrack initialized from OSD lock: bbox=%s native_id=%d',
                         [round(v) for v in bbox], det['tracker_id'])
                if state.pgie_element and state.pgie_one_shot:
                    state.pgie_element.set_property('interval', 10000)
                    log.info('PGIE sleep mode active.')

        # ── C. IDLE state — show instructions ─────────────────────────────
        if state.tracking_state == TrackingState.IDLE:
            _write_status_text(batch_meta, fm,
                               '[IDLE]  Press "s" to start selecting a target',
                               (0.8, 0.8, 0.8, 1.0))

        # ── C. SELECTING state — OSD candidate cycling ─────────────────────
        elif state.tracking_state == TrackingState.SELECTING:
            if not dets:
                _write_status_text(batch_meta, fm,
                                   '[SELECTING]  No detections yet... waiting',
                                   (1.0, 1.0, 0.0, 1.0))
            else:
                # Clamp index
                state.candidate_index = state.candidate_index % len(dets)
                idx = state.candidate_index

                for i, det in enumerate(dets):
                    bbox = [det['x'], det['y'], det['w'], det['h']]
                    if i == idx:
                        # ── CANDIDATE: thick bright yellow box, prominent label ──
                        _write_osd_box(batch_meta, fm, bbox,
                                       '>>> %s ID:%d <<<' % (det['label'], det['tracker_id']),
                                       (1.0, 1.0, 0.0, 1.0),   # Bright Yellow
                                       border_w=8)
                    # All other detections are hidden during SELECTING
                    # to avoid visual confusion.

                _write_status_text(batch_meta, fm,
                                   '[SELECT %d/%d] n=next  p=prev  l=lock  q=cancel' % (
                                       idx + 1, len(dets)),
                                   (1.0, 1.0, 0.0, 1.0))

        # ── D. LOCKED / SEARCHING / STALE state — tracking ────────────────
        elif state.tracking_state in (TrackingState.LOCKED, TrackingState.SEARCHING):
            rgb = get_rgb()
            if rgb is None:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue

            results = state.manager.update(rgb, state.frame_idx)

            for obj_id, sutrack_bbox in results.items():
                inst = state.manager.active_trackers.get(obj_id)
                conf = getattr(inst, 'confidence', 1.0)

                if state.tracking_state == TrackingState.LOCKED:
                    if conf < LOW_CONF_THRESHOLD:
                        state.lost_frames += 1
                        if state.lost_frames > state.hibernate_after:
                            log.info('Target lost → SEARCHING (conf=%.3f, frames=%d)',
                                     conf, state.lost_frames)
                            state.tracking_state = TrackingState.SEARCHING
                    else:
                        state.lost_frames = 0
                        state.id_history.update(rgb, sutrack_bbox, state.frame_idx)
                        # Keep target_id in sync with native tracker
                        for det in dets:
                            if get_iou(sutrack_bbox,
                                       [det['x'], det['y'], det['w'], det['h']]) > 0.5:
                                if det['tracker_id'] != state.target_id:
                                    log.debug('native ID: %d → %d',
                                              state.target_id, det['tracker_id'])
                                    state.target_id = det['tracker_id']
                                break

                    label = '[LOCKED] (conf=%.2f)' % conf
                    _write_osd_box(batch_meta, fm, sutrack_bbox, label, COLOR_LOCKED)

                elif state.tracking_state == TrackingState.SEARCHING:
                    state.lost_frames += 1

                    match = _reid_scan(rgb, fm, state.id_history,
                                       exclude_id=state.target_id)
                    if match:
                        log.info('Re-ID → re-init on native_id=%d', match['tracker_id'])
                        new_bbox = [match['x'], match['y'], match['w'], match['h']]
                        state.manager.active_trackers.clear()
                        state.manager.initialize(rgb, new_bbox, frame_idx=state.frame_idx)
                        state.target_id      = match['tracker_id']
                        state.id_history.update(rgb, new_bbox, state.frame_idx)
                        state.tracking_state = TrackingState.LOCKED
                        state.lost_frames    = 0
                    elif state.lost_frames > state.hibernate_after:
                        log.info('Re-ID timeout → STALE')
                        state.tracking_state = TrackingState.STALE
                        state.manager.active_trackers.clear()
                    else:
                        label = '[SEARCHING...]'
                        _write_osd_box(batch_meta, fm, sutrack_bbox, label,
                                       COLOR_SEARCHING, border_w=2)

        elif state.tracking_state == TrackingState.STALE:
            _write_status_text(batch_meta, fm,
                               '[TARGET LOST]  Press "s" to select a new target',
                               COLOR_STALE)

        # ── E. Debug: yellow last-known bbox ─────────────────────────────
        if state.debug_boxes and state.id_history.last_bbox is not None:
            try:
                dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                dm.num_rects = 1
                r  = dm.rect_params[0]
                lb = state.id_history.last_bbox
                r.left, r.top, r.width, r.height = lb[0], lb[1], lb[2], lb[3]
                r.border_width = 2
                r.border_color.set(1.0, 1.0, 0.0, 0.8)
                r.has_bg_color = 0
                pyds.nvds_add_display_meta_to_frame(fm, dm)
            except Exception:
                pass

        state.frame_idx   += 1
        state.frame_count += 1

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------

def build_pipeline(state, pipe_cfg):
    Gst.init(None)
    pipeline = Gst.Pipeline.new('sutrack-v3-pipeline')
    width    = int(pipe_cfg.get('width',  1280))
    height   = int(pipe_cfg.get('height', 720))
    inp      = pipe_cfg.get('input_source') or ''

    if inp.startswith('rtsp://') or inp.startswith('http://'):
        src     = Gst.ElementFactory.make('uridecodebin', 'src')
        src.set_property('uri', inp)
        decoder = None
    else:
        src = (Gst.ElementFactory.make('filesrc', 'src') if inp
               else Gst.ElementFactory.make('v4l2src', 'src'))
        if not inp:
            src.set_property('device', '/dev/video0')
        else:
            src.set_property('location', inp)
        decoder = Gst.ElementFactory.make('decodebin', 'decoder')

    nvconv1 = Gst.ElementFactory.make('nvvideoconvert', 'nvconv1')
    caps1   = Gst.ElementFactory.make('capsfilter', 'caps1')
    caps1.set_property('caps', Gst.Caps.from_string(
        'video/x-raw(memory:NVMM),format=NV12'))

    mux = Gst.ElementFactory.make('nvstreammux', 'mux')
    mux.set_property('batch-size', 1)
    mux.set_property('width', width)
    mux.set_property('height', height)
    mux.set_property('batched-push-timeout', 4000000)
    mux.set_property('live-source', 0)

    pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
    pgie.set_property('config-file-path', state.pgie_config_path)
    state.pgie_element = pgie

    tracker = Gst.ElementFactory.make('nvtracker', 'nvtracker')
    lib_candidates = [
        '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so',
        '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvdcf.so',
        '/usr/lib/aarch64-linux-gnu/deepstream/libnvds_nvmultiobjecttracker.so',
    ]
    found_lib = next((p for p in lib_candidates if os.path.exists(p)), lib_candidates[0])
    logging.info('nvtracker library: %s', found_lib)
    tracker.set_property('ll-lib-file', found_lib)

    nvdcf_cfg = os.path.normpath(
        os.path.join(ROOT, pipe_cfg.get('nvdcf_config', 'configs/nvdcf_config.txt')))
    tracker.set_property('ll-config-file', nvdcf_cfg)
    tracker.set_property('tracker-width',  int(pipe_cfg.get('nvtracker_width',  640)))
    tracker.set_property('tracker-height', int(pipe_cfg.get('nvtracker_height', 384)))
    tracker.set_property('gpu-id', 0)

    nvconv_bridge = Gst.ElementFactory.make('nvvideoconvert', 'nvconv_bridge')
    caps_bridge   = Gst.ElementFactory.make('capsfilter', 'caps_bridge')
    caps_bridge.set_property('caps', Gst.Caps.from_string(
        'video/x-raw(memory:NVMM),format=RGBA'))

    osd = Gst.ElementFactory.make('nvdsosd', 'osd')
    osd.set_property('process-mode', 0)

    q_disp    = Gst.ElementFactory.make('queue', 'q_disp')
    sink_disp = (Gst.ElementFactory.make('nv3dsink', 'sink_disp')
                 or Gst.ElementFactory.make('fakesink', 'sink_disp'))
    sink_disp.set_property('sync', False)

    elements = [src, nvconv1, caps1, mux, pgie, tracker,
                nvconv_bridge, caps_bridge, osd, q_disp, sink_disp]
    if decoder:
        elements.append(decoder)
    for el in elements:
        pipeline.add(el)

    def on_pad_added(element, pad, dest):
        if pad.query_caps(None).get_structure(0).get_name().startswith('video/'):
            pad.link(dest.get_static_pad('sink'))

    if decoder:
        src.link(decoder)
        decoder.connect('pad-added', on_pad_added, nvconv1)
    else:
        src.connect('pad-added', on_pad_added, nvconv1)

    nvconv1.link(caps1)
    caps1.get_static_pad('src').link(mux.get_request_pad('sink_0'))
    mux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvconv_bridge)
    nvconv_bridge.link(caps_bridge)
    caps_bridge.link(osd)
    osd.link(q_disp)
    q_disp.link(sink_disp)

    osd.get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER, tracker_probe, state)
    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SUTrack Phase 10: Live OSD Selection + ID Persistence')
    parser.add_argument('--config',      default='deepstream/configs/tracker_config.yml')
    parser.add_argument('--input', '-i', default=None,
                        help='Video file, RTSP URL, or omit for camera')
    parser.add_argument('--no-one-shot', action='store_true',
                        help='Keep PGIE active after selection (no power-save)')
    parser.add_argument('--debug-boxes', action='store_true',
                        help='Show last-known bbox in yellow (drift analysis)')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging(cfg.get('pipeline', {}).get('log_level', 'INFO'))
    log = logging.getLogger('main')

    if not PYDS_AVAILABLE:
        log.error('pyds is required for Phase 10.')
        sys.exit(1)

    pipe_cfg = cfg.get('pipeline', {})
    if args.input:
        pipe_cfg['input_source'] = args.input

    model_cfg   = cfg.get('model', {})
    engine_path = os.path.normpath(
        os.path.join(ROOT, model_cfg.get('engine_path', '../sutrack_fp32.engine')))
    engine  = SUTrackEngine(engine_path=engine_path)
    manager = TrackerManager(engine=engine)

    pgie_cfg_path = os.path.normpath(
        os.path.join(ROOT, pipe_cfg.get('pgie_config', 'configs/pgie_config.txt')))
    pgie_one_shot = pipe_cfg.get('pgie_one_shot', True) and not args.no_one_shot

    state = AppState(
        manager=manager, cfg=cfg,
        headless=pipe_cfg.get('headless', False),
        pgie_config_path=pgie_cfg_path,
        pgie_one_shot=pgie_one_shot,
        debug_boxes=args.debug_boxes)

    pipeline   = build_pipeline(state, pipe_cfg)
    loop       = GLib.MainLoop()
    state.loop = loop

    pipeline.set_state(Gst.State.PLAYING)
    log.info('Pipeline PLAYING in IDLE mode.')
    log.info('Controls (no Enter needed): s=select  n=next  p=prev  l=lock  q=cancel  x=exit')

    # ── Key Reader Thread ────────────────────────────────────────────────────
    key_reader = KeyReader()
    key_reader.start()

    def process_key(ch):
        """Handle a single keypress dispatched from the main-thread poll loop."""
        if ch == 's':
            if state.tracking_state in (TrackingState.IDLE, TrackingState.STALE,
                                        TrackingState.LOCKED, TrackingState.SEARCHING):
                log.info('Entering SELECTING mode.')
                state.candidate_index = 0
                state.tracking_state  = TrackingState.SELECTING

        elif ch == 'n':
            if state.tracking_state == TrackingState.SELECTING:
                state.candidate_index = (state.candidate_index + 1) % max(len(state.current_dets), 1)
                log.info('Candidate → %d/%d', state.candidate_index + 1, len(state.current_dets))

        elif ch == 'p':
            if state.tracking_state == TrackingState.SELECTING:
                state.candidate_index = (state.candidate_index - 1) % max(len(state.current_dets), 1)
                log.info('Candidate → %d/%d', state.candidate_index + 1, len(state.current_dets))

        elif ch in ('l', ' '):
            if state.tracking_state == TrackingState.SELECTING and state.current_dets:
                det = state.current_dets[state.candidate_index % len(state.current_dets)]
                bbox = [det['x'], det['y'], det['w'], det['h']]
                log.info('Locking onto: %s (native_id=%d)', bbox, det['tracker_id'])

                rgb = None  # RGB not available here; probe will initialize on next frame
                # We set a special pending_init dict; probe picks it up next frame
                state._pending_lock = det
                state.tracking_state = TrackingState.LOCKED  # probe checks for _pending_lock

        elif ch == 'q':
            log.info('Tracking cancelled → IDLE')
            state.tracking_state = TrackingState.IDLE
            state.manager.active_trackers.clear()

        elif ch in ('x', '\x03'):    # x or Ctrl+C
            log.info('Exit requested.')
            state.exit_requested = True
            if state.loop:
                state.loop.quit()

    # ── GLib loop in background thread ───────────────────────────────────────
    glib_t = threading.Thread(target=loop.run, daemon=True, name='glib')
    glib_t.start()

    # ── Main thread: poll keys ───────────────────────────────────────────────
    try:
        while not state.exit_requested:
            ch = key_reader.get()
            if ch is not None:
                process_key(ch)
            import time; time.sleep(0.03)    # ~30 Hz poll — negligible CPU
    except KeyboardInterrupt:
        pass
    finally:
        key_reader.stop()
        state.exit_requested = True
        if state.loop:
            state.loop.quit()
        pipeline.set_state(Gst.State.NULL)
        log.info('Stopped. total_frames=%d', state.frame_count)


if __name__ == '__main__':
    main()

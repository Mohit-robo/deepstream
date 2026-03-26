"""
SUTrack Server Application (Phase 12 / V5)
Headless DeepStream pipeline — RTSP output + REST control API.

Pipeline:
  source -> decoder -> nvconv1 -> mux -> pgie -> nvtracker
         -> nvconv_bridge -> osd -> tee
              Branch A: queue -> fakesink          (consume frames, no display)
              Branch B: queue -> nvconv_enc -> nvv4l2h264enc -> rtph264pay -> udpsink

GstRtspServer wraps udpsink:  rtsp://<ip>:8554/sutrack
REST API on port 8000:
  GET  /api/state    -> {"state", "fps", "target_id", "frame_idx"}
  POST /api/command  -> {"action": "select"|"next"|"prev"|"lock"|"cancel"|"click",
                          "x": float, "y": float}  # x,y only for click (normalised 0.0-1.0)

Usage (run from repo root on Jetson):
    python deepstream/apps/deepstream_server_app.py \\
        --config deepstream/configs/tracker_config.yml \\
        --input /path/to/video.mp4
"""# ── EGL / Display bootstrap (must happen before any GStreamer / DeepStream import) ──
# nvinfer on Jetson always uses EGL for NVMM buffer allocation — even in
# headless mode. We need a reachable X11 display.  Strategy:
#
#  1. If DISPLAY is already set and reachable (e.g. running on the Jetson
#     desktop or with X forwarding), use it as-is.
#  2. Otherwise launch Xvfb on :99 so we get a virtual framebuffer with no
#     physical monitor required (works over plain SSH).
#
# This block runs before the LD_PRELOAD re-exec so DISPLAY is inherited by
# the re-launched process.

import os
import sys

def _ensure_display():
    """
    Prepare a usable EGL environment for nvinfer on Jetson.

    X11-forwarded DISPLAY values (e.g. 'localhost:10.0') are NOT backed by a
    local DRM/tegra device, so NVIDIA's EGL stack cannot create EGLImages from
    NVMM buffers — even though xdpyinfo reports the display as reachable.

    Strategy:
      1. Remove any X11-forwarded DISPLAY (remote host != '') so it doesn't
         confuse libEGL.
      2. Set EGL_PLATFORM=surfaceless — uses EGL_EXT_platform_device which
         binds directly to the GPU device, no X11/DRM window system needed.
         This is NVIDIA's recommended approach for headless Jetson inference.
      3. Optionally start Xvfb for tools that still need an X11 socket, but
         EGL_PLATFORM=surfaceless takes priority for nvinfer.
    """
    import os, shutil, subprocess, time

    # Drop X11-forwarding DISPLAY — 'localhost:N' means the display is on a
    # remote machine and the local NVIDIA EGL cannot use it for NVMM operations.
    disp = os.environ.get('DISPLAY', '')
    if disp and (':' in disp) and not disp.startswith(':'):
        # e.g. 'localhost:10.0' or '192.168.x.x:0' — remote, not local
        print('[server] Dropping X11-forwarded DISPLAY=%s (not usable by nvinfer).' % disp,
              flush=True)
        del os.environ['DISPLAY']

    # Always use surfaceless EGL — works on Jetson with no display at all.
    os.environ['EGL_PLATFORM'] = 'surfaceless'
    print('[server] EGL_PLATFORM=surfaceless (headless nvinfer — no display required).',
          flush=True)

_ensure_display()

# DeepStream/NVIDIA hints for headless execution
os.environ['NV_DS_HEADLESS']                = '1'
os.environ['QT_QPA_PLATFORM']               = 'offscreen'
os.environ['USE_NVBUF_SURFACE_FOR_METADATA'] = '1'

# ── LD_PRELOAD self-re-exec (libgomp.so.1 on Jetson) ──────────────────────
_GOMP = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'
if os.path.exists(_GOMP) and 'LD_PRELOAD' not in os.environ:
    os.environ['LD_PRELOAD'] = _GOMP
    if '__SUTRACK_RELAUNCHED' not in os.environ:
        os.environ['__SUTRACK_RELAUNCHED'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)

# ── Normal imports ──────────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import cv2

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
try:
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import GstRtspServer  # noqa: F401 — must be imported before setup_rtsp_server
    _RTSP_LIB_OK = True
except Exception:
    _RTSP_LIB_OK = False
from gi.repository import Gst, GLib

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app_utils import (
    load_yaml, setup_logging, get_local_ip, setup_rtsp_server,
    get_iou, IDHistory, TrackingState,
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
# Constants (identical to desktop_app)
# ---------------------------------------------------------------------------
COLOR_LOCKED    = (0.0, 1.0, 0.5, 1.0)
COLOR_SEARCHING = (1.0, 0.5, 0.0, 1.0)
COLOR_STALE     = (1.0, 0.2, 0.2, 0.9)

LOW_CONF_THRESHOLD  = 0.20
REID_IOU_THRESHOLD  = 0.35
REID_HIST_THRESHOLD = 0.58

HIST_UPDATE_INTERVAL = 20
UI_UPDATE_EVERY      = 5

# ---------------------------------------------------------------------------
# FPSCounter (identical to desktop_app)
# ---------------------------------------------------------------------------

class FPSCounter:
    """Sliding-window FPS counter."""
    def __init__(self, window=60):
        self._times  = []
        self._window = window

    def tick(self):
        now = time.monotonic()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return 0.0 if elapsed <= 0 else (len(self._times) - 1) / elapsed

# ---------------------------------------------------------------------------
# TRTWorkerThread (identical to desktop_app)
# ---------------------------------------------------------------------------

class TRTWorkerThread(threading.Thread):
    """
    Runs SUTrack TRT inference in a dedicated background thread.

    The GStreamer probe submits frames non-blocking; the worker processes them
    and stores results atomically (GIL-safe replacement of dict references).
    All manager mutations happen exclusively here — no data races on
    TrackerInstance.state.
    """

    def __init__(self, manager):
        super().__init__(daemon=True, name='trt-worker')
        self.manager     = manager
        self._lock       = threading.Lock()
        self._pending    = None
        self._trigger    = threading.Event()
        self._running    = True
        self.last_result = {}
        self.last_conf   = {}
        self.start()

    def submit_track(self, rgb, frame_idx):
        with self._lock:
            self._pending = ('track', rgb, frame_idx)
        self._trigger.set()

    def submit_init(self, rgb, bbox, frame_idx):
        with self._lock:
            self._pending = ('init', rgb, bbox, frame_idx)
        self._trigger.set()

    def submit_clear(self):
        with self._lock:
            self._pending = ('clear',)
        self._trigger.set()

    def stop(self):
        self._running = False
        self._trigger.set()

    def run(self):
        log = logging.getLogger('trt-worker')
        while self._running:
            self._trigger.wait(timeout=0.5)
            self._trigger.clear()
            if not self._running:
                break
            with self._lock:
                item = self._pending
                self._pending = None
            if item is None:
                continue
            cmd = item[0]
            if cmd == 'clear':
                self.manager.active_trackers.clear()
                self.last_result = {}
                self.last_conf   = {}
            elif cmd == 'init':
                _, rgb, bbox, frame_idx = item
                self.manager.active_trackers.clear()
                self.manager.initialize(rgb, [int(v) for v in bbox], frame_idx=frame_idx)
                self.last_result = {}
                self.last_conf   = {}
                log.info('Initialized bbox=%s', [round(v) for v in bbox])
            elif cmd == 'track':
                _, rgb, frame_idx = item
                if not self.manager.active_trackers:
                    continue
                try:
                    results  = self.manager.update(rgb, frame_idx)
                    conf_map = {}
                    for oid in results:
                        inst = self.manager.active_trackers.get(oid)
                        conf_map[oid] = getattr(inst, 'confidence', 1.0)
                    self.last_result = results
                    self.last_conf   = conf_map
                except Exception as e:
                    log.warning('Update error: %s', e)

# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, manager, cfg, pgie_config_path, pgie_one_shot=True, debug_boxes=False):
        self.manager     = manager
        self.cfg         = cfg
        self.trt_worker  = TRTWorkerThread(manager)
        self.fps_counter = FPSCounter()
        self.id_history  = IDHistory()

        self.pgie_config_path = pgie_config_path
        self.pgie_one_shot    = pgie_one_shot
        self.pgie_element     = None
        self.use_pgie         = True

        self.tracking_state   = TrackingState.IDLE
        self.target_id        = -1
        self.candidate_index  = 0
        self.lost_frames      = 0
        self.hibernate_after  = int(cfg.get('tracker', {}).get('hibernate_after', 150))
        self.debug_boxes      = debug_boxes

        self.current_dets     = []
        self._pending_lock    = None
        self.auto_lock        = False   # set from --auto-lock: pick largest det automatically
        self.frame_idx        = 0
        self.frame_count      = 0

        # UI link (unused on server, kept for structure)
        self.ui_update_fn     = None

        # V5 addition: pipeline frame dimensions for REST click normalisation
        pipe_cfg = cfg.get('pipeline', {})
        self.frame_width     = int(pipe_cfg.get('width',  1280))
        self.frame_height    = int(pipe_cfg.get('height',  720))
        self.click_bbox_size = 80  # Default bbox size for manual init


# ---------------------------------------------------------------------------
# OSD / Probe Helpers (identical to desktop_app)
# ---------------------------------------------------------------------------

def _capture_dets(frame_meta):
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
                'x': int(rect.left),  'y': int(rect.top),
                'w': int(rect.width), 'h': int(rect.height),
                'class_id':   int(om.class_id),
                'label':      str(om.obj_label) if om.obj_label else '',
                'confidence': float(om.confidence) if hasattr(om, 'confidence') else 1.0,
                'tracker_id': int(om.object_id),
            })
        try:
            l_obj = l_obj.next
        except StopIteration:
            break
    dets.sort(key=lambda d: d['x'] + d['w'] // 2)
    return dets


def _write_osd_box(batch_meta, frame_meta, bbox, label, color_rgba, border_w=3):
    try:
        om = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
        om.unique_component_id = 10
        om.object_id = 0
        r = om.rect_params
        r.left, r.top, r.width, r.height = (
            float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        r.border_width = border_w
        r.border_color.set(*color_rgba)
        r.has_bg_color = 0
        t = om.text_params
        t.display_text = label
        t.x_offset = int(bbox[0])
        t.y_offset = max(int(bbox[1]) - 22, 0)
        t.font_params.font_name = 'Serif'
        t.font_params.font_size = 11
        t.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        t.set_bg_clr = 1
        t.text_bg_clr.set(0.0, 0.35, 0.15, 0.85)
        pyds.nvds_add_obj_meta_to_frame(frame_meta, om, None)
    except Exception:
        pass


def _write_status_text(batch_meta, frame_meta, text, color_rgba=(1.0, 1.0, 1.0, 1.0)):
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
# Core Probe (identical to desktop_app; ui_update_fn guard handles no-GTK)
# ---------------------------------------------------------------------------

def tracker_probe(pad, info, state):
    if not PYDS_AVAILABLE:
        return Gst.PadProbeReturn.OK

    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    try:
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    except Exception:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        state.fps_counter.tick()

        _rgb_cache = [None]

        def get_rgb():
            if _rgb_cache[0] is not None:
                return _rgb_cache[0]
            try:
                n    = pyds.get_nvds_buf_surface(hash(buf), fm.batch_id)
                rgba = np.array(n, copy=True, order='C')
                _rgb_cache[0] = rgba[:, :, :3]
                return _rgb_cache[0]
            except Exception:
                return None

        dets = _capture_dets(fm)
        state.current_dets = dets

        # Consume pending lock queued by REST handler
        if state._pending_lock is not None:
            det = state._pending_lock
            state._pending_lock = None
            rgb = get_rgb()
            if rgb is not None:
                bbox = [det['x'], det['y'], det['w'], det['h']]
                state.trt_worker.submit_init(rgb, bbox, state.frame_idx)
                state.target_id = det['tracker_id']
                state.id_history.clear()
                state.id_history.update(rgb, bbox, state.frame_idx)
                state.lost_frames    = 0
                state.tracking_state = TrackingState.LOCKED
                logging.getLogger('probe').info(
                    'SUTrack init queued: bbox=%s native_id=%d', bbox, det['tracker_id'])
                if state.pgie_element and state.pgie_one_shot:
                    state.pgie_element.set_property('interval', 10000)

        if state.tracking_state == TrackingState.IDLE:
            if state.use_pgie and dets:
                if state.auto_lock:
                    # Automatically initialize on the largest detected object
                    largest = max(dets, key=lambda d: d['w'] * d['h'])
                    state._pending_lock = largest
                    logging.getLogger('probe').info(
                        'Auto-lock: %s area=%d', largest['label'],
                        largest['w'] * largest['h'])
                else:
                    # Show all detections; user clicks to select target
                    for det in dets:
                        _write_osd_box(
                            batch_meta, fm,
                            [det['x'], det['y'], det['w'], det['h']],
                            '%s' % det['label'],
                            (0.7, 0.7, 1.0, 0.9), border_w=2)
                    _write_status_text(
                        batch_meta, fm,
                        '[%d DETECTED]  Click on target in client' % len(dets),
                        (0.7, 0.9, 1.0, 1.0))
            else:
                msg = ('[WAITING]  No detections yet...' if state.use_pgie
                       else '[MANUAL MODE]  Click on stream point to track')
                _write_status_text(batch_meta, fm, msg, (0.8, 0.8, 0.8, 1.0))

        elif state.tracking_state == TrackingState.SELECTING:
            if not dets:
                _write_status_text(batch_meta, fm,
                                   '[SELECTING]  No detections yet...',
                                   (1.0, 1.0, 0.0, 1.0))
            else:
                state.candidate_index = state.candidate_index % len(dets)
                idx = state.candidate_index
                for i, det in enumerate(dets):
                    bbox = [det['x'], det['y'], det['w'], det['h']]
                    if i == idx:
                        _write_osd_box(
                            batch_meta, fm, bbox,
                            '>>> %s ID:%d <<<' % (det['label'], det['tracker_id']),
                            (1.0, 1.0, 0.0, 1.0), border_w=8)
                _write_status_text(
                    batch_meta, fm,
                    '[SELECT %d/%d]  Send next/prev/lock command' % (idx + 1, len(dets)),
                    (1.0, 1.0, 0.0, 1.0))

        elif state.tracking_state in (TrackingState.LOCKED, TrackingState.SEARCHING):
            rgb = get_rgb()
            if rgb is not None:
                state.trt_worker.submit_track(rgb, state.frame_idx)

            results  = state.trt_worker.last_result
            conf_map = state.trt_worker.last_conf

            for obj_id, bbox in results.items():
                conf = conf_map.get(obj_id, 1.0)

                if state.tracking_state == TrackingState.LOCKED:
                    if conf < LOW_CONF_THRESHOLD:
                        state.lost_frames += 1
                        if state.lost_frames > state.hibernate_after:
                            state.tracking_state = TrackingState.SEARCHING
                    else:
                        state.lost_frames = 0
                        if rgb is not None and state.frame_idx % HIST_UPDATE_INTERVAL == 0:
                            state.id_history.update(rgb, bbox, state.frame_idx)
                    _write_osd_box(batch_meta, fm, bbox,
                                   '[LOCKED] %.2f' % conf, COLOR_LOCKED, border_w=4)

                elif state.tracking_state == TrackingState.SEARCHING:
                    state.lost_frames += 1
                    if rgb is not None:
                        best_det, best_score = None, 0.0
                        for det in dets:
                            d_bbox = [det['x'], det['y'], det['w'], det['h']]
                            if state.id_history.last_bbox and \
                               get_iou(d_bbox, state.id_history.last_bbox) >= REID_IOU_THRESHOLD:
                                best_det, best_score = det, 1.0
                                break
                            score = state.id_history.match_score(rgb, d_bbox)
                            if score > best_score:
                                best_det, best_score = det, score
                        if best_det and best_score >= REID_HIST_THRESHOLD:
                            new_bbox = [best_det['x'], best_det['y'],
                                        best_det['w'], best_det['h']]
                            state.trt_worker.submit_init(rgb, new_bbox, state.frame_idx)
                            state.target_id = best_det['tracker_id']
                            state.id_history.update(rgb, new_bbox, state.frame_idx)
                            state.tracking_state = TrackingState.LOCKED
                            state.lost_frames    = 0
                        elif state.lost_frames > state.hibernate_after:
                            state.tracking_state = TrackingState.STALE
                            state.trt_worker.submit_clear()
                        else:
                            _write_osd_box(batch_meta, fm, bbox,
                                           '[SEARCHING...]', COLOR_SEARCHING, border_w=2)

        elif state.tracking_state == TrackingState.STALE:
            if state.use_pgie and dets:
                for det in dets:
                    _write_osd_box(
                        batch_meta, fm,
                        [det['x'], det['y'], det['w'], det['h']],
                        '%s' % det['label'],
                        (1.0, 0.6, 0.2, 0.9), border_w=2)
                _write_status_text(
                    batch_meta, fm,
                    '[TARGET LOST]  %d detected — click to re-acquire' % len(dets),
                    COLOR_STALE)
            else:
                _write_status_text(batch_meta, fm,
                                   '[TARGET LOST]  Click on target to re-acquire',
                                   COLOR_STALE)

        state.frame_idx   += 1
        state.frame_count += 1

        # No GTK on server; ui_update_fn stays None — guard prevents GLib.idle_add
        if state.frame_idx % UI_UPDATE_EVERY == 0 and state.ui_update_fn is not None:
            GLib.idle_add(state.ui_update_fn)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

# ---------------------------------------------------------------------------
# REST API Thread
# ---------------------------------------------------------------------------

class RestAPIThread(threading.Thread):
    """
    Lightweight REST daemon for V5 remote control.

    The inner _Handler class is defined inside run() so it closes over `state`
    via the outer scope — the cleanest way to share state without class attrs.
    BaseHTTPRequestHandler processes one request at a time (single-thread server).
    Request rates are low (polling + occasional commands) so this is sufficient.
    """

    def __init__(self, state, host='0.0.0.0', port=8000):
        super().__init__(daemon=True, name='rest-api')
        self._state  = state
        self._host   = host
        self._port   = port
        self._server = None

    def run(self):
        state = self._state
        log   = logging.getLogger('rest-api')

        class _Handler(BaseHTTPRequestHandler):

            def log_message(self, fmt, *args):
                log.debug('HTTP ' + fmt, *args)

            def _send_json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header('Content-Type',                'application/json')
                self.send_header('Content-Length',              str(len(body)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == '/api/state':
                    self._send_json(200, {
                        'state':     state.tracking_state.value,
                        'fps':       round(state.fps_counter.fps, 1),
                        'target_id': state.target_id,
                        'frame_idx': state.frame_idx,
                    })
                else:
                    self._send_json(404, {'error': 'not found'})

            def do_POST(self):
                if self.path != '/api/command':
                    self._send_json(404, {'error': 'not found'})
                    return
                length = int(self.headers.get('Content-Length', 0))
                try:
                    cmd = json.loads(self.rfile.read(length))
                except Exception as e:
                    self._send_json(400, {'error': str(e)})
                    return

                action = cmd.get('action', '')
                log.info('Command: %s', cmd)

                if action == 'select':
                    state.candidate_index = 0
                    state.tracking_state  = TrackingState.SELECTING

                elif action == 'next':
                    if state.tracking_state == TrackingState.SELECTING:
                        n = max(len(state.current_dets), 1)
                        state.candidate_index = (state.candidate_index + 1) % n

                elif action == 'prev':
                    if state.tracking_state == TrackingState.SELECTING:
                        n = max(len(state.current_dets), 1)
                        state.candidate_index = (state.candidate_index - 1) % n

                elif action == 'lock':
                    if (state.tracking_state == TrackingState.SELECTING
                            and state.current_dets):
                        idx = state.candidate_index % len(state.current_dets)
                        state._pending_lock  = state.current_dets[idx]
                        state.tracking_state = TrackingState.LOCKED

                elif action == 'cancel':
                    state.tracking_state = TrackingState.IDLE
                    state.trt_worker.submit_clear()

                elif action == 'click':
                    # Normalised (0.0-1.0) click; map to pixel space
                    nx = float(cmd.get('x', 0.5))
                    ny = float(cmd.get('y', 0.5))
                    px = nx * state.frame_width
                    py = ny * state.frame_height
                    dets = state.current_dets

                    if dets:
                        # Find nearest detection centroid — latency-insensitive spatial matching.
                        best = min(
                            dets,
                            key=lambda d: math.hypot(
                                d['x'] + d['w'] / 2.0 - px,
                                d['y'] + d['h'] / 2.0 - py))
                        state._pending_lock  = best
                        state.tracking_state = TrackingState.LOCKED
                        log.info('Click lock: (%.3f, %.3f) -> nearest det %d',
                                 nx, ny, best['tracker_id'])
                    else:
                        # No detections (PGIE disabled or no targets); implement Manual Init.
                        sz = state.click_bbox_size
                        manual_det = {
                            'x': px - sz/2.0,
                            'y': py - sz/2.0,
                            'w': sz,
                            'h': sz,
                            'tracker_id': -1,
                            'label': 'manual'
                        }
                        state._pending_lock  = manual_det
                        state.tracking_state = TrackingState.LOCKED
                        log.info('Manual Click lock: (%.3f, %.3f) -> seed bbox at focal point',
                                 nx, ny)

                else:
                    self._send_json(400, {'error': 'unknown action: %s' % action})
                    return

                self._send_json(200, {'ok': True})

        self._server = HTTPServer((self._host, self._port), _Handler)
        log.info('REST API listening on %s:%d', self._host, self._port)
        self._server.serve_forever()

    def stop(self):
        if self._server:
            self._server.shutdown()

# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------

def build_pipeline(state, pipe_cfg):
    Gst.init(None)
    pipeline = Gst.Pipeline.new('sutrack-server')

    width  = int(pipe_cfg.get('width',  1280))
    height = int(pipe_cfg.get('height', 720))
    inp    = pipe_cfg.get('input_source') or ''

    # Source
    if inp.startswith('rtsp://') or inp.startswith('http://'):
        src     = Gst.ElementFactory.make('uridecodebin', 'src')
        src.set_property('uri', inp)
        decoder = None
    else:
        src = (Gst.ElementFactory.make('filesrc', 'src') if inp
               else Gst.ElementFactory.make('v4l2src', 'src'))
        if inp:
            src.set_property('location', inp)
        decoder = Gst.ElementFactory.make('decodebin', 'decoder')

    nvconv1 = Gst.ElementFactory.make('nvvideoconvert', 'nvconv1')
    nvconv1.set_property('nvbuf-memory-type', 4)  # 4 = Surface Array (Headless-safe)
    caps1   = Gst.ElementFactory.make('capsfilter', 'caps1')
    caps1.set_property('caps', Gst.Caps.from_string(
        'video/x-raw(memory:NVMM),format=NV12'))

    mux = Gst.ElementFactory.make('nvstreammux', 'mux')
    mux.set_property('batch-size', 1)
    mux.set_property('width', width)
    mux.set_property('height', height)
    mux.set_property('batched-push-timeout', 4000000)
    mux.set_property('live-source', 0)
    mux.set_property('nvbuf-memory-type', 4)
  # Use CUDA Device memory

    use_pgie = getattr(state, 'use_pgie', True)
    if use_pgie:
        pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
        pgie.set_property('config-file-path', state.pgie_config_path)
        state.pgie_element = pgie
    else:
        pgie = None

    tracker = Gst.ElementFactory.make('nvtracker', 'nvtracker')
    lib_candidates = [
        '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so',
        '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvdcf.so',
    ]
    lib = next((p for p in lib_candidates if os.path.exists(p)), lib_candidates[0])
    tracker.set_property('ll-lib-file', lib)
    nvdcf_cfg = os.path.normpath(
        os.path.join(ROOT, pipe_cfg.get('nvdcf_config', 'configs/nvdcf_config.txt')))
    tracker.set_property('ll-config-file', nvdcf_cfg)
    tracker.set_property('tracker-width',  int(pipe_cfg.get('nvtracker_width',  640)))
    tracker.set_property('tracker-height', int(pipe_cfg.get('nvtracker_height', 384)))
    tracker.set_property('gpu-id', 0)

    # RGBA bridge for OSD (same as desktop_app)
    nvconv_bridge = Gst.ElementFactory.make('nvvideoconvert', 'nvconv_bridge')
    nvconv_bridge.set_property('nvbuf-memory-type', 4)
    caps_bridge   = Gst.ElementFactory.make('capsfilter', 'caps_bridge')
    caps_bridge.set_property('caps', Gst.Caps.from_string(
        'video/x-raw(memory:NVMM),format=RGBA'))

    osd = Gst.ElementFactory.make('nvdsosd', 'osd')
    osd.set_property('process-mode', 0)

    # Tee: Branch A (fakesink — drop frames) + Branch B (RTSP H.264 encode)
    tee      = Gst.ElementFactory.make('tee',     'tee')
    q_fake   = Gst.ElementFactory.make('queue',   'q_fake')
    fakesink = Gst.ElementFactory.make('fakesink', 'fakesink')
    fakesink.set_property('sync', False)

    # RTSP encode branch
    udp_port     = int(pipe_cfg.get('rtsp_udp_port', 5400))
    rtsp_bitrate = int(pipe_cfg.get('rtsp_bitrate',  4000000))

    q_rtsp     = Gst.ElementFactory.make('queue',          'q_rtsp')
    nvconv_enc = Gst.ElementFactory.make('nvvideoconvert', 'nvconv_enc')
    nvconv_enc.set_property('nvbuf-memory-type', 4)
    encoder    = Gst.ElementFactory.make('nvv4l2h264enc',  'encoder')
    encoder.set_property('bitrate',        rtsp_bitrate)
    encoder.set_property('iframeinterval', 30)
    pay        = Gst.ElementFactory.make('rtph264pay', 'pay')
    pay.set_property('config-interval', 1)
    pay.set_property('pt', 96)
    udpsink    = Gst.ElementFactory.make('udpsink', 'udpsink')
    udpsink.set_property('host', '127.0.0.1')
    udpsink.set_property('port', udp_port)
    udpsink.set_property('sync', False)

    all_el = [
        src, nvconv1, caps1, mux, tracker,
        nvconv_bridge, caps_bridge, osd, tee,
        q_fake, fakesink,
        q_rtsp, nvconv_enc, encoder, pay, udpsink,
    ]
    if pgie:
        all_el.append(pgie)
    if decoder:
        all_el.append(decoder)
    for el in all_el:
        pipeline.add(el)

    def on_pad_added(element, pad, dest):
        caps = pad.query_caps(None)
        if caps and caps.get_structure(0).get_name().startswith('video/'):
            pad.link(dest.get_static_pad('sink'))

    if decoder:
        src.link(decoder)
        decoder.connect('pad-added', on_pad_added, nvconv1)
    else:
        src.connect('pad-added', on_pad_added, nvconv1)

    nvconv1.link(caps1)
    caps1.get_static_pad('src').link(mux.get_request_pad('sink_0'))
    if pgie:
        mux.link(pgie)
        pgie.link(tracker)
    else:
        mux.link(tracker)
    tracker.link(nvconv_bridge)
    nvconv_bridge.link(caps_bridge)
    caps_bridge.link(osd)
    osd.link(tee)

    # Branch A: fakesink
    tee.get_request_pad('src_%u').link(q_fake.get_static_pad('sink'))
    q_fake.link(fakesink)

    # Branch B: RTSP encode
    tee.get_request_pad('src_%u').link(q_rtsp.get_static_pad('sink'))
    q_rtsp.link(nvconv_enc)
    nvconv_enc.link(encoder)
    encoder.link(pay)
    pay.link(udpsink)

    osd.get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER, tracker_probe, state)

    return pipeline

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SUTrack Server App (Phase 12 / V5)')
    parser.add_argument('--config',      default='deepstream/configs/tracker_config.yml')
    parser.add_argument('--input', '-i', default=None,
                        help='Video file or RTSP URL (overrides config)')
    parser.add_argument('--api-port',    type=int, default=8000,
                        help='REST API port (default: 8000)')
    parser.add_argument('--no-pgie',     action='store_true',
                        help='Skip PGIE detector (fallback: manual click initialisation)')
    parser.add_argument('--auto-lock',   action='store_true',
                        help='Automatically lock to the largest PGIE detection (no user input)')
    parser.add_argument('--loop',        action='store_true',
                        help='Loop file input indefinitely (seek to start on EOS)')
    parser.add_argument('--no-one-shot', action='store_true',
                        help='Keep PGIE running every frame after selection')
    parser.add_argument('--debug-boxes', action='store_true',
                        help='Show raw nvtracker boxes in OSD')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging(cfg.get('pipeline', {}).get('log_level', 'INFO'))
    log = logging.getLogger('main')

    if not PYDS_AVAILABLE:
        log.error('pyds not found — cannot run server (requires DeepStream).')
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
        pgie_config_path=pgie_cfg_path,
        pgie_one_shot=pgie_one_shot,
        debug_boxes=args.debug_boxes)
    state.use_pgie   = not args.no_pgie
    state.auto_lock  = args.auto_lock

    pipeline = build_pipeline(state, pipe_cfg)

    # RTSP server (wraps the udpsink output as an RTSP endpoint)
    rtsp_port = int(pipe_cfg.get('rtsp_port',     8554))
    udp_port  = int(pipe_cfg.get('rtsp_udp_port', 5400))
    rtsp_path = pipe_cfg.get('rtsp_path', '/sutrack')

    rtsp_server = setup_rtsp_server(rtsp_port, udp_port, rtsp_path)
    local_ip    = get_local_ip()
    if rtsp_server:
        log.info('RTSP stream:  rtsp://%s:%d%s', local_ip, rtsp_port, rtsp_path)
    else:
        log.warning('GstRtspServer unavailable — RTSP output disabled.')

    log.info('REST API:     http://%s:%d/api/', local_ip, args.api_port)
    log.info('Controls:     POST /api/command {action: select|next|prev|lock|cancel|click}')

    # Start REST API daemon
    rest = RestAPIThread(state, host='0.0.0.0', port=args.api_port)
    rest.start()

    # GLib main loop — drives GstRtspServer and GStreamer bus
    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    def on_eos(b, m):
        if args.loop:
            log.info('EOS — looping file from start (restarting pipeline).')
            pipeline.set_state(Gst.State.READY)
            state.frame_idx = 0
            state.id_history.clear()
            state.candidate_index = 0
            state.current_dets = []
            pipeline.set_state(Gst.State.PLAYING)
        else:
            log.info('EOS — stream finished.')
            loop.quit()

    def on_error(b, m):
        err, dbg = m.parse_error()
        log.error('GStreamer error: %s | %s', err, dbg)
        loop.quit()

    bus.connect('message::eos',   on_eos)
    bus.connect('message::error', on_error)

    log.info('Starting server pipeline...')
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        log.error('Pipeline failed to enter PLAYING state.')
        sys.exit(1)

    try:
        loop.run()
    except KeyboardInterrupt:
        log.info('Interrupted.')
    finally:
        pipeline.set_state(Gst.State.NULL)
        state.trt_worker.stop()
        rest.stop()
        log.info('Shutdown. total_frames=%d', state.frame_count)


if __name__ == '__main__':
    main()

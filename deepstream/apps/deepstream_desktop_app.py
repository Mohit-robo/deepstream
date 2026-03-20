"""
SUTrack Desktop Application (Phase 10)
GTK3-wrapped DeepStream pipeline with hardware-accelerated video embedding.

Layout:
  ┌─────────────────────────────────────┐
  │         Video Canvas (nv3dsink)     │
  ├──────────────────────────────────────│
  │  FPS: 28.5  │  State: IDLE          │
  ├──────────────────────────────────────│
  │ [Select] [◀ Prev] [Next ▶] [Lock] [Cancel] │
  └─────────────────────────────────────┘

Performance notes:
  The main FPS bottleneck in LOCKED mode is TRT inference (~30 ms/frame on Jetson)
  blocking the GStreamer streaming thread. TRTWorkerThread decouples inference from
  the pipeline: the probe submits frames non-blocking and reads the last result,
  letting the pipeline run at source rate (60 FPS) while TRT runs independently.

Usage:
    export DISPLAY=:0
    python deepstream/apps/deepstream_desktop_app.py \\
        --config deepstream/configs/tracker_config.yml \\
        --input /path/to/video.mp4

Key issue on Jetson:
  libgomp.so.1 must be preloaded before any GLib/Python-C extension is loaded.
  Set LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 or set it inline below.
"""

# ── Must be first — before any C-extension import ─────────────────────────
import os
import sys
_GOMP = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'
if os.path.exists(_GOMP) and 'LD_PRELOAD' not in os.environ:
    os.environ['LD_PRELOAD'] = _GOMP
    # Re-exec with LD_PRELOAD so the dynamic linker picks it up
    if '__SUTRACK_RELAUNCHED' not in os.environ:
        os.environ['__SUTRACK_RELAUNCHED'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)

# ── Normal imports ─────────────────────────────────────────────────────────
import argparse
import logging
import threading
import time
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import cv2

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GLib', '2.0')
gi.require_version('Gtk', '3.0')
gi.require_version('GdkX11', '3.0')
from gi.repository import Gst, GstVideo, GLib, Gtk, GdkX11, Gdk

# ---------------------------------------------------------------------------
# Project Path Setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app_utils import (
    load_yaml, setup_logging,
    get_iou,
    IDHistory, TrackingState,
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
# Constants
# ---------------------------------------------------------------------------
COLOR_LOCKED    = (0.0, 1.0, 0.5, 1.0)
COLOR_SEARCHING = (1.0, 0.5, 0.0, 1.0)
COLOR_STALE     = (1.0, 0.2, 0.2, 0.9)

LOW_CONF_THRESHOLD  = 0.20
REID_IOU_THRESHOLD  = 0.35
REID_HIST_THRESHOLD = 0.58

# Performance tuning
HIST_UPDATE_INTERVAL = 20   # update appearance histogram every N LOCKED frames
UI_UPDATE_EVERY      = 5    # refresh GTK labels every N frames (~12 Hz at 60 FPS)

# ---------------------------------------------------------------------------
# FPS Counter
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
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed

# ---------------------------------------------------------------------------
# Async TRT Worker
# ---------------------------------------------------------------------------

class TRTWorkerThread(threading.Thread):
    """
    Runs SUTrack TRT inference in a dedicated background thread.

    Design:
      - GStreamer probe calls submit_track(rgb, frame_idx) — non-blocking.
        If the worker is still processing the previous frame, the new frame
        overwrites the pending slot (always use the freshest frame available).
      - Probe reads last_result / last_conf for OSD (1-2 frame lag — imperceptible).
      - All manager mutations (initialize / update / clear) happen exclusively here,
        eliminating any data-race on TrackerInstance.state.

    Thread safety:
      - _lock protects _pending (written by probe, read by worker).
      - last_result / last_conf are replaced atomically (Python dict assignment
        is GIL-safe; probe reads them without holding any lock).
    """

    def __init__(self, manager):
        super().__init__(daemon=True, name='trt-worker')
        self.manager   = manager
        self._lock     = threading.Lock()
        self._pending  = None       # ('track', rgb, idx) | ('init', rgb, bbox, idx) | ('clear',)
        self._trigger  = threading.Event()
        self._running  = True
        # Results — replaced atomically, GIL-safe reads from probe
        self.last_result = {}       # {obj_id: [x, y, w, h]}
        self.last_conf   = {}       # {obj_id: float}
        self.start()

    # ── Submission API (called from GStreamer streaming thread) ───────────

    def submit_track(self, rgb, frame_idx):
        """Non-blocking. Overwrites pending slot if worker is still busy."""
        with self._lock:
            self._pending = ('track', rgb, frame_idx)
        self._trigger.set()

    def submit_init(self, rgb, bbox, frame_idx):
        """Queue a tracker initialization (takes priority over pending track)."""
        with self._lock:
            self._pending = ('init', rgb, bbox, frame_idx)
        self._trigger.set()

    def submit_clear(self):
        """Clear all active trackers (user cancelled)."""
        with self._lock:
            self._pending = ('clear',)
        self._trigger.set()

    def stop(self):
        self._running = False
        self._trigger.set()

    # ── Worker loop ───────────────────────────────────────────────────────

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
                self.manager.initialize(rgb, [int(v) for v in bbox],
                                        frame_idx=frame_idx)
                self.last_result = {}
                self.last_conf   = {}
                log.info('TRT worker: initialized bbox=%s', [round(v) for v in bbox])

            elif cmd == 'track':
                _, rgb, frame_idx = item
                if not self.manager.active_trackers:
                    continue
                try:
                    results  = self.manager.update(rgb, frame_idx)
                    conf_map = {}
                    for obj_id in results:
                        inst = self.manager.active_trackers.get(obj_id)
                        conf_map[obj_id] = getattr(inst, 'confidence', 1.0)
                    # Atomic replacement — GIL-safe reads from probe
                    self.last_result = results
                    self.last_conf   = conf_map
                except Exception as e:
                    log.warning('TRT update error: %s', e)

# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, manager, cfg, pgie_config_path, pgie_one_shot=True, debug_boxes=False):
        self.manager          = manager
        self.cfg              = cfg
        self.pgie_config_path = pgie_config_path
        self.pgie_one_shot    = pgie_one_shot
        self.pgie_element     = None

        self.tracking_state   = TrackingState.IDLE
        self.target_id        = -1
        self.id_history       = IDHistory()

        self.frame_idx   = 0
        self.frame_count = 0

        self.current_dets    = []
        self.candidate_index = 0
        self._pending_lock   = None

        self.lost_frames     = 0
        self.hibernate_after = int(cfg.get('tracker', {}).get('hibernate_after', 150))

        self.debug_boxes = debug_boxes
        self.fps_counter = FPSCounter()
        self.ui_update_fn = None

        # Async TRT worker — owns all manager mutations
        self.trt_worker = TRTWorkerThread(manager)

# ---------------------------------------------------------------------------
# OSD / Probe Helpers
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
    dets.sort(key=lambda d: d['x'] + d['w'] // 2)
    return dets


def _write_osd_box(batch_meta, frame_meta, bbox, label, color_rgba, border_w=3):
    try:
        om = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
        om.unique_component_id = 10
        om.object_id = 0
        r = om.rect_params
        r.left, r.top, r.width, r.height = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
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
# Core Probe
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

        # Lazy pixel extraction — only when needed (avoids NVMM copy in IDLE/SELECTING)
        _rgb_cache = [None]

        def get_rgb():
            if _rgb_cache[0] is not None:
                return _rgb_cache[0]
            try:
                n    = pyds.get_nvds_buf_surface(hash(buf), fm.batch_id)
                rgba = np.array(n, copy=True, order='C')
                _rgb_cache[0] = rgba[:, :, :3]   # RGBA→RGB via slice (no extra alloc)
                return _rgb_cache[0]
            except Exception:
                return None

        dets = _capture_dets(fm)
        state.current_dets = dets

        # ── B. Consume pending lock (user pressed Lock button) ────────────
        if state._pending_lock is not None:
            det = state._pending_lock
            state._pending_lock = None
            rgb = get_rgb()
            if rgb is not None:
                bbox = [det['x'], det['y'], det['w'], det['h']]
                # Async: worker does clear + initialize; probe continues immediately
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

        # ── State rendering ───────────────────────────────────────────────

        if state.tracking_state == TrackingState.IDLE:
            _write_status_text(batch_meta, fm,
                               '[IDLE]  Click "Select" to choose a target',
                               (0.8, 0.8, 0.8, 1.0))

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
                        _write_osd_box(batch_meta, fm, bbox,
                                       '>>> %s ID:%d <<<' % (det['label'], det['tracker_id']),
                                       (1.0, 1.0, 0.0, 1.0), border_w=8)
                _write_status_text(batch_meta, fm,
                                   '[SELECT %d/%d]  Next / Prev / Lock' % (idx + 1, len(dets)),
                                   (1.0, 1.0, 0.0, 1.0))

        elif state.tracking_state in (TrackingState.LOCKED, TrackingState.SEARCHING):
            # ── Submit frame to TRT worker (non-blocking) ─────────────────
            rgb = get_rgb()
            if rgb is not None:
                state.trt_worker.submit_track(rgb, state.frame_idx)

            # ── Read last TRT result for OSD (1-2 frame lag, imperceptible) ──
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
                        # Throttled histogram update — Re-ID template stays fresh
                        # without computing a histogram on every single frame
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
            _write_status_text(batch_meta, fm,
                               '[TARGET LOST]  Click "Select" to choose a new target',
                               COLOR_STALE)

        state.frame_idx   += 1
        state.frame_count += 1

        # Throttle GTK label updates — every UI_UPDATE_EVERY frames is plenty
        if state.frame_idx % UI_UPDATE_EVERY == 0 and state.ui_update_fn is not None:
            GLib.idle_add(state.ui_update_fn)

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
    pipeline = Gst.Pipeline.new('sutrack-desktop')

    width  = int(pipe_cfg.get('width',  1280))
    height = int(pipe_cfg.get('height', 720))
    inp    = pipe_cfg.get('input_source') or ''

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
    ]
    lib = next((p for p in lib_candidates if os.path.exists(p)), lib_candidates[0])
    tracker.set_property('ll-lib-file', lib)
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

    q_disp = Gst.ElementFactory.make('queue', 'q_disp')

    # Prefer nv3dsink → nveglglessink → glimagesink (all support GstVideoOverlay on Jetson)
    sink = None
    for sink_name in ('nv3dsink', 'nveglglessink', 'glimagesink'):
        sink = Gst.ElementFactory.make(sink_name, 'sink')
        if sink:
            logging.getLogger('pipeline').info('Using sink: %s', sink_name)
            break
    if sink is None:
        sink = Gst.ElementFactory.make('fakesink', 'sink')
    sink.set_property('sync', False)

    elements = [src, nvconv1, caps1, mux, pgie, tracker,
                nvconv_bridge, caps_bridge, osd, q_disp, sink]
    if decoder:
        elements.append(decoder)
    for el in elements:
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
    mux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvconv_bridge)
    nvconv_bridge.link(caps_bridge)
    caps_bridge.link(osd)
    osd.link(q_disp)
    q_disp.link(sink)

    osd.get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER, tracker_probe, state)

    return pipeline, sink

# ---------------------------------------------------------------------------
# GTK3 Main Window
# ---------------------------------------------------------------------------

class SUTrackApp(Gtk.Window):

    def __init__(self, state, pipeline, sink, video_width=1280, video_height=720):
        super().__init__(title='SUTrack — DeepStream Tracker')
        self.state    = state
        self.pipeline = pipeline
        self.sink     = sink
        self.log      = logging.getLogger('app')
        self._xid     = None

        self.set_default_size(video_width, video_height + 90)
        self.connect('destroy', self._on_close)
        self.connect('key-press-event', self._on_key)

        # ── Video Canvas ──────────────────────────────────────────────────
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(vbox)

        self.video_canvas = Gtk.DrawingArea()
        self.video_canvas.set_size_request(video_width, video_height)
        # Black background while pipeline warms up
        self.video_canvas.override_background_color(
            Gtk.StateFlags.NORMAL, Gdk.RGBA(0, 0, 0, 1))
        vbox.pack_start(self.video_canvas, True, True, 0)

        # ── Info Bar ──────────────────────────────────────────────────────
        info_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        info_bar.set_margin_start(12)
        info_bar.set_margin_end(12)
        info_bar.set_margin_top(4)
        info_bar.set_margin_bottom(2)
        vbox.pack_start(info_bar, False, False, 0)

        self.fps_label   = Gtk.Label(label='FPS: --')
        self.state_label = Gtk.Label(label='State: IDLE')
        self.fps_label.set_xalign(0)
        self.state_label.set_xalign(0)
        info_bar.pack_start(self.fps_label,   False, False, 0)
        info_bar.pack_start(Gtk.Label(label=' | '), False, False, 0)
        info_bar.pack_start(self.state_label, False, False, 0)

        # ── Control Buttons ───────────────────────────────────────────────
        btn_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        btn_bar.set_margin_start(12)
        btn_bar.set_margin_end(12)
        btn_bar.set_margin_top(4)
        btn_bar.set_margin_bottom(8)
        vbox.pack_start(btn_bar, False, False, 0)

        def make_btn(label, callback, css_class=None):
            b = Gtk.Button(label=label)
            b.connect('clicked', callback)
            if css_class:
                b.get_style_context().add_class(css_class)
            btn_bar.pack_start(b, True, True, 0)
            return b

        self.btn_select = make_btn('🎯  Select',  self._on_select)
        self.btn_prev   = make_btn('◀  Prev',     self._on_prev)
        self.btn_next   = make_btn('Next  ▶',     self._on_next)
        self.btn_lock   = make_btn('✅  Lock',    self._on_lock,   'suggested-action')
        self.btn_cancel = make_btn('✖  Cancel',   self._on_cancel, 'destructive-action')

        css_prov = Gtk.CssProvider()
        css_prov.load_from_data(b"""
            window { background-color: #1c1c1c; }
            label  { color: #e0e0e0; font-size: 13px; }
            button { font-size: 13px; min-height: 34px; }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), css_prov,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        state.ui_update_fn = self._refresh_labels

        # ── GStreamer bus: handle prepare-window-handle message ───────────
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.enable_sync_message_emission()
        bus.connect('sync-message::element', self._on_sync_msg)
        bus.connect('message::eos',   self._on_eos)
        bus.connect('message::error', self._on_error)

        self.show_all()

        # ── Start pipeline after window is fully drawn ────────────────────
        # 300 ms delay ensures the X11 window ID is available
        GLib.timeout_add(300, self._start_pipeline)

    def _start_pipeline(self):
        """Called once, 300 ms after show_all, to safely start the pipeline."""
        self._xid = self.video_canvas.get_window().get_xid()
        self.log.info('Canvas X11 xid=%d  — starting pipeline.', self._xid)
        self.pipeline.set_state(Gst.State.PLAYING)
        return False  # don't repeat

    def _on_sync_msg(self, bus, msg):
        """Handle the 'prepare-window-handle' message from the video sink."""
        if msg.get_structure().get_name() == 'prepare-window-handle':
            self.log.info('Sink requested window handle — injecting xid.')
            msg.src.set_window_handle(self._xid or
                                      self.video_canvas.get_window().get_xid())

    def _on_eos(self, bus, msg):
        self.log.info('EOS — stream finished.')
        self.pipeline.set_state(Gst.State.NULL)

    def _on_error(self, bus, msg):
        err, dbg = msg.parse_error()
        self.log.error('GStreamer error: %s | %s', err, dbg)

    # ── Label Refresh ─────────────────────────────────────────────────────

    def _refresh_labels(self):
        self.fps_label.set_text('FPS: %.1f' % self.state.fps_counter.fps)
        self.state_label.set_text('State: %s' % self.state.tracking_state.value)
        return False

    # ── Button Handlers ───────────────────────────────────────────────────

    def _on_select(self, _btn):
        self.log.info('Select')
        self.state.candidate_index = 0
        self.state.tracking_state  = TrackingState.SELECTING

    def _on_prev(self, _btn):
        if self.state.tracking_state == TrackingState.SELECTING:
            n = max(len(self.state.current_dets), 1)
            self.state.candidate_index = (self.state.candidate_index - 1) % n

    def _on_next(self, _btn):
        if self.state.tracking_state == TrackingState.SELECTING:
            n = max(len(self.state.current_dets), 1)
            self.state.candidate_index = (self.state.candidate_index + 1) % n

    def _on_lock(self, _btn):
        if self.state.tracking_state == TrackingState.SELECTING and self.state.current_dets:
            idx = self.state.candidate_index % len(self.state.current_dets)
            det = self.state.current_dets[idx]
            self.log.info('Lock → native_id=%d  %s', det['tracker_id'], det)
            self.state._pending_lock  = det
            self.state.tracking_state = TrackingState.LOCKED

    def _on_cancel(self, _btn):
        self.log.info('Cancel → IDLE')
        self.state.tracking_state = TrackingState.IDLE
        self.state.trt_worker.submit_clear()   # worker clears manager asynchronously

    # ── Keyboard Shortcuts ────────────────────────────────────────────────

    def _on_key(self, _widget, event):
        k = event.keyval
        if k in (Gdk.KEY_s, Gdk.KEY_S):                            self._on_select(None)
        elif k == Gdk.KEY_n:                                        self._on_next(None)
        elif k == Gdk.KEY_p:                                        self._on_prev(None)
        elif k in (Gdk.KEY_l, Gdk.KEY_L, Gdk.KEY_Return):         self._on_lock(None)
        elif k in (Gdk.KEY_q, Gdk.KEY_Q, Gdk.KEY_Escape):         self._on_cancel(None)
        elif k in (Gdk.KEY_x, Gdk.KEY_X):                         self._on_close(None)

    def _on_close(self, _widget):
        self.log.info('Closing.')
        self.pipeline.set_state(Gst.State.NULL)
        self.state.trt_worker.stop()
        Gtk.main_quit()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SUTrack Desktop App — Phase 10')
    parser.add_argument('--config',      default='deepstream/configs/tracker_config.yml')
    parser.add_argument('--input', '-i', default=None)
    parser.add_argument('--no-one-shot', action='store_true')
    parser.add_argument('--debug-boxes', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging(cfg.get('pipeline', {}).get('log_level', 'INFO'))
    log = logging.getLogger('main')

    if not PYDS_AVAILABLE:
        log.error('pyds not found — cannot run.')
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

    pipeline, sink = build_pipeline(state, pipe_cfg)

    width  = int(pipe_cfg.get('width',  1280))
    height = int(pipe_cfg.get('height', 720))

    log.info('Controls:  s=Select  n=Next  p=Prev  l/Enter=Lock  q=Cancel  x=Exit')
    SUTrackApp(state, pipeline, sink, video_width=width, video_height=height)
    Gtk.main()
    log.info('Exit. total_frames=%d', state.frame_count)


if __name__ == '__main__':
    main()

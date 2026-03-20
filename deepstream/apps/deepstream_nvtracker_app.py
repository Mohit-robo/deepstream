"""
DeepStream SUTrack Tracker Application - Phase 8 (Hybrid Tracking: nvtracker + SUTrack)

Pipeline:
    src -> decoder -> nvconv1 -> mux 
        -> nvinfer (PGIE detector) 
        -> nvtracker (Multi-object tracking)
        -> nvconv_bridge -> osd -> tee -> [display, RTSP]
"""

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
import math

# ---------------------------------------------------------------------------
# Project Setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# App Utils and Tracker core
from app_utils import (
    load_yaml, setup_logging, setup_rtsp_server, get_local_ip,
    manual_roi_select, select_bbox_click_to_select, CLASS_COLORS,
    get_iou, RTSP_AVAILABLE
)
from tracker.sutrack_engine import SUTrackEngine
from tracker.tracker_manager import TrackerManager

# pyds requirement
try:
    import pyds
    PYDS_AVAILABLE = True
except ImportError:
    pyds = None
    PYDS_AVAILABLE = False

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, manager, cfg, headless, pgie_enabled=False, pgie_config_path='', pgie_one_shot=True, no_one_shot=False, debug_boxes=False):
        self.manager     = manager
        self.cfg         = cfg
        self.headless    = headless

        self.pgie_enabled = pgie_enabled
        self.pgie_config_path = pgie_config_path
        self.pgie_one_shot = pgie_one_shot
        self.pgie_element = None 
        
        self.nvtracker_enabled = True # Phase 8 core
        self.target_id = -1          # nvtracker ID we want SUTrack to follow
        
        self.first_frame_detections = [] 
        self.initialized = False
        self.init_frame  = None    
        self.frame_ready = threading.Event()  
        self.init_done   = threading.Event()  
        self.init_bbox   = None    
        self.frame_idx   = 0
        self.frame_count = 0
        self.total_time  = 0.0
        self.loop        = None    

        # For mid-stream switching
        self.trigger_reselect = False
        self.no_one_shot = no_one_shot
        self.debug_boxes = debug_boxes

# ---------------------------------------------------------------------------
# Core Probe
# ---------------------------------------------------------------------------

def tracker_probe(pad, info, state):
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

        # Move pixel extraction inside logic to avoid overhead for non-tracking frames
        frame_rgb = None
        def get_rgb():
            nonlocal frame_rgb
            if frame_rgb is not None: return frame_rgb
            try:
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frame_rgba_local = np.array(n_frame, copy=True, order='C')
                frame_rgb = cv2.cvtColor(frame_rgba_local, cv2.COLOR_RGBA2RGB)
                return frame_rgb
            except Exception as e:
                log.warning('Frame extraction failed: %s', e)
                return None

        # --- Phase 8 Initialization & Detections Capture ---
        if not state.initialized or state.trigger_reselect:
            # For initial selection, wait ~10 frames for detector to warm up.
            ready_to_capture = (state.frame_idx >= 10 if not state.initialized else True)
            
            if not state.headless and ready_to_capture:
                # Need pixels for GUI
                try:
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    state.init_frame = np.array(n_frame, copy=True, order='C')
                except: pass

                dets = []
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try: obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    except StopIteration: break
                    if obj_meta.object_id != -1 or obj_meta.unique_component_id in [1, 2]:
                        rect = obj_meta.rect_params
                        dets.append({
                            'x': int(rect.left), 'y': int(rect.top),
                            'w': int(rect.width), 'h': int(rect.height),
                            'class_id': int(obj_meta.class_id),
                            'label': str(obj_meta.obj_label) if obj_meta.obj_label else '',
                            'confidence': float(obj_meta.confidence) if hasattr(obj_meta, 'confidence') else 1.0,
                            'tracker_id': int(obj_meta.object_id)
                        })
                    try: l_obj = l_obj.next
                    except StopIteration: break
                
                state.first_frame_detections = dets
                state.trigger_reselect = False 
                state.init_done.clear()        
                state.frame_ready.set()
                
                log.info('System paused for %s selection (Frame %d).', 
                         'initial' if not state.initialized else 'mid-stream', state.frame_idx)

                state.init_done.wait()
                
                if state.initialized and state.pgie_element and state.pgie_one_shot and not state.no_one_shot:
                    log.info('PGIE entered Sleep mode.')
                    state.pgie_element.set_property('interval', 10000)

        # --- Hybrid Tracking Logic ---
        if state.initialized:
            frgb = get_rgb()
            if frgb is not None:
                # SUTrack update
                results = state.manager.update(frgb, state.frame_idx)
                
                # Find the native target object if we have a target_id
                native_target_bbox = None
                if state.target_id != -1:
                    curr_obj = frame_meta.obj_meta_list
                    while curr_obj is not None:
                        try: o_meta = pyds.NvDsObjectMeta.cast(curr_obj.data)
                        except StopIteration: break
                        if o_meta.object_id == state.target_id:
                            r = o_meta.rect_params
                            native_target_bbox = [r.left, r.top, r.width, r.height]
                            break
                        try: curr_obj = curr_obj.next
                        except StopIteration: break

                if results:
                    for obj_id, sutrack_bbox in results.items():
                        sx, sy, sw, sh = sutrack_bbox
                        
                        # RE-SYNC Logic (Skeptical Hybrid - Phase 9)
                        if native_target_bbox is not None:
                            # 1. IOU and Center distance
                            iou_with_native = get_iou(sutrack_bbox, native_target_bbox)
                            scx, scy = sx + sw/2, sy + sh/2
                            ncx, ncy = native_target_bbox[0] + native_target_bbox[2]/2, native_target_bbox[1] + native_target_bbox[3]/2
                            dist = math.sqrt((scx-ncx)**2 + (scy-ncy)**2)
                            dist_threshold = (sw + native_target_bbox[2]) / 6.0 # ~33% of avg width
                            
                            inst = state.manager.active_trackers.get(obj_id)
                            conf = inst.confidence if inst else 1.0
                            
                            # CRITICAL: Only trust Native if SUTrack is struggling OR if They are very far apart
                            # This prevents Native from "dragging" a healthy SUTrack into a distractor
                            emergency_resync = (dist > dist_threshold * 2) and (iou_with_native < 0.1)
                            low_conf_resync = (conf < 0.2) and (iou_with_native < 0.3)
                            
                            if emergency_resync or low_conf_resync:
                                log.info('SUTrack Re-Sync [%d] Triggered (LowConf=%s, Emergency=%s)', 
                                         obj_id, low_conf_resync, emergency_resync)
                                if inst: inst.state = native_target_bbox
                                sutrack_bbox = native_target_bbox
                                sx, sy, sw, sh = sutrack_bbox

                        # --- Debug Boxes (Phase 9 - DisplayMeta) ---
                        if state.debug_boxes:
                            d_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                            d_meta.num_rects = 0
                            
                            # 1. Raw SUTrack (Cyan)
                            r_s = d_meta.rect_params[d_meta.num_rects]
                            r_s.left, r_s.top, r_s.width, r_s.height = sx, sy, sw, sh
                            r_s.border_width = 3
                            r_s.border_color.set(0.0, 1.0, 1.0, 1.0) # Full Cyan
                            r_s.has_bg_color = 0
                            d_meta.num_rects += 1
                            
                            if native_target_bbox:
                                # 2. Raw Native (Yellow)
                                r_n = d_meta.rect_params[d_meta.num_rects]
                                r_n.left, r_n.top, r_n.width, r_n.height = native_target_bbox[0], native_target_bbox[1], native_target_bbox[2], native_target_bbox[3]
                                r_n.border_width = 3
                                r_n.border_color.set(1.0, 1.0, 0.0, 1.0) # Full Yellow
                                r_n.has_bg_color = 0
                                d_meta.num_rects += 1
                                
                            pyds.nvds_add_display_meta_to_frame(frame_meta, d_meta)

                        # Sync labels to NvDsObjectMeta
                        matched_native = False
                        curr_obj = frame_meta.obj_meta_list
                        while curr_obj is not None:
                            try: o_meta = pyds.NvDsObjectMeta.cast(curr_obj.data)
                            except StopIteration: break
                            if o_meta.object_id != -1 or o_meta.unique_component_id in [1, 2]:
                                r = o_meta.rect_params
                                nv_bbox = [r.left, r.top, r.width, r.height]
                                if get_iou(sutrack_bbox, nv_bbox) > 0.4:
                                    r.left, r.top, r.width, r.height = sx, sy, sw, sh
                                    r.border_width = 3
                                    r.border_color.set(0.0, 1.0, 0.0, 1.0)
                                    o_meta.text_params.display_text = "Stable [%d]" % o_meta.object_id
                                    matched_native = True
                                    break
                            try: curr_obj = curr_obj.next
                            except StopIteration: break

                        if not matched_native:
                            # Add SUTrack overlay
                            new_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                            new_meta.unique_component_id = 8
                            new_meta.object_id = int(obj_id)
                            r = new_meta.rect_params
                            r.left, r.top, r.width, r.height = sx, sy, sw, sh
                            r.border_width = 4
                            r.border_color.set(0.0, 1.0, 0.5, 1.0)
                            txt = new_meta.text_params
                            txt.display_text = 'SUTrack [%d]' % int(obj_id)
                            txt.x_offset, txt.y_offset = int(sx), max(int(sy)-20, 0)
                            txt.font_params.font_name, txt.font_params.font_size = 'Serif', 11
                            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                            txt.set_bg_clr, txt.text_bg_clr.set(0.0, 0.4, 0.2, 0.8)
                            pyds.nvds_add_obj_meta_to_frame(frame_meta, new_meta, None)

        # Global counters synchronized
        state.frame_idx += 1
        state.frame_count += 1

        try: l_frame = l_frame.next
        except StopIteration: break
    return Gst.PadProbeReturn.OK

# ---------------------------------------------------------------------------
# Pipeline Builder (Phase 7 Hybrid)
# ---------------------------------------------------------------------------

def build_pipeline(state, pipe_cfg):
    Gst.init(None)
    pipeline = Gst.Pipeline.new('sutrack-hybrid-pipeline')
    width, height = int(pipe_cfg.get('width', 1280)), int(pipe_cfg.get('height', 720))
    input_source = pipe_cfg.get('input_source') or ''

    # Elements
    if input_source.startswith('rtsp://') or input_source.startswith('http://'):
        src = Gst.ElementFactory.make('uridecodebin', 'src')
        src.set_property('uri', input_source)
        decoder = None
    else:
        src = Gst.ElementFactory.make('filesrc', 'src') if input_source else Gst.ElementFactory.make('v4l2src', 'src')
        if not input_source: src.set_property('device', '/dev/video0')
        else: src.set_property('location', input_source)
        decoder = Gst.ElementFactory.make('decodebin', 'decoder')

    nvconv1 = Gst.ElementFactory.make('nvvideoconvert', 'nvconv1')
    caps1 = Gst.ElementFactory.make('capsfilter', 'caps1')
    caps1.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM),format=NV12'))

    mux = Gst.ElementFactory.make('nvstreammux', 'mux')
    mux.set_property('batch-size', 1)
    mux.set_property('width', width)
    mux.set_property('height', height)
    mux.set_property('batched-push-timeout', 4000000)
    mux.set_property('live-source', 0)

    pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
    pgie.set_property('config-file-path', state.pgie_config_path)
    state.pgie_element = pgie

    # Phase 8: NVTRACKER - Dynamic library path discovery
    tracker = Gst.ElementFactory.make('nvtracker', 'nvtracker')
    
    # Common Jetson paths for tracker libraries (User search found libnvds_nvmultiobjecttracker.so)
    possible_lib_paths = [
        '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so',
        '/opt/nvidia/deepstream/deepstream-6.3/lib/libnvds_nvmultiobjecttracker.so',
        '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvdcf.so',
        '/usr/lib/aarch64-linux-gnu/deepstream/libnvds_nvmultiobjecttracker.so',
    ]
    
    found_lib = None
    for path in possible_lib_paths:
        if os.path.exists(path):
            found_lib = path
            break
    
    if found_lib:
        logging.info('Found tracker library at: %s', found_lib)
        tracker.set_property('ll-lib-file', found_lib)
    else:
        # Fallback to the default but log a warning
        logging.error('CRITICAL: libnvds_nvdcf.so not found in standard paths.')
        tracker.set_property('ll-lib-file', possible_lib_paths[0])

    nvdcf_cfg  = os.path.normpath(os.path.join(ROOT, pipe_cfg.get('nvdcf_config', 'configs/nvdcf_config.txt')))
    trk_width  = int(pipe_cfg.get('nvtracker_width',  640))
    trk_height = int(pipe_cfg.get('nvtracker_height', 384))
    
    tracker.set_property('ll-config-file', nvdcf_cfg)
    tracker.set_property('tracker-width', trk_width)
    tracker.set_property('tracker-height', trk_height)
    tracker.set_property('gpu-id', 0)
    # tracker.set_property('enable-batch-process', 1) # Not supported in all versions

    nvconv_bridge = Gst.ElementFactory.make('nvvideoconvert', 'nvconv_bridge')
    caps_bridge = Gst.ElementFactory.make('capsfilter', 'caps_bridge')
    caps_bridge.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM),format=RGBA'))

    osd = Gst.ElementFactory.make('nvdsosd', 'osd')
    osd.set_property('process-mode', 0) # 0=CPU mode for stability
    tee = Gst.ElementFactory.make('tee', 'tee')

    # Display / RTSP branches (simplified for brevity, mirroring RTSP app logic)
    q_disp = Gst.ElementFactory.make('queue', 'q_disp')
    sink_disp = Gst.ElementFactory.make('nv3dsink', 'sink_disp') or Gst.ElementFactory.make('fakesink', 'sink_disp')
    sink_disp.set_property('sync', False)

    # Link everything
    pipeline.add(src)
    if decoder: pipeline.add(decoder)
    pipeline.add(nvconv1); pipeline.add(caps1); pipeline.add(mux)
    pipeline.add(pgie); pipeline.add(tracker); pipeline.add(nvconv_bridge); pipeline.add(caps_bridge); pipeline.add(osd); pipeline.add(tee)
    pipeline.add(q_disp); pipeline.add(sink_disp)

    # ... linkage logic ...
    def on_pad_added(src, pad, dest):
        if pad.query_caps(None).get_structure(0).get_name().startswith('video/'):
            pad.link(dest.get_static_pad('sink'))
    
    if decoder:
        src.link(decoder)
        decoder.connect('pad-added', on_pad_added, nvconv1)
    else:
        src.connect('pad-added', on_pad_added, nvconv1)
    
    nvconv1.link(caps1)
    caps1.get_static_pad('src').link(mux.get_request_pad('sink_0'))
    mux.link(pgie); pgie.link(tracker); tracker.link(nvconv_bridge); nvconv_bridge.link(caps_bridge); caps_bridge.link(osd); osd.link(tee)
    tee.get_request_pad('src_0').link(q_disp.get_static_pad('sink'))
    q_disp.link(sink_disp)

    osd.get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER, tracker_probe, state)
    return pipeline

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Phase 7: Hybrid Tracking')
    parser.add_argument('--config', default='deepstream/configs/tracker_config.yml')
    parser.add_argument('--input', '-i', default=None)
    parser.add_argument('--no-one-shot', action='store_true',
                        help='Disable PGIE power-save (keeps detector active)')
    parser.add_argument('--debug-boxes', action='store_true',
                        help='Draw raw SUTrack and Native boxes for drift analysis')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging(cfg.get('pipeline', {}).get('log_level', 'INFO'))
    log = logging.getLogger('main')

    if not PYDS_AVAILABLE:
        log.error('pyds required for Phase 7.'); sys.exit(1)

    # TRT Engine + Tracker Manager initialization (reused from RTSP app)
    pipe_cfg = cfg.get('pipeline', {})
    if args.input:
        pipe_cfg['input_source'] = args.input
        
    model_cfg = cfg.get('model', {})
    tracker_cfg = cfg.get('tracker', {})
    headless = pipe_cfg.get('headless', False)
    
    engine_path = os.path.normpath(os.path.join(ROOT, model_cfg.get('engine_path', '../sutrack_fp32.engine')))
    engine = SUTrackEngine(engine_path=engine_path)
    manager = TrackerManager(engine=engine)
    
    pgie_config_path = os.path.normpath(os.path.join(ROOT, pipe_cfg.get('pgie_config', 'configs/pgie_config.txt')))
    pgie_one_shot = not args.no_one_shot if args.no_one_shot else pipe_cfg.get('pgie_one_shot', True)

    state = AppState(manager=manager, cfg=cfg, headless=headless, 
                     pgie_enabled=True, pgie_config_path=pgie_config_path,
                     pgie_one_shot=pgie_one_shot,
                     no_one_shot=args.no_one_shot,
                     debug_boxes=args.debug_boxes)
    
    pipeline = build_pipeline(state, pipe_cfg)
    loop = GLib.MainLoop()
    state.loop = loop
    
    pipeline.set_state(Gst.State.PLAYING)
    
    # GUI selection
    if not state.initialized:
        state.frame_ready.wait()
        state.frame_ready.clear()   # Must clear before select_loop thread starts
        selected_bbox = None
        if state.first_frame_detections:
            selected_bbox = select_bbox_click_to_select(state.init_frame, state.first_frame_detections)

        if selected_bbox is None:
            selected_bbox = manual_roi_select(state.init_frame)

        if selected_bbox:
            if isinstance(selected_bbox, dict):
                x, y, w, h = selected_bbox['x'], selected_bbox['y'], selected_bbox['w'], selected_bbox['h']
                state.target_id = selected_bbox.get('tracker_id', -1)
            else:
                x, y, w, h = selected_bbox
                state.target_id = -1
            
            state.manager.initialize(cv2.cvtColor(state.init_frame, cv2.COLOR_RGBA2RGB), [int(x), int(y), int(w), int(h)], frame_idx=0)
            state.initialized = True
            log.info('Initial target selected: %s (Native ID: %d)', selected_bbox, state.target_id)

        state.init_done.set()

    # Dynamic re-selection loop (Phase 7)
    def select_loop():
        while True:
            state.frame_ready.wait()
            state.frame_ready.clear()
            log.info('Switching target: showing detections...')
            selected = select_bbox_click_to_select(state.init_frame, state.first_frame_detections)
            if selected is None:
                selected = manual_roi_select(state.init_frame)

            if selected:
                if isinstance(selected, dict):
                    # Click-to-select path
                    x, y, w, h = selected['x'], selected['y'], selected['w'], selected['h']
                    state.target_id = selected.get('tracker_id', -1)
                else:
                    # Manual ROI path (list)
                    x, y, w, h = selected
                    state.target_id = -1
                
                # Clear all existing trackers before initializing new target
                state.manager.active_trackers.clear()
                cur_idx = state.frame_idx
                state.manager.initialize(cv2.cvtColor(state.init_frame, cv2.COLOR_RGBA2RGB),
                                         [int(x), int(y), int(w), int(h)], frame_idx=cur_idx)
                state.initialized = True
                log.info('New target selected: %s (Native ID: %d)', selected, state.target_id)
            
            state.init_done.set() # Resume pipeline

    if not headless:
        t = threading.Thread(target=select_loop, daemon=True)
        t.start()
        
        # Stdin listener for re-selection trigger
        def input_listener():
            log.info('Phase 7 Hybrid Online. Type "s" + ENTER to switch target.')
            while True:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 's':
                    log.info('Reselection triggered by user.')
                    state.trigger_reselect = True
        
        it = threading.Thread(target=input_listener, daemon=True)
        it.start()

    try: 
        loop.run()
    except KeyboardInterrupt: pass
    finally: pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()

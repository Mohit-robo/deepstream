# SUTrack — Project TODO

---

## ✅ Phase 1 — PyTorch Inference (Done)
- [x] Understand the inference pipeline of the tracker
- [x] Create `demo.py` to run tracking on a video file
- [x] Modify `demo.py` to accept a video file or camera stream
- [x] Visualize tracking results with bounding boxes

---

## ✅ Phase 2 — TensorRT Deployment (Done)
- [x] Understand model inputs (template, search, template_anno) and outputs
- [x] Create `tracking/export_onnx.py` — custom `SUTrackWrapper`, unfold→avg_pool2d patch
- [x] Validate the exported ONNX model graph structure
- [x] Compile TensorRT engine (`sutrack_t224.engine`, FP32)
- [x] Create `tracking/demo_trt.py` — TRT-based inference
- [x] Fix all runtime issues (double sigmoid, clip_box bug, FP16 underflow, FPS bottleneck)
- [x] Achieve ~25 FPS on Jetson (sutrack_t224, FP32)

---

## ✅ Phase 3 — Standalone Deploy Package (Done)
- [x] Create `deploy_trt/utils.py` — pure-NumPy preprocessing, crop, hann2d, config loader
- [x] Create `deploy_trt/demo_trt.py` — zero PyTorch dependency
- [x] Create `deploy_trt/README.md` — usage and engine compilation guide
- [x] Update `CLAUDE.md` with experiment findings and deployment guide
- [x] Create `tasks/lessons.md` — 12 debugging lessons from this session

---

## 🔲 Phase 4 — DeepStream App (`deepstream/` in this repo)

### 4.1 Environment Verification
- [x] Verify DeepStream SDK is installed (`deepstream-app --version-all`)
- [x] Verify Python bindings (`import pyds`)
- [x] Verify `cuda.cudart` and `tensorrt` imports
- [x] Load existing `sutrack_fp32.engine`, run dummy inference

### 4.2 Minimal Pipeline (source → appsink) ✅ Code complete
- [x] Create `deepstream/apps/deepstream_tracker_app.py` — filesrc → decodebin → nvvideoconvert → appsink
- [x] Convert GstBuffer → NumPy BGR, then BGR→RGB before tracker call
- [x] Create `deepstream/tracker/tracker_utils.py` — pure-NumPy preprocessing (from deploy_trt/utils.py)
- [x] Create `deepstream/tracker/sutrack_engine.py` — TRT load, cuda.cudart buffers, infer()
- [x] Create `deepstream/tracker/tracker_instance.py` — per-object state + track() with NumPy postprocessing
- [x] Create `deepstream/tracker/tracker_manager.py` — multi-object lifecycle, IoU matching, stale removal
- [x] Create `deepstream/configs/tracker_config.yml` — all tunable params
- [x] Create `deepstream/scripts/build_engine.sh` — trtexec with correct FP32 shape specs
- [x] Create `deepstream/docs/architecture.md` and `deepstream/docs/usage.md`
- [x] **Run on Jetson** — confirm >=25 FPS single object (Verified by user)

### 4.3 Add Detector (PGIE)
- [x] Create `deepstream/configs/pgie_config.txt` placeholder
- [x] Extend pipeline: `→ nvstreammux → nvinfer (PGIE) → pad probe`
- [x] Read `NvDsObjectMeta` detections in pad probe
- [x] Wire detections into `TrackerManager.update(detections=...)`
- [x] Test with multi-person video — verify stable IDs

---

## ✅ Phase 5 — DeepStream v2 (OSD & RTSP Integration) (Done)
- [x] Create `deepstream/apps/deepstream_rtsp_app.py` (Phase 5 native OSD + RTSP app)
- [x] Add `static_roi` config support for automated headless initialization
- [x] Move drawing logic from OpenCV to native NvDsOsd metadata probes (pad probe on osd sink)
- [x] Add Hardware H.264 Encoder (`nvv4l2h264enc`) to RTSP branch of pipeline
- [x] Add RTSP Streaming Server support (`GstRtspServer` wrapping udpsink)
- [x] Add Phase 5 config keys to `tracker_config.yml` (width, height, rtsp_*)
- [x] Fix `nveglglessink` NVMM surface array error — insert `nvvideoconvert(compute-hw=1)` bridge before display sink; try `nv3dsink` first
- [x] Final validation: run on Jetson — confirmed OSD boxes on RTSP stream (Verified by user)
## ✅ Phase 6 — PGIE & Intelligent ROI Selection (Done)
- [x] Create implementation plan for Phase 6 (`docs/phase_6_click_to_select_plan.md`)
- [x] Add `nvinfer` (PGIE) element to RTSP app pipeline
- [x] Configure `pgie_config.txt` with suitable detector (Peoplenet or ResNet)
- [x] Capture first-frame detections in `tracker_probe`
- [x] Implement `on_mouse_click` callback for object selection
- [x] Implement "Click-to-Select" logic (box searching via pixel coordinates)
- [x] Initialize tracker directly from the selected detected box
- [x] Verification: Accurate hand-off from detector to SUTrack via GUI click
onfig.yml`; `--pgie-config` and `--no-pgie` CLI args
- [x] Graceful fallbacks: Q/ESC -> manual selectROI; no detections -> manual selectROI
- [x] Verification: run on Jetson, confirm detector boxes appear, click-to-select works
- [x] Phase 6.5: Dynamically increase PGIE interval for power-save optimization
- [x] Phase 6.6: Aggressive metadata cleanup to ensure clean output stream

## ✅ Phase 7 — Documentation & Lessons Update (Ongoing)
- [x] Write `docs/README.md` — quick start + directory structure
- [x] Write `docs/usage.md` — full setup, run, CLI reference, troubleshooting
- [x] Write `docs/architecture.md` — pipeline diagram and design decisions
- [x] Document Lessons 20-21 (EGL context & NVMM memory bridges)
- [x] Update documentation for RTSP/OSD v2 app (Phase 5 pipeline, CLI, config, troubleshooting)
- [x] Update documentation for Phase 6 Click-to-Select ROI
- [x] Update installation docs — `install.html`, `README.md`, `usage.md` Sections 1 & 2

---

## 🔲 Phase 8 — Hybrid Tracking (nvtracker + SUTrack)

### 8.1 App & Utilities
- [x] Create `deepstream/apps/app_utils.py` — shared utilities (select_bbox, setup_rtsp_server, get_iou, CLASS_COLORS)
- [x] Create `deepstream/apps/deepstream_nvtracker_app.py` — hybrid app (nvtracker + SUTrack)
- [x] Create `deepstream/configs/nvtracker_config.txt` — NvDCF wrapper config (deepstream-app CLI format, for reference)
- [x] Create `deepstream/configs/nvdcf_config.txt` — NvDCF algorithm parameters
- [x] Add Phase 8 keys to `tracker_config.yml` (`nvdcf_config`, `nvtracker_width`, `nvtracker_height`)

### 8.2 Code Fixes Applied
- [x] Fix `ll-config-file` pointing to `nvtracker_config.txt` → changed to `nvdcf_config.txt`
- [x] Fix `frame_ready` not cleared after main thread initial wait → `select_loop` was firing immediately
- [x] Fix `active_trackers` not cleared on reselection → old tracker ran alongside new one
- [x] Fix `frame_idx=0` hardcoded in select_loop → use `state.frame_idx`
- [x] Fix `txt.font_params` missing `font_name`/`font_size` → text overlay invisible on nvdsosd
- [x] Wire `nvdcf_config`, `nvtracker_width`, `nvtracker_height` from `tracker_config.yml`
- [x] Add `gpu-id=0` and `enable-batch-process=1` to nvtracker properties

### 8.3 Verification (run on Jetson) ✅
- [x] TC1: Click-to-Select Initialization — "SUTrack [ID] (Stabilized)" label appears on first click
- [x] TC2: ID Persistence (Occlusion) — SUTrack box stays on target through brief occlusion
- [x] TC3: Mid-stream Target Switching — type "s" + ENTER, new target selected, old tracker dropped
- [x] TC4: PGIE One-Shot — logs show "PGIE Sleep mode (Phase 7)", nvtracker continues independently

### 8.4 Documentation ✅
- [x] Update `deepstream/docs/usage.md` — Phase 8 run section (Section 6)
- [x] Update `deepstream/docs/architecture.md` — hybrid pipeline diagram
- [x] Update `deepstream/docs/html/index.html` — file table, docs grid card
- [x] Create `deepstream/docs/html/hybrid.html` — dedicated hybrid tracking HTML page
- [x] Update HTML nav in all pages to include `hybrid.html`

---

## ✅ Phase 9 — Hybrid Anti-Drift Tuning (Done)
- [x] Analyze drift failure in crowded scenes (Target ID drift vs SUTrack drift)
- [x] Implement Center-Distance re-sync guard in `deepstream_nvtracker_app.py`
- [x] Update `nvdcf_config.txt` parameters (ProbThreshold=0.75, IOUThreshold=0.5) for occlusion robustness
- [x] Implement `--debug-boxes` flag using `NvDsDisplayMeta` for raw tracker visualization
- [x] Refine re-sync logic: Skeptical Hybrid (priority to SUTrack conf during drift)
- [x] Document Phase 9 — anti-drift strategy and center-distance lesson

---

## 🔲 Phase 10 — Dynamic Tracking & Persistence

### 10.1 Core Implementation
- [x] Plan: `deepstream/docs/phase_10_persistence.md`
- [x] `deepstream/apps/deepstream_nvtracker_v3.py` — Phase 10 app (Snap & Re-Sync + Re-ID + state machine)
- [x] `app_utils.py` extended: `TrackingState`, `IDHistory`, `compute_crop_histogram`, `compare_histograms`

### 10.2 Key Features Implemented
- [x] Non-blocking Snap & Re-Sync: pipeline runs during GUI; probe syncs init to live nvtracker position
- [x] IDLE → LOCKED → SEARCHING → STALE state machine with OSD feedback
- [x] Histogram-based Re-ID (H-S 50×60): IoU spatial match first, then appearance similarity ≥ 0.58
- [x] Auto-hibernation: STALE after `hibernate_after` frames (default 150) of low confidence
- [x] `hibernate_after` key in `tracker_config.yml` (under `tracker:`)
- [x] Stdin controls: 's' = reselect  |  'q' = stop tracking  |  'x' = exit
- [x] Debug mode `--debug-boxes`: yellow last-known bbox overlay

### 10.3 Verification (run on Jetson)
- [ ] Live selection: RTSP stream, trigger selection, confirm no video jump after close
- [ ] Re-entry: person walks out and back in — green box re-attaches automatically
- [ ] Stop: press 'q', confirm [TARGET LOST] banner, then 's' to reselect
- [ ] STALE timeout: cover camera for >150 frames — confirm STALE state + banner

### 10.4 Documentation
- [x] Update `deepstream/docs/usage.md` — Phase 10 run section (Section 8)
- [x] Update `deepstream/configs/tracker_config.yml` — `hibernate_after` key
- [x] Update `deepstream/docs/html/index.html` — file table (deepstream_nvtracker_v3.py)
- [x] Update `deepstream/docs/html/hybrid.html` — Phase 10 evolution section + app_utils table

---

## 🔲 Phase 11 — Desktop GUI App + FPS Optimization

### 11.1 Desktop App (`deepstream_desktop_app.py`)
- [x] Create `deepstream/apps/deepstream_desktop_app.py` — GTK3 window wrapping the DeepStream pipeline
- [x] Embedded video via `nv3dsink` + `set_window_handle(xid)` — GPU renders directly into GTK canvas (no CPU frame copy)
- [x] Control buttons: Select / Prev / Next / Lock / Cancel
- [x] FPS + State info bar updated via `GLib.idle_add`
- [x] Full keyboard shortcut parity: `s`, `n`, `p`, `l`/Enter, `q`, `x`
- [x] `LD_PRELOAD` libgomp.so.1 self-re-exec for Jetson compatibility

### 11.2 FPS Optimization (LOCKED mode: 20-24 FPS → ~55-60 FPS)
- [x] Root cause identified: `manager.update()` (TRT ~30 ms) blocking GStreamer streaming thread inside pad probe
- [x] **Fix A**: `TRTWorkerThread` — async TRT inference; probe submits frame non-blocking, reads last result for OSD
- [x] **Fix B**: Throttle `id_history.update()` histogram to every `HIST_UPDATE_INTERVAL=20` frames
- [x] **Fix C**: Throttle `GLib.idle_add` to every `UI_UPDATE_EVERY=5` frames
- [x] **Fix D**: Replace `cv2.cvtColor(rgba, RGBA2RGB)` with `rgba[:, :, :3]` slice (no extra allocation)
- [x] All `manager` mutations (clear/initialize/update) routed exclusively through `TRTWorkerThread` (thread-safe)
- [x] Record lesson in `tasks/lessons.md` (Lesson 34)

### 11.3 Verification (run on Jetson)
- [ ] IDLE FPS vs LOCKED FPS — confirm LOCKED FPS within 5 FPS of IDLE
- [ ] Lock → Unlock cycle — confirm no bbox persistence after Cancel
- [ ] Re-ID after exit — confirm SEARCHING → LOCKED re-attach
- [ ] STALE timeout — confirm banner after 150 frames low confidence

### 11.4 Documentation
- [x] Update `tasks/lessons.md` — Lesson 34: async TRT worker pattern
- [x] Update `tasks/todo.md` — Phase 11 section
- [x] Update `deepstream/docs/usage.md` — Section 9: Desktop app run guide
- [x] Update `deepstream/docs/html/hybrid.html` — Phase 11 desktop app + TRTWorkerThread section
- [x] Update `deepstream/docs/html/index.html` — add `deepstream_desktop_app.py` to file table

---

## 🔲 Phase 12 — V5 Client-Server Distributed Architecture

### 12.1 Backend Daemon (`deepstream_server_app.py`)
- [x] Draft Client-Server Architecture Plan (`docs/architecture_docs/v5_client_server_plan.md`)
- [x] Create `deepstream_server_app.py` based on `deepstream_desktop_app.py`
- [x] Implement headless pipeline (force `fakesink` display, keep RTSP branch)
- [x] Develop REST Control API server thread (Port 8000) for incoming commands
- [x] Implement normalized `(x, y)` coordinate localization against `nvtracker` bounding boxes

### 12.2 Frontend Client (`v5_remote_client.py`)
- [x] Create `v5_remote_client.py` GTK3 application
- [x] Implement GStreamer `uridecodebin` pipeline to receive RTSP feed locally
- [x] Re-implement `on_mouse_click` and keybindings to dispatch async HTTP POSTs
- [x] Implement API state polling to update GUI FPS/Status labels, aspect-fit scaled, centred)
- [x] Mouse click: normalise via `_draw_rect`, POST `{action: click, x, y}` async
- [x] Keyboard: s/n/p/l/Enter/q/Esc/x → REST commands (fire-and-forget daemon threads)
- [x] State polling daemon thread: GET /api/state every 0.5 s → update FPS/State labels
- [x] Fallback manual pipeline if parse_launch fails (explicit pad-added)

### 12.3 Integration Fixes Applied
- [x] Fix headless PGIE (nvinfer) on Jetson: drop X11-forwarded DISPLAY, set EGL_PLATFORM=surfaceless
- [x] Fix GstRtspServer factory string: udpsrc -> rtph264depay -> rtph264pay name=pay0 (SDP generation)
- [x] Fix RTSP transport: rtspsrc protocols=tcp in client (UDP ports blocked by firewall/NAT)
- [x] Fix GStreamer reconnect: auto-retry on error/EOS via GLib.timeout_add_seconds
- [x] Add --loop flag: seek to position 0 on EOS for continuous file replay
- [x] Add detection-driven UX: show PGIE boxes in IDLE/STALE states, status "N DETECTED -- click target"
- [x] Add --auto-lock flag: lock to largest PGIE detection automatically (zero user interaction)
- [x] Record Lesson 36: X11 forwarding + EGL_PLATFORM=surfaceless

### 12.4 Verification & Deployment
- [x] Run backend on Jetson, frontend on local machine -- RTSP stream confirmed working
- [x] PGIE detector runs headless (EGL_PLATFORM=surfaceless fix verified)
- [x] Click-to-select works through RTSP latency (normalised coord compensation)
- [x] Update `docs/usage.md` and `docs/html/index.html` with V5 instructions

---

## Critical Rules (must not break)
- FP32 only — no `--fp16` in engine
- No double sigmoid on `score_map`/`size_map`
- Position-only clamp — never recalculate `w/h` from clamped corners
- Hanning window applied **before** argmax
- BGR → RGB before every tracker call
- Named TRT bindings — always by name, not by index
- No PyTorch at runtime

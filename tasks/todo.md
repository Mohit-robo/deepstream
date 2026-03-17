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
- [ ] Extend pipeline: `→ nvstreammux → nvinfer (PGIE) → pad probe`
- [ ] Read `NvDsObjectMeta` detections in pad probe
- [ ] Wire detections into `TrackerManager.update(detections=...)`
- [ ] Test with multi-person video — verify stable IDs

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
- [ ] Add automated ROI init via standard PGIE detector (Phase 4.3 dependency)



## ✅ Phase 6 — Documentation & Lessons Update (Done)
- [x] Write `docs/README.md` — quick start + directory structure
- [x] Write `docs/usage.md` — full setup, run, CLI reference, troubleshooting
- [x] Write `docs/architecture.md` — pipeline diagram and design decisions
- [x] Document Lessons 20-21 (EGL context & NVMM memory bridges)
- [x] Update documentation for RTSP/OSD v2 app (Phase 5 pipeline, CLI, config, troubleshooting)

---

## Critical Rules (must not break)
- FP32 only — no `--fp16` in engine
- No double sigmoid on `score_map`/`size_map`
- Position-only clamp — never recalculate `w/h` from clamped corners
- Hanning window applied **before** argmax
- BGR → RGB before every tracker call
- Named TRT bindings — always by name, not by index
- No PyTorch at runtime

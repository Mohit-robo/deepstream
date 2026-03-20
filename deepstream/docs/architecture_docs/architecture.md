# DeepStream SUTrack - Architecture

## Pipeline (Phase 8/9 - Hybrid Tracking, CURRENT)

The hybrid pipeline leverages **nvtracker** (NvDCF) for global object persistence and **SUTrack** for high-precision bounding box refinement of a selected target.

```
Video / Camera
      |
      V
  filesrc / v4l2src
      |
  decodebin (HW decode)
      |
  nvvideoconvert (NVMM:RGBA)
      |
  nvstreammux
      |
  nvinfer (PGIE) -> [Detector boxes attached to obj_meta]
      |
  nvtracker (NvDCF) -> [Candidate IDs assigned to obj_meta]
      |
  nvvideoconvert (Bridge)
      |
      V
  [PAD PROBE on nvdsosd sink pad]
      |
      |   1. Get frame_rgb (cv2.cvtColor from NVMM surface)
      |   2. Find Native Leader: Scan obj_meta for state.target_id
      |   3. Run SUTrack: state.manager.update(frame_rgb)
      |   4. Skeptical Re-Sync (Phase 9):
      |      Only trust Native Leader if SUTrack confidence is low (< 0.2)
      |      OR if geometric distance > 50% box width.
      |   5. Debug Overlays (Phase 9):
      |      If --debug-boxes: pyds.nvds_acquire_display_meta_from_pool()
      |                        Draw Teal (Raw SUTrack) and Yellow (Raw Native)
      |   6. Meta Correction:
      |      Update the matched Native obj_meta.rect_params with SUTrack output.
      |      Set label to "STABLE [ID]".
      |
  nvdsosd (NVMM Rendering)
      |
   Branch A (Display) / Branch B (RTSP)
```

## Pipeline (Phase 6 - PGIE + Click-to-Select ROI)

```
Video / Camera
      |
      V
  filesrc / v4l2src
      |
  decodebin  (HW decode; dynamic pad -> on_decoder_pad_added)
      |
  nvvideoconvert (compute-hw=1, VIC) -> video/x-raw(memory:NVMM),format=RGBA
      |
  nvstreammux (width x height, batch-size=1)
      |
  nvinfer (PGIE) [optional -- only when pgie_config is set in tracker_config.yml]
      |              ResNet-10 INT8 detector; populates NvDsObjectMeta per object
      |
  nvvideoconvert (compute-hw=1)  [bridge required between nvinfer and nvdsosd]
      |
      V
  [PAD PROBE on nvdsosd sink pad]
      |
      |  pyds.get_nvds_buf_surface() -> RGBA NumPy (CPU copy)
      |  On frame 0: capture frame + read obj_meta_list detections
      |              state.frame_ready.set()        (signal main thread)
      |              state.init_done.wait()          (block until ROI selected)
      |  Frame 1+:  cv.cvtColor RGBA->RGB
      |              TrackerManager.update()
      |              For each bbox: pyds.nvds_acquire_obj_meta_from_pool()
      |                             populate rect_params + text_params
      |                             pyds.nvds_add_obj_meta_to_frame()
      |
  nvdsosd (process-mode=0, CPU)  <- draws NvDsObjectMeta boxes in GPU memory
      |
      +------- tee --------+
      |                    |
   Branch A             Branch B
  (Display)             (RTSP)
      |                    |
  queue               queue
      |                    |
  nvvideoconvert      nvvideoconvert (compute-hw=1)
  (compute-hw=1)           |
      |               nvv4l2h264enc (HW H.264 NVENC)
  nv3dsink /               |
  nveglglessink        rtph264pay
                           |
                       udpsink (port 5400)

  GstRtspServer (port 8554):
      udpsrc(5400) -> rtsp://<jetson-ip>:8554/sutrack

  Main thread (ROI selection, runs while probe is blocked):
      state.frame_ready.wait()     <- waits for probe to capture first frame
      select_bbox_click_to_select()  <- OpenCV window, PGIE boxes drawn, user clicks
          OR manual_roi_select()     <- fallback: draw box with mouse
      manager.initialize(frame_rgb, bbox)
      state.init_done.set()        <- unblocks probe thread
```

## Pipeline (Phase 5 - Native OSD + RTSP Streaming, without PGIE)

```
Video / Camera
      |
      V
  filesrc / v4l2src
      |
  decodebin  (HW decode; dynamic pad -> on_decoder_pad_added)
      |
  nvvideoconvert (compute-hw=1, VIC) -> video/x-raw(memory:NVMM),format=RGBA
      |
  nvstreammux (width x height, batch-size=1)
      |
      V
  [PAD PROBE on nvdsosd sink -- SUTrack logic runs here]
      |
      |  pyds.get_nvds_buf_surface() -> RGBA NumPy (CPU copy)
      |  cv.cvtColor RGBA->RGB
      |  TrackerManager.update()  (initialize or track)
      |  For each bbox: pyds.nvds_acquire_obj_meta_from_pool()
      |                 populate rect_params + text_params
      |                 pyds.nvds_add_obj_meta_to_frame()
      |
  nvdsosd (process-mode=0, CPU)  <- draws NvDsObjectMeta boxes in GPU memory
      |
      +------- tee --------+
      |                    |
   Branch A             Branch B
  (Display)             (RTSP)
      |                    |
  queue               queue
      |                    |
  nvvideoconvert      nvvideoconvert (compute-hw=1)
  (compute-hw=1)           |
      |               nvv4l2h264enc (HW H.264 NVENC)
  nv3dsink /               |
  nveglglessink        rtph264pay
                           |
                       udpsink (port 5400)

  GstRtspServer (port 8554):
      udpsrc(5400) -> rtsp://<jetson-ip>:8554/sutrack
```

## Pipeline (Phase 4.2 - Appsink / Manual Init, Legacy)

```
Video / Camera
      |
      V
  filesrc / v4l2src
      |
  decodebin / nvv4l2decoder  (HW decode on Jetson)
      |
  nvvideoconvert (VIC/GPU) -> video/x-raw,format=RGBA
      |
  videoconvert (CPU bridge) -> video/x-raw,format=RGBA
      |
  appsink  (Python callback, 1 buffer max, drop=true)
      |
      V
  nvbuf_to_bgr()          map GstBuffer -> NumPy BGR array
      |
  cv.cvtColor BGR->RGB     (Lesson 12: model expects RGB)
      |
  TrackerManager.update()
      |
      +-- TrackerInstance.track()  x  N active tracks
      |       |
      |       +-- sample_target()     search crop extraction
      |       +-- preprocess()        normalize -> (1,6,224,224) float32
      |       +-- SUTrackEngine.infer()  <- shared single TRT context
      |       +-- NumPy postprocessing
      |               +-- hann2d * score_map   (Lesson 6: before argmax)
      |               +-- argmax -> peak (idx_x, idx_y)
      |               +-- decode cx/cy/w/h from size_map + offset_map
      |               +-- map crop-space -> image coords
      |               +-- position-only clamp  (Lesson 4)
      |
  Draw bboxes + IDs  ->  cv.imshow / VideoWriter
```

## Component Responsibilities

| File | Responsibility |
|------|---------------|
| `deepstream/tracker/tracker_utils.py` | Pure-NumPy preprocessing, crop extraction, Hanning window, IoU |
| `deepstream/tracker/sutrack_engine.py` | TRT engine load, buffer allocation (cuda.cudart), infer() |
| `deepstream/tracker/tracker_instance.py` | Per-object state: template buffers, state [x,y,w,h], track() |
| `deepstream/tracker/tracker_manager.py` | Pool of TrackerInstances, IoU matching, stale removal |
| `deepstream/apps/deepstream_rtsp_app.py` | **Phase 5**: GStreamer pad probe, NvDsOsd, RTSP streaming |
| `deepstream/apps/deepstream_tracker_app.py` | Phase 4 (legacy): GStreamer pipeline, appsink callback, video I/O |
| `deepstream/configs/tracker_config.yml` | All tunable parameters (no hardcoded values in code) |

## Key Design Decisions

### Shared TRT Engine
All `TrackerInstance` objects share one `SUTrackEngine` (one context, one buffer set).
Inference is sequential - the manager calls `infer()` per tracker per frame.
This avoids GPU memory duplication and is correct because sutrack_t224 is stateless
(template buffers live in Python, not in the engine).

### No PyTorch at Runtime
All preprocessing and postprocessing is pure NumPy.
`cuda.cudart` replaces `pycuda` for Jetson compatibility (Lesson 7).

### IoU Matching (Phase 4.3+)
When a PGIE detector is attached, `TrackerManager.update()` greedily matches
new detections to active tracks by IoU. Unmatched detections above
`min_confidence` spawn new `TrackerInstance` objects.

### FP32 Only
The TRT engine must be compiled without `--fp16`. The SUTrack decoder uses
sigmoid activations that underflow to zero in FP16 (Lesson 2).

### Hardware Abstraction
The pipeline uses `nvvideoconvert` with `compute-hw=1` (VIC) for headless SSH stability,
or `compute-hw=0` (GPU) for maximum performance when a display context is available (Lessons 20, 21).

### ASCII Compliance
All scripts and documents use standard ASCII to ensure compatibility with embedded
locale settings (Lesson 19).

### Engine Location
The compiled engine (`sutrack_fp32.engine`) lives at the SUTrack repo root.
`tracker_config.yml` references it as `../sutrack_fp32.engine` (relative to `deepstream/`).

### Phase 6 — PGIE Click-to-Select ROI
An `nvinfer` element runs a DeepStream ResNet-10 INT8 detector on the first frame.
The probe captures the frame and all `NvDsObjectMeta` detections, then signals the
main thread via `threading.Event` (Lesson 26). The main thread opens an OpenCV window
with per-class coloured boxes drawn and waits for a mouse click. When the user clicks
on a detection, the smallest enclosing bounding box is selected and passed to
`TrackerManager.initialize()`. The probe is unblocked and tracking begins from frame 1.

Fallback chain (three tiers):
1. `static_roi` in config (or `--init_bbox` CLI) — headless, no window
2. PGIE detections present — `select_bbox_click_to_select()` click window
3. PGIE disabled or no detections — `manual_roi_select()` (draw box with mouse)

`nvinfer` is wired between `nvstreammux` and the `nvvideoconvert` bridge before `nvdsosd`.
A second `nvvideoconvert(compute-hw=1)` is required after `nvinfer` and before `nvdsosd`
following DeepStream sample pipeline best practices.

Thread safety: all OpenCV GUI calls (`namedWindow`, `imshow`, `setMouseCallback`) are
confined to the Python main thread, never called inside the GStreamer pad probe (Lesson 25).

### Phase 5 — Native OSD + RTSP (Zero-Copy Drawing)
BBox drawing is moved from OpenCV CPU to DeepStream `nvdsosd` (GPU memory).
The pad probe on the `nvdsosd` sink pad populates `NvDsObjectMeta` (`rect_params`,
`text_params`) per frame; `nvdsosd` then renders boxes entirely in GPU memory.
This eliminates the CPU copy for drawing and enables headless, multi-client RTSP
streaming without a local display.

### Phase 5 — Display Branch Fix (nveglglessink NVMM surface array)
`nveglglessink` cannot directly consume NVMM surface arrays output by `nvdsosd`.
A `nvvideoconvert(compute-hw=1)` bridge is required between `q_disp` and the display
sink to translate the surface to a compatible format. `nv3dsink` (Jetson-native) is
tried first as it handles NVMM natively without the EGL copy path (Lesson 24).

### Phase 5 — RTSP Streaming
The pipeline branches at a `tee` element. The RTSP branch encodes with `nvv4l2h264enc`
(NVENC hardware block) and pipes to `udpsink` on port 5400.
A `GstRtspServer` media factory wraps that `udpsrc` and serves the stream at
`rtsp://<jetson-ip>:8554/sutrack`.  Multiple VLC clients can connect simultaneously.

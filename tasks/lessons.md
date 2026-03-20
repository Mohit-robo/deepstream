# SUTrack — Lessons Learned

Debugging and optimization lessons from the TensorRT deployment of the `sutrack_t224` model on Jetson.

---

## Lesson 1 — TensorRT input bindings must be mapped by name, not index

**Mistake:** Fed data to TRT bindings using integer index (0, 1, 2, …).

**Root cause:** TensorRT's internal binding order does not always match the ONNX `input_names` order. Feeding by index silently scrambled which tensor received which data.

**Fix:** Use `engine.get_tensor_name(i)` to build named dictionaries:
```python
inputs['template'], inputs['search'], inputs['template_anno']
```
and always copy data using those named keys.

**Rule:** Always use named tensor binding in TensorRT. Never assume index order matches ONNX declaration order.

---

## Lesson 2 — FP16 precision causes sigmoid underflow in the decoder

**Mistake:** Compiled the TensorRT engine with `--fp16`, expecting a speed boost.

**Root cause:** The SUTrack decoder uses multiple sigmoid activations. In FP16, activations with large negative logits underflow to zero, producing a `score_map` max of ~0.02 instead of ~0.55+. The tracker locked onto nothing.

**Fix:** Compile with FP32 (default — just omit `--fp16`).

**Rule:** Always use FP32 for TensorRT inference on models with sigmoid-heavy decoders unless you have validated FP16 numerics explicitly.

---

## Lesson 3 — Double sigmoid corrupts size_map and score_map

**Mistake:** Applied `sigmoid()` to `score_map` and `size_map` in the Python post-processing script.

**Root cause:** `decoder.py`'s `get_score_map()` already applies `_sigmoid()` to both maps **inside the exported ONNX graph**. Applying sigmoid a second time squashed values into the range [0.5, 1.0], effectively erasing the spatial information and producing bounding boxes covering the whole frame.

**Fix:** Never apply sigmoid to TRT outputs that already passed through the sigmoid-activated decoder graph.

**Rule:** Before post-processing TRT outputs, check `decoder.py` to see which activations are already applied inside the graph.

---

## Lesson 4 — clip_box recalculates w/h from clamped corners, causing box explosion

**Mistake:** Used the standard `clip_box` from `lib/utils/box_ops.py`.

**Root cause:** The function clamps `x1, x2, y1, y2` to frame boundaries, then recalculates `w = x2_clamped - x1_clamped`. When the prediction drifted slightly off-screen, this set `w = frame_width`. That enormous box then fed back into `sample_target`, producing a crop factor nearly equal to the full frame.  On the next frame, `resize_factor` was tiny (~0.058), which scaled the predicted box output back up by ×17.

**Fix:** Replace `clip_box` with a position-only clamp:
```python
x1 = max(0.0, min(cx - 0.5*w, W - 1.0))
y1 = max(0.0, min(cy - 0.5*h, H - 1.0))
# w and h remain unchanged
```

**Rule:** When clamping bounding boxes in a tracker, **never recalculate w/h from clamped corners**. Only clamp the position (x1, y1) while preserving the predicted size.

---

## Lesson 5 — CPU-based CenterPredictor causes 0.9 FPS

**Mistake:** Instantiated a full PyTorch `CenterPredictor` CPU model every frame to replicate `cal_bbox` logic.

**Root cause:** The PyTorch model initialization overhead plus CPU execution reduced throughput to 0.9 FPS on Jetson, overwhelming the TRT speedup entirely.

**Fix:** Replace with a ~12-line pure NumPy implementation:
```python
score_flat = score_map.reshape(-1)
idx = int(np.argmax(score_flat))
idx_y, idx_x = idx // feat_sz, idx % feat_sz
cx_norm = (idx_x + off_x) / feat_sz
cy_norm = (idx_y + off_y) / feat_sz
w = size_map[0, idx] * scale
h = size_map[1, idx] * scale
```
This lifted FPS from **0.9 → ~25 FPS**.

**Rule:** All post-processing in a TRT inference loop must be pure NumPy or CUDA — never instantiate a PyTorch model in the tracking loop.

---

## Lesson 6 — Hanning window must be applied BEFORE argmax, not after

**Mistake:** Applied the Hanning window penalty to the score map *after* finding the peak location.

**Root cause:** The window is a spatial suppression mechanism. Applying it after `argmax` has no effect on peak selection.

**Fix:** Apply the Hanning window to the full score map first:
```python
score_map = score_map * hann_window  # (feat_sz, feat_sz)
idx = int(np.argmax(score_map.reshape(-1)))
```

**Rule:** Always apply the Hanning window to the full score map before the argmax operation.

---

## Lesson 7 — pycuda unavailable on Jetson; use cuda.cudart instead

**Mistake:** Initially used `pycuda` for GPU memory allocation.

**Root cause:** `pycuda` is not available on all Jetson configurations and may conflict with the system TensorRT environment.

**Fix:** Replace all `pycuda` calls with `cuda.cudart` ctypes bindings:
```python
from cuda import cudart
err, stream = cudart.cudaStreamCreate()
err, device_mem = cudart.cudaMalloc(nbytes)
err, host_mem = cudart.cudaMallocHost(nbytes)
host_np = np.frombuffer((ctypes.c_char * nbytes).from_address(int(host_mem)), dtype=dtype)
```

**Rule:** Use `cuda-python` (`cuda.cudart`) in deployments targeting Jetson. It is the officially supported Python CUDA interface in JetPack environments.

---

## Lesson 8 — cudaMallocHost returns an int pointer, not a ctypes buffer

**Mistake:** Tried to use the MRO chain to create a buffer from the host pointer returned by `cudaMallocHost`.

**Root cause:** `cudart.cudaMallocHost` returns a plain Python `int` (the raw memory address). Directly passing it to NumPy's `frombuffer` fails.

**Fix:** Wrap with `ctypes`:
```python
host_np = np.frombuffer(
    (ctypes.c_char * nbytes).from_address(int(host_mem)),
    dtype=dtype).reshape(shape)
```

**Rule:** Always wrap raw `cudaMallocHost` pointers using `ctypes.c_char.from_address()` before passing to NumPy.

---

## Lesson 9 — OpenCV Qt builds require imshow before selectROI

**Mistake:** Called `cv.selectROI(win, disp)` immediately after `cv.namedWindow()`.

**Root cause:** On Qt-based OpenCV builds, `namedWindow` alone does not fully initialize the window's mouse callback handler. `selectROI` internally calls `cvSetMouseCallback`, which fails with a null window handler.

**Fix:** Call `cv.imshow(win, disp)` and `cv.waitKey(1)` before `selectROI`:
```python
cv.imshow(win, disp)
cv.waitKey(1)
x, y, w, h = cv.selectROI(win, disp, fromCenter=False)
```

**Rule:** On Qt OpenCV builds, always call `imshow` before `selectROI` to populate and initialize the window.

---

## Lesson 10 — ONNX export requires unfold → avg_pool2d replacement

**Mistake:** Assumed the base SUTrack model could be directly exported via `torch.onnx.export`.

**Root cause:** `prepare_tokens_with_masks` uses `torch.Tensor.unfold()` to downsample the template mask. ONNX opset 11 does not support this operation pattern.

**Fix:** Monkey-patch the function to use `F.avg_pool2d` with `kernel_size=patch_size, stride=patch_size`, which is ONNX-compatible and produces an identical result.

**Rule:** Before exporting to ONNX, trace through the full forward pass looking for Python control flow and unsupported ops (`unfold`, dynamic shapes, Python conditionals, list inputs). Wrap or replace each one.

---

## Lesson 11 — trtexec requires explicit shape specs for dynamic-axis ONNX models

**Mistake:** Running `trtexec --onnx=model.onnx --saveEngine=model.engine` without shape specs.

**Root cause:** The exported ONNX has dynamic batch/spatial axes. Without explicit `--minShapes`/`--optShapes`/`--maxShapes`, `trtexec` either fails or produces an engine that errors at runtime.

**Fix:**
```bash
trtexec --onnx=sutrack_t224.onnx --saveEngine=sutrack_t224.engine \
    --minShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --optShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --maxShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --memPoolSize=workspace:4096MiB
```

**Rule:** Always provide `--minShapes`, `--optShapes`, and `--maxShapes` when the ONNX model has dynamic axes. Use `--memPoolSize` instead of the deprecated `--workspace`.

---

## Lesson 12 — BGR vs RGB colour space mismatch

**Mistake:** Passed the raw OpenCV frame (BGR) directly to the tracker's `preprocess()` function.

**Root cause:** OpenCV reads video frames in BGR format, but SUTrack (like most vision models) expects RGB input. This produced subtly wrong embeddings and poor tracking.

**Fix:** Convert before every tracker call:
```python
frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
tracker.track(frame_rgb)
```

**Rule:** Always convert BGR → RGB when passing OpenCV frames to a PyTorch or NumPy preprocessing pipeline trained on RGB images.

---

## Lesson 13 — GStreamer appsink must have emit-signals and drop enabled

**Context:** DeepStream / GStreamer Python pipeline for live frame processing.

**Mistake:** Used appsink without configuring it properly, causing pipeline stalls or frames piling up.

**Root cause:** Without `drop=true`, the pipeline stalls waiting for Python to consume every frame. Without `emit-signals=true`, the Python `new-sample` callback is never triggered.

**Fix:**
```python
pipeline_str = (
    '... ! appsink name=appsink '
    'emit-signals=true max-buffers=1 drop=true sync=false'
)
appsink.connect('new-sample', on_new_sample, state)
```

**Rule:** Always set `emit-signals=true max-buffers=1 drop=true sync=false` on appsink in real-time tracking pipelines.

---

## Lesson 14 — NvBufSurface produces different pixel formats depending on the pipeline

**Context:** Converting GStreamer buffer → NumPy array inside the appsink callback.

**Mistake:** Assumed the buffer always arrives as `RGBA` and used a fixed reshape.

**Root cause:** Different decode paths produce different formats: `nvvideoconvert` on Jetson produces `RGBA`, software `videoconvert` may produce `BGR`, camera sources often emit `NV12` or `I420`. Using a fixed format causes reshape failures or silent colour corruption.

**Fix:** Read the caps `format` string and branch explicitly:
```python
fmt = caps.get_structure(0).get_string('format')
if fmt in ('RGBA', 'RGBx'):
    img = data.reshape(H, W, 4)
    bgr = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
elif fmt in ('NV12',):
    yuv = data.reshape(H * 3 // 2, W)
    bgr = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_NV12)
```

**Rule:** Never assume a fixed pixel format from appsink. Always read `format` from GStreamer caps and handle each case explicitly.

---

## Lesson 15 — Video writer must be lazily initialized after the first frame

**Context:** Saving tracked output video in the appsink callback.

**Mistake:** Tried to create `cv.VideoWriter` at pipeline startup before any frames arrive.

**Root cause:** Frame dimensions are not known until the first decoded frame arrives from the pipeline.

**Fix:** Initialize lazily on the first frame:
```python
if state.writer is None and state.output_path:
    H, W = frame_bgr.shape[:2]
    state.writer = cv.VideoWriter(path, fourcc, fps, (W, H))
```

**Rule:** Never create `cv.VideoWriter` before receiving the first frame. Always initialize it lazily with actual decoded frame dimensions.

---

## Lesson 16 — Engine path in config must be anchored to the script, not cwd

**Context:** `tracker_config.yml` stores `engine_path: ../sutrack_fp32.engine`.

**Mistake:** Resolved the path relative to `os.getcwd()`.

**Root cause:** If the script is run from a different directory the relative path resolves incorrectly and the engine fails to load.

**Fix:** Anchor relative paths to `__file__`, not cwd:
```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
engine_path = os.path.normpath(os.path.join(ROOT, model_cfg['engine_path']))
```

**Rule:** Always resolve config-relative paths from the project root computed via `__file__`. Document this convention in config file comments.

---

## Lesson 17 — GStreamer pipelines require GLib.MainLoop for EOS/error handling

**Context:** Building a GStreamer pipeline with bus-based EOS/error callbacks.

**Mistake:** Used a `while True` polling loop instead of a GLib event loop.

**Root cause:** GStreamer buses are event-driven. Without a GLib main loop, bus callbacks (EOS, error) never fire and `Ctrl+C` leaves dangling GStreamer threads.

**Fix:**
```python
loop = GLib.MainLoop()
bus.add_signal_watch()
bus.connect('message::eos',   on_eos,   state)
bus.connect('message::error', on_error, state)
pipeline.set_state(Gst.State.PLAYING)
try:
    loop.run()
finally:
    pipeline.set_state(Gst.State.NULL)
```

**Rule:** Always use `GLib.MainLoop` with `bus.add_signal_watch()` for GStreamer Python pipelines. Never poll with a busy loop.

---

## Lesson 18 — Multi-tracker with one TRT engine requires sequential inference (batch=1)

**Context:** Multiple `TrackerInstance` objects sharing one `SUTrackEngine`.

**Mistake:** Attempted to batch all tracker inferences into a single TRT call.

**Root cause:** The engine is compiled with fixed batch size 1 (`minShapes`/`maxShapes` both set to 1). Template buffers differ per tracker, so passing batch > 1 would require recompiling with a dynamic batch dimension.

**Fix:** Run `engine.infer()` sequentially once per active tracker. Each `TrackerInstance` stores its own template buffers and copies them into the shared engine buffers just before `infer()`.

**Rule:** For a batch=1 engine, use sequential-per-tracker inference. Only add batching complexity after profiling confirms it as the bottleneck - and only after recompiling the engine with a dynamic batch dimension.

---

## Lesson 19 - Avoid non-ASCII characters in deployment scripts

**Context:** Coding docstrings and comments for Python scripts on Jetson/embedded.

**Mistake:** Used characters like `—` (em-dash), `→`, `←`, or `×` in the file.

**Root cause:** Many embedded environments (Jetson/headless) default to `ASCII` or `C.UTF-8` locale which might cause `SyntaxError` when Python 3 encounters non-ASCII bytes in a docstring if the encoding isn't explicitly declared or handled properly by the terminal.

**Fix:** Standardize on standard ASCII characters for all code, comments, and docstrings (`-`, `->`, `<-`, `x`).

**Rule:** Use only standard ASCII characters in `.py` and `.sh` scripts. Never use special symbols like smart quotes, long dashes, or mathematical operators unless required by data.

---

---

## Lesson 21 - Standard videoconvert cannot handle NVMM memory

**Context:** Bridging hardware decoder output to an appsink.

**Mistake:** Attempted to use software `videoconvert` directly after `nvv4l2decoder` (or `decodebin`).

**Root cause:** Hardware decoders on Jetson output `NVMM` (GPU) memory buffers. The standard GStreamer `videoconvert` only works with system RAM. Connecting them directly leads to a `"not-negotiated (-4)"` streaming error.

**Fix:** Always use `nvvideoconvert` first to bridge out of `NVMM` memory into system RAM before passing to standard `videoconvert` or `appsink`.

**Rule:** Never connect a standard `videoconvert` directly to a hardware decoder. Always use `nvvideoconvert` as the bridge from GPU memory to system RAM.

---

## Lesson 22 - Phase 5 pipeline requires element-by-element construction (not parse_launch)

**Context:** Building the Phase 5 nvstreammux + nvdsosd + tee pipeline.

**Mistake (avoided):** Using Gst.parse_launch() for a pipeline containing decodebin.

**Root cause:** decodebin creates its src pads dynamically (pad-added signal). parse_launch links are resolved statically at launch time and cannot handle dynamic pads that appear after the pipeline enters PLAYING state.

**Fix:** Build the pipeline element-by-element and connect the `pad-added` signal manually.

**Rule:** Always use element-by-element construction when any element uses dynamic pads (decodebin, uridecodebin, qtdemux).

---

## Lesson 23 - nvdsosd process-mode=0 (CPU) is safer for initial deployment

**Context:** Phase 5 nvdsosd configuration on Jetson.

**Mistake (potential):** Setting process-mode=1 (GPU) on a system without EGL display.

**Root cause:** GPU mode (process-mode=1) in nvdsosd requires an EGL context. Without DISPLAY=:0, it fails with EGL init errors.

**Fix:** Default to process-mode=0 (CPU) during initial deployment.

**Rule:** Default nvdsosd to CPU mode (0) for headless/SSH deployments until EGL availability is confirmed.

---

## Lesson 24 - nveglglessink cannot consume NVMM surface arrays from nvdsosd

**Context:** Phase 5 display branch: `nvdsosd -> tee -> queue -> nveglglessink`.

**Mistake:** Linking `q_disp` directly to `nveglglessink` after the tee.

**Root cause:** `nveglglessink` fails to copy the NVMM surface array output by `nvdsosd` without a conversion bridge. This causes a `GST_FLOW_ERROR` that crashes the pipeline.

**Fix:** Insert `nvvideoconvert(compute-hw=1)` before the display sink or use `nv3dsink`.

**Rule:** Always bridge `nvdsosd` to `nveglglessink` using `nvvideoconvert`. Prefer `nv3dsink` on Jetson.

---

## Lesson 25 - OpenCV GUI functions must run in the Python Main Thread

**Context:** Opening `cv2.selectROI` from a GStreamer Pad Probe.

**Mistake:** Calling `cv2.namedWindow` or `cv2.selectROI` directly inside the `tracker_probe` callback.

**Root cause:** GStreamer probes run in a background streaming thread. OpenCV's GUI backends (Qt/GTK) are not thread-safe and require all window management to occur on the application's primary thread. Calling them from a probe causes `assertion 'acquired_context' failed` or segmentation faults.

**Fix:** Capture the frame in the probe, signal a `threading.Event`, and perform the GUI selection in the `main()` function thread before starting the GLib loop.

**Rule:** Never call OpenCV GUI functions (`imshow`, `selectROI`, `waitKey`) inside a GStreamer callback or probe. Always delegate GUI interaction to the main thread.

---

## Lesson 26 - Use threading.Event to synchronize interactive ROI with video streams

**Context:** Pausing a live stream while a user draws a bounding box.

**Mistake:** Attempting to asynchronously pause the pipeline while the user is selecting an ROI.

**Root cause:** GStreamer pipelines are high-speed. Simple async pauses often take several frames to propagate, causing the user to select an ROI on "Frame 1" but the tracker to start on "Frame 50".

**Fix:** Use a `threading.Event` to physically **block** the GStreamer probe thread after it captures the first frame. The stream will "freeze" exactly at the point of capture and only resume after the main thread signals that the selection is complete.

**Rule:** Use `Event.wait()` inside a pad probe to force-synchronize human interaction with real-time streaming threads.

---

## Lesson 27 - Metadata cleanup required after one-shot PGIE detection

**Context:** Using a PGIE detector only for the first frame to enable "Click-to-Select" ROI.

**Mistake:** Assuming that filtering the selection logic in Python would hide the detector's boxes.

**Root cause:** DeepStream's `nvinfer` (PGIE) continues to run and attach bounding boxes (in its own thread) to every frame. Even if Python ignores them, the downstream `nvdsosd` element will automatically draw every box it finds in the metadata.

**Fix:** Inside the `tracker_probe`, explicitly clear the `obj_meta_list` using `pyds.nvds_remove_obj_meta_from_frame` for all frames after initialization.

**Rule:** If a detector is used for initialization only, its metadata must be explicitly stripped from the pipeline after selection to prevent "distraction" boxes from appearing on the stream.

---

## Lesson 28 - Dynamically increase PGIE interval for "One-Shot" power-save

**Context:** Reducing GPU load once a detector's task is done.

**Mistake:** Letting the detector run at `interval=0` for the entire session.

**Root cause:** Running a PGIE (like ResNet-10) on every frame consumes significant GPU cycles and power, even if the results are being discarded/hidden.

**Fix:** Store a reference to the `nvinfer` GSt element and set its `interval` property to a very high number (e.g., 10,000) as soon as the tracker is initialized. This effectively pauses the detector without needing a pipeline state change.

**Rule:** Always optimize "initialization-only" elements by dynamically adjusting their properties to a "sleep" state once they are no longer needed.

---

## Lesson 29 - Detectors can attach info to both obj_meta and display_meta

**Context:** Cleaning up metadata to ensure a single-object output.

**Mistake:** Only clearing `obj_meta_list`.

**Root cause:** Some detectors or GIE configurations attach text labels or analytics to `display_meta_list` instead of (or in addition to) the bounding box in `obj_meta_list`. If only one is cleared, "ghost" labels or confidence scores might persist.

**Fix:** Clear both lists in the probe:
```python
while frame_meta.obj_meta_list:
    pyds.nvds_remove_obj_meta_from_frame(frame_meta, ...)
while frame_meta.display_meta_list:
    pyds.nvds_remove_display_meta_from_frame(frame_meta, ...)
```

**Rule:** When stripping metadata for a clean output, always check and clear both `obj_meta_list` and `display_meta_list`.

---

## Lesson 30 — Center-distance is more robust than IOU for drift detection

**Context:** Detecting when SUTrack drifts onto a distractor in a crowded scene.

**Problem:** In overlapping crowds, IOU can remain high even if the tracker has switched from the target's chest to a distractor's shoulder.

**Root cause:** IOU measures area overlap, but not alignment.

**Fix:** Implement a center-distance guard. Calculate the Euclidean distance between the SUTrack box center and the Native (NvDCF) leader center. If the distance exceeds a percentage of the box width (e.g., 15-25%), trigger a re-sync.

**Rule:** Use geometric center-distance as a primary "Drift Guard" in hybrid tracking scenarios.

---

## Lesson 31 — Native tracker "Leader Poisoning" in Hybrid Mode

**Context:** Re-syncing SUTrack to the NvDCF bounding box when drift is detected.

**Problem:** If the Native tracker (NvDCF) ITSELF drifts onto a distractor, it "drags" a healthy SUTrack along with it.

**Root cause:** Blindly trusting the Native leader as the ground truth.

**Fix:** Implement "Skeptical Hybrid" logic. Only trigger a re-sync if SUTrack's internal confidence is also low (< 0.2). If SUTrack is confident but far from the leader, it might actually be correct while the leader is drifting.

**Rule:** Never allow an external tracker to force a re-sync on a high-confidence tracker.

---

## Lesson 32 — NvDsDisplayMeta is required for "Forced" debug overlays

**Context:** Implementing `--debug-boxes` to see raw tracker states.

**Problem:** Raw boxes added via `NvDsObjectMeta` would disappear because downstream components or UI logic were filtering objects by `class_id` or `unique_component_id`.

**Root cause:** `obj_meta` is part of the inference stream and is subject to filtering/aggregation logic.

**Fix:** Use `NvDsDisplayMeta` for raw debug visualization. Items in DisplayMeta (lines, rects, text) are "passive" overlays that are drawn directly by OSD and are not filtered as "objects".

**Rule:** Use `DisplayMeta` for internal debug visualization to ensure standard object-filtering logic doesn't hide your debug data.

---

## Lesson 33 — DeepStream OSD requires has_bg_color to be explicitly zeroed

**Context:** Drawing raw rectangles using `NvDsDisplayMeta`.

**Problem:** Debug boxes appeared as solid colored blocks instead of hollow rectangles.

**Root cause:** The `has_bg_color` field in `NvDsRectParams` (inside DisplayMeta) defaults to a non-zero value or uninitialized garbage, causing the OSD to fill the rectangle.

**Fix:** Explicitly set `r_meta.has_bg_color = 0` for every debug rectangle.

**Rule:** Always explicitly initialize `has_bg_color` and `border_width` for every rectangle in `DisplayMeta`.

---

## Lesson 34 — TRT inference inside a GStreamer pad probe blocks the entire pipeline

**Context:** `deepstream_desktop_app.py` (Phase 10 desktop app) — FPS dropped from 60 to 20-24 when entering LOCKED state.

**Mistake:** Called `manager.update()` (SUTrack TRT inference, ~30 ms) directly inside the GStreamer pad probe callback.

**Root cause:** A GStreamer pad probe runs on the streaming thread. Any blocking work inside the probe stalls the entire pipeline until it returns. With TRT inference taking ~30 ms per frame, the pipeline was capped at ~33 FPS maximum, and additional work (frame copy, histogram) pushed it to 20-24 FPS.

The pattern looks innocent:
```python
def tracker_probe(pad, info, state):
    ...
    results = state.manager.update(rgb, frame_idx)  # BLOCKS for ~30 ms
    ...
    return Gst.PadProbeReturn.OK  # pipeline resumes here
```

**Fix:** Move all `manager` operations into a dedicated `TRTWorkerThread`. The probe:
1. Copies the frame reference into the worker's pending slot (< 0.1 ms)
2. Signals the worker thread
3. Reads the *last computed result* for OSD rendering
4. Returns immediately

The worker thread processes frames at TRT speed (~30 FPS) while the GStreamer pipeline runs at source rate (60 FPS). OSD shows 1-2 frame lag on the bbox — imperceptible.

```python
class TRTWorkerThread(threading.Thread):
    def submit_track(self, rgb, frame_idx):
        with self._lock:
            self._pending = ('track', rgb, frame_idx)  # overwrite if busy
        self._trigger.set()

# In probe:
state.trt_worker.submit_track(rgb, state.frame_idx)  # non-blocking
results = state.trt_worker.last_result               # last known bbox
```

Additional gains from the same fix:
- Throttle `id_history.update()` to every 20 frames (histogram every frame was ~3 ms/frame wasted)
- Throttle `GLib.idle_add` to every 5 frames (label updates at 12 Hz is sufficient)
- Use `rgba[:, :, :3]` slice instead of `cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)` (avoids one full-frame allocation)

**Result:** Pipeline FPS in LOCKED mode: 20-24 FPS → ~55-60 FPS. TRT inference rate unchanged (~28-33 FPS, hardware-limited).

**Rule:** Never run TRT inference (or any operation > 2 ms) directly inside a GStreamer pad probe. Use a worker thread with a pending-slot double-buffer pattern. The probe submits work and reads the last result immediately.

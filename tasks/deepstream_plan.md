You are responsible for building a production-ready NVIDIA DeepStream application that integrates the SUTrack tracker.

Your goal is to create a DeepStream pipeline that uses SUTrack as the tracking module while maintaining high performance and modular code design.

You must follow strict engineering practices and produce maintainable code.

---

# Primary Objective

Develop a DeepStream-based tracking application that:

1. Receives video input from file, RTSP stream, or camera
2. Uses SUTrack as the tracking module
3. Maintains object identities across frames
4. Outputs tracking results with bounding boxes and IDs

The system should support real-time inference on Jetson hardware.

---

# What We Already Have

From the TensorRT deployment experiment, the following is complete and working:

## ONNX Export — `tracking/export_onnx.py`

The `sutrack_t224` model has been exported to ONNX using a custom wrapper.

Key customisations that were required:
- `SUTrackWrapper`: collapses the multi-list Python API into three named tensors
- `unfold()` replaced with `F.avg_pool2d(kernel_size=patch_size, stride=patch_size)` for ONNX compatibility

Export command:
```bash
python tracking/export_onnx.py --param sutrack_t224
```

## TensorRT Engine

Compile with (Jetson, FP32):
```bash
trtexec --onnx=sutrack_t224.onnx --saveEngine=sutrack_t224.engine \
    --minShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --optShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --maxShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --memPoolSize=workspace:4096MiB
```

> **Critical:** Do NOT use `--fp16`. SUTrack decoder uses sigmoid activations that
> underflow to zero in FP16, producing near-zero scores and broken tracking.

## Standalone TRT Package — `deploy_trt/`

A fully working, PyTorch-free inference package already exists:

```
deploy_trt/
├── demo_trt.py   ← main tracking script
├── utils.py      ← pure-NumPy preprocessing, crop extraction, hann2d, config loader
└── README.md
```

Runtime dependencies (no PyTorch required):
- `numpy`, `opencv-python`, `tensorrt`, `cuda-python`, `pyyaml`

Confirmed performance on Jetson (sutrack_t224, FP32 engine):
- Average FPS: **~25 FPS** over 184 frames

---

# Model Specification (sutrack_t224)

These are the validated, correct I/O specs for the TensorRT engine.

## TRT Engine Inputs

| Name | Shape | Description |
|------|-------|-------------|
| `template` | `(1, 6, 112, 112)` | Template crop. 6ch = RGB duplicated (multi-modal compatibility) |
| `search` | `(1, 6, 224, 224)` | Search area crop. 6ch = RGB duplicated |
| `template_anno` | `(1, 4)` | Normalized `[x1, y1, w, h]` annotation of target in template |

## TRT Engine Outputs

| Name | Shape | Notes |
|------|-------|-------|
| `score_map` | `(1, 1, 14, 14)` | Already sigmoid-activated inside graph |
| `size_map` | `(1, 2, 14, 14)` | Already sigmoid-activated inside graph — `[w_norm, h_norm]` |
| `offset_map` | `(1, 2, 14, 14)` | Raw logits — sub-cell offset `[off_x, off_y]` |
| `pred_boxes` | `(1, 4)` | Internal cal_bbox output — not used (bypassed by Hanning window logic) |

Feature map size: `search_size / encoder_stride = 224 / 16 = 14`

## Preprocessing

- Extract crop using `sample_target(image_rgb, state, factor, output_size)`
  - template: `factor=2.0, output_size=112`
  - search: `factor=4.0, output_size=224`
- Normalize (ImageNet mean/std): `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`
- Stack 3-ch → 6-ch: `np.concatenate([img, img], axis=0)`
- Expand batch dim → shape `(1, 6, H, W)`, dtype `float32`

## Post-processing (validated, pure NumPy)

```python
score_map  = trt_out['score_map'].reshape(14, 14)
size_map   = trt_out['size_map'].reshape(2, 14*14)
offset_map = trt_out['offset_map'].reshape(2, 14*14)

# 1. Apply Hanning window BEFORE argmax
score_map = score_map * hann2d(14, 14)

# 2. Find peak
idx = int(np.argmax(score_map.reshape(-1)))
idx_y, idx_x = idx // 14, idx % 14

# 3. Decode position and size
scale = 224 / resize_factor  # crop_sz
cx = (idx_x + offset_map[0, idx]) / 14 * scale
cy = (idx_y + offset_map[1, idx]) / 14 * scale
w  = size_map[0, idx] * scale
h  = size_map[1, idx] * scale

# 4. Map from crop-space to original image
cx_real = cx + (cx_prev - scale/2)
cy_real = cy + (cy_prev - scale/2)

# 5. Position-only clamp (NEVER recalculate w/h from clamped corners)
x1 = max(0, min(cx_real - w/2, W - 1))
y1 = max(0, min(cy_real - h/2, H - 1))
state = [x1, y1, w, h]
```

---

# SUTrack Integration Strategy for DeepStream

SUTrack is a **single-object tracker** — one engine instance tracks one target.

For multi-object tracking in DeepStream, use the **Tracker Manager** approach:

## TrackerManager Design

```python
class TrackerInstance:
    object_id: int
    state: [x, y, w, h]       # current bounding box
    template_img: np.ndarray  # (1, 6, 112, 112) float32
    template_anno: np.ndarray # (1, 4) float32
    last_seen_frame: int
    confidence: float

class TrackerManager:
    active_trackers: dict[int, TrackerInstance]

    def initialize(object_id, frame_rgb, bbox): ...
    def update(frame_rgb) -> dict[int, bbox]: ...
    def remove_stale(current_frame, max_age=30): ...
```

## Tracker Lifecycle

1. `PGIE` detects objects → bounding boxes with class labels
2. For each **new** detection: `TrackerManager.initialize(id, frame, bbox)`
3. Each frame: `TrackerManager.update(frame)` → returns {id: bbox} for all active trackers
4. Boxes not updated for `max_age` frames are removed

## Shared TRT Engine (Important)

All `TrackerInstance` objects should share **one TRT context** to avoid GPU memory duplication. The template and search buffers are swapped per-tracker before inference.

---

# System Architecture

```
Video Source (file / RTSP / camera)
    │
nvdec (hardware decoder)
    │
nvstreammux
    │
nvinfer (PGIE — object detector, e.g. YOLOv8-NMS)
    │
Python Probe / Custom nvtracker plugin
    │   ├── extract frame (NvBufSurface → np / cv2)
    │   ├── TrackerManager.update(frame_rgb, detections)
    │   └── write bboxes back to NvDsObjectMeta
    │
nvdsosd (on-screen display)
    │
sink (file / RTSP / display)
```

---

# Repository Structure

```
deepstream_sutrack/
├── apps/
│   └── deepstream_tracker_app.py     # main pipeline entry point
├── tracker/
│   ├── sutrack_engine.py             # TRT engine wrapper (load, allocate, infer)
│   ├── tracker_instance.py           # single-object tracker state
│   ├── tracker_manager.py            # multi-object lifecycle manager
│   └── tracker_utils.py             # preprocessing, postprocessing (from deploy_trt/utils.py)
├── models/
│   ├── sutrack_t224.onnx             # exported ONNX
│   └── sutrack_t224.engine           # compiled TRT engine
├── configs/
│   ├── deepstream_app_config.txt     # DeepStream pipeline config
│   ├── pgie_config.txt               # detector config
│   └── tracker_config.yml            # tracker thresholds, max_age, confidence
├── scripts/
│   ├── export_onnx.py                # ONNX export (from tracking/export_onnx.py)
│   └── build_engine.sh               # trtexec command
└── docs/
    ├── architecture.md
    └── usage.md
```

---

# Configuration File Template

`configs/tracker_config.yml`:
```yaml
model:
  engine_path: models/sutrack_t224.engine
  template_size: 112
  search_size: 224
  template_factor: 2.0
  search_factor: 4.0
  encoder_stride: 16
  use_hanning_window: true

tracker:
  max_age: 30           # frames before removing a lost tracker
  min_confidence: 0.25  # minimum detector confidence to initialize a tracker
  iou_match_threshold: 0.3
```

---

# Known Pitfalls (from experiment)

These must be baked into the implementation from the start:

| # | Pitfall | Rule |
|---|---------|------|
| 1 | FP16 engine causes sigmoid underflow → scores ~0.02 | Always compile with FP32 |
| 2 | Double sigmoid on `score_map`/`size_map` | They are already sigmoided inside TRT graph — do not apply again |
| 3 | `clip_box` recalculates `w/h` from clamped corners | Position-only clamp only — preserve predicted `w, h` |
| 4 | Hanning window applied after argmax | Always apply window **before** argmax |
| 5 | BGR input to tracker | Always `cvtColor(BGR → RGB)` before preprocessing |
| 6 | TRT binding by index | Always bind by name (`inputs['template']`, etc.) |
| 7 | `pycuda` unavailable on Jetson | Use `cuda.cudart` with ctypes wrapping |

---

# Performance Requirements

Target platform: **Jetson Orin / Xavier (JetPack 5+)**

| Metric | Target |
|--------|--------|
| Tracker FPS (TRT, 1 object) | ≥ 25 FPS |
| Multi-object overhead | Linear in active tracker count |
| GPU preprocessing | Preferred over CPU resize |
| Inference precision | FP32 (mandatory) |

Avoid:
- Instantiating any PyTorch model in the tracking loop
- CPU-based bounding box computations (use NumPy, not PyTorch CPU)
- Re-loading the TRT engine per frame

---

# Coding Standards

1. Modular design: separate pipeline, tracking logic, and inference
2. Class-based: `SUTrackEngine`, `TrackerInstance`, `TrackerManager`
3. Docstrings on all modules
4. No hardcoded paths — use `tracker_config.yml`
5. Logging for: init, update, delete, fps, inference time
6. Review `tasks/lessons.md` before implementing any inference logic

---

# Validation Strategy

Test in order:

1. **Single object, full video** — verify IDs stay consistent
2. **Object leaving frame** — verify tracker is removed after `max_age`
3. **Multiple objects** — verify separate IDs, no ID swaps
4. **Object re-entry** — verify new ID is assigned (expected)

---

# Deliverables

1. Working DeepStream pipeline (`deepstream_tracker_app.py`)
2. `SUTrackEngine` — TRT inference wrapper using `cuda.cudart`
3. `TrackerManager` — multi-object lifecycle
4. `tracker_utils.py` — pure-NumPy preprocessing (reuse from `deploy_trt/utils.py`)
5. `build_engine.sh` — trtexec with correct shape specs
6. `configs/tracker_config.yml` — full config template
7. `docs/architecture.md` — pipeline diagram and design decisions

---

# Constraints

1. Do not modify the original SUTrack repository — use wrappers only
2. No PyTorch at runtime — NumPy + TensorRT + cuda.cudart only
3. Must be compatible with DeepStream SDK 6.x / 7.x (Python bindings)
4. The `deploy_trt/utils.py` already contains the validated pure-NumPy utilities — reuse directly

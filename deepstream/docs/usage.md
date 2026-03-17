# DeepStream SUTrack -- User Manual

Full setup-to-run guide for the DeepStream SUTrack tracker on Jetson.

---

## 1. System Requirements

| Item | Requirement |
|------|------------|
| Device | NVIDIA Jetson (Orin / Xavier / AGX) |
| Software | JetPack 5.x with DeepStream 6.x or 7.x |
| Python | 3.8+ (pre-installed with JetPack) |
| Engine | sutrack_fp32.engine (FP32 mandatory -- see section 3) |

---

## 2. Install Dependencies

```bash
# DeepStream Python bindings (already in JetPack; install the wheel)
pip install /opt/nvidia/deepstream/deepstream/lib/pyds-*.whl

# Python packages (no PyTorch required)
pip install numpy opencv-python pyyaml cuda-python

# GStreamer Python bindings
sudo apt-get install -y \
    python3-gi python3-gst-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good

# Verify
python -c "import pyds; print('OK pyds')"
python -c "from cuda import cudart; print('OK cudart')"
python -c "import tensorrt as trt; print('OK TRT', trt.__version__)"
python -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; print('OK GStreamer')"
```

---

## 3. Build the TensorRT Engine (one-time)

Run from the `~/SUTrack` repo root.

```bash
cd ~/SUTrack

# Step 3a -- export ONNX
python tracking/export_onnx.py --param sutrack_t224
# Output: sutrack.onnx  (repo root)

# Step 3b -- compile TRT engine
bash deepstream/scripts/build_engine.sh sutrack.onnx sutrack_fp32.engine
# Output: sutrack_fp32.engine  (repo root)
```

> **WARNING: Do NOT add `--fp16`.**
> The SUTrack decoder uses sigmoid activations throughout.
> FP16 causes them to underflow to zero -- tracker sees no peaks -- broken output.
> Always compile FP32 (the default when `--fp16` is omitted).

Expected build time on Jetson Orin: ~3-5 minutes.

---

## 4. Configure

Edit `deepstream/configs/tracker_config.yml`:

```yaml
model:
  engine_path: ../sutrack_fp32.engine   # relative to deepstream/ -> repo root
  template_size: 112
  search_size:   224
  template_factor: 2.0
  search_factor:   4.0
  encoder_stride:  16
  use_hanning_window: true

tracker:
  max_age:            30    # frames before removing a lost tracker
  min_confidence:     0.25
  iou_match_threshold: 0.3
  static_roi: ""            # "X,Y,W,H" for headless init; "" = interactive

pipeline:
  input_source: ""     # "" = camera; file path or rtsp:// for video
  output_path:  ""     # Phase 4 only: save annotated MP4 here
  headless:     false
  log_level:    INFO

  # Phase 5: nvstreammux resolution (must match or be >= source resolution)
  width:  1280
  height:  720

  # Phase 5: RTSP streaming
  rtsp_enabled:   true
  rtsp_port:      8554
  rtsp_udp_port:  5400
  rtsp_path:      /sutrack
  rtsp_bitrate:   4000000   # H.264 target bitrate in bps

  # Phase 6: PGIE click-to-select ROI
  # Path relative to deepstream/ directory; leave empty or omit to disable.
  # When set, nvinfer runs on frame 0 and shows detected boxes for user selection.
  pgie_config: configs/pgie_config.txt
```

Key parameter guide:

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `engine_path` | Path to `.engine` (relative to `deepstream/`) | `../sutrack_fp32.engine` |
| `template_factor` | Crop size for template = `factor x bbox_size` | `2.0` |
| `search_factor` | Crop size for search = `factor x bbox_size` | `4.0` |
| `max_age` | Frames a track can miss before being removed | `30` |
| `use_hanning_window` | Spatial suppression of distractors | `true` |
| `static_roi` | Skip GUI; format `"X,Y,W,H"` e.g. `"300,200,100,100"` | `""` |
| `headless` | Run with no display (Phase 4 legacy flag) | `false` |
| `rtsp_enabled` | Enable RTSP stream output (Phase 5) | `true` |
| `rtsp_port` | GstRtspServer listen port | `8554` |
| `rtsp_bitrate` | H.264 encoder bitrate in bps | `4000000` |
| `pgie_config` | Path to PGIE config file (relative to `deepstream/`). Enables click-to-select ROI on first frame. Empty = disabled. | `configs/pgie_config.txt` |

---

## 5. Run -- Phase 5 (Native OSD + RTSP, Recommended)

All commands are run from `~/SUTrack` (repo root).

### 5a -- Headless with static ROI (most common production use)

Set `static_roi` in `tracker_config.yml` and run:

```bash
cd ~/SUTrack
python deepstream/apps/deepstream_rtsp_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```

On another machine, open the stream:

```bash
vlc rtsp://<jetson-ip>:8554/sutrack
```

---

### 5b -- With local display

If a monitor is connected to the Jetson, the pipeline opens `nv3dsink` for local
preview in addition to the RTSP stream:

```bash
export DISPLAY=:0
python deepstream/apps/deepstream_rtsp_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```

---

### 5c -- Camera input (Phase 5)

Leave `input_source` empty (or pass `--input ""`):

```bash
python deepstream/apps/deepstream_rtsp_app.py \
    --config deepstream/configs/tracker_config.yml
```

A camera on `/dev/video0` is used. Set `static_roi` in the config for headless init.

---

### 5d -- Phase 6: PGIE Click-to-Select ROI (Recommended for interactive use)

Ensure `pgie_config` is set in `tracker_config.yml` (default: `configs/pgie_config.txt`),
then run normally:

```bash
export DISPLAY=:0    # required for OpenCV window on Jetson
python deepstream/apps/deepstream_rtsp_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```

On the first frame, a window titled **"Click-to-Select Object"** opens with all detector
boxes drawn in per-class colours. Click on the object you want to track. The SUTrack
engine initialises on that box and begins tracking. The tracked green box appears on the
RTSP stream immediately after selection.

**GPU Optimization (One-Shot Detection):**
To maximize performance, the PGIE detector automatically "goes to sleep" (sets `interval=10000`)
immediately after an object is selected. This ensures zero detector overhead during active
tracking while maintaining a low-latency pipeline.

**ROI selection fallback chain:**
1. `static_roi` set in config (or `--init_bbox` from CLI) → no window, headless
2. PGIE detections found → click-to-select window (Phase 6)
3. PGIE disabled (`--no-pgie`) or no detections found → manual draw-a-box window

**Disable PGIE at runtime (keep config unchanged):**

```bash
python deepstream/apps/deepstream_rtsp_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4 \
    --no-pgie
```

**Override PGIE config path:**

```bash
python deepstream/apps/deepstream_rtsp_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4 \
    --pgie-config /custom/path/to/pgie_config.txt
```

---

## 6. Run -- Phase 4 Legacy App (Appsink + OpenCV)

The Phase 4 app is kept as a fallback for non-RTSP use cases.

### 6a -- GUI ROI Selection (mouse draw)

```bash
cd ~/SUTrack
python deepstream/apps/deepstream_tracker_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```

When the first frame appears, draw a rectangle around the target with the mouse,
then press SPACE or ENTER to start tracking.

---

### 6b -- Headless (no display, preset bounding box)

```bash
python deepstream/apps/deepstream_tracker_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4 \
    --init_bbox 300 200 120 140 \
    --output out.mp4 \
    --headless
```

`--init_bbox X Y W H` -- top-left corner (X, Y), width W, height H in pixels.

---

### 6c -- Non-Jetson / Software Decode Fallback

Use `--no-nvmm` on a desktop Linux machine (no NVMM memory, software decode):

```bash
python deepstream/apps/deepstream_tracker_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input video.mp4 \
    --no-nvmm
```

---

## 7. CLI Reference

### Phase 5 App (`deepstream_rtsp_app.py`)

```
usage: deepstream_rtsp_app.py [--config CONFIG] [--input INPUT]
```

| Flag | Type | Description |
|------|------|-------------|
| `--config` | str | Path to `tracker_config.yml` |
| `--input` | str | Video file, RTSP URL, or omit for camera |
| `--pgie-config` | str | Override `pgie_config` path from YAML |
| `--no-pgie` | flag | Disable PGIE at runtime regardless of config |

All other ROI init and RTSP settings are controlled via `tracker_config.yml`.

### Phase 4 App (`deepstream_tracker_app.py`)

```
usage: deepstream_tracker_app.py [--config CONFIG] [--input INPUT]
                                  [--init_bbox X Y W H] [--output OUTPUT]
                                  [--headless] [--no-nvmm]
```

| Flag | Type | Description |
|------|------|-------------|
| `--config` | str | Path to `tracker_config.yml` |
| `--input` | str | Video file, RTSP URL, or omit for camera |
| `--init_bbox X Y W H` | 4x float | Skip GUI -- use this bounding box |
| `--output` | str | Save annotated output to this `.mp4` path |
| `--headless` | flag | No display window. Requires `--init_bbox` |
| `--no-nvmm` | flag | Use software decode (non-Jetson fallback) |

---

## 8. Expected Output

**Phase 6 console (PGIE click-to-select):**
```
INFO  Engine loaded: /home/phoenix/SUTrack/sutrack_fp32.engine
INFO  PGIE config: /home/phoenix/SUTrack/deepstream/configs/pgie_config.txt
INFO  RTSP stream: rtsp://0.0.0.0:8554/sutrack
INFO  Pipeline PLAYING
INFO  PGIE detections on first frame: 3
INFO  [click-to-select window opens -- user clicks]
INFO  Selected bbox: [305, 198, 62, 180]
INFO  Tracker initialized on object_id=0
INFO  EOS -- frames=300  avg_fps=24.8
```

**Phase 5 console (static_roi or no PGIE):**
```
INFO  Engine loaded: /home/phoenix/SUTrack/sutrack_fp32.engine
INFO  Static ROI: [201, 294, 116, 288]
INFO  RTSP stream: rtsp://0.0.0.0:8554/sutrack
INFO  Pipeline PLAYING
INFO  EOS -- frames=300  avg_fps=25.4
```

**Phase 4 console:**
```
10:30:01 [INFO] main: Engine path: /home/phoenix/SUTrack/sutrack_fp32.engine
10:30:03 [INFO] main: Starting pipeline...
10:30:04 [INFO] appsink: Writer opened: out.mp4 (1280x720)
10:30:35 [INFO] pipeline: EOS -- frames=184  total=7.42s  avg_fps=24.80
```

**Video output:** Green bounding box with `ID <n>` label drawn by `nvdsosd` (Phase 5)
or OpenCV (Phase 4).

---

## 9. Validation Checklist

| Test | Expected Result |
|------|----------------|
| Single object, full video | Stable green box, consistent ID throughout |
| Object leaves frame | Tracker disappears after `max_age=30` frames |
| FPS summary (Jetson, Orin, FP32) | >= 25 FPS average |
| RTSP stream (Phase 5) | VLC connects to `rtsp://<ip>:8554/sutrack`, boxes visible |
| Latency (Phase 5) | < 200ms end-to-end measured from source to VLC |
| PGIE click-to-select (Phase 6) | Window opens on frame 0 with detection boxes; clicking initializes tracker |
| PGIE fallback (no detections) | Falls back to manual `selectROI` without crash |
| No PyTorch imported | `python -c "import deepstream.tracker.tracker_utils; import sys; assert 'torch' not in sys.modules; print('OK')"` |

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Score map max ~0.024 / box jumps wildly | FP16 engine used | Recompile with FP32 (omit `--fp16`) |
| Box expands to full frame | Double sigmoid in post-processing | `score_map`/`size_map` already sigmoided in TRT graph |
| `nvbufsurface: Failed to create EGLImage` | GPU conversion over SSH | Run `export DISPLAY=:0` or use VIC mode (`compute-hw=1`) |
| Streaming stopped, `not-negotiated` (-4) | Incorrect memory bridge | Re-insert `nvvideoconvert compute-hw=1` bridge (Lesson 21) |
| `eglglessink cannot handle NVRM surface array` | Missing nvvideoconvert bridge before display sink | Phase 5 app already inserts `conv_disp`; ensure you run `deepstream_rtsp_app.py`, not the Phase 4 app |
| Pipeline dies after 3 frames, `h264parse` error | Cascade from display sink crash | See above -- NVMM surface bridge fix (Lesson 24) |
| NULL window handler (`selectROI`) | Qt OpenCV initialization | Code already calls `imshow` before `selectROI`; ensure display is connected |
| Click-to-select window does not open | `DISPLAY` not set | `export DISPLAY=:0` before running when connected to Jetson via SSH |
| PGIE detections: 0 on first frame | Detector engine not compiled for this JetPack | Engine file `resnet10.caffemodel_b1_gpu0_int8.engine` must exist; run `deepstream-app --version-all` to confirm SDK is intact |
| `nvinfer: Failed to create engine` | Wrong path in `pgie_config.txt` | Verify `model-engine-file` path exists: `ls /opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/` |
| Click hits between boxes (no box selected) | Click point not inside any detection | Click directly on the object center; window stays open — try again |
| `ModuleNotFoundError: pyds` | Python binding not installed | `pip install /opt/nvidia/deepstream/.../pyds-*.whl` |
| `cannot import gi` / GStreamer missing | GObject not installed | `sudo apt-get install python3-gi python3-gst-1.0` |
| FPS drops to 0.9 | PyTorch model in loop | Pure-NumPy `tracker_utils.py` must be used in the loop |
| No RTSP stream despite `rtsp_enabled: true` | `GstRtspServer` not installed | `sudo apt-get install gstreamer1.0-rtsp libgstrtspserver-1.0-dev` |

---

## 11. SSH & Headless Advanced Execution

If you are running the application via SSH, handle the X11/EGL context as follows.

### Phase 5 App (Recommended -- no EGL needed)

The Phase 5 app uses `fakesink` for the display branch when `headless=true` in the
config, and all `nvvideoconvert` elements use `compute-hw=1` (VIC). No `DISPLAY`
environment variable is needed.

```bash
# Set in tracker_config.yml:
#   pipeline.headless: true   (or simply don't connect a display)
#   tracker.static_roi: "X,Y,W,H"
python deepstream/apps/deepstream_rtsp_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```

### Phase 4 App

#### Option A: Use a Physical Display

If a monitor is connected, bridge your SSH session to it:
```bash
export DISPLAY=:0
python deepstream/apps/deepstream_tracker_app.py ...
```

#### Option B: VIC-Only Mode

If no display is connected, pass `--headless --init_bbox X Y W H`.
The pipeline skips `imshow` and uses VIC for all conversions.

---

## 12. File Reference

| File | Purpose |
|------|---------|
| `apps/deepstream_rtsp_app.py` | **Phase 5** -- Native OSD + RTSP streaming (recommended) |
| `apps/deepstream_tracker_app.py` | Phase 4 (legacy) -- appsink + OpenCV drawing |
| `tracker/sutrack_engine.py` | TRT engine load + inference (`cuda.cudart`) |
| `tracker/tracker_instance.py` | Per-object state + `track()` |
| `tracker/tracker_manager.py` | Multi-object lifecycle, IoU matching, stale removal |
| `tracker/tracker_utils.py` | Pure-NumPy: `preprocess`, `sample_target`, `hann2d`, `iou` |
| `configs/tracker_config.yml` | Runtime config (all tunable values, Phase 4 + 5) |
| `configs/pgie_config.txt` | DeepStream ResNet-10 INT8 detector config (Phase 6 click-to-select) |
| `scripts/build_engine.sh` | Correct `trtexec` command with shape specs |
| `docs/architecture.md` | Pipeline diagram + design decisions |

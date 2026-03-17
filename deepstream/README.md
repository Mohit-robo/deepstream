# SUTrack DeepStream Tracker

Single / multi-object tracker using **SUTrack (TensorRT FP32)** inside a **GStreamer / DeepStream** pipeline.

No PyTorch at runtime. Runs on Jetson (JetPack 5+).

---

## Quick Start

```bash
# From SUTrack repo root
cd ~/SUTrack

# 1. Export ONNX (one-time)
python tracking/export_onnx.py --param sutrack_t224

# 2. Compile TRT engine (one-time)
bash deepstream/scripts/build_engine.sh sutrack.onnx sutrack_fp32.engine

# 3. Run tracker (requires display context for PGIE selection)
export DISPLAY=:0
python deepstream/apps/deepstream_rtsp_app.py \
    --config deepstream/configs/tracker_config.yml
```

> **Phase 6 Feature:** On startup, click any detected box in the "Click-to-Select" window to initialize tracking. The detector then automatically optimizes itself to save power.

---

## Directory Structure

```
SUTrack/
├── sutrack.onnx                         # ONNX export (repo root)
├── sutrack_fp32.engine                  # TRT engine (repo root)
└── deepstream/
    ├── apps/
    │   ├── deepstream_rtsp_app.py       # Current production app (OSD + RTSP + PGIE)
    │   └── deepstream_tracker_app.py    # Legacy appsink app
    ├── tracker/
    │   ├── sutrack_engine.py            # TRT load + inference (cuda.cudart)
    │   ├── tracker_instance.py          # Per-object state + track()
    │   ├── tracker_manager.py           # Multi-object lifecycle
    │   └── tracker_utils.py             # Pure-NumPy: preprocess, crop, hann2d, IoU
    ├── configs/
    │   ├── tracker_config.yml           # All tunable parameters
    │   └── pgie_config.txt              # Primary detector config
    ├── scripts/build_engine.sh          # trtexec wrapper with correct shape flags
    └── docs/                            # Markdown and HTML documentation
        ├── usage.md                     # Full usage guide
        └── architecture.md              # Pipeline diagram + design decisions
```

---

## Dependencies

### 1. System: JetPack 5.x + DeepStream SDK

Verify your JetPack and DeepStream versions first:

```bash
# JetPack version
cat /etc/nv_tegra_release

# DeepStream SDK version (must succeed)
deepstream-app --version-all
```

If `deepstream-app` is missing, install DeepStream:

```bash
sudo apt-get install -y deepstream-7.0
# or deepstream-6.4 depending on your JetPack version
```

### 2. System packages (GStreamer + RTSP server)

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-gi python3-gst-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-rtsp \
    libgstrtspserver-1.0-dev \
    gir1.2-gst-rtsp-server-1.0
```

### 3. DeepStream Python bindings (pyds)

```bash
# Locate the wheel (path varies by DeepStream version)
find /opt/nvidia/deepstream -name "pyds*.whl" 2>/dev/null

# Install it
pip install /opt/nvidia/deepstream/deepstream/lib/pyds-*.whl
```

### 4. Python packages (no PyTorch required)

```bash
pip install numpy opencv-python pyyaml cuda-python
```

> TensorRT and CUDA are pre-installed with JetPack and do not require separate installation.

### 5. Verify all components

```bash
python -c "import pyds; print('OK pyds')"
python -c "from cuda import cudart; print('OK cudart')"
python -c "import tensorrt as trt; print('OK TRT', trt.__version__)"
python -c "import numpy; print('OK numpy', numpy.__version__)"
python -c "import cv2; print('OK OpenCV', cv2.__version__)"
python -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; print('OK GStreamer')"
python -c "import gi; gi.require_version('GstRtspServer','1.0'); from gi.repository import GstRtspServer; print('OK GstRtspServer')"
```

All 7 lines must print `OK`. See [docs/usage.md](docs/usage.md) for troubleshooting if any fail.

---

## See Also

- [docs/usage.md](docs/usage.md) - detailed run instructions and CLI reference
- [docs/architecture.md](docs/architecture.md) - pipeline diagram and design decisions
- [../tasks/lessons.md](../tasks/lessons.md) - bugs and fixes from this deployment

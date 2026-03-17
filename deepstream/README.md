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

# 3. Run tracker (requires display context for GPU conversion)
export DISPLAY=:0
python deepstream/apps/deepstream_tracker_app.py \
    --config deepstream/configs/tracker_config.yml
```

---

## Directory Structure

```
SUTrack/
├── sutrack.onnx                         # ONNX export (repo root)
├── sutrack_fp32.engine                  # TRT engine (repo root)
└── deepstream/
    ├── apps/deepstream_tracker_app.py   # Main entry point
    ├── tracker/
    │   ├── sutrack_engine.py            # TRT load + inference (cuda.cudart)
    │   ├── tracker_instance.py          # Per-object state + track()
    │   ├── tracker_manager.py           # Multi-object lifecycle
    │   └── tracker_utils.py             # Pure-NumPy: preprocess, crop, hann2d, IoU
    ├── configs/
    │   ├── tracker_config.yml           # All tunable parameters
    │   └── pgie_config.txt              # Detector config (Phase 4.3+)
    ├── scripts/build_engine.sh          # trtexec wrapper with correct shape flags
    └── docs/
        ├── usage.md                     # Full usage guide
        └── architecture.md              # Pipeline diagram + design decisions
```

---

## Dependencies

```bash
# DeepStream Python bindings (pre-installed with JetPack)
pip install /opt/nvidia/deepstream/deepstream/lib/pyds-*.whl

# Runtime Python deps (no PyTorch required)
pip install numpy opencv-python pyyaml cuda-python

# GStreamer Python
sudo apt-get install python3-gi python3-gst-1.0 \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good
```

> TensorRT and CUDA are pre-installed with JetPack and do not require separate installation.

---

## See Also

- [docs/usage.md](docs/usage.md) - detailed run instructions and CLI reference
- [docs/architecture.md](docs/architecture.md) - pipeline diagram and design decisions
- [../tasks/lessons.md](../tasks/lessons.md) - bugs and fixes from this deployment

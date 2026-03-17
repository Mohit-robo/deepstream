# SUTrack - Jetson TensorRT & DeepStream Deployment

This repository focuses on the high-performance deployment of the **SUTrack** object tracker on NVIDIA Jetson platforms using **TensorRT** and **DeepStream**.

---

## 🚀 Status: Production-Ready Deployment
We have successfully ported the SUTrack PyTorch model to a standalone, PyTorch-free C++/Python runtime achieving **~25 FPS** on Jetson Orin.

### Key Deployment Modules

1. **[SUTrack_deploy_trt/](SUTrack_deploy_trt/)**: Pure NumPy + TensorRT deployment package.
   - Zero-dependency on PyTorch at runtime.
   - Uses `cuda.cudart` for memory management.
   - Verified FP32 precision for model accuracy.

2. **[deepstream/](deepstream/)**: Production GStreamer pipeline integration.
   - **Intelligent ROI Selection:** Click-to-select objects using integrated PGIE detectors (Phase 6).
   - **RTSP Streaming:** Native H.264 broadcast of tracked streams (Phase 5).
   - **Multi-Object Tracking:** Manages multiple concurrent tracks with ID persistence.
   - **One-Shot Optimization:** Dynamically disables detector overhead after ROI selection to save GPU cycles.
   - **Hardware Acceleration:** Uses VIC (`compute-hw=1`) or GPU (`compute-hw=0`) for color conversion.

---

## 📚 Documentation Index

| Document | Purpose |
| :--- | :--- |
| **[deepstream/README.md](deepstream/README.md)** | **Start Here.** Complete guide for setup, build, and execution. |
| **[tasks/lessons.md](tasks/lessons.md)** | **21 Lessons Learned.** Critical debugging and optimization findings. |
| **[deepstream/docs/architecture.md](deepstream/docs/architecture.md)** | Pipeline diagrams and design decisions. |
| **[CLAUDE.md](CLAUDE.md)** | developer guide and technical constraints. |

---

## 🛠 Quick Start

### 1. Build the Engine
```bash
./tracking/export_onnx.py --param sutrack_t224
./deepstream/scripts/build_engine.sh sutrack.onnx sutrack_fp32.engine
```

### 2. Run the App
```bash
export DISPLAY=:0
python deepstream/apps/deepstream_rtsp_app.py --config deepstream/configs/tracker_config.yml
```

---

## 📈 Performance (Jetson Orin)
- **Framework:** TensorRT 8.6+
- **Precision:** FP32 (Mandatory)
- **Throughput:** ~25 FPS
- **Latency:** ~40ms per frame (inference + pre/post-processing)

---

## 📝 Acknowledgments
Original SUTrack implementation based on the AAAI 2025 paper:
*SUTrack: Towards Simple and Unified Single Object Tracking*.
For research-specific instructions (training, original benchmarks), see the legacy documentation or the official paper.

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
| **[deepstream/README.md](deepstream/README.md)** | **Start Here.**## The Applications

We developed this application in four distinct stages to solve Jetson performance bottlenecks step-by-step. 

1.  **V1 Pipeline (Legacy)**: `deepstream_tracker_app.py`  
    *Appsink to CPU → OpenCV drawing → Appsrc out*. Easy to write, but severely caps FPS due to CPU memory copies.
2.  **V2 Pipeline**: `deepstream_rtsp_app.py`  
    *Native `nvdsosd` drawing + RTSP Server*. Zero CPU copies for display; introduced PGIE "click-to-select" bounding box initialization.
3.  **V3 Pipeline**: `deepstream_nvtracker_app.py`  
    *Hybrid Tracking*. Runs NVIDIA's `nvtracker` for multi-object tracking at 60 FPS, and selectively correlates a single target using SUTrack via Spatial IoU.
4.  **V4 Pipeline (Recommended)**: `deepstream_desktop_app.py`  
    *GTK3 Desktop GUI & Async TRT*. Wraps the pipeline in a GTK UI (no terminal needed) and decouples TRT inference into a background thread, unlocking maximum 60 FPS performance even during active tracking.

## Getting Started

1.  **Compilation**: Compile the ONNX model into an FP32 TensorRT engine. *(Do not use FP16)*.
2.  **Docs**: See all detailed technical explanations in `deepstream/docs/html/index.html`.
3.  **Run Guide**: For full CLI commands and application walkthroughs, read [`deepstream/docs/usage.md`](deepstream/docs/usage.md).

```bash
# Example: Run the recommended V4 GTK Desktop App
export DISPLAY=:0
python deepstream/apps/deepstream_desktop_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```

## Performance (Jetson Orin)
- **Framework:** TensorRT 8.6+
- **Precision:** FP32 (Mandatory)
- **Throughput:** ~60 FPS (V4 Async TRT Worker)
- **Latency:** ~30ms per frame TRT inference (background thread)

---

## 📝 Acknowledgments
Original SUTrack implementation based on the AAAI 2025 paper:
*SUTrack: Towards Simple and Unified Single Object Tracking*.
For research-specific instructions (training, original benchmarks), see the legacy documentation or the official paper.

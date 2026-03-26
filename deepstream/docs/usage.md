# SUTrack DeepStream User Guide

This guide explains the progression of the SUTrack DeepStream application. We developed it in four major stages, each solving specific performance and usability bottlenecks for Jetson deployment. 

For local Jetson deployment with a connected monitor, we recommend **V4**. For remote/headless deployment where the operator is on a separate PC, use **V5**.

---

## The 5 Application Versions

| Version | File | Key Features | Performance |
|---------|------|--------------|-------------|
| **V1** | `deepstream_tracker_app.py` | OpenCV appsink, host-memory drawing | Poor (CPU bound) |
| **V2** | `deepstream_rtsp_app.py` | `nvdsosd` native drawing, RTSP out, PGIE selection | Good (GPU bound) |
| **V3** | `deepstream_nvtracker_app.py` | Hybrid Tracking (`nvtracker` + `SUTrack`), robust ID management | Good (GPU bound) |
| **V4** | `deepstream_desktop_app.py` | GTK3 GUI, `TRTWorkerThread`, zero-terminal deployment | **Excellent (~60 FPS)** |
| **V5** (Remote) | `deepstream_server_app.py` + `v5_remote_client.py` | Headless Jetson server, GTK client on any PC, REST API | **Excellent (~60 FPS)** |

---

## V1: Appsink & OpenCV Processing (Legacy)
`deepstream/apps/deepstream_tracker_app.py`

**The Goal:** The initial proof-of-concept. It bridged DeepStream with SUTrack by pulling frames out of the video pipeline into host memory (CPU) via an `appsink`, drawing bounding boxes using OpenCV, and pushing them back via `appsrc`.

**Limitations:** Extremely slow. Copying frames to the CPU memory and using OpenCV for display limits the Jetson to 10-15 FPS.

**Run Command:**
```bash
export DISPLAY=:0
python deepstream/apps/deepstream_tracker_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```

---

## V2: Hardware OSD & RTSP Streaming
`deepstream/apps/deepstream_rtsp_app.py`

**The Goal:** Eliminate CPU bottlenecks. We replaced the OpenCV drawing logic with NVIDIA's native `nvdsosd` (On-Screen Display). Bounding boxes are now written to `NvDsObjectMeta` metadata, and hardware draws them directly in GPU memory. We also added an RTSP server for network streaming.

**Key Addition: PGIE Click-to-Select.** A ResNet-10 detector runs on the very first frame. A snapshot window pops up showing all detected objects. The user clicks one, and SUTrack takes over tracking that object for the rest of the video.

**Run Command:**
```bash
export DISPLAY=:0
python deepstream/apps/deepstream_rtsp_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```
*Stream available at: `rtsp://<jetson-ip>:8554/sutrack`*

---

## V3: Hybrid Tracking & Multi-Object Management
`deepstream/apps/deepstream_nvtracker_app.py`

**The Goal:** SUTrack is incredibly accurate but computationally heavy, making it difficult to track dozens of objects simultaneously. In V3, we introduced **Hybrid Tracking**.

**How Hybrid Tracking Works:**
1. We run NVIDIA's highly efficient `nvtracker` (NvDCF) on the pipeline. It assigns a unique ID to every object in the frame at 60+ FPS.
2. SUTrack runs in the background targeting *only one specific object of interest*.
3. We compute the **Spatial IoU (Intersection over Union)** between SUTrack's output and `nvtracker`'s bounding boxes to correlate the SUTrack target with a native DeepStream ID. 
4. This allows us to seamlessly swap targets mid-stream and rely on `nvtracker`'s temporal robustness. 

**Run Command:**
```bash
export DISPLAY=:0
python deepstream/apps/deepstream_nvtracker_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```

---

## V4: Desktop GUI & Async TRT (Recommended)
`deepstream/apps/deepstream_desktop_app.py`

**The Goal:** Real-world deployment. End-users cannot use a command-line terminal to select targets. Furthermore, running TRT inference synchronously inside the GStreamer probe caps the pipeline at ~30 FPS on Jetson. 

**Key Features:**
1. **GTK3 Desktop App**: The DeepStream `nv3dsink` is embedded directly into a native Linux GUI window using `set_window_handle()`. 
2. **On-Screen Controls**: Select, Next, Prev, Lock, and Cancel buttons replace terminal inputs.
3. **Live OSD Cycling**: Press "Select" to freeze tracking and cycle through highlighted candidates on the live feed. Press "Lock" to lock on.
4. **Appearance Re-ID**: When SUTrack loses confidence, it enters a `SEARCHING` state. It compares incoming detections against a historically saved Color Histogram signature (Bhattacharyya distance) to automatically re-acquire the target.
5. **TRTWorkerThread (FPS Unlock)**: TRT inference is moved to a background daemon thread. The pipeline probe submits frames non-blocking and reads the *previous* TRT result. This allows the video to render at 60 FPS while inference runs at its maximum hardware capability asynchronously.

**Run Command:**
```bash
export DISPLAY=:0
python deepstream/apps/deepstream_desktop_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4
```

---

---

## V5: Client-Server (Remote Deployment)
**Server:** `deepstream/apps/deepstream_server_app.py` (runs on Jetson — headless)
**Client:** `deepstream/apps/v5_remote_client.py` (runs on any Linux/Mac PC)

**The Goal:** Decouple the Jetson compute node from the operator's workstation. The Jetson runs a fully headless pipeline — no monitor, no X11, no display server required — and streams annotated video over RTSP. The operator receives the stream on a GTK window on their PC and clicks to select tracking targets.

**Architecture:**
```
Jetson (headless)                         Operator PC
------------------------------            ---------------------------
deepstream_server_app.py                  v5_remote_client.py
  |                                         |
  +--> PGIE detector (ResNet-10)            +--> rtspsrc (TCP)
  +--> nvtracker (NvDCF)                    +--> h264 decode
  +--> SUTrack TRT inference                +--> GTK window (click-to-select)
  +--> nvdsosd (OSD annotations)            +--> HTTP REST client
  +--> nvv4l2h264enc --> RTSP :8554   <----+     (POST /api/command)
  +--> REST API :8000   <------------------+
```

**Key Design Decisions:**
- **No ROI drawing**: The PGIE detector automatically finds objects every frame. The stream shows labeled boxes for all detections. The operator clicks on the desired target; the server resolves the click to the nearest detection centroid in the live frame — no manual bounding-box drawing at any point.
- **EGL_PLATFORM=surfaceless**: `nvinfer` on Jetson requires EGL for NVMM buffer operations. In headless SSH sessions `DISPLAY=localhost:N` is an X11-forwarded socket that NVIDIA's libEGL cannot use. The server always sets `EGL_PLATFORM=surfaceless` (and drops any forwarded DISPLAY) to bind EGL directly to the GPU device.
- **TCP RTSP transport**: `rtspsrc protocols=tcp` in the client forces RTP-over-TCP (interleaved on port 8554) so no random UDP ports need to be open in firewalls between Jetson and PC.
- **Normalised click coordinates**: Client sends `{x: 0.0-1.0, y: 0.0-1.0}`. Server denormalises and finds the nearest live detection — compensates for ~0.5-1.5 s RTSP latency.

**Run — Server (Jetson, SSH session):**
```bash
cd ~/SUTrack
python3 deepstream/apps/deepstream_server_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4 \
    --loop
```

**Run — Client (operator PC):**
```bash
python deepstream/apps/v5_remote_client.py --host 192.168.128.178
```

**Bounding Box Extraction (Real-Time):**
The server exposes the live tracking coordinates via the `GET /api/state` endpoint. The `v5_remote_client.py` automatically polls this and prints the bounding box to the terminal (`stdout`) in the format:
`BBOX,<frame_idx>,<x>,<y>,<w>,<h>`

You can easily pipe or redirect this output to save it on your local machine:
```bash
python deepstream/apps/v5_remote_client.py --host 192.168.128.178 | grep "BBOX" > tracking_results.csv
```

**Fully automatic (no operator interaction):**
```bash
# Server: lock to the largest PGIE detection on the first frame automatically
python3 deepstream/apps/deepstream_server_app.py \
    --config deepstream/configs/tracker_config.yml \
    --input /path/to/video.mp4 \
    --auto-lock --loop
```

**V5 CLI Reference:**

| Flag | App | Description |
|------|-----|-------------|
| `--config` | Server | Path to `tracker_config.yml` |
| `--input`, `-i` | Server | Video file, RTSP URL, or omit for camera |
| `--loop` | Server | Seek to start on EOS (continuous file replay) |
| `--auto-lock` | Server | Automatically lock to the largest PGIE detection |
| `--no-pgie` | Server | Skip PGIE (fallback: manual click initialisation with no detection boxes) |
| `--no-one-shot` | Server | Keep PGIE running every frame after lock (default: PGIE sleeps post-lock) |
| `--debug-boxes` | Server | Show raw nvtracker boxes in OSD |
| `--api-port` | Server | REST API port (default: 8000) |
| `--host` | Client | Jetson IP address (default: 127.0.0.1) |
| `--rtsp-port` | Client | RTSP server port (default: 8554) |
| `--api-port` | Client | REST API port (default: 8000) |

**OSD Status Messages:**

| OSD Text | Meaning |
|----------|---------|
| `[N DETECTED]  Click on target in client` | PGIE active, N objects visible — waiting for operator click |
| `[WAITING]  No detections yet...` | PGIE active but no detections on current frame |
| `[LOCKED] 0.92` | SUTrack tracking with confidence 0.92 |
| `[SEARCHING...]` | SUTrack lost confidence — running Re-ID |
| `[TARGET LOST]  N detected — click to re-acquire` | STALE state — new detections shown, click to restart |

---

## General CLI Reference (All Versions)

| Flag | Description |
|------|-------------|
| `--config` | Path to `tracker_config.yml` |
| `--input`, `-i` | Video file, RTSP URL, or omit for camera |
| `--no-one-shot` | Keep PGIE active after lock (default: PGIE sleeps to save power) |
| `--debug-boxes` | Show last-known bbox in yellow for drift analysis |

---

## Critical Rules

1. **FP32 Engine Only:** Always compile the TRT engine without `--fp16`. FP16 underflows sigmoid outputs to near-zero, breaking the tracker completely.
2. **No Double Sigmoid:** `score_map` and `size_map` are already sigmoid-activated inside the TRT graph. Do not activate them again in Python.
3. **Hanning Window First:** Apply the spatial Hanning penalty to the score map *before* taking the `argmax`.
4. **RGB Conversion:** Always convert frames to `RGB` before inference. DeepStream outputs `RGBA` via NumPy view. Extracting `rgba[:,:,:3]` avoids excessive memory allocations. 

*(See `deepstream/docs/html/index.html` for deep-dive technical explanations)*

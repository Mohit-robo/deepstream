# V5: Distributed Client-Server Architecture (Phase 12)

## Motivation
With the V4 desktop application, SUTrack successfully provides a native, low-latency GTK3 desktop GUI. However, executing this heavy GStreamer/TensorRT tracking pipeline restricts the user to always operate locally on the Jetson edge device.

**V5 Architecture Objective:** Run the entire heavy machine learning and DeepStream pipeline headless on the Jetson (Backend), and allow the user to visualize the stream and interact with the tracker seamlessly from their Local Laptop/PC via a dedicated client GUI (Frontend).

## High-Level Architecture

```mermaid
graph TD
    subgraph Jetson Device (Backend)
        A[Camera / File Source] --> B(DeepStream Pipeline)
        B --> C[nvtracker + SUTrack]
        C --> D{Tee}
        D -->|RTSP H.264| E[RTSP Server / Port: 8554]
        D -->|OSD drawing| F[fakesink]
        
        G[REST/Socket Control API / Port: 8000] -.->|Commands| C
    end

    subgraph User Desktop (Frontend)
        H[GTK Desktop App] -->|Reads RTSP| E
        H -->|HTTP/REST commands| G
        User([User]) -->|Clicks & Keyboard| H
    end
```

## Backend: Node `deepstream_server_app.py`
The Jetson will run the complete V4 tracking logic with all performance optimizations (like `TRTWorkerThread`) but operate entirely headless.

*   **Display Disabled:** The `nv3dsink` sink used in V4 is replaced strictly with a `fakesink`. X11 display context (`DISPLAY=:0`) is absolutely not required. The Jetson simply encodes the annotated frame buffer into H.264 packets for RTSP.
*   **REST/Socket API:** A lightweight daemon thread using Python `http.server` or `Flask` will be bound to port `8000`. 
    *   `/api/state` (GET) → Exposes current TrackerState (LOCKED, IDLE, SEARCHING), FPS, and target tracking ID.
    *   `/api/command` (POST) → Ingests `{action: "..."}` payloads directly from the frontend to trigger internal `TrackerManager` states.
**Key Design Decisions:**
- **No ROI drawing**: The PGIE detector automatically finds objects every frame. The stream shows labeled boxes for all detections. The operator clicks on the desired target; the server resolves the click to the nearest detection centroid in the live frame — no manual bounding-box drawing at any point.
- **EGL_PLATFORM=surfaceless**: `nvinfer` on Jetson requires EGL for NVMM buffer operations. In headless SSH sessions `DISPLAY=localhost:N` is an X11-forwarded socket that NVIDIA's libEGL cannot use. The server always sets `EGL_PLATFORM=surfaceless` (and drops any forwarded DISPLAY) to bind EGL directly to the GPU device.
- **TCP RTSP transport**: `rtspsrc protocols=tcp` in the client forces RTP-over-TCP (interleaved on port 8554) so no random UDP ports need to be open in firewalls between Jetson and PC.
- **Normalised click coordinates**: Client sends `{x: 0.0-1.0, y: 0.0-1.0}`. Server denormalises and finds the nearest live detection — compensates for ~0.5-1.5 s RTSP latency.
- **Real-Time Bounding Box Sync**: The server exposes the active target's coordinates via `GET /api/state`. The client polls this and outputs `BBOX,idx,x,y,w,h` to stdout for programmatic consumption on the remote operator's machine.
*   **Latency Compensation Strategy:** The most critical challenge. Given an RTSP network delay of ~0.5 - 1.5s, clicking on a moving bounding box frame locally acts on where the box *used to be*. 
    *   **Solution**: The client sends *normalized screen coordinates* `(x: float 0.0-1.0, y: float 0.0-1.0)`. The `deepstream_server_app` cross-references these static spatial coordinates with the active `nvtracker` metadata pool in its *current* live frame, calculating Euclidean distance and acquiring the structurally nearest valid candidate to lock on.

## Frontend: Node `v5_remote_client.py` 
A Python GTK3 desktop application intended to run effortlessly on the user's local, remote machine.

*   **Video Integration:** Leverages a simple receiving `uridecodebin` GStreamer pipeline inside a GTK `DrawingArea` (the exact same architecture used in `deepstream_desktop_app.py`), directly rendering the `rtsp://<jetson-ip>:8554/sutrack` stream locally.
*   **Interaction Processing:** 
    *   **Mouse Callback:** Evaluates click positions on the GTK Canvas, normalizes them against the respective element dimensions relative to current Window Size, and performs an HTTP POST dispatch to the Jetson API.
    *   **Keyboard Binding:** Overrides `s`, `p`, `n`, `l`, `q` keys, mirroring the exact functional flow of V4, dispatching equivalent network API POST requests on trigger.
*   **Asynchronicity:** All HTTP requests to the Jetson backend will be dispatched via Python `threading.Thread` or `requests.post()` async wrappers to ensure that network latency does not stutter the local video playback thread.

## Execution Sequence (V5 Deployment)

1. **Jetson:** `$ python apps/deepstream_server_app.py --config configs/tracker_config.yml` (Runs daemonized, serves RTSP and API)
2. **Local PC:** `$ python apps/v5_remote_client.py --host 192.168.1.100` (Opens GTK window, syncs with streams)
3. **Usage:** User clicks objects directly inside the GTK app's video canvas, the remote backend locks on seamlessly, bounding boxes continue rendering synchronously across RTSP frames.

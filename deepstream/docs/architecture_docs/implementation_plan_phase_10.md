# Phase 10 Implementation Plan: Desktop GUI Integration

This phase upgrades the DeepStream pipeline from a terminal-bound script into a complete Desktop Application. We will embed the hardware-accelerated video stream (`nv3dsink`) directly into a GTK3 window, and provide point-and-click buttons to control the OSD target cycling logic.

## Goal
Build `deepstream_desktop_app.py` using PyGObject (GTK3). The app will host the GStreamer video, provide "Select", "Next", "Prev", and "Lock" buttons to control the tracker state seamlessly, and display a real-time FPS counter.

## Proposed Changes

### [NEW] deepstream/apps/deepstream_desktop_app.py
Create a completely new entrypoint script that encapsulates the UI logic:
* **GTK3 Layout**: 
  * A main window containing a large `Gtk.DrawingArea` (the video canvas).
  * A bottom Control Panel with buttons: `[Start Selection]`, `[◀ Prev]`, `[Next ▶]`, `[Lock]`, `[Cancel]`.
  * An info label showing status (`IDLE`, `SELECTING`, etc.) and the **Current FPS**.
* **GStreamer Integration**:
  * Extract the `window.get_xid()` from the `Gtk.DrawingArea` and pass it to `sink_disp.set_window_handle()`. This forces `nv3dsink`/`nveglglessink` to render inside the GTK app instead of floating loosely.
* **State Management**:
  * Button clicks simply update the underlying [AppState](file:///run/user/1002/gvfs/sftp:host=192.168.128.177,user=phoenix/home/phoenix/SUTrack/deepstream/apps/deepstream_rtsp_app.py#114-136) object (e.g. `state.tracking_state = TrackingState.SELECTING` or `state._pending_lock = det`).
  * The [tracker_probe](file:///run/user/1002/gvfs/sftp:host=192.168.128.177,user=phoenix/home/phoenix/SUTrack/deepstream/apps/deepstream_nvtracker_app.py#91-287) continues running untouched in the background GStreamer thread, drawing the yellow OSD highlight based on the UI's state variables.
* **FPS Counter**:
  * Add an FPS sliding window in the probe.
  * Use `GLib.timeout_add` to update the GUI thread label every 1.0 seconds with the calculated FPS.

### [MODIFY] deepstream/apps/deepstream_nvtracker_v3.py
* Extract the GStreamer pipeline construction logic ([build_pipeline](file:///run/user/1002/gvfs/sftp:host=192.168.128.177,user=phoenix/home/phoenix/SUTrack/deepstream/apps/deepstream_nvtracker_app.py#292-398)) so it can be cleanly imported by the new Desktop App, isolating the backend logic from the frontend UI.
* Keep the `tty` terminal entrypoint active as a developer headless fallback, but default to the GTK app for production.

## Verification Plan

### Manual Verification
1. Run `deepstream_desktop_app.py --config config.yml --input video.mp4`.
2. Confirm a native desktop window opens containing the high-quality video feed.
3. Observe the FPS label in `IDLE` mode.
4. Click `[Start Selection]`. Verify the prominent yellow highlight appears via OSD on the video.
5. Click `[Next]` and `[Prev]` to verify the candidate highlight cycles instantly.
6. Click `[Lock]`. Verify SUTrack initiates tracking and the FPS updates to reflect the tracking overhead.
7. Click `[Cancel]` to verify it stops tracking and returns to `IDLE`.

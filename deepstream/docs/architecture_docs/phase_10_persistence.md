# Phase 10: Desktop App Integration & Dynamic Persistence

## Objective
To deploy the DeepStream hardware-accelerated tracking pipeline within a fully managed desktop user interface (GUI). Because end-users won't have access to a terminal in deployment, we embed the GStreamer video sink directly into a desktop application window, provide on-screen buttons that trigger the OSD-based target cycling system, and offload TRT inference to a background worker thread so the GStreamer pipeline runs at full source FPS.

---

## 1. GUI Architecture (GTK3 via PyGObject)

We wrap the GStreamer pipeline inside a GTK3 Desktop Application (`deepstream_desktop_app.py`).

### Video Embedding
Instead of letting `nv3dsink` create a floating X11 window, we:
1. Create a `Gtk.DrawingArea` inside the desktop window.
2. Start the pipeline via `GLib.timeout_add(300ms)` after `show_all()`.
3. Intercept the `sync-message::element` bus signal for `prepare-window-handle` and inject the canvas XID via `msg.src.set_window_handle(xid)`.
4. The DeepStream live feed renders natively **inside** the GTK window at full hardware-accelerated FPS.

### UI Controls (Replacing the Terminal `tty`)

| Button | Keyboard | Action |
|--------|----------|--------|
| 🎯 Select | `s` | Enter SELECTING mode; OSD highlights first candidate |
| ◀ Prev | `p` | Cycle backward through detected objects |
| Next ▶ | `n` | Cycle forward through detected objects |
| ✅ Lock | `l` / `Enter` | Lock SUTrack onto the highlighted candidate |
| ✖ Cancel | `q` / `Esc` | Stop tracking, return to IDLE |
| — | `x` | Exit application |

---

## 2. TRTWorkerThread (Key Performance Architecture)

The main FPS bottleneck in `LOCKED` mode is TRT inference (~30 ms/frame on Jetson), which would block the GStreamer streaming thread and cap the pipeline at ~33 FPS.

**Solution:** A dedicated `TRTWorkerThread` owns all `TrackerManager` mutations. The probe submits frames non-blocking and reads the last result with a 1-2 frame lag (imperceptible to the user). The pipeline can now run at source rate (e.g. 60 FPS) while TRT inference runs independently.

```
GStreamer probe thread:
  submit_track(rgb) ──non-blocking──► TRTWorkerThread.run()
  reads last_result / last_conf ◄──── (atomic dict replacement, GIL-safe)
```

**Thread safety:** A `threading.Lock` protects the pending slot. `last_result`/`last_conf` are replaced by atomic dict assignment (GIL-safe), so the probe reads them without holding any lock.

---

## 3. Additional Optimisations

| Optimisation | Detail |
|---|---|
| `RGBA→RGB slice` | `rgba[:, :, :3]` instead of `cv2.cvtColor` — avoids extra allocation |
| `HIST_UPDATE_INTERVAL = 20` | Re-ID histogram recomputed every 20 LOCKED frames, not every frame |
| `UI_UPDATE_EVERY = 5` | GTK label updates every 5 frames (~12 Hz at 60 FPS) |
| Lazy NVMM copy | `get_rgb()` only called in LOCKED/SEARCHING, not in IDLE/SELECTING |
| `LD_PRELOAD` re-exec | Script re-launches itself with `LD_PRELOAD=libgomp.so.1` to fix Jetson `nvtracker` TLS error |

---

## 4. Appearance-Based Re-Identification (Re-ID)

When a target leaves the frame, `NvDCF` drops the ID. SUTrack enters `SEARCHING`:
* **Signature**: H-S Color Histogram computed every `HIST_UPDATE_INTERVAL` frames while LOCKED.
* **Probing**: New detections are checked against the signature via Bhattacharyya distance AND spatial IoU.
* **Re-lock**: If match found, `submit_init()` is called asynchronously to re-initialize SUTrack.

---

## 5. State Machine

| State | Trigger | Description |
|-------|---------|-------------|
| `IDLE` | Start / Cancel | Pipeline runs; no SUTrack active |
| `SELECTING` | Select button | OSD cycles through current PGIE detections |
| `LOCKED` | Lock button | SUTrack + TRTWorkerThread tracking target |
| `SEARCHING` | Confidence drops | Scanning new detections for Re-ID match |
| `STALE` | Search timeout | Target lost; returns to IDLE |

# Phase 6: PGIE Integration & Click-to-Select ROI

This document outlines the technical plan for implementing an intelligent ROI selection method. Instead of manually drawing a box, the system will run a detector on the first frame and allow the user to select an object by simply clicking on it.

## 1. Objective
Enable faster and more accurate tracker initialization by leveraging a Primary GIE (PGIE) detector to suggest bounding boxes, which are then selected via a mouse click.

## 2. Technical Approach

### Pipeline Architecture
The pipeline will be extended to include `nvinfer` (PGIE) before the OSD and probe logic:
`source → mux → nvinfer (PGIE) → nvvideoconvert → nvosd → probe → sink`

### Component Changes

#### PGIE Configuration (`deepstream/configs/pgie_config.txt`)
- Path to standard ResNet-10 or Peoplenet engine.
- Configuration for object classes (Person, Vehicle, etc.).
- Clustering and thresholding parameters.

#### Application State (`AppState`)
- `detections`: A list to store `[x, y, w, h]` of all objects found by PGIE on the first frame.
- `click_point`: Store the `(x, y)` coordinates from the OpenCV mouse callback.

#### Click-to-Select Logic
1. **Capture**: The `tracker_probe` waits for the first frame that contains `NvDsFrameMeta` with detections.
2. **Snapshot**: The probe captures the frame and signals the Main Thread.
3. **GUI Selection**:
   - The Main Thread opens a window displaying the first frame.
   - Detected boxes are overlaid as visual guides.
   - A mouse callback `on_mouse_click` is registered.
4. **Box Matching**:
   - When a user clicks at `(px, py)`, the code iterates through all `detections`.
   - It identifies the box where `x <= px <= x+w` and `y <= py <= y+h`.
   - If multiple boxes overlap the click point, the one with the smallest area is selected.
5. **Handoff**: The selected box is passed to `state.manager.initialize()`, and the pipeline resumes.

## 3. Verification Plan
- **Detection Check**: Verify that `nvinfer` is correctly producing `NvDsObjectMeta` on frame #0.
- **Callback Check**: Verify that mouse clicks correctly resolve to pixel coordinates.
- **Initialization Check**: Verify that the SUTrack green box matches the detector's box after selection.
- **RTSP Latency**: Ensure the additional PGIE element does not push latency above the 200ms target.

---

## 4. Why This Approach?
- **Speed**: One click is faster than drawing a box.
- **Precision**: Uses the detector's perfectly aligned pixel boundaries instead of human-drawn approximations.
- **Performance**: The PGIE can be set to `interval=0` only for the first few frames and then disabled (or kept at a high interval) to save GPU cycles once tracking is active.

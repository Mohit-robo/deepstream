"""
SUTrack TensorRT Demo — Self-Contained, No PyTorch Dependency
============================================================
Dependencies: numpy, opencv-python, tensorrt, cuda-python (cuda.cudart)

Usage:
    # GUI selection of target:
    python demo_trt.py video.mp4 --engine model.engine --config ../experiments/sutrack/sutrack_t224.yaml

    # Headless with saved output:
    python demo_trt.py video.mp4 --engine model.engine \
        --config ../experiments/sutrack/sutrack_t224.yaml \
        --init_bbox 300 200 120 140 \
        --headless --save_path out.mp4
"""

import os
import sys
import ctypes
import argparse
import time
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import cv2 as cv
import tensorrt as trt
from cuda import cudart

# Utility functions — pure NumPy, no PyTorch
from utils import TrackerConfig, preprocess, sample_target, transform_image_to_crop, hann2d

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# TensorRT buffer helpers
# ---------------------------------------------------------------------------

class HostDeviceMem:
    """Paired host (CPU) and device (GPU) memory buffers."""
    def __init__(self, host_mem: np.ndarray, device_mem):
        self.host   = host_mem
        self.device = device_mem


def allocate_buffers(engine, batch_size: int = 1):
    inputs, outputs, bindings = {}, {}, []
    err, stream = cudart.cudaStreamCreate()

    for i in range(engine.num_io_tensors):
        name      = engine.get_tensor_name(i)
        dtype     = trt.nptype(engine.get_tensor_dtype(name))
        shape     = engine.get_tensor_shape(name)
        # Replace dynamic dims (−1) with batch_size
        shape     = tuple(batch_size if d == -1 else d for d in shape)
        nbytes    = int(np.prod(shape)) * np.dtype(dtype).itemsize

        # Pinned host memory — wrap raw int pointer with ctypes then numpy
        err, host_mem = cudart.cudaMallocHost(nbytes)
        host_np = np.frombuffer(
            (ctypes.c_char * nbytes).from_address(int(host_mem)),
            dtype=dtype).reshape(shape)

        # Device memory
        err, device_mem = cudart.cudaMalloc(nbytes)
        bindings.append(int(device_mem))

        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        mem = HostDeviceMem(host_np, device_mem)
        (inputs if is_input else outputs)[name] = mem

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream) -> dict:
    """Copy inputs → GPU, run TRT, copy outputs → CPU."""
    for inp in inputs.values():
        cudart.cudaMemcpyAsync(
            inp.device, inp.host.ctypes.data, inp.host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream)
    for out in outputs.values():
        cudart.cudaMemcpyAsync(
            out.host.ctypes.data, out.device, out.host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)
    return {name: out.host for name, out in outputs.items()}


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TRTSUTrack:
    """
    TensorRT SUTrack tracker — pure NumPy post-processing.

    Engine inputs  : template (1,6,112,112), search (1,6,224,224), template_anno (1,4)
    Engine outputs : pred_boxes (1,4), score_map (1,1,F,F), size_map (1,2,F,F), offset_map (1,2,F,F)
                     where F = search_size / encoder_stride  (typically 14)
    """

    def __init__(self, engine_path: str, config_yaml: str):
        cfg = TrackerConfig(config_yaml)
        self.cfg          = cfg
        self.search_size  = cfg.search_size       # 224
        self.template_size = cfg.template_size     # 112
        self.search_factor = cfg.search_factor     # 4.0
        self.template_factor = cfg.template_factor # 2.0
        self.feat_sz      = cfg.search_data_size // cfg.encoder_stride  # 224//16 = 14

        # Load TensorRT engine
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape('template',      (1, 6, self.template_size, self.template_size))
        self.context.set_input_shape('search',        (1, 6, self.search_size,   self.search_size))
        self.context.set_input_shape('template_anno', (1, 4))

        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

        # Hanning spatial penalty window (feat_sz × feat_sz)
        self.hann_window = None
        if cfg.use_window:
            self.hann_window = hann2d(self.feat_sz, self.feat_sz, centered=True)

        self.state: list = None         # [x, y, w, h] in original image pixels
        self._template_img:  np.ndarray = None  # (1, 6, template_size, template_size)
        self._template_anno: np.ndarray = None  # (1, 4)

    # ------------------------------------------------------------------
    def initialize(self, image_rgb: np.ndarray, init_bbox: list):
        """
        Prepare template buffers from the first frame.

        Args:
            image_rgb:  H×W×3 uint8 RGB image
            init_bbox:  [x, y, w, h] pixel bounding box of the target
        """
        self.state = list(init_bbox)

        # Extract and preprocess template crop
        z_crop, rf = sample_target(image_rgb, self.state, self.template_factor,
                                   self.template_size)
        self._template_img = preprocess(z_crop)  # (1, 6, 112, 112) float32

        # Normalized template annotation for the encoder's token-type mask
        anno = transform_image_to_crop(self.state, self.state, rf, self.template_size)
        self._template_anno = anno[np.newaxis].astype(np.float32)  # (1, 4)

    # ------------------------------------------------------------------
    def track(self, image_rgb: np.ndarray) -> list:
        """
        Track one frame and return updated [x, y, w, h].

        Args:
            image_rgb: H×W×3 uint8 RGB image

        Returns:
            [x, y, w, h] bounding box in pixel coordinates
        """
        H, W = image_rgb.shape[:2]

        # Extract and preprocess search crop
        x_crop, resize_factor = sample_target(image_rgb, self.state,
                                              self.search_factor, self.search_size)
        search_np = preprocess(x_crop)  # (1, 6, 224, 224) float32

        # Copy to TRT host buffers
        np.copyto(self.inputs['template'].host,      self._template_img)
        np.copyto(self.inputs['search'].host,        search_np)
        np.copyto(self.inputs['template_anno'].host, self._template_anno)

        # Run TRT inference
        trt_out = do_inference(self.context, self.bindings,
                               self.inputs, self.outputs, self.stream)

        # --- Post-processing (pure NumPy) ---
        # score_map and size_map are already sigmoid-activated inside TRT graph.
        # offset_map is raw (no sigmoid).
        score_map  = trt_out['score_map'].reshape(self.feat_sz, self.feat_sz)
        size_map   = trt_out['size_map'].reshape(2, self.feat_sz * self.feat_sz)
        offset_map = trt_out['offset_map'].reshape(2, self.feat_sz * self.feat_sz)

        # Apply Hanning window BEFORE argmax (spatial suppression of distractors)
        if self.hann_window is not None:
            score_map = score_map * self.hann_window

        # Find peak (mirrors CenterPredictor.cal_bbox in decoder.py)
        score_flat = score_map.reshape(-1)
        idx    = int(np.argmax(score_flat))
        idx_y  = idx // self.feat_sz
        idx_x  = idx % self.feat_sz

        w_norm = float(size_map[0, idx])    # sigmoided → [0,1] fraction of search img
        h_norm = float(size_map[1, idx])
        off_x  = float(offset_map[0, idx])  # sub-cell offset (raw logit)
        off_y  = float(offset_map[1, idx])

        cx_norm = (idx_x + off_x) / self.feat_sz
        cy_norm = (idx_y + off_y) / self.feat_sz

        # Scale to crop-space pixels (scale = crop_sz)
        scale = self.search_size / resize_factor
        cx = cx_norm * scale
        cy = cy_norm * scale
        w  = w_norm  * scale
        h  = h_norm  * scale

        # Map from crop-space → original image
        cx_prev   = self.state[0] + 0.5 * self.state[2]
        cy_prev   = self.state[1] + 0.5 * self.state[3]
        cx_real   = cx + (cx_prev - 0.5 * scale)
        cy_real   = cy + (cy_prev - 0.5 * scale)

        # Convert to [x, y, w, h] — clamp position, preserve predicted size
        x1 = max(0.0, min(cx_real - 0.5 * w, W - 1.0))
        y1 = max(0.0, min(cy_real - 0.5 * h, H - 1.0))
        self.state = [x1, y1, w, h]
        return self.state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SUTrack TensorRT Demo (no PyTorch)')
    parser.add_argument('video_path',              help='Input video path')
    parser.add_argument('--engine',   required=True, help='Path to .engine file')
    parser.add_argument('--config',   required=True,
                        help='Path to YAML config (e.g. experiments/sutrack/sutrack_t224.yaml)')
    parser.add_argument('--init_bbox', type=float, nargs=4,
                        metavar=('X', 'Y', 'W', 'H'),
                        help='Initial bounding box. Skips GUI selection if provided.')
    parser.add_argument('--headless', action='store_true',
                        help='Disable display (requires --init_bbox)')
    parser.add_argument('--save_path', default=None,
                        help='Optional: save annotated output video to this path')
    args = parser.parse_args()

    if args.headless and args.init_bbox is None:
        parser.error('--headless requires --init_bbox')

    # Load tracker
    print(f'Loading engine: {args.engine}')
    tracker = TRTSUTrack(args.engine, args.config)

    # Open video
    cap = cv.VideoCapture(args.video_path)
    if not cap.isOpened():
        sys.exit(f'Cannot open video: {args.video_path}')

    ret, frame = cap.read()
    if not ret:
        sys.exit('Empty video')

    # Get initial bounding box
    if args.init_bbox is not None:
        x, y, w, h = [int(v) for v in args.init_bbox]
    else:
        win = 'SUTrack TRT'
        cv.namedWindow(win, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(win, 960, 720)
        disp = frame.copy()
        cv.putText(disp, 'Draw ROI then press SPACE / ENTER', (20, 40),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 200, 0), 2)
        # imshow must be called BEFORE selectROI to initialize the Qt window handler
        cv.imshow(win, disp)
        cv.waitKey(1)
        x, y, w, h = cv.selectROI(win, disp, fromCenter=False)

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    init_state = [int(x), int(y), int(w), int(h)]
    print(f'Initializing with bbox {init_state}')
    tracker.initialize(frame_rgb, init_state)

    # Optional video writer
    writer = None
    if args.save_path:
        fps_src = cap.get(cv.CAP_PROP_FPS) or 30.0
        W_src   = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        H_src   = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        writer  = cv.VideoWriter(args.save_path,
                                 cv.VideoWriter_fourcc(*'mp4v'),
                                 fps_src, (W_src, H_src))
        print(f'Saving output to {args.save_path}')

    win = None if args.headless else 'SUTrack TRT'
    if win:
        cv.namedWindow(win, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

    frame_count = 0
    total_time  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        t0    = time.perf_counter()
        state = tracker.track(frame_rgb)
        t1    = time.perf_counter()

        dt          = t1 - t0
        total_time += dt
        frame_count += 1
        fps = 1.0 / dt

        # Draw result
        sx, sy, sw, sh = [int(v) for v in state]
        disp = frame.copy()
        cv.rectangle(disp, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        cv.putText(disp, f'TRT {fps:.1f} FPS', (20, 50),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 2)

        if writer:
            writer.write(disp)
        if win:
            cv.imshow(win, disp)
            if cv.waitKey(1) == ord('q'):
                break

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f'\n--- Done ---')
    print(f'Frames    : {frame_count}')
    print(f'Total time: {total_time:.2f}s')
    print(f'Avg FPS   : {avg_fps:.2f}')

    cap.release()
    if writer:
        writer.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

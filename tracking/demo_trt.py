import os
import sys
import argparse
import time
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import cv2 as cv
import tensorrt as trt
from cuda import cudart

import torch

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.parameter.sutrack import parameters
from lib.test.tracker.utils import Preprocessor, transform_image_to_crop, sample_target
from lib.test.utils.hann import hann2d

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine, batch_size=1):
    inputs = {}
    outputs = {}
    bindings = []
    import ctypes
    
    # Create stream
    _, stream = cudart.cudaStreamCreate()
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        dims = engine.get_tensor_shape(name)
        
        # Handle dynamic batch size
        if dims[0] == -1:
            dims = (batch_size,) + dims[1:]
            
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(dims) * np.dtype(dtype).itemsize
        
        # Allocate host and device buffers
        _, host_mem = cudart.cudaMallocHost(size)
        _, device_mem = cudart.cudaMalloc(size)
        
        # Cast to numpy array for easier interaction on host
        # host_mem is an int (memory address), we need to cast it using ctypes
        ctypes_type = ctypes.c_uint8 * size
        buffer_ptr = ctypes.cast(host_mem, ctypes.POINTER(ctypes_type)).contents
        host_mem_np = np.frombuffer(buffer_ptr, dtype=dtype).reshape(dims)
        
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if is_input:
            inputs[name] = HostDeviceMem(host_mem_np, device_mem)
        else:
            outputs[name] = HostDeviceMem(host_mem_np, device_mem)
            
    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    for name, inp in inputs.items():
        cudart.cudaMemcpyAsync(inp.device, inp.host.ctypes.data, inp.host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream)
    
    # Transfer predictions back from the GPU.
    for name, out in outputs.items():
        cudart.cudaMemcpyAsync(out.host.ctypes.data, out.device, out.host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        
    # Synchronize the stream
    cudart.cudaStreamSynchronize(stream)
    return {name: out.host for name, out in outputs.items()}

class TRTSUTrack:
    def __init__(self, engine_path, param_name):
        self.params = parameters(param_name)
        self.cfg = self.params.cfg
        
        # Load TRT Engine
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # We enforce batch_size=1 for the demo
        self.context.set_input_shape('template', (1, 6, 112, 112)) # template
        self.context.set_input_shape('search', (1, 6, 224, 224)) # search
        self.context.set_input_shape('template_anno', (1, 4)) # template_anno
        
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, batch_size=1)
        
        self.preprocessor = Preprocessor()
        
        # Feature map size (search_size / stride = 224 / 16 = 14)
        self.feat_sz = int(self.cfg.DATA.SEARCH.SIZE / self.cfg.MODEL.ENCODER.STRIDE)
        
        # Pre-compute Hanning window as numpy array for the feature map
        self.output_window = None
        if self.cfg.TEST.WINDOW:
            win = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True)
            self.output_window = win.squeeze().numpy()  # (feat_sz, feat_sz)
        
        self.state = None
        self.template_list = []
        self.template_anno_list = []
        
        # For mapping boxes back
        self.search_size = self.cfg.TEST.SEARCH_SIZE
        self.template_size = self.cfg.TEST.TEMPLATE_SIZE
        # In this demo we assume num_templates=1 for simplicity
        self.num_template = 1

    def initialize(self, image, init_bbox):
        self.state = init_bbox
        
        # Get template
        z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor, output_sz=self.template_size)
        template = self.preprocessor.process(z_patch_arr)
        
        # Convert RGB to 6 channel (Multi-Modal fallback in SUTrack)
        if template.shape[1] == 3:
            template = torch.cat((template, template), axis=1)
            
        # Get template annotations
        prev_box_crop = transform_image_to_crop(
            torch.tensor(self.state, dtype=torch.float32),
            torch.tensor(self.state, dtype=torch.float32),
            resize_factor,
            torch.tensor([self.template_size, self.template_size], dtype=torch.float32),
            normalize=True
        )
        
        self.template_list = [template.contiguous().numpy().astype(np.float32)]
        self.template_anno_list = [prev_box_crop.unsqueeze(0).contiguous().numpy().astype(np.float32)]
        
    def track(self, image):
        H, W, _ = image.shape
        # Get search area
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor, output_sz=self.search_size)
        
        search = self.preprocessor.process(x_patch_arr)
        if search.shape[1] == 3:
            search = torch.cat((search, search), axis=1)
            
        search_np = search.contiguous().numpy().astype(np.float32)
            
        # Format TRT inputs
        np.copyto(self.inputs['template'].host, self.template_list[0])
        np.copyto(self.inputs['search'].host, search_np)
        np.copyto(self.inputs['template_anno'].host, self.template_anno_list[0])
        
        # Infer
        trt_outputs = do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        
        # Unpack TRT outputs as numpy
        # NOTE: score_map and size_map have sigmoid ALREADY applied inside the TRT graph.
        # offset_map is raw logits (no sigmoid in training code).
        score_map = trt_outputs['score_map'].reshape(self.feat_sz, self.feat_sz)
        size_map  = trt_outputs['size_map'].reshape(2, self.feat_sz * self.feat_sz)
        offset_map = trt_outputs['offset_map'].reshape(2, self.feat_sz * self.feat_sz)
        
        # Apply Hanning window penalty (must be done BEFORE argmax)
        if self.output_window is not None:
            score_map = score_map * self.output_window
        
        # Pure-numpy cal_bbox (mirrors decoder.py CenterPredictor.cal_bbox exactly)
        score_flat = score_map.reshape(-1)
        idx = int(np.argmax(score_flat))
        conf_score = float(score_flat[idx])
        
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz
        
        # size and offset at the peak location
        w_norm = float(size_map[0, idx])   # already sigmoided [0,1]
        h_norm = float(size_map[1, idx])   # already sigmoided [0,1]
        off_x  = float(offset_map[0, idx]) # raw logit
        off_y  = float(offset_map[1, idx]) # raw logit
        
        cx_norm = (idx_x + off_x) / self.feat_sz
        cy_norm = (idx_y + off_y) / self.feat_sz
        
        # Scale from [0,1] to original image pixel space
        scale = self.search_size / resize_factor  # = crop_sz
        cx = cx_norm * scale
        cy = cy_norm * scale
        w  = w_norm  * scale
        h  = h_norm  * scale
        
        # Map from crop-space back to original image coords
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        half_side = 0.5 * scale
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        
        # Convert cx,cy,w,h -> x,y,w,h and clamp position only
        x1 = cx_real - 0.5 * w
        y1 = cy_real - 0.5 * h
        x1 = max(0.0, min(x1, W - 1.0))
        y1 = max(0.0, min(y1, H - 1.0))
        
        self.state = [x1, y1, w, h]
        
        return self.state
        
def main():
    parser = argparse.ArgumentParser(description='Run custom TRT tracker on a video file.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT .engine file.')
    parser.add_argument('--param', type=str, default='sutrack_t224', help='Config parameter name.')
    parser.add_argument('--init_bbox', type=float, nargs=4, help='Initial bbox [x,y,w,h]. If provided, bypasses GUI selection.')
    parser.add_argument('--headless', action='store_true', help='Disable video rendering for raw performance testing.')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save output video (.mp4 format).')
    
    args = parser.parse_args()

    print(f"Loading TRT Engine from {args.engine}...")
    tracker = TRTSUTrack(args.engine, args.param)
    
    cap = cv.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Failed to open video {args.video_path}")
        return
        
    ret, frame = cap.read()
    if not ret:
        return
    if args.init_bbox is not None:
        x, y, w, h = [int(v) for v in args.init_bbox]
    else:
        if args.headless:
            print("Error: --headless requires --init_bbox since GUI selection is disabled.")
            return
            
        display_name = 'TensorRT SUTrack Demo'
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        
        frame_disp = frame.copy()
        cv.putText(frame_disp, 'Select target ROI and press SPACE', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
        x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    init_state = [x, y, w, h]
    print(f"Initializing tracker with box {init_state}")
    tracker.initialize(frame_rgb, init_state)
    
    # Initialize Video Writer if requested
    video_writer = None
    if args.save_path:
        fps_out = cap.get(cv.CAP_PROP_FPS)
        width_out = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height_out = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(args.save_path, fourcc, fps_out, (width_out, height_out))
        print(f"Saving tracking output to {args.save_path}")
    
    frame_count = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if frame is None:
            break
            
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
        start_time = time.perf_counter()
        state = tracker.track(frame_rgb)
        end_time = time.perf_counter()
        
        frame_time = end_time - start_time
        total_time += frame_time
        frame_count += 1
        fps = 1.0 / frame_time

        state = [int(s) for s in state]
        frame_disp = frame.copy()
        cv.rectangle(frame_disp, (state[0], state[1]), (state[0]+state[2], state[1]+state[3]), (0, 255, 0), 2)
        cv.putText(frame_disp, f'TRT Tracking: {fps:.1f} FPS', (20, 50), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        if video_writer:
            video_writer.write(frame_disp)

        if not args.headless:
            cv.imshow(display_name, frame_disp)
            
            if cv.waitKey(1) == ord('q'):
                break

    print(f"\n--- Performance Summary ---")
    print(f"Processed {frame_count} frames.")
    print(f"Average FPS: {frame_count / total_time:.2f}")

    cap.release()
    if video_writer:
        video_writer.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

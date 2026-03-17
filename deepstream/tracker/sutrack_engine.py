"""
SUTrackEngine — Shared TensorRT Inference Engine

Loads the sutrack_t224 FP32 TRT engine once. All TrackerInstance objects
share this single engine context and call infer() sequentially per frame.

Design rules enforced here:
- FP32 only (no fp16)
- Named TRT bindings — never by index (Lesson 1)
- cuda.cudart with ctypes wrapping — no pycuda (Lessons 7, 8)
- score_map / size_map are NOT double-sigmoided (sigmoid already in graph, Lesson 3)
"""

import ctypes
import logging
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import tensorrt as trt
from cuda import cudart

logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

class HostDeviceMem:
    """Paired pinned host and device memory buffers for one TRT tensor."""
    def __init__(self, host_mem: np.ndarray, device_mem, name: str):
        self.host   = host_mem
        self.device = device_mem
        self.name   = name


def _allocate_buffers(engine, template_size: int, search_size: int):
    """
    Allocate pinned host + device buffers for all engine I/O tensors.

    Uses named tensor access (engine.get_tensor_name) — never by index (Lesson 1).
    """
    err, stream = cudart.cudaStreamCreate()
    if err.value != 0:
        raise RuntimeError(f'cudaStreamCreate failed: {err}')

    inputs, outputs, bindings = {}, {}, []

    for i in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = engine.get_tensor_shape(name)
        shape = tuple(1 if d == -1 else d for d in shape)
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        # Pinned host memory — wrap raw int pointer with ctypes (Lesson 8)
        err2, host_ptr = cudart.cudaMallocHost(nbytes)
        if err2.value != 0:
            raise RuntimeError(f'cudaMallocHost failed for {name}: {err2}')
        host_np = np.frombuffer(
            (ctypes.c_char * nbytes).from_address(int(host_ptr)),
            dtype=dtype).reshape(shape)

        err3, device_ptr = cudart.cudaMalloc(nbytes)
        if err3.value != 0:
            raise RuntimeError(f'cudaMalloc failed for {name}: {err3}')
        bindings.append(int(device_ptr))

        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        mem = HostDeviceMem(host_np, device_ptr, name)
        (inputs if is_input else outputs)[name] = mem
        logger.debug('Allocated %s: %s shape=%s', 'input' if is_input else 'output', name, shape)

    return inputs, outputs, bindings, stream


# ---------------------------------------------------------------------------
# SUTrackEngine
# ---------------------------------------------------------------------------

class SUTrackEngine:
    """
    Loads a sutrack_t224 FP32 TensorRT engine and exposes infer().

    All TrackerInstance objects share one SUTrackEngine — they copy their own
    template/search arrays into the shared buffers before calling infer().

    Args:
        engine_path:    Path to the compiled .engine file
        template_size:  Template crop size in pixels (default 112)
        search_size:    Search crop size in pixels (default 224)
        encoder_stride: Encoder patch stride (default 16)
    """

    def __init__(self, engine_path: str,
                 template_size: int = 112,
                 search_size: int = 224,
                 encoder_stride: int = 16):
        self.template_size  = template_size
        self.search_size    = search_size
        self.encoder_stride = encoder_stride
        self.feat_sz        = search_size // encoder_stride  # 224 // 16 = 14

        logger.info('Loading TRT engine: %s', engine_path)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.context.set_input_shape('template',      (1, 6, template_size, template_size))
        self.context.set_input_shape('search',        (1, 6, search_size,   search_size))
        self.context.set_input_shape('template_anno', (1, 4))

        self.inputs, self.outputs, self.bindings, self.stream = \
            _allocate_buffers(self.engine, template_size, search_size)

        logger.info('SUTrackEngine ready. feat_sz=%d  inputs=%s', self.feat_sz, list(self.inputs))

    # ------------------------------------------------------------------
    def infer(self, template_img: np.ndarray,
              search_img: np.ndarray,
              template_anno: np.ndarray) -> dict:
        """
        Run one TRT forward pass.

        Args:
            template_img:   (1, 6, template_size, template_size) float32
            search_img:     (1, 6, search_size, search_size) float32
            template_anno:  (1, 4) float32 — normalized [x1, y1, w, h] in [0,1]

        Returns:
            dict with keys: 'score_map', 'size_map', 'offset_map', 'pred_boxes'
            score_map and size_map are already sigmoid-activated inside the TRT graph.
            Do NOT apply sigmoid again (Lesson 3).
        """
        np.copyto(self.inputs['template'].host,      template_img)
        np.copyto(self.inputs['search'].host,        search_img)
        np.copyto(self.inputs['template_anno'].host, template_anno)

        for inp in self.inputs.values():
            cudart.cudaMemcpyAsync(
                inp.device, inp.host.ctypes.data, inp.host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)

        for out in self.outputs.values():
            cudart.cudaMemcpyAsync(
                out.host.ctypes.data, out.device, out.host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

        cudart.cudaStreamSynchronize(self.stream)
        return {name: mem.host for name, mem in self.outputs.items()}

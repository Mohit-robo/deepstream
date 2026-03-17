# SUTrack — TensorRT Deployment

Self-contained TensorRT inference package. **No PyTorch required** at runtime.

## Files

| File | Purpose |
|------|---------|
| `demo_trt.py` | Main tracking demo — run this |
| `utils.py` | Pure-NumPy utilities (preprocessing, crop extraction, Hanning window, config loader) |

## Dependencies

```
numpy
opencv-python
tensorrt        (from NVIDIA JetPack / TensorRT SDK)
cuda-python     (pip install cuda-python)
pyyaml          (pip install pyyaml)
```

## Usage

### Interactive (GUI selection)
```bash
python demo_trt.py video.mp4 \
    --engine /path/to/sutrack_t224.engine \
    --config /path/to/experiments/sutrack/sutrack_t224.yaml
```

### Headless with saved output
```bash
python demo_trt.py video.mp4 \
    --engine /path/to/sutrack_t224.engine \
    --config /path/to/experiments/sutrack/sutrack_t224.yaml \
    --init_bbox 300 200 120 140 \
    --headless \
    --save_path output.mp4
```

### Arguments
| Argument | Description |
|----------|-------------|
| `video_path` | Input video file |
| `--engine` | Path to compiled `.engine` file |
| `--config` | Path to SUTrack YAML experiment config |
| `--init_bbox X Y W H` | Initial bounding box (skips GUI) |
| `--headless` | No display window (requires `--init_bbox`) |
| `--save_path` | Save annotated video to this path |

## Engine Compilation (on Jetson)

First export to ONNX from the SUTrack project root:
```bash
python tracking/export_onnx.py --param sutrack_t224
```

Then compile to a TensorRT engine:
```bash
trtexec \
    --onnx=sutrack_t224.onnx \
    --saveEngine=sutrack_t224.engine \
    --minShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --optShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --maxShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --memPoolSize=workspace:4096MiB
```

> **Important:** Do **not** add `--fp16`. FP16 causes underflow in the decoder's
> sigmoid activations, producing near-zero scores and broken tracking. Leave it
> as FP32 (the default).

## Benchmark (Jetson, sutrack_t224, FP32)

| Metric | Value |
|--------|-------|
| Avg FPS | **~25 FPS** |
| Frames processed | 184 |
| Hanning window | ✅ Enabled |

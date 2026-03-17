# SUTrack - ONNX Runtime Deployment

Self-contained ONNX Runtime inference package. **No PyTorch or TensorRT required** at runtime.

## Files

| File | Purpose |
|------|---------|
| `demo_onnx.py` | Main tracking demo - run this |
| `utils.py` | Pure-NumPy utilities (preprocessing, crop extraction, Hanning window, config loader) |

## Dependencies

```bash
pip install numpy opencv-python onnxruntime-gpu pyyaml
```
*Use `onnxruntime` if you don't have a GPU.*

## Usage

### Interactive (GUI selection)
```bash
python demo_onnx.py video.mp4 \
    --model sutrack.onnx \
    --config sutrack_t224.yaml
```

### Headless with saved output
```bash
python demo_onnx.py video.mp4 \
    --model sutrack.onnx \
    --config sutrack_t224.yaml \
    --init_bbox 300 200 120 140 \
    --headless \
    --save_path output.mp4
```

### Arguments
| Argument | Description |
|----------|-------------|
| `video_path` | Input video file |
| `--model` | Path to `.onnx` file |
| `--config` | Path to SUTrack YAML experiment config |
| `--init_bbox X Y W H` | Initial bounding box (skips GUI) |
| `--headless` | No display window (requires `--init_bbox`) |
| `--save_path` | Save annotated video to this path |

## Exporting the ONNX Model

Export to ONNX from the SUTrack project root:
```bash
python tracking/export_onnx.py --param sutrack_t224 --output SUTrack_deploy_onnx/sutrack.onnx
```

## Performance Note
ONNX Runtime with `CUDAExecutionProvider` provides good performance on Jetson, though usually slightly lower than native TensorRT. It is excellent for quick validation and cross-platform compatibility.

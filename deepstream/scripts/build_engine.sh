#!/usr/bin/env bash
# Build the sutrack_t224 TensorRT FP32 engine from ONNX.
#
# IMPORTANT: Do NOT add --fp16. The SUTrack decoder uses sigmoid activations
# that underflow to zero in FP16, producing broken near-zero scores (Lesson 2).
#
# Run from the SUTrack/ project root:
#   bash deepstream/scripts/build_engine.sh [path/to/sutrack.onnx] [output/sutrack_fp32.engine]

ONNX="${1:-sutrack.onnx}"
ENGINE="${2:-sutrack_fp32.engine}"

echo "Building TRT engine..."
echo "  ONNX   : $ONNX"
echo "  Engine : $ENGINE"

trtexec \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --minShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --optShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --maxShapes=template:1x6x112x112,search:1x6x224x224,template_anno:1x4 \
    --memPoolSize=workspace:4096MiB

echo "Done: $ENGINE"

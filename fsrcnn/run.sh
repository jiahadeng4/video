#!/bin/bash
# Quick start script for FSRCNN real-time video processing

set -e

# Default values
ENGINE="${1:-models/fsrcnn_x2_fp16.engine}"
INPUT="${2:-0}"
PYTORCH="${3:-}"

echo "=========================================="
echo "FSRCNN Real-time Video Processor"
echo "=========================================="

# Check if engine exists, if not try to convert
if [ ! -f "$ENGINE" ]; then
    echo "⚠ Engine file not found: $ENGINE"
    
    if [ -n "$PYTORCH" ] && [ -f "$PYTORCH" ]; then
        echo "Found PyTorch model: $PYTORCH"
        echo "Starting auto-conversion..."
        python3 auto_convert_to_tensorrt.py --pytorch "$PYTORCH" --engine "$ENGINE"
    else
        echo "Please provide PyTorch model path:"
        echo "  ./run.sh [engine] [input] [pytorch_model]"
        echo ""
        echo "Or convert manually:"
        echo "  python3 auto_convert_to_tensorrt.py --pytorch results/fsrcnn_x2/best.pth.tar"
        exit 1
    fi
fi

# Run the processor
echo "Starting video processing..."
python3 realtime_video_processor.py \
    --engine "$ENGINE" \
    --input "$INPUT"


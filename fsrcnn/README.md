# FSRCNN TensorRT Real-time Video Processing

This directory contains optimized scripts for real-time video super-resolution using FSRCNN with TensorRT acceleration, designed for low-latency applications like remote forklift control.

## Quick Start

### 1. Automatic Conversion and Processing (Recommended)

```bash
# Process video with auto-conversion (if engine doesn't exist)
python3 realtime_video_processor.py --pytorch results/fsrcnn_x2/best.pth.tar

# Or just run with default settings
python3 realtime_video_processor.py
```

### 2. Manual Conversion

```bash
# Step 1: Convert PyTorch to ONNX
python3 convert_pytorch_to_onnx.py \
    --input results/fsrcnn_x2/best.pth.tar \
    --output models/fsrcnn_x2.onnx

# Step 2: Convert ONNX to TensorRT
python3 convert_onnx_to_tensorrt.py \
    --onnx models/fsrcnn_x2.onnx \
    --engine models/fsrcnn_x2_fp16.engine \
    --fp16

# Step 3: Run real-time processing
python3 realtime_video_processor.py --engine models/fsrcnn_x2_fp16.engine
```

### 3. One-Click Auto Conversion

```bash
# Automatically convert PyTorch model to TensorRT engine
python3 auto_convert_to_tensorrt.py --pytorch results/fsrcnn_x2/best.pth.tar
```

## File Descriptions

### Core Scripts

- **`realtime_video_processor.py`** - Main real-time video processing script
  - Supports camera and video file input
  - Automatic model conversion
  - Command-line interface
  - Performance monitoring

- **`auto_convert_to_tensorrt.py`** - Automated conversion pipeline
  - PyTorch → ONNX → TensorRT in one command
  - Smart file detection
  - Interactive prompts

- **`convert_pytorch_to_onnx.py`** - PyTorch to ONNX converter
- **`convert_onnx_to_tensorrt.py`** - ONNX to TensorRT converter
- **`tensorrt_inference.py`** - TensorRT inference engine class

## Usage Examples

### Process Camera Input

```bash
# Default camera (index 0)
python3 realtime_video_processor.py

# Specific camera
python3 realtime_video_processor.py --input 1

# No preview (faster)
python3 realtime_video_processor.py --input 0 --no-preview
```

### Process Video File

```bash
# Process and save
python3 realtime_video_processor.py \
    --input input_video.mp4 \
    --output enhanced_video.mp4

# Process without saving
python3 realtime_video_processor.py --input input_video.mp4
```

### Custom Engine Path

```bash
python3 realtime_video_processor.py \
    --engine /path/to/your/engine.engine \
    --input 0
```

### Auto-Convert from PyTorch

```bash
# Auto-convert and process
python3 realtime_video_processor.py \
    --pytorch results/fsrcnn_x2/best.pth.tar \
    --input 0
```

## Command-Line Options

### realtime_video_processor.py

```
--engine PATH          Path to TensorRT engine file
--pytorch PATH         PyTorch model path (for auto-conversion)
--input SOURCE         Camera index (0, 1, 2...) or video file path
--output PATH          Output video path (optional)
--upscale FACTOR       Upscale factor: 2, 3, or 4 (default: 2)
--fps FPS              Target FPS (default: 30)
--no-preview           Disable preview window
--no-auto-convert      Disable automatic model conversion
```

### auto_convert_to_tensorrt.py

```
--pytorch PATH         PyTorch model file path
--engine PATH          Output TensorRT engine path (auto-generated if not specified)
--upscale FACTOR       Upscale factor (default: 2)
--input_height HEIGHT  Input image height (default: 480)
--input_width WIDTH    Input image width (default: 640)
--fp16                 Use FP16 precision (default)
--fp32                 Use FP32 precision
--workspace_size SIZE  TensorRT workspace size in bytes (default: 1GB)
--force                Force rebuild even if files exist
```

## Performance

Expected performance on RTX 3060 with FP16:
- **Latency**: <20ms per frame (480p input)
- **FPS**: 50+ FPS (480p input)
- **Memory**: <500MB (FP16 engine)

## Troubleshooting

### Engine file not found

```bash
# Option 1: Auto-convert
python3 realtime_video_processor.py --pytorch results/fsrcnn_x2/best.pth.tar

# Option 2: Manual conversion
python3 auto_convert_to_tensorrt.py --pytorch results/fsrcnn_x2/best.pth.tar
```

### Camera not working

```bash
# Check available cameras
ls /dev/video*

# Try different camera index
python3 realtime_video_processor.py --input 1
```

### Import errors

```bash
# Check TensorRT installation
python3 -c "import tensorrt as trt; print(trt.__version__)"

# Check PyCUDA
python3 -c "import pycuda.driver as cuda; print('OK')"
```

## Requirements

- Python 3.7+
- PyTorch
- TensorRT 8.0+
- CUDA 11.0+
- OpenCV
- PyCUDA
- ONNX

## Notes

- The model processes only the Y channel in YCbCr color space
- Cb and Cr channels are upsampled using bicubic interpolation
- Input images are automatically downsampled to 480p if larger
- All processing is done on GPU for maximum performance


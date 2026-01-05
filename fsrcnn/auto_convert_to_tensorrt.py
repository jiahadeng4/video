# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Automated conversion from PyTorch model to TensorRT engine."""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if file exists and print status."""
    if os.path.exists(filepath):
        print(f"✓ {description} found: {filepath}")
        return True
    else:
        print(f"✗ {description} not found: {filepath}")
        return False


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed!")
        return False
    print(f"✓ {description} completed successfully!")
    return True


def convert_pytorch_to_onnx(pytorch_path, onnx_path, upscale_factor, input_size):
    """Convert PyTorch model to ONNX."""
    script_path = os.path.join(os.path.dirname(__file__), "convert_pytorch_to_onnx.py")
    if not os.path.exists(script_path):
        print(f"✗ Error: convert_pytorch_to_onnx.py not found at {script_path}")
        return False
    
    cmd = [
        sys.executable,
        script_path,
        "--input", pytorch_path,
        "--output", onnx_path,
        "--upscale_factor", str(upscale_factor),
        "--input_height", str(input_size[0]),
        "--input_width", str(input_size[1])
    ]
    return run_command(cmd, "PyTorch → ONNX Conversion")


def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16, workspace_size):
    """Convert ONNX model to TensorRT engine."""
    script_path = os.path.join(os.path.dirname(__file__), "convert_onnx_to_tensorrt.py")
    if not os.path.exists(script_path):
        print(f"✗ Error: convert_onnx_to_tensorrt.py not found at {script_path}")
        return False
    
    cmd = [
        sys.executable,
        script_path,
        "--onnx", onnx_path,
        "--engine", engine_path
    ]
    
    if fp16:
        cmd.append("--fp16")
    
    if workspace_size:
        cmd.extend(["--workspace_size", str(workspace_size)])
    
    return run_command(cmd, "ONNX → TensorRT Conversion")


def auto_convert(pytorch_path, engine_path, upscale_factor=2, 
                 input_size=(480, 640), fp16=True, workspace_size=1<<30,
                 force_rebuild=False):
    """
    Automatically convert PyTorch model to TensorRT engine.
    
    Args:
        pytorch_path: Path to PyTorch model (.pth.tar)
        engine_path: Output path for TensorRT engine (.engine)
        upscale_factor: Super-resolution upscale factor
        input_size: Input image size (height, width)
        fp16: Use FP16 precision
        workspace_size: TensorRT workspace size in bytes
        force_rebuild: Force rebuild even if engine exists
    """
    print("=" * 60)
    print("FSRCNN Automatic TensorRT Conversion")
    print("=" * 60)
    
    # Check if engine already exists
    if os.path.exists(engine_path) and not force_rebuild:
        print(f"\n✓ TensorRT engine already exists: {engine_path}")
        response = input("Do you want to rebuild it? (y/N): ").strip().lower()
        if response != 'y':
            print("Using existing engine file.")
            return True
    
    # Determine intermediate file paths
    onnx_path = engine_path.replace('.engine', '.onnx')
    if not onnx_path.endswith('.onnx'):
        onnx_path = engine_path.rsplit('.', 1)[0] + '.onnx'
    
    # Create output directory
    os.makedirs(os.path.dirname(engine_path) if os.path.dirname(engine_path) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(onnx_path) if os.path.dirname(onnx_path) else '.', exist_ok=True)
    
    # Step 1: Check PyTorch model
    print(f"\n[1/3] Checking PyTorch model...")
    if not check_file_exists(pytorch_path, "PyTorch model"):
        print(f"\n✗ Error: PyTorch model not found: {pytorch_path}")
        print("Please train the model first or provide the correct path.")
        return False
    
    # Step 2: Convert PyTorch to ONNX (if needed)
    print(f"\n[2/3] Converting PyTorch to ONNX...")
    if os.path.exists(onnx_path) and not force_rebuild:
        print(f"✓ ONNX file already exists: {onnx_path}")
        response = input("Do you want to regenerate it? (y/N): ").strip().lower()
        if response == 'y':
            if not convert_pytorch_to_onnx(pytorch_path, onnx_path, upscale_factor, input_size):
                return False
    else:
        if not convert_pytorch_to_onnx(pytorch_path, onnx_path, upscale_factor, input_size):
            return False
    
    # Verify ONNX file
    if not check_file_exists(onnx_path, "ONNX model"):
        return False
    
    # Step 3: Convert ONNX to TensorRT
    print(f"\n[3/3] Converting ONNX to TensorRT...")
    if not convert_onnx_to_tensorrt(onnx_path, engine_path, fp16, workspace_size):
        return False
    
    # Verify engine file
    if not check_file_exists(engine_path, "TensorRT engine"):
        return False
    
    print("\n" + "=" * 60)
    print("✓ Conversion completed successfully!")
    print("=" * 60)
    print(f"PyTorch model: {pytorch_path}")
    print(f"ONNX model:    {onnx_path}")
    print(f"TensorRT engine: {engine_path}")
    print(f"Engine size:   {os.path.getsize(engine_path) / (1024*1024):.2f} MB")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Automatically convert PyTorch FSRCNN model to TensorRT engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-convert with default settings
  python3 auto_convert_to_tensorrt.py --pytorch results/fsrcnn_x2/best.pth.tar
  
  # Specify output engine path
  python3 auto_convert_to_tensorrt.py \\
      --pytorch results/fsrcnn_x2/best.pth.tar \\
      --engine models/fsrcnn_x2_fp16.engine
  
  # Force rebuild
  python3 auto_convert_to_tensorrt.py \\
      --pytorch results/fsrcnn_x2/best.pth.tar \\
      --force
        """
    )
    
    parser.add_argument(
        '--pytorch',
        type=str,
        default='results/fsrcnn_x2/best.pth.tar',
        help='Path to PyTorch model file (default: results/fsrcnn_x2/best.pth.tar)'
    )
    parser.add_argument(
        '--engine',
        type=str,
        default=None,
        help='Output TensorRT engine path (default: auto-generated from pytorch path)'
    )
    parser.add_argument(
        '--upscale',
        type=int,
        default=2,
        help='Upscale factor (default: 2)'
    )
    parser.add_argument(
        '--input_height',
        type=int,
        default=480,
        help='Input image height (default: 480)'
    )
    parser.add_argument(
        '--input_width',
        type=int,
        default=640,
        help='Input image width (default: 640)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='Use FP16 precision (default: True)'
    )
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Use FP32 precision (overrides --fp16)'
    )
    parser.add_argument(
        '--workspace_size',
        type=int,
        default=1<<30,  # 1GB
        help='TensorRT workspace size in bytes (default: 1GB)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if files exist'
    )
    
    args = parser.parse_args()
    
    # Determine engine path
    if args.engine is None:
        # Auto-generate from pytorch path
        pytorch_dir = os.path.dirname(args.pytorch)
        pytorch_name = os.path.basename(args.pytorch).rsplit('.', 2)[0]  # Remove .pth.tar
        precision = "fp16" if (args.fp16 and not args.fp32) else "fp32"
        args.engine = os.path.join("models", f"{pytorch_name}_{precision}.engine")
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(args.engine) if os.path.dirname(args.engine) else '.', exist_ok=True)
    
    # Run conversion
    success = auto_convert(
        pytorch_path=args.pytorch,
        engine_path=args.engine,
        upscale_factor=args.upscale,
        input_size=(args.input_height, args.input_width),
        fp16=args.fp16 and not args.fp32,
        workspace_size=args.workspace_size,
        force_rebuild=args.force
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


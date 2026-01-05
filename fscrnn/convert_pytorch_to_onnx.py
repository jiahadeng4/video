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
"""Convert PyTorch model to ONNX format for TensorRT optimization."""
import argparse
import os

import torch
import torch.onnx

import config
from model import FSRCNN


def convert_to_onnx(pytorch_model_path: str, onnx_model_path: str, 
                    upscale_factor: int = 2, input_size: tuple = (480, 640)) -> None:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        pytorch_model_path: Path to PyTorch model checkpoint
        onnx_model_path: Output path for ONNX model
        upscale_factor: Super-resolution upscale factor
        input_size: Input image size (height, width)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
    
    # Load PyTorch model
    print(f"Loading PyTorch model from {pytorch_model_path}...")
    model = FSRCNN(upscale_factor).to(config.device)
    checkpoint = torch.load(pytorch_model_path, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Create dummy input (batch_size=1, channels=1, height, width)
    dummy_input = torch.randn(1, 1, input_size[0], input_size[1]).to(config.device)
    
    # Export to ONNX
    print(f"Exporting to ONNX format...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {2: 'height', 3: 'width'},
                'output': {2: 'height', 3: 'width'}
            }
        )
    
    print(f"ONNX model saved to: {onnx_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch FSRCNN model to ONNX format")
    parser.add_argument(
        '--input', 
        type=str, 
        default='results/fsrcnn_x2/best.pth.tar',
        help='Input PyTorch model path'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='models/fsrcnn_x2.onnx',
        help='Output ONNX model path'
    )
    parser.add_argument(
        '--upscale_factor', 
        type=int, 
        default=2,
        help='Super-resolution upscale factor'
    )
    parser.add_argument(
        '--input_height', 
        type=int, 
        default=480,
        help='Input image height'
    )
    parser.add_argument(
        '--input_width', 
        type=int, 
        default=640,
        help='Input image width'
    )
    
    args = parser.parse_args()
    
    convert_to_onnx(
        pytorch_model_path=args.input,
        onnx_model_path=args.output,
        upscale_factor=args.upscale_factor,
        input_size=(args.input_height, args.input_width)
    )


if __name__ == "__main__":
    main()


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
"""Convert ONNX model to TensorRT engine for optimized inference."""
import argparse
import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_file_path: str, engine_file_path: str, 
                 max_batch_size: int = 1, fp16_mode: bool = True, 
                 max_workspace_size: int = 1 << 30) -> trt.ICudaEngine:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_file_path: Path to ONNX model file
        engine_file_path: Output path for TensorRT engine
        max_batch_size: Maximum batch size for inference
        fp16_mode: Enable FP16 precision mode for faster inference
        max_workspace_size: Maximum workspace size in bytes
    
    Returns:
        TensorRT engine object
    """
    # Check if input file exists
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_file_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(engine_file_path) if os.path.dirname(engine_file_path) else '.', exist_ok=True)
    
    # Create TensorRT builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print(f"Parsing ONNX model from {onnx_file_path}...")
    try:
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                error_msg = "ERROR: Failed to parse ONNX file\n"
                for error in range(parser.num_errors):
                    error_msg += f"  {parser.get_error(error)}\n"
                raise RuntimeError(error_msg)
    except Exception as e:
        print(f"✗ Error parsing ONNX file: {e}")
        return None
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # Enable FP16 precision if supported
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 precision enabled")
    else:
        print("FP16 precision not available, using FP32")
    
    # Build engine
    print("Building TensorRT engine... This may take a while...")
    try:
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Engine build returned None")
    except Exception as e:
        print(f"✗ ERROR: Engine build failed: {e}")
        return None
    
    # Save engine to file
    print(f"Saving TensorRT engine to {engine_file_path}...")
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved successfully!")
    print(f"Engine file size: {os.path.getsize(engine_file_path) / (1024 * 1024):.2f} MB")
    
    return engine


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine")
    parser.add_argument(
        '--onnx', 
        type=str, 
        default='models/fsrcnn_x2.onnx',
        help='Input ONNX model path'
    )
    parser.add_argument(
        '--engine', 
        type=str, 
        default='models/fsrcnn_x2_fp16.engine',
        help='Output TensorRT engine path'
    )
    parser.add_argument(
        '--max_batch_size', 
        type=int, 
        default=1,
        help='Maximum batch size'
    )
    parser.add_argument(
        '--fp16', 
        action='store_true',
        help='Enable FP16 precision mode'
    )
    parser.add_argument(
        '--workspace_size', 
        type=int, 
        default=1 << 30,  # 1GB
        help='Maximum workspace size in bytes'
    )
    
    args = parser.parse_args()
    
    build_engine(
        onnx_file_path=args.onnx,
        engine_file_path=args.engine,
        max_batch_size=args.max_batch_size,
        fp16_mode=args.fp16,
        max_workspace_size=args.workspace_size
    )


if __name__ == "__main__":
    main()


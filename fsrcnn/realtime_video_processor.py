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
"""Real-time video processing for low-latency applications (e.g., remote forklift control)."""
import cv2
import time
import os
import sys
import argparse
import subprocess
import numpy as np
from collections import deque

from tensorrt_inference import FSRCNNTensorRTInference


class RealtimeVideoProcessor:
    """Real-time video processor optimized for low-latency scenarios."""
    
    def __init__(self, engine_path: str, upscale_factor: int = 2, target_fps: int = 30):
        """
        Initialize real-time video processor.
        
        Args:
            engine_path: Path to TensorRT engine file
            upscale_factor: Super-resolution upscale factor
            target_fps: Target frame rate for processing
        """
        self.inference_engine = FSRCNNTensorRTInference(engine_path, upscale_factor)
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.latency_history = deque(maxlen=30)
        
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process single frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Tuple of (processed_frame, latency_ms)
        """
        start_time = time.time()
        
        # Downsample to 480p if input resolution is too large
        h, w = frame.shape[:2]
        if h > 480 or w > 640:
            scale = min(480 / h, 640 / w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Perform super-resolution
        sr_frame = self.inference_engine.infer(frame)
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.latency_history.append(latency)
        
        return sr_frame, latency
    
    def process_video_stream(self, input_source, output_path: str = None, 
                            show_preview: bool = True) -> None:
        """
        Process video stream from camera or video file.
        
        Args:
            input_source: Input source (camera index or video file path)
            output_path: Output video path (optional)
            show_preview: Whether to show preview window
        """
        cap = cv2.VideoCapture(input_source)
        
        # Get input video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.target_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set output video properties
        out_width = width * self.inference_engine.upscale_factor
        out_height = height * self.inference_engine.upscale_factor
        
        # Create video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        frame_count = 0
        total_time = 0
        
        print(f"Starting video stream processing: {width}x{height} @ {fps}fps")
        print(f"Output resolution: {out_width}x{out_height}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                sr_frame, latency = self.process_frame(frame)
                total_time += latency
                frame_count += 1
                
                # Display information
                avg_latency = total_time / frame_count
                current_fps = 1000.0 / latency if latency > 0 else 0
                
                if show_preview:
                    # Display performance information on image
                    info_text = [
                        f"Latency: {latency:.1f}ms",
                        f"Avg Latency: {avg_latency:.1f}ms",
                        f"FPS: {current_fps:.1f}",
                        f"Frame: {frame_count}"
                    ]
                    for i, text in enumerate(info_text):
                        cv2.putText(sr_frame, text, (10, 30 + i * 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('FSRCNN Real-time Processing', sr_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write output video
                if output_path:
                    out.write(sr_frame)
                
                # Print statistics every 10 frames
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count} frames, "
                          f"Avg Latency: {avg_latency:.1f}ms, "
                          f"Current FPS: {current_fps:.1f}")
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"\nProcessing completed!")
            print(f"Total frames: {frame_count}")
            print(f"Average latency: {total_time / frame_count:.2f}ms")
            print(f"Average FPS: {1000.0 / (total_time / frame_count):.2f}")


def ensure_engine_exists(engine_path, pytorch_path=None, auto_convert=True):
    """
    Ensure TensorRT engine exists, auto-convert if needed.
    
    Args:
        engine_path: Desired engine file path
        pytorch_path: PyTorch model path (for auto-conversion)
        auto_convert: Whether to auto-convert if engine doesn't exist
    
    Returns:
        bool: True if engine exists or was created successfully
    """
    if os.path.exists(engine_path):
        return True
    
    print(f"\n⚠ Engine file not found: {engine_path}")
    
    if not auto_convert:
        return False
    
    # Try to find PyTorch model
    if pytorch_path is None:
        # Common locations
        possible_paths = [
            "results/fsrcnn_x2/best.pth.tar",
            "results/fsrcnn_x2/last.pth.tar",
            "samples/fsrcnn_x2/best.pth.tar",
            "../FSRCNN-PyTorch/results/fsrcnn_x2/best.pth.tar",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytorch_path = path
                break
    
    if pytorch_path and os.path.exists(pytorch_path):
        print(f"✓ Found PyTorch model: {pytorch_path}")
        print("Starting automatic conversion to TensorRT...")
        
        # Check if auto_convert script exists
        auto_convert_script = os.path.join(os.path.dirname(__file__), "auto_convert_to_tensorrt.py")
        if os.path.exists(auto_convert_script):
            cmd = [
                sys.executable,
                auto_convert_script,
                "--pytorch", pytorch_path,
                "--engine", engine_path
            ]
            result = subprocess.run(cmd)
            if result.returncode == 0 and os.path.exists(engine_path):
                print(f"✓ Auto-conversion successful!")
                return True
        else:
            # Fallback: manual conversion steps
            print("Auto-convert script not found. Please run conversion manually:")
            onnx_path = engine_path.replace('.engine', '.onnx')
            print(f"  1. python3 convert_pytorch_to_onnx.py --input {pytorch_path} --output {onnx_path}")
            print(f"  2. python3 convert_onnx_to_tensorrt.py --onnx {onnx_path} --engine {engine_path} --fp16")
    
    print("\n✗ Cannot auto-convert. Please convert the model manually.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSRCNN Real-time Video Processor for Low-Latency Applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default camera with default engine
  python3 realtime_video_processor.py
  
  # Specify engine path
  python3 realtime_video_processor.py --engine models/fsrcnn_x2_fp16.engine
  
  # Process video file
  python3 realtime_video_processor.py --input video.mp4 --output enhanced.mp4
  
  # Use different camera, no preview
  python3 realtime_video_processor.py --input 1 --no-preview
  
  # Auto-convert from PyTorch model if engine doesn't exist
  python3 realtime_video_processor.py --pytorch results/fsrcnn_x2/best.pth.tar
        """
    )
    
    parser.add_argument(
        '--engine',
        type=str,
        default='models/fsrcnn_x2_fp16.engine',
        help='Path to TensorRT engine file (default: models/fsrcnn_x2_fp16.engine)'
    )
    parser.add_argument(
        '--pytorch',
        type=str,
        default=None,
        help='Path to PyTorch model (for auto-conversion if engine not found)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='0',
        help='Input source: camera index (0, 1, 2...) or video file path (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path (optional, leave empty to not save)'
    )
    parser.add_argument(
        '--upscale',
        type=int,
        default=2,
        choices=[2, 3, 4],
        help='Upscale factor: 2, 3, or 4 (default: 2)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target FPS (default: 30)'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Disable preview window (faster processing)'
    )
    parser.add_argument(
        '--no-auto-convert',
        action='store_true',
        help='Disable automatic model conversion'
    )
    
    args = parser.parse_args()
    
    # Validate engine file
    if not os.path.exists(args.engine):
        if not args.no_auto_convert:
            if not ensure_engine_exists(args.engine, args.pytorch, auto_convert=True):
                print(f"\n✗ ERROR: Engine file not found: {args.engine}")
                print("\nPlease:")
                print("  1. Convert the model to TensorRT first, or")
                print("  2. Provide --pytorch path for auto-conversion, or")
                print("  3. Use --no-auto-convert to disable auto-conversion")
                sys.exit(1)
        else:
            print(f"\n✗ ERROR: Engine file not found: {args.engine}")
            print("Please convert the model to TensorRT first.")
            sys.exit(1)
    
    # Convert input source
    try:
        input_source = int(args.input)
        input_type = "camera"
    except ValueError:
        input_source = args.input
        input_type = "video file"
        if not os.path.exists(input_source):
            print(f"\n✗ ERROR: Input file not found: {input_source}")
            sys.exit(1)
    
    print("=" * 60)
    print("FSRCNN Real-time Video Processor")
    print("=" * 60)
    print(f"Engine: {args.engine}")
    print(f"Input: {input_source} ({input_type})")
    print(f"Upscale: {args.upscale}x")
    print(f"Target FPS: {args.fps}")
    if args.output:
        print(f"Output: {args.output}")
    print("=" * 60)
    
    # Initialize processor
    try:
        processor = RealtimeVideoProcessor(
            engine_path=args.engine,
            upscale_factor=args.upscale,
            target_fps=args.fps
        )
    except Exception as e:
        print(f"\n✗ ERROR: Failed to initialize processor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Process video stream
    try:
        processor.process_video_stream(
            input_source=input_source,
            output_path=args.output,
            show_preview=not args.no_preview
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


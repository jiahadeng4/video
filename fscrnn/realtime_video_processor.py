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


if __name__ == "__main__":
    processor = RealtimeVideoProcessor(
        engine_path="models/fsrcnn_x2_fp16.engine",
        upscale_factor=2,
        target_fps=30
    )
    
    # Process camera input (camera index 0)
    processor.process_video_stream(
        input_source=0,
        output_path=None,  # Don't save file
        show_preview=True
    )
    
    # Or process video file
    # processor.process_video_stream(
    #     input_source="input_video.mp4",
    #     output_path="output_video.mp4",
    #     show_preview=True
    # )


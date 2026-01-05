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
"""Performance optimization configuration for low-latency scenarios."""
# TensorRT engine configuration
TENSORRT_CONFIG = {
    "fp16_mode": True,           # Use FP16 precision (2x speedup)
    "int8_mode": False,          # INT8 precision (requires calibration, faster but accuracy loss)
    "max_workspace_size": 1 << 30,  # 1GB workspace
    "max_batch_size": 1,         # Single frame processing
}

# Input/Output configuration
INPUT_CONFIG = {
    "max_width": 640,            # Maximum input width
    "max_height": 480,           # Maximum input height
    "resize_method": "linear",   # Downsampling method
}

# Processing configuration
PROCESSING_CONFIG = {
    "target_fps": 30,            # Target frame rate
    "skip_frames": 0,            # Number of frames to skip (0 = no skipping)
    "async_mode": False,         # Asynchronous mode (requires multi-threading)
}

# Memory optimization
MEMORY_CONFIG = {
    "preallocate_buffers": True,  # Pre-allocate buffers
    "clear_cache_interval": 100,  # Clear cache every N frames
}

# Combined optimization configuration
OPTIMIZATION_CONFIG = {
    "tensorrt": TENSORRT_CONFIG,
    "input": INPUT_CONFIG,
    "processing": PROCESSING_CONFIG,
    "memory": MEMORY_CONFIG,
}


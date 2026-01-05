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
"""TensorRT inference engine for FSRCNN model."""
import os

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

import imgproc


class FSRCNNTensorRTInference:
    """FSRCNN TensorRT inference engine for low-latency video processing."""
    
    def __init__(self, engine_path: str, upscale_factor: int = 2):
        """
        Initialize TensorRT inference engine.
        
        Args:
            engine_path: Path to TensorRT engine file
            upscale_factor: Super-resolution upscale factor
        """
        self.upscale_factor = upscale_factor
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
    def load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """
        Load TensorRT engine from file.
        
        Args:
            engine_path: Path to TensorRT engine file
        
        Returns:
            TensorRT engine object
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            return runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self):
        """
        Allocate GPU memory buffers for input and output.
        
        Returns:
            Tuple of (inputs, outputs, bindings, stream)
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def infer(self, input_image: np.ndarray) -> np.ndarray:
        """
        Perform inference on input image.
        
        Args:
            input_image: Input image (numpy array, BGR format, 0-255 range)
        
        Returns:
            Super-resolved image (numpy array, BGR format, 0-255 range)
        """
        # Convert to YCbCr and extract Y channel
        ycbcr_image = imgproc.bgr2ycbcr(input_image.astype(np.float32) / 255.0)
        y_channel = ycbcr_image[:, :, 0]
        cb_channel = ycbcr_image[:, :, 1]
        cr_channel = ycbcr_image[:, :, 2]
        
        # Prepare input data (add batch and channel dimensions)
        input_data = y_channel.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)  # Add channel dimension
        
        # Copy input data to GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output data back to CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # Get output
        output_shape = self.context.get_binding_shape(1)
        output_data = self.outputs[0]['host'][:np.prod(output_shape)].reshape(output_shape)
        output_y = np.squeeze(output_data)
        output_y = np.clip(output_y, 0, 1.0)
        
        # Upsample Cb and Cr channels using bicubic interpolation
        h, w = output_y.shape
        cb_upsampled = cv2.resize(cb_channel, (w, h), interpolation=cv2.INTER_CUBIC)
        cr_upsampled = cv2.resize(cr_channel, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Merge YCbCr channels and convert back to BGR
        ycbcr_output = np.stack([output_y, cb_upsampled, cr_upsampled], axis=2)
        bgr_output = imgproc.ycbcr2bgr(ycbcr_output)
        bgr_output = np.clip(bgr_output * 255.0, 0, 255).astype(np.uint8)
        
        return bgr_output


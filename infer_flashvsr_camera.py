#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 FlashVSR 实时处理 USB 摄像头画面
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from einops import rearrange
import argparse
from collections import deque
import time

from diffsynth import ModelManager, FlashVSRTinyPipeline
from utils.utils import Buffer_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder


def tensor2video(frames: torch.Tensor):
    """将张量转换为视频帧列表"""
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return frames


def largest_8n1_leq(n):  # 8n+1
    """计算最大的8n+1形式的值"""
    return 0 if n < 1 else ((n - 1)//8)*8 + 1


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    """将PIL图像转换为[-1,1]范围的张量"""
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0  # CHW in [-1,1]
    return t.to(dtype)


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    """计算缩放和目标尺寸"""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    """放大并中心裁剪图像"""
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def init_pipeline():
    """初始化 FlashVSR 管道"""
    print(f"GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([
        "./FlashVSR/diffusion_pytorch_model_streaming_dmd.safetensors",
    ])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    LQ_proj_in_path = "./FlashVSR/LQ_proj_in.ckpt"
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)

    pipe.denoising_model().LQ_proj_in.to('cuda')

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
    mis = pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR/TCDecoder.ckpt"), strict=False)
    print(f"TCDecoder loading: {mis}")

    pipe.to('cuda')
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit","vae"])
    print("Pipeline initialized successfully!")
    return pipe


def process_frame_batch(pipe, frame_batch, scale, tH, tW, F, sparse_ratio=2.0, seed=0):
    """处理一批帧"""
    # 准备输入张量
    frames_tensor = []
    for frame in frame_batch:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
        frames_tensor.append(pil_to_tensor_neg1_1(img_out, dtype=torch.bfloat16, device='cuda'))
    
    LQ = torch.stack(frames_tensor, 0).permute(1,0,2,3).unsqueeze(0)  # 1 C F H W
    
    # 推理
    with torch.no_grad():
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
            LQ_video=LQ, num_frames=F, height=tH, width=tW, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(tH*tW), 
            kv_ratio=3.0,
            local_range=11,
            color_fix=True,
        )
    
    # 转换为numpy数组
    video_frames = tensor2video(video)  # T H W C
    return video_frames


def main():
    parser = argparse.ArgumentParser(description='使用 FlashVSR 处理 USB 摄像头画面')
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID (默认: 0)')
    parser.add_argument('--width', type=int, default=640, help='摄像头宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480, help='摄像头高度 (默认: 480)')
    parser.add_argument('--scale', type=float, default=4.0, help='超分辨率倍数 (默认: 4.0)')
    parser.add_argument('--sparse_ratio', type=float, default=1.5, help='稀疏比率 (默认: 1.5, 推荐: 1.5-2.0)')
    parser.add_argument('--save', action='store_true', help='保存处理后的视频')
    parser.add_argument('--output', type=str, default='./results/camera_output.mp4', help='输出视频路径')
    parser.add_argument('--display', action='store_true', default=True, help='显示处理结果 (默认: True)')
    parser.add_argument('--batch_size', type=int, default=9, help='批处理帧数 (默认: 9, 必须是8n+1)')
    
    args = parser.parse_args()
    
    # 确保 batch_size 是 8n+1 格式
    F = largest_8n1_leq(args.batch_size)
    if F == 0:
        F = 9  # 默认值
        print(f"警告: batch_size 无效，使用默认值 {F}")
    else:
        F = args.batch_size if args.batch_size == F else F
        print(f"使用批处理大小: {F} (8n+1格式)")
    
    # 初始化摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {args.camera}")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"摄像头分辨率: {actual_width}x{actual_height}, FPS: {fps}")
    
    # 计算目标尺寸
    sW, sH, tW, tH = compute_scaled_and_target_dims(actual_width, actual_height, scale=args.scale, multiple=128)
    print(f"缩放分辨率 (x{args.scale}): {sW}x{sH} -> 目标分辨率 (128倍数): {tW}x{tH}")
    
    # 初始化管道
    print("正在初始化 FlashVSR 管道...")
    pipe = init_pipeline()
    
    # 帧缓冲区
    frame_buffer = deque(maxlen=F)
    output_frames = []
    last_output_frame = None  # 用于保存最后一帧输出
    
    # 视频写入器
    video_writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (tW, tH))
        print(f"将保存视频到: {args.output}")
    
    print("\n开始处理...")
    print("按 'q' 键退出, 按 's' 键保存当前帧")
    
    frame_count = 0
    process_count = 0
    last_process_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            frame_count += 1
            frame_buffer.append(frame.copy())
            
            # 当缓冲区有足够帧时进行处理
            if len(frame_buffer) == F:
                process_start = time.time()
                
                # 处理帧批次
                processed_frames = process_frame_batch(
                    pipe, list(frame_buffer), args.scale, tH, tW, F, 
                    sparse_ratio=args.sparse_ratio, seed=0
                )
                
                process_time = time.time() - process_start
                process_count += 1
                fps_actual = 1.0 / process_time if process_time > 0 else 0
                
                print(f"处理批次 {process_count}: {process_time:.2f}s ({fps_actual:.2f} FPS)")
                
                # 保存处理后的帧（取中间帧作为输出，避免边界效应）
                # FlashVSR 输出 8n-3 帧（因为输入是 8n+1，填充了4帧）
                valid_frames = len(processed_frames) - 4  # 减去填充的帧
                if valid_frames > 0:
                    mid_idx = valid_frames // 2
                    output_frame = processed_frames[mid_idx]  # H W C
                    output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    last_output_frame = output_frame_bgr
                    
                    if args.save and video_writer:
                        video_writer.write(output_frame_bgr)
                    
                    output_frames.append(output_frame_bgr)
                
                # 清空缓冲区，重新开始收集帧（简单模式）
                # 注意：这会导致一些延迟，但实现更简单
                frame_buffer.clear()
                
            # 显示结果（如果有输出帧）
            if args.display and last_output_frame is not None:
                # 调整原始帧大小以便显示
                orig_display = cv2.resize(frame, (actual_width, actual_height))
                # 调整处理后的帧大小以便显示（如果太大）
                if tW > 1920 or tH > 1080:
                    display_scale = min(1920/tW, 1080/tH)
                    display_w = int(tW * display_scale)
                    display_h = int(tH * display_scale)
                    proc_display = cv2.resize(last_output_frame, (display_w, display_h))
                else:
                    proc_display = last_output_frame
                
                # 并排显示
                if orig_display.shape[0] != proc_display.shape[0]:
                    # 调整高度一致
                    h = min(orig_display.shape[0], proc_display.shape[0])
                    orig_display = cv2.resize(orig_display, (int(orig_display.shape[1]*h/orig_display.shape[0]), h))
                    proc_display = cv2.resize(proc_display, (int(proc_display.shape[1]*h/proc_display.shape[0]), h))
                
                combined = np.hstack([orig_display, proc_display])
                
                # 添加文字信息
                fps_display = fps_actual if process_count > 0 else 0
                info_text = f"Original | FlashVSR x{args.scale} | Batch: {process_count} | FPS: {fps_display:.1f}"
                cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('FlashVSR Camera Processing', combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出")
                    break
                elif key == ord('s') and last_output_frame is not None:
                    save_path = f"./results/camera_frame_{frame_count}.jpg"
                    cv2.imwrite(save_path, last_output_frame)
                    print(f"已保存帧到: {save_path}")
            
            # 清空显存缓存（定期）
            if frame_count % 10 == 0:
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        print(f"\n处理完成! 共处理 {process_count} 个批次, {frame_count} 帧")
        if args.save:
            print(f"视频已保存到: {args.output}")


if __name__ == "__main__":
    main()


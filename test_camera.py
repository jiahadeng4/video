#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的 USB 摄像头测试脚本
用于检测摄像头是否可用并显示画面
"""

import cv2
import sys
import argparse


def test_camera(camera_id=0, width=640, height=480):
    """测试指定 ID 的摄像头"""
    print(f"正在测试摄像头 ID: {camera_id}")
    print(f"尝试打开 /dev/video{camera_id} (如果存在)...")
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开摄像头 {camera_id}")
        print(f"   请检查:")
        print(f"   1. 摄像头是否已连接")
        print(f"   2. 设备 ID 是否正确 (尝试 0, 1, 2...)")
        print(f"   3. 是否有其他程序占用摄像头")
        print(f"   4. 权限是否足够 (Linux: sudo usermod -a -G video $USER)")
        return False
    
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ 摄像头打开成功!")
    print(f"   分辨率: {actual_width}x{actual_height}")
    print(f"   帧率: {fps if fps > 0 else '未知'}")
    print(f"\n按 'q' 键退出测试")
    print(f"按 's' 键保存当前帧")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"❌ 无法读取画面 (帧 {frame_count})")
                break
            
            frame_count += 1
            
            # 在画面上显示信息
            info_text = f"Camera ID: {camera_id} | Frame: {frame_count} | {actual_width}x{actual_height}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示画面
            cv2.imshow(f'Camera Test - ID {camera_id}', frame)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户退出")
                break
            elif key == ord('s'):
                save_path = f"./results/test_camera_{camera_id}_frame_{frame_count}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"已保存帧到: {save_path}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n测试完成! 共读取 {frame_count} 帧")
    
    return True


def list_available_cameras(max_test=5):
    """列出所有可用的摄像头"""
    print("正在检测可用的摄像头...")
    print("-" * 50)
    
    available = []
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available.append({
                    'id': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                print(f"✅ 摄像头 ID {i}: {width}x{height} @ {fps} FPS")
            cap.release()
        else:
            print(f"❌ 摄像头 ID {i}: 不可用")
    
    print("-" * 50)
    
    if available:
        print(f"\n找到 {len(available)} 个可用摄像头:")
        for cam in available:
            print(f"  ID {cam['id']}: /dev/video{cam['id']} - {cam['width']}x{cam['height']}")
        return available
    else:
        print("\n未找到可用的摄像头")
        return []


def main():
    parser = argparse.ArgumentParser(description='测试 USB 摄像头')
    parser.add_argument('--camera', type=int, default=None, 
                       help='摄像头设备ID (默认: 自动检测)')
    parser.add_argument('--width', type=int, default=640, 
                       help='摄像头宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480, 
                       help='摄像头高度 (默认: 480)')
    parser.add_argument('--list', action='store_true', 
                       help='只列出可用摄像头，不进行测试')
    
    args = parser.parse_args()
    
    # 创建结果目录
    import os
    os.makedirs("./results", exist_ok=True)
    
    # 列出所有可用摄像头
    available_cameras = list_available_cameras()
    
    if args.list:
        # 只列出，不测试
        sys.exit(0)
    
    # 确定要测试的摄像头 ID
    if args.camera is not None:
        camera_id = args.camera
    elif available_cameras:
        camera_id = available_cameras[0]['id']
        print(f"\n使用第一个可用摄像头: ID {camera_id}")
    else:
        print("\n未找到可用摄像头，尝试默认 ID 0...")
        camera_id = 0
    
    # 测试摄像头
    print("\n" + "=" * 50)
    success = test_camera(camera_id, args.width, args.height)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()


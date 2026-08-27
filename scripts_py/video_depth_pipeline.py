import os
import sys
import time
import argparse
import numpy as np
import cv2
import yaml

# 自动确保 nvcc 在 PATH 中
if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from stereo_depth_pipeline_gpu import StereoCalibrator, GPUStereoDepthEngine, colorize_disparity, colorize_depth


def process_video(
    video_path,
    calib_path,
    output_path,
    model_type='ess',
    layout='side_by_side',
    max_frames=0,
    min_depth=0.3,
    max_depth=5.0
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")
        
    print(f"\n==================================================================")
    print(f"   启动视频深度全 GPU 流水线处理: [{model_type}]")
    print(f"   输入视频: {video_path}")
    print(f"   标定文件: {calib_path}")
    print(f"   排布模式: {layout}")
    print(f"==================================================================")

    # 1. 载入标定参数
    calibrator = None
    orig_w, orig_h = 1920, 1200
    fx, fy, cx, cy = 1304.75, 1304.75, 959.75, 643.68
    baseline = 0.06321
    
    if os.path.exists(calib_path):
        calibrator = StereoCalibrator(calib_path)
        orig_w, orig_h = calibrator.w, calibrator.h
        fx, fy, cx, cy = calibrator.fx, calibrator.fy, calibrator.cx, calibrator.cy
        baseline = calibrator.baseline
    else:
        print(f"[Warning] Calib file not found at {calib_path}, using default 1920x1200 pinhole parameters.")

    # 2. 初始化 All-CUDA 推理引擎
    engine = GPUStereoDepthEngine(
        model_type=model_type,
        orig_w=orig_w, orig_h=orig_h,
        fx=fx, baseline=baseline
    )

    # 3. 打开输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0
        
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video Info] Total Frames: {total_frames}, Input FPS: {fps_in:.1f}, Resolution: {in_w}x{in_h}")

    # 4. 配置输出分辨率
    if layout == 'side_by_side':
        out_w, out_h = 1920, 600
    elif layout == 'full':
        out_w, out_h = 3840, 1200
    elif layout == 'single':
        out_w, out_h = 1920, 1200
    elif layout == 'quad':
        out_w, out_h = 1920, 1200
    else:
        out_w, out_h = 1920, 600

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (out_w, out_h))
    if not writer.isOpened():
        # Fallback to avc1 or MJPG
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (out_w, out_h))

    frames_to_process = total_frames if max_frames <= 0 else min(total_frames, max_frames)
    print(f"[Processing] Will process {frames_to_process} frames -> Saving to: {output_path}\n")

    frame_idx = 0
    t_start = time.perf_counter()
    latencies = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        t0 = time.perf_counter()
        
        # 1. 切分左右目
        h, w = frame.shape[:2]
        if w == 4000:
            left_raw = frame[:, 160:2080]
            right_raw = frame[:, 2080:4000]
        elif w == 3840:
            left_raw = frame[:, 0:1920]
            right_raw = frame[:, 1920:3840]
        else:
            mid = w // 2
            left_raw = frame[:, :mid]
            right_raw = frame[:, mid:]

        # 2. 极线校正
        if calibrator is not None:
            left_rect, right_rect = calibrator.rectify(left_raw, right_raw)
        else:
            left_rect, right_rect = left_raw, right_raw

        # 3. All-CUDA 深度推理
        need_disp = (layout == 'quad')
        if need_disp:
            depth, disparity = engine.infer_depth_gpu(left_rect, right_rect, return_disparity=True)
        else:
            depth = engine.infer_depth_gpu(left_rect, right_rect, return_disparity=False)

        # 4. 色彩映射
        depth_vis = colorize_depth(depth, min_d=min_depth, max_d=max_depth)

        # 5. 排布画面
        if layout == 'side_by_side':
            l_small = cv2.resize(left_rect, (960, 600), interpolation=cv2.INTER_LINEAR)
            d_small = cv2.resize(depth_vis, (960, 600), interpolation=cv2.INTER_LINEAR)
            out_frame = np.hstack([l_small, d_small])
        elif layout == 'full':
            out_frame = np.hstack([left_rect, depth_vis])
        elif layout == 'single':
            out_frame = depth_vis
        elif layout == 'quad':
            disp_vis = colorize_disparity(disparity, vmin=0, vmax=128)
            top_row = np.hstack([cv2.resize(left_rect, (960, 600)), cv2.resize(right_rect, (960, 600))])
            bot_row = np.hstack([cv2.resize(disp_vis, (960, 600)), cv2.resize(depth_vis, (960, 600))])
            out_frame = np.vstack([top_row, bot_row])
        else:
            out_frame = np.hstack([cv2.resize(left_rect, (960, 600)), cv2.resize(depth_vis, (960, 600))])

        writer.write(out_frame)
        
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)

        if frame_idx % 10 == 0 or frame_idx == frames_to_process:
            avg_fps = 1000.0 / np.mean(latencies[-30:])
            elapsed = time.perf_counter() - t_start
            eta_sec = (frames_to_process - frame_idx) / (avg_fps + 1e-5)
            print(f"  Frame [{frame_idx:4d}/{frames_to_process:4d}] - Realtime Speed: {avg_fps:5.1f} FPS | Elapsed: {elapsed:5.1f}s | ETA: {eta_sec:4.1f}s")

    cap.release()
    writer.release()
    
    total_time = time.perf_counter() - t_start
    overall_fps = frame_idx / total_time if total_time > 0 else 0
    print("\n==================================================================")
    print(f"   视频深度流生成完成: [{output_path}]")
    print(f"   总处理帧数 : {frame_idx} 帧")
    print(f"   全流程总耗时 : {total_time:.2f} 秒")
    print(f"   端到端综合平均帧率: {overall_fps:.2f} FPS")
    print("==================================================================\n")


def main():
    parser = argparse.ArgumentParser(description='All-CUDA Stereo Video Depth Generation Pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to input stereo MP4 video')
    parser.add_argument('--model', type=str, default='ess', choices=['foundation', 'ess', 'light_ess'], help='Model backend')
    parser.add_argument('--calib', type=str, default=None, help='Path to camchain calibration YAML')
    parser.add_argument('--output', type=str, default=None, help='Output MP4 video path')
    parser.add_argument('--layout', type=str, default='side_by_side', choices=['side_by_side', 'full', 'single', 'quad'], help='Output layout')
    parser.add_argument('--max_frames', type=int, default=0, help='Max frames to process (0 for full video)')
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = args.calib
    if calib_path is None:
        calib_path = os.path.join(curr_dir, 'calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml')
        
    output_path = args.output
    if output_path is None:
        model_tag = args.model
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(curr_dir, f"output_results/depth_{base_name}_{model_tag}.mp4")

    process_video(
        video_path=args.video,
        calib_path=calib_path,
        output_path=output_path,
        model_type=args.model,
        layout=args.layout,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()

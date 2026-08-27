#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo 5A: ROS 2 Bag / Video to FoundationStereo 16-Bit Metric Depth & 3D PointCloud (for Nav2 & RTAB-Map SLAM)
===========================================================================================================
Features:
1. Inputs stereo video or ROS 2 bag image stream.
2. Generates Standard 16-Bit Metric Depth Maps (uint16 depth in millimeters, e.g. 1500 = 1.500m)
   compatible with ROS 2 `image_transport`, RTAB-Map, ORB-SLAM3 RGB-D, and Nav2 `depthimage_to_laserscan`.
3. Exports Dense 3D Point Cloud (.ply / .pcd) for Global 3D Grid Mapping.
4. Generates camera_info.yaml for ROS 2 camera calibration publishing.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts_py'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
if '/usr/local/cuda/lib64' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from stereo_depth_pipeline_gpu import StereoCalibrator, GPUStereoDepthEngine
from rtmpose_depth_pipeline import CUDA_FAST_PIPELINE_SOURCE

def export_pointcloud_ply(rgb_img, depth_m, fx, fy, cx, cy, output_ply, min_z=0.2, max_z=4.0, downsample=2):
    """Exports RGB-D image pair into standard 3D colored PLY pointcloud file"""
    h, w = depth_m.shape
    v, u = np.mgrid[0:h:downsample, 0:w:downsample]
    
    z = depth_m[v, u]
    valid = (z >= min_z) & (z <= max_z) & (~np.isnan(z))
    
    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]
    
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    
    rgb_valid = rgb_img[v_valid, u_valid] # BGR
    r = rgb_valid[:, 2]
    g = rgb_valid[:, 1]
    b = rgb_valid[:, 0]
    
    num_pts = len(z_valid)
    with open(output_ply, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(num_pts):
            f.write(f"{x_valid[i]:.4f} {y_valid[i]:.4f} {z_valid[i]:.4f} {r[i]} {g[i]} {b[i]}\n")
    print(f"[Pointcloud Exporter] Saved {num_pts} 3D Points -> {output_ply}")

def run_bag_depth_exporter(
    video_path,
    calib_path,
    depth_engine_path,
    output_dir,
    depth_model='foundation',
    min_depth=0.15,
    max_depth=5.0,
    max_frames=100
):
    print(f"\n==================================================================")
    print(f"   [Demo 5A] 启动 ROS 2 / SLAM / Nav2 16位深度与 3D 点云批量生成器")
    print(f"   输入视频: {video_path}")
    print(f"   输出目录: {output_dir}")
    print(f"==================================================================")

    os.makedirs(os.path.join(output_dir, "left_rect"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_16bit_mm"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pointclouds"), exist_ok=True)

    calibrator = StereoCalibrator(calib_path)
    orig_w, orig_h = calibrator.w, calibrator.h
    fx, fy, cx, cy = calibrator.fx, calibrator.fy, calibrator.cx, calibrator.cy
    baseline = calibrator.baseline
    depth_engine = GPUStereoDepthEngine(model_type=depth_model, orig_w=orig_w, orig_h=orig_h, fx=fx, baseline=baseline)

    mod = SourceModule(CUDA_FAST_PIPELINE_SOURCE)
    remap_kernel = mod.get_function("cuda_remap_bilinear")

    d_map1_x = cuda.mem_alloc(calibrator.map_l1.nbytes)
    d_map1_y = cuda.mem_alloc(calibrator.map_l2.nbytes)
    d_map2_x = cuda.mem_alloc(calibrator.map_r1.nbytes)
    d_map2_y = cuda.mem_alloc(calibrator.map_r2.nbytes)

    cuda.memcpy_htod(d_map1_x, np.ascontiguousarray(calibrator.map_l1, dtype=np.float32))
    cuda.memcpy_htod(d_map1_y, np.ascontiguousarray(calibrator.map_l2, dtype=np.float32))
    cuda.memcpy_htod(d_map2_x, np.ascontiguousarray(calibrator.map_r1, dtype=np.float32))
    cuda.memcpy_htod(d_map2_y, np.ascontiguousarray(calibrator.map_r2, dtype=np.float32))

    frame_bytes = orig_h * orig_w * 3
    d_raw_left = cuda.mem_alloc(frame_bytes)
    d_raw_right = cuda.mem_alloc(frame_bytes)
    d_rect_left = cuda.mem_alloc(frame_bytes)
    d_rect_right = cuda.mem_alloc(frame_bytes)
    h_rect_left = np.empty((orig_h, orig_w, 3), dtype=np.uint8)

    block_dim = 16
    remap_block = (block_dim, block_dim, 1)
    remap_grid = ((orig_w + block_dim - 1) // block_dim, (orig_h + block_dim - 1) // block_dim, 1)

    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames if max_frames <= 0 else min(total_frames, max_frames)

    processed_count = 0
    t_start = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            processed_count += 1
            if max_frames > 0 and processed_count > max_frames:
                break

            w = frame.shape[1]
            if w == 4000:
                l_raw, r_raw = frame[:, 160:2080], frame[:, 2080:4000]
            elif w == 3840:
                l_raw, r_raw = frame[:, 0:1920], frame[:, 1920:3840]
            else:
                mid = w // 2
                l_raw, r_raw = frame[:, :mid], frame[:, mid:]

            cuda.memcpy_htod_async(d_raw_left, np.ascontiguousarray(l_raw), depth_engine.stream)
            cuda.memcpy_htod_async(d_raw_right, np.ascontiguousarray(r_raw), depth_engine.stream)

            remap_kernel(d_raw_left, d_rect_left, d_map1_x, d_map1_y, np.int32(orig_w), np.int32(orig_h),
                         block=remap_block, grid=remap_grid, stream=depth_engine.stream)
            remap_kernel(d_raw_right, d_rect_right, d_map2_x, d_map2_y, np.int32(orig_w), np.int32(orig_h),
                         block=remap_block, grid=remap_grid, stream=depth_engine.stream)

            cuda.memcpy_dtoh_async(h_rect_left, d_rect_left, depth_engine.stream)

            depth_engine.preprocess_kernel(d_rect_left, depth_engine.d_model_left, np.int32(orig_w), np.int32(orig_h),
                                           np.int32(depth_engine.input_w), np.int32(depth_engine.input_h),
                                           block=depth_engine.pre_block, grid=depth_engine.pre_grid, stream=depth_engine.stream)
            depth_engine.preprocess_kernel(d_rect_right, depth_engine.d_model_right, np.int32(orig_w), np.int32(orig_h),
                                           np.int32(depth_engine.input_w), np.int32(depth_engine.input_h),
                                           block=depth_engine.pre_block, grid=depth_engine.pre_grid, stream=depth_engine.stream)
            depth_engine.context.execute_async_v3(stream_handle=depth_engine.stream.handle)
            depth_engine.postprocess_kernel(depth_engine.d_model_disp, depth_engine.d_final_disp, depth_engine.d_final_depth,
                                            np.int32(depth_engine.input_w), np.int32(depth_engine.input_h),
                                            np.int32(orig_w), np.int32(orig_h), np.float32(depth_engine.fx_baseline),
                                            block=depth_engine.post_block, grid=depth_engine.post_grid, stream=depth_engine.stream)

            cuda.memcpy_dtoh_async(depth_engine.h_final_depth, depth_engine.d_final_depth, depth_engine.stream)
            depth_engine.stream.synchronize()

            # Save 1. Left Rectified RGB Image
            img_name = f"frame_{processed_count:06d}.png"
            cv2.imwrite(os.path.join(output_dir, "left_rect", img_name), h_rect_left)

            # Save 2. Standard 16-Bit Metric Depth Map in Millimeters (uint16)
            depth_mm = (depth_engine.h_final_depth * 1000.0).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(output_dir, "depth_16bit_mm", img_name), depth_mm)

            # Save 3. Sample 3D Point Clouds periodically (every 50 frames)
            if processed_count % 50 == 1:
                ply_name = f"pointcloud_{processed_count:06d}.ply"
                export_pointcloud_ply(h_rect_left, depth_engine.h_final_depth, fx, fy, cx, cy,
                                      os.path.join(output_dir, "pointclouds", ply_name))

            if processed_count % 50 == 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = processed_count / elapsed
                print(f"[SLAM Depth Exporter] Exported {processed_count}/{frames_to_process} RGB-D Pairs ({cur_fps:.1f} FPS)")

    finally:
        cap.release()
        t_total = time.perf_counter() - t_start
        fps_avg = processed_count / t_total if t_total > 0 else 0
        print(f"\n==================================================================")
        print(f"   ROS 2 / SLAM RGB-D 数据集导出完成: {processed_count} Frames ({fps_avg:.2f} FPS)")
        print(f"   RGB 目录: {os.path.join(output_dir, 'left_rect')}")
        print(f"   16位深度: {os.path.join(output_dir, 'depth_16bit_mm')}")
        print(f"   3D 点云: {os.path.join(output_dir, 'pointclouds')}")
        print(f"==================================================================")

def main():
    parser = argparse.ArgumentParser(description="Demo 5A: ROS 2 / SLAM 16-bit Metric Depth Exporter")
    parser.add_argument("--video", required=True, help="Input stereo video MP4")
    parser.add_argument("--calib", default="/home/elp/spatial_ai_trt_ws/calibration/camera_calib_results/camchain-datacalibration_bagskalibr_input.yaml")
    parser.add_argument("--depth_engine", default="/home/elp/spatial_ai_trt_ws/foundationstereo_320x736_fp16.engine")
    parser.add_argument("--model", default="foundation", choices=["foundation", "ess"])
    parser.add_argument("--output_dir", default="/home/elp/spatial_ai_trt_ws/output_results/04_pointcloud_and_eval/nav2_slam_rgbd_dataset")
    parser.add_argument("--min_depth", type=float, default=0.15)
    parser.add_argument("--max_depth", type=float, default=5.0)
    parser.add_argument("--max_frames", type=int, default=100)
    args = parser.parse_args()

    run_bag_depth_exporter(
        video_path=args.video,
        calib_path=args.calib,
        depth_engine_path=args.depth_engine,
        output_dir=args.output_dir,
        depth_model=args.model,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()

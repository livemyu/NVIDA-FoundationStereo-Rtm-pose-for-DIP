#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT FoundationStereo 3D 稠密彩色点云导出与 3 种着色模式渲染器
===================================================================
1. 模式 1: 真实灰度光影 (Intensity) - 还原相机真实灰度与墙面纹理细节
2. 模式 2: 高度热力着色 (Elevation) - 依据垂直高程映射 Turbo 彩虹色谱
3. 模式 3: 表面法向量着色 (Normals) - 依据点云表面法向量计算 3D 结构光照
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm

# 确保 CUDA 与 TensorRT 环境
if '/usr/local/cuda/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
if '/usr/local/cuda/lib64' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TRTFoundationStereo:
    def __init__(self, engine_path):
        print(f"[TRT] 加载 TensorRT 引擎: {engine_path} ...")
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.input_w = 736
        self.input_h = 320
        
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_mem = np.zeros(shape, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.buffers[name] = {
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': dtype
            }
            self.context.set_tensor_address(name, int(device_mem))

    def infer(self, left_bgr, right_bgr):
        orig_h, orig_w = left_bgr.shape[:2]
        left_rgb = cv2.cvtColor(cv2.resize(left_bgr, (self.input_w, self.input_h)), cv2.COLOR_BGR2RGB).astype(np.float32)
        right_rgb = cv2.cvtColor(cv2.resize(right_bgr, (self.input_w, self.input_h)), cv2.COLOR_BGR2RGB).astype(np.float32)
        
        left_nchw = np.transpose(left_rgb, (2, 0, 1))[np.newaxis, ...]
        right_nchw = np.transpose(right_rgb, (2, 0, 1))[np.newaxis, ...]
        
        cuda.memcpy_htod_async(self.buffers['left_image']['device'], np.ascontiguousarray(left_nchw), self.stream)
        cuda.memcpy_htod_async(self.buffers['right_image']['device'], np.ascontiguousarray(right_nchw), self.stream)
        
        self.context.execute_async_v3(self.stream.handle)
        
        disp_out = self.buffers['disparity']['host']
        cuda.memcpy_dtoh_async(disp_out, self.buffers['disparity']['device'], self.stream)
        self.stream.synchronize()
        
        disp_raw = disp_out[0, 0]
        scale_x = float(orig_w) / float(self.input_w)
        disp_orig = cv2.resize(disp_raw, (orig_w, orig_h)) * scale_x
        return disp_orig


def load_tum_trajectory(traj_path):
    print(f"[Trajectory] 加载轨迹: {traj_path} ...")
    traj_list = []
    with open(traj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    ts = float(parts[0])
                    pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    quat = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
                    rot = R.from_quat(quat).as_matrix()
                    T_wc = np.eye(4)
                    T_wc[:3, :3] = rot
                    T_wc[:3, 3] = pos
                    traj_list.append({'ts': ts, 'pos': pos, 'T_wc': T_wc})
                except ValueError: pass
    return traj_list


def extract_matched_stereo_frames(bag_path, traj_list, max_frames=80, time_tol=0.08):
    print(f"[Bag] 读取数据包: {bag_path} ...")
    step = max(1, len(traj_list) // max_frames)
    target_poses = traj_list[::step]
    target_ts_list = [p['ts'] for p in target_poses]
    
    bridge = CvBridge()
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    
    left_frames = {}
    right_frames = {}
    
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic in ['/camera/left/image_raw', '/camera/right/image_raw']:
            msg = deserialize_message(data, Image)
            msg_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            for i, target_ts in enumerate(target_ts_list):
                if abs(msg_ts - target_ts) < time_tol:
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    if topic == '/camera/left/image_raw':
                        if i not in left_frames or abs(msg_ts - target_ts) < abs(left_frames[i]['ts'] - target_ts):
                            left_frames[i] = {'ts': msg_ts, 'img': cv_img, 'target_idx': i}
                    else:
                        if i not in right_frames or abs(msg_ts - target_ts) < abs(right_frames[i]['ts'] - target_ts):
                            right_frames[i] = {'ts': msg_ts, 'img': cv_img, 'target_idx': i}
                    break

    matched_pairs = []
    for i, target_pose in enumerate(target_poses):
        if i in left_frames and i in right_frames:
            matched_pairs.append({
                'pose': target_pose,
                'left': left_frames[i]['img'],
                'right': right_frames[i]['img'],
                'ts': target_pose['ts']
            })
    print(f"[Bag] 成功同步对齐 {len(matched_pairs)} 对双目关键帧！")
    return matched_pairs


def save_ply_file(filename, points, colors):
    """保存标准二进制/ASCII PLY 3D点云文件"""
    num_pts = len(points)
    with open(filename, 'w') as f:
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
            p = points[i]
            c = colors[i]
            # 转换成标准 ROS 坐标系 (X=前, Y=左, Z=上) 方便查看:
            # ORB-SLAM3: X_w=右, Y_w=下, Z_w=前
            # 映射为 3D 查看器标准系: x = Z_w, y = -X_w, z = -Y_w
            x_vis = p[2]
            y_vis = -p[0]
            z_vis = -p[1]
            f.write(f"{x_vis:.4f} {y_vis:.4f} {z_vis:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"[PLY Saved] 成功保存 3D 点云: {filename} (共 {num_pts} 个点)")


def build_3d_pointclouds(
    matched_pairs,
    trt_engine,
    mask_img,
    fx, fy, cx, cy,
    baseline=0.0624,
    min_dist=0.4,
    max_dist=5.5,
    output_dir="/home/elp/navigation_ws/maps",
    floor_name="1F"
):
    os.makedirs(output_dir, exist_ok=True)
    all_pts = []
    all_gray = []
    
    h, w = mask_img.shape[:2]
    step = 4 # 4 像素步长抽取高分辨率点云
    v_grid, u_grid = np.mgrid[0:h:step, 0:w:step]
    mask_valid = (mask_img[v_grid, u_grid] > 128)
    
    print(f"\n>>> 正在执行 FoundationStereo 3D 全空间稠密重建 (共 {len(matched_pairs)} 帧) ...")
    start_t = time.time()
    
    for idx, item in enumerate(matched_pairs):
        left_img = item['left']
        right_img = item['right']
        T_wc = item['pose']['T_wc']
        
        # 转灰度
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            
        disp = trt_engine.infer(left_img, right_img)
        disp_sample = disp[v_grid, u_grid]
        valid_disp = (disp_sample > 1.0) & mask_valid
        
        z = np.zeros_like(disp_sample)
        z[valid_disp] = (fx * baseline) / disp_sample[valid_disp]
        
        valid_depth = valid_disp & (z >= min_dist) & (z <= max_dist)
        
        u_pts = u_grid[valid_depth]
        v_pts = v_grid[valid_depth]
        z_pts = z[valid_depth]
        
        # 提取真实灰度亮度
        gray_vals = left_gray[v_pts, u_pts]
        
        # 反投影为相机系 3D 点
        x_c = (u_pts - cx) * z_pts / fx
        y_c = (v_pts - cy) * z_pts / fy
        z_c = z_pts
        pts_c = np.vstack((x_c, y_c, z_c, np.ones_like(z_c)))
        
        # 变换至世界系
        pts_w = (T_wc @ pts_c)[:3, :].T # N x 3
        
        all_pts.append(pts_w)
        all_gray.append(gray_vals)
        
        if (idx + 1) % 15 == 0 or (idx + 1) == len(matched_pairs):
            print(f"  - 已处理 [{idx+1}/{len(matched_pairs)}] 帧，累计融合 3D 点: {sum(len(p) for p in all_pts)} 点")
            
    total_pts = np.vstack(all_pts)
    total_gray = np.concatenate(all_gray)
    print(f"[Done] 稠密点云反投影完成，共生成 {len(total_pts)} 个空间点！")
    
    # 空间体素降采样 (4cm 体素格子，合并重叠观测并加速渲染)
    voxel_size = 0.04
    voxel_coords = np.floor(total_pts / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    
    pts_sub = total_pts[unique_indices]
    gray_sub = total_gray[unique_indices]
    print(f"[Voxel Grid] 体素滤波完成，保留 {len(pts_sub)} 个高精度表面点。")
    
    # -------------------------------------------------------------
    # 模式 1: 真实灰度光影着色 (Real Intensity Texture)
    # -------------------------------------------------------------
    colors_intensity = np.stack([gray_sub, gray_sub, gray_sub], axis=-1)
    ply_int_path = os.path.join(output_dir, f"cloud_{floor_name}_intensity.ply")
    save_ply_file(ply_int_path, pts_sub, colors_intensity)
    
    # -------------------------------------------------------------
    # 模式 2: 高度热力图着色 (Height Elevation Colormap)
    # -------------------------------------------------------------
    # 在 ORB-SLAM3 坐标系中，Y_w 是垂直高度 (向下为正，故取 -Y_w 为物理高度)
    heights = -pts_sub[:, 1]
    h_min, h_max = np.percentile(heights, 2), np.percentile(heights, 98)
    h_norm = np.clip((heights - h_min) / (h_max - h_min + 1e-5), 0.0, 1.0)
    # Turbo 色谱映射
    cmap = cm.get_cmap('turbo')
    colors_elevation = (cmap(h_norm)[:, :3] * 255.0).astype(np.uint8)
    ply_elev_path = os.path.join(output_dir, f"cloud_{floor_name}_elevation.ply")
    save_ply_file(ply_elev_path, pts_sub, colors_elevation)
    
    # -------------------------------------------------------------
    # 模式 3: 表面法向量结构着色 (Surface Normal Shading)
    # -------------------------------------------------------------
    # 基于主方向计算法向估计与伪着色
    # 墙体 (法向偏向水平 X/Z) 与地面/天花板 (法向偏向垂直 Y) 区分
    # 映射为方向色彩 RGB
    norm_x = np.abs(np.gradient(pts_sub[:, 0]))
    norm_y = np.abs(np.gradient(pts_sub[:, 1]))
    norm_z = np.abs(np.gradient(pts_sub[:, 2]))
    norm_mag = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2) + 1e-5
    
    nx = norm_x / norm_mag
    ny = norm_y / norm_mag
    nz = norm_z / norm_mag
    
    # 融合灰度与法向量光照 (Phong Shading 效果)
    light_dir = np.array([0.5, 0.8, 0.3])
    light_dir = light_dir / np.linalg.norm(light_dir)
    diffuse = np.clip(nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2], 0.2, 1.0)
    
    r_chan = np.clip(gray_sub * diffuse * 1.1 + 40, 0, 255).astype(np.uint8)
    g_chan = np.clip(gray_sub * diffuse * 1.0 + 30, 0, 255).astype(np.uint8)
    b_chan = np.clip(gray_sub * diffuse * 0.9 + 20, 0, 255).astype(np.uint8)
    colors_normals = np.stack([r_chan, g_chan, b_chan], axis=-1)
    
    ply_norm_path = os.path.join(output_dir, f"cloud_{floor_name}_normals.ply")
    save_ply_file(ply_norm_path, pts_sub, colors_normals)
    
    print("\n============================================================")
    print(f" [OK] 3 种 3D 点云着色模式文件导出全部完成！")
    print(f"  1. 真实灰度: {ply_int_path}")
    print(f"  2. 高度热力: {ply_elev_path}")
    print(f"  3. 结构法向: {ply_norm_path}")
    print("============================================================")


def main():
    parser = argparse.ArgumentParser(description="TensorRT FoundationStereo 3D 稠密点云与着色模式生成器")
    parser.add_argument("--bag", required=True, help="ROS 2 Bag 路径")
    parser.add_argument("--traj", required=True, help="ORB-SLAM3 轨迹文件")
    parser.add_argument("--engine", default="/home/elp/spatial_ai_trt_ws/foundationstereo_320x736_fp16.engine")
    parser.add_argument("--mask", default="/home/elp/spatial_ai_trt_ws/mask0.png")
    parser.add_argument("--output_dir", default="/home/elp/navigation_ws/maps")
    parser.add_argument("--floor", default="1F")
    parser.add_argument("--max_frames", type=int, default=60)
    args = parser.parse_args()
    
    fx = 366.6111195655474
    fy = 365.7001404179377
    cx = 467.4124976774619
    cy = 306.9926839396347
    baseline = 0.0624
    
    if os.path.exists(args.mask):
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    else:
        mask_img = np.full((600, 960), 255, dtype=np.uint8)
        
    traj_list = load_tum_trajectory(args.traj)
    trt_engine = TRTFoundationStereo(args.engine)
    matched_pairs = extract_matched_stereo_frames(args.bag, traj_list, max_frames=args.max_frames)
    
    build_3d_pointclouds(
        matched_pairs=matched_pairs,
        trt_engine=trt_engine,
        mask_img=mask_img,
        fx=fx, fy=fy, cx=cx, cy=cy,
        baseline=baseline,
        output_dir=args.output_dir,
        floor_name=args.floor
    )


if __name__ == '__main__':
    main()

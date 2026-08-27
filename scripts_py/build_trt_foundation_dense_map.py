#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT FoundationStereo 稠密 2D 占据栅格地图重建引擎 (带全视线 Raycasting 自由空间雕刻)
========================================================================================
1. 红色/黑色 (0): 障碍物与墙壁 (Occupied Obstacles)
2. 白色 (254): 自由通行区域 (Free Navigable Space, 由相机光线投射 Raycast 完整覆盖)
3. 灰色 (205): 未探索未知区域 (Unknown Space)
4. 坐标系:
   - X_world: 水平横向 (米)
   - Z_world: 水平纵向 (米)
   - Y_world: 垂直高度轴 (切片过滤地面与天花板)
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

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
        print(f"[TRT] 正在加载 TensorRT 引擎: {engine_path} ...")
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.input_w = 736
        self.input_h = 320
        
        # 分配显存
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
        print(f"[TRT] 引擎初始化就绪 (输入尺寸: {self.input_w}x{self.input_h})！")

    def infer(self, left_bgr, right_bgr):
        orig_h, orig_w = left_bgr.shape[:2]
        
        # 缩放至 (320, 736) RGB
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
        
        disp_raw = disp_out[0, 0] # (320, 736)
        scale_x = float(orig_w) / float(self.input_w)
        disp_orig = cv2.resize(disp_raw, (orig_w, orig_h)) * scale_x
        return disp_orig


def load_tum_trajectory(traj_path):
    print(f"[Trajectory] 正在加载轨迹文件: {traj_path} ...")
    traj_list = []
    with open(traj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    ts = float(parts[0])
                    pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    quat = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]) # qx, qy, qz, qw
                    rot = R.from_quat(quat).as_matrix()
                    
                    T_wc = np.eye(4)
                    T_wc[:3, :3] = rot
                    T_wc[:3, 3] = pos
                    
                    traj_list.append({
                        'ts': ts,
                        'pos': pos,
                        'quat': quat,
                        'T_wc': T_wc
                    })
                except ValueError:
                    continue
    print(f"[Trajectory] 成功加载 {len(traj_list)} 个位姿节点 (时间跨度: {traj_list[-1]['ts'] - traj_list[0]['ts']:.2f} s)。")
    return traj_list


def extract_matched_stereo_frames(bag_path, traj_list, max_frames=100, time_tol=0.08):
    print(f"[Bag] 正在读取数据包: {bag_path} ...")
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
    print(f"[Bag] 成功完成 {len(matched_pairs)} 对双目关键帧与 6-DoF 位姿的时间同步对齐！")
    return matched_pairs


def build_dense_map_topdown(
    matched_pairs,
    trt_engine,
    mask_img,
    fx, fy, cx, cy,
    baseline=0.0624,
    grid_res=0.05,
    min_dist=0.35,
    max_dist=5.5,
    output_dir="/home/elp/navigation_ws/maps",
    map_name="map_1F"
):
    """
    执行俯视俯瞰 2D 占据栅格地图重建 (Bird's Eye View Top-Down)
    世界坐标系映射:
      - X_world: 地图水平 X 坐标 (米)
      - Z_world: 地图水平 Y 坐标 (米)
      - Y_world: 垂直高度轴 (向下为正, 过滤地面与天花板)
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_obstacles = [] # list of (cam_pos_2d, obs_pts_2d)
    all_obstacle_pts = []
    all_cam_positions = []
    
    h, w = mask_img.shape[:2]
    step = 4
    v_grid, u_grid = np.mgrid[0:h:step, 0:w:step]
    mask_valid = (mask_img[v_grid, u_grid] > 128)
    
    print(f"\n>>> 正在执行 FoundationStereo 俯视稠密深度推理与点云提取 (共 {len(matched_pairs)} 帧) ...")
    start_t = time.time()
    
    for idx, item in enumerate(matched_pairs):
        left_img = item['left']
        right_img = item['right']
        T_wc = item['pose']['T_wc']
        cam_pos = T_wc[:3, 3]
        cam_pos_2d = np.array([cam_pos[0], cam_pos[2]])
        all_cam_positions.append(cam_pos_2d)
        
        # 1. 深度推理
        disp = trt_engine.infer(left_img, right_img)
        disp_sample = disp[v_grid, u_grid]
        valid_disp = (disp_sample > 1.0) & mask_valid
        
        z = np.zeros_like(disp_sample)
        z[valid_disp] = (fx * baseline) / disp_sample[valid_disp]
        
        valid_depth = valid_disp & (z >= min_dist) & (z <= max_dist)
        
        u_pts = u_grid[valid_depth]
        v_pts = v_grid[valid_depth]
        z_pts = z[valid_depth]
        
        # 2. 相机系 3D 点反投影
        x_c = (u_pts - cx) * z_pts / fx
        y_c = (v_pts - cy) * z_pts / fy
        z_c = z_pts
        
        pts_c = np.vstack((x_c, y_c, z_c, np.ones_like(z_c))) # 4 x N
        
        # 3. 转换至世界坐标系
        pts_w = T_wc @ pts_c # 4 x N
        
        # 4. 垂直高度切片 (Y_world 轴为高度方向)
        rel_y = pts_w[1, :] - cam_pos[1]
        obs_mask = (rel_y > -0.75) & (rel_y < 0.25)
        
        if np.any(obs_mask):
            obs_2d = np.vstack((pts_w[0, obs_mask], pts_w[2, obs_mask])).T # N x 2
            all_obstacle_pts.append(obs_2d)
            frame_obstacles.append((cam_pos_2d, obs_2d))
            
        if (idx + 1) % 15 == 0 or (idx + 1) == len(matched_pairs):
            print(f"  - 已处理 [{idx+1}/{len(matched_pairs)}] 帧，累计融合俯视障碍物点: {sum(len(p) for p in all_obstacle_pts)} 点")

    total_time = time.time() - start_t
    print(f"[Done] 俯视稠密重建推理完成，平均每帧耗时: {total_time/len(matched_pairs):.3f} s。")
    
    if not all_obstacle_pts:
        print("[Error] 未提取到有效障碍物点！")
        return
        
    obstacles = np.vstack(all_obstacle_pts)
    cam_traj = np.array(all_cam_positions)
    
    # 5. 构建俯视 2D 占据栅格地图
    print("\n>>> 正在执行 Raycasting 光线投射与自由空间雕刻 (Free Space Carving) ...")
    min_x = min(np.min(obstacles[:, 0]), np.min(cam_traj[:, 0])) - 1.5
    max_x = max(np.max(obstacles[:, 0]), np.max(cam_traj[:, 0])) + 1.5
    min_y = min(np.min(obstacles[:, 1]), np.min(cam_traj[:, 1])) - 1.5
    max_y = max(np.max(obstacles[:, 1]), np.max(cam_traj[:, 1])) + 1.5
    
    width_cells = int(np.ceil((max_x - min_x) / grid_res))
    height_cells = int(np.ceil((max_y - min_y) / grid_res))
    
    # 205 (未知空间), 254 (自由可行走空间), 0 (障碍物占据)
    grid_img = np.full((height_cells, width_cells), 205, dtype=np.uint8)
    
    # 执行 Raycast: 从相机点向每个障碍物点投射视线，将视线内部标记为自由空间 254
    for cam_pos_2d, obs_2d in frame_obstacles:
        gx_cam = int((cam_pos_2d[0] - min_x) / grid_res)
        gy_cam = int((cam_pos_2d[1] - min_y) / grid_res)
        
        # 降采样投射光线，加速大厅区域雕刻
        sub_obs = obs_2d[::3]
        for pt in sub_obs:
            gx_obs = int((pt[0] - min_x) / grid_res)
            gy_obs = int((pt[1] - min_y) / grid_res)
            if 0 <= gx_obs < width_cells and 0 <= gy_obs < height_cells:
                cv2.line(grid_img, (gx_cam, gy_cam), (gx_obs, gy_obs), 254, 1)
                
    # 强化轨迹邻域为 100% 自由区域
    for pos in cam_traj:
        gx = int((pos[0] - min_x) / grid_res)
        gy = int((pos[1] - min_y) / grid_res)
        if 0 <= gx < width_cells and 0 <= gy < height_cells:
            cv2.circle(grid_img, (gx, gy), int(0.60 / grid_res), 254, -1)
            
    # 覆盖标记所有真实障碍物点为 0 (黑色)
    for pt in obstacles:
        gx = int((pt[0] - min_x) / grid_res)
        gy = int((pt[1] - min_y) / grid_res)
        if 0 <= gx < width_cells and 0 <= gy < height_cells:
            grid_img[gy, gx] = 0
            
    # 满足 ROS 标准保存为 PGM (行 0 为图像顶部，故翻转存储)
    grid_img_flipped = cv2.flip(grid_img, 0)
    
    # 6. 保存 PGM 与 YAML
    pgm_path = os.path.join(output_dir, f"{map_name}.pgm")
    yaml_path = os.path.join(output_dir, f"{map_name}.yaml")
    
    cv2.imwrite(pgm_path, grid_img_flipped)
    print(f"[Save] 俯视 2D 占据栅格图像已保存: {pgm_path} (尺寸: {width_cells}x{height_cells})")
    
    yaml_content = {
        'image': f"{map_name}.pgm",
        'resolution': float(grid_res),
        'origin': [float(min_x), float(min_y), 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.25,
        'mode': 'trinary'
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"[Save] Nav2 地图描述文件已保存: {yaml_path}")
    print(f"============================================================")
    print(f" [OK] {map_name} 俯视鸟瞰导航地图 (全视线雕刻) 构建成功！")
    print(f"============================================================")


def main():
    parser = argparse.ArgumentParser(description="TensorRT FoundationStereo 俯视 2D 导航建图器")
    parser.add_argument("--bag", required=True, help="ROS 2 Bag 路径")
    parser.add_argument("--traj", required=True, help="ORB-SLAM3 轨迹文件路径")
    parser.add_argument("--engine", default="/home/elp/spatial_ai_trt_ws/foundationstereo_320x736_fp16.engine", help="TRT 模型路径")
    parser.add_argument("--mask", default="/home/elp/spatial_ai_trt_ws/mask0.png", help="mask0.png 路径")
    parser.add_argument("--output_dir", default="/home/elp/navigation_ws/maps", help="输出地图目录")
    parser.add_argument("--map_name", default="map_1F", help="地图名称")
    parser.add_argument("--max_frames", type=int, default=80, help="处理关键帧数量上限")
    args = parser.parse_args()
    
    fx = 366.6111195655474
    fy = 365.7001404179377
    cx = 467.4124976774619
    cy = 306.9926839396347
    baseline = 0.0624
    
    if os.path.exists(args.mask):
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        print(f"[Mask] 成功加载鱼眼边缘掩码: {args.mask} (尺寸: {mask_img.shape})")
    else:
        print("[Mask] 未找到 mask 图像，使用全白掩码。")
        mask_img = np.full((600, 960), 255, dtype=np.uint8)
        
    traj_list = load_tum_trajectory(args.traj)
    if not traj_list:
        print("[Error] 轨迹为空！")
        return
        
    trt_engine = TRTFoundationStereo(args.engine)
    matched_pairs = extract_matched_stereo_frames(args.bag, traj_list, max_frames=args.max_frames)
    if not matched_pairs:
        print("[Error] 未能在 Bag 中匹配到对应的双目图像对！")
        return
        
    build_dense_map_topdown(
        matched_pairs=matched_pairs,
        trt_engine=trt_engine,
        mask_img=mask_img,
        fx=fx, fy=fy, cx=cx, cy=cy,
        baseline=baseline,
        grid_res=0.05,
        output_dir=args.output_dir,
        map_name=args.map_name
    )


if __name__ == '__main__':
    main()

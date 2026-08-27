#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Precision 2D Occupancy Grid Map Builder for Dataset 2026-08-20
Direct SQLite-Indexed Keyframe Retrieval + FoundationStereo ONNX + Raycasting
"""

import os
import sys
import gc
import cv2
import shutil
import sqlite3
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def load_vins_trajectory(csv_path):
    print(f"[1/4] 正在加载 180° VIO 轨迹: {csv_path} ...")
    traj = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                try:
                    raw_ts = float(parts[0])
                    ts = raw_ts / 1e9 if raw_ts > 1e12 else raw_ts
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    traj.append({
                        'ts': ts,
                        'pos': np.array([tx, ty, tz]),
                        'quat': np.array([qx, qy, qz, qw])
                    })
                except ValueError:
                    continue
    print(f"      成功加载 {len(traj)} 个高精位姿 (起始: {traj[0]['ts']:.2f}s, 结束: {traj[-1]['ts']:.2f}s)")
    return traj

def build_dense_2d_map():
    workspace_dir = "/home/elp/picture_resize_recording_NVIDA"
    db_path = os.path.join(workspace_dir, "datasets/my_dataset_20260820_040247/my_dataset_20260820_040247_0.db3")
    csv_path = os.path.join(workspace_dir, "output/vio.csv")
    model_path = os.path.join(workspace_dir, "models_and_assets/foundationstereo_320x736.onnx")
    out_dir = os.path.join(workspace_dir, "maps")
    os.makedirs(out_dir, exist_ok=True)
    
    out_pgm = os.path.join(out_dir, "map_20260820.pgm")
    out_yaml = os.path.join(out_dir, "map_20260820.yaml")
    out_png = os.path.join(out_dir, "map_20260820_render.png")
    artifact_png = "/home/elp/.gemini/antigravity-ide/brain/abcbe3da-080c-4a47-a448-d28b563c2de4/map_20260820_render.png"
    
    traj = load_vins_trajectory(csv_path)
    traj_pts = np.array([p['pos'] for p in traj])
    
    print("[2/4] 正在初始化 FoundationStereo 深度神经网络引擎与 SQLite 索引...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in session.get_inputs()]
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id = 5')
    min_record_ts, max_record_ts = cursor.fetchone()
    
    vins_start_ts = traj[0]['ts']
    vins_end_ts = traj[-1]['ts']
    
    # 45 target keyframe poses evenly across trajectory (~2.4m/frame)
    num_keyframes = 45
    step = max(1, len(traj) // num_keyframes)
    target_poses = traj[::step]
    
    fx, fy, cx, cy = 366.61, 365.70, 467.41, 306.99
    baseline = 0.0624
    
    T_body_cam0 = np.array([
        [ 0.25091278,  0.00548910,  0.96799413, -0.10491511],
        [-0.96797581,  0.00979053,  0.25085251,  0.09482819],
        [-0.00810022, -0.99993701,  0.00776989,  0.00509889],
        [ 0.0,         0.0,         0.0,         1.0 ]
    ])
    
    # Map spatial bounds
    res = 0.03  # 3cm high-definition resolution
    margin = 2.0
    min_x, max_x = np.min(traj_pts[:, 0]) - margin - 3.5, np.max(traj_pts[:, 0]) + margin + 3.5
    min_y, max_y = np.min(traj_pts[:, 1]) - margin - 3.5, np.max(traj_pts[:, 1]) + margin + 3.5
    
    width = int(np.ceil((max_x - min_x) / res))
    height = int(np.ceil((max_y - min_y) / res))
    print(f"      地图尺寸: {width} x {height} 像素, 精度: {res*100:.1f} cm/格")
    print(f"      X: [{min_x:.2f}, {max_x:.2f}] m, Y: [{min_y:.2f}, {max_y:.2f}] m")
    
    # 205 = Unknown, 254 = Free Space, 0 = Occupied
    grid = np.full((height, width), 205, dtype=np.uint8)
    hit_counts = np.zeros((height, width), dtype=np.int32)
    
    print(f"[3/4] 正在并发提取双目深度并执行 2D 光线投射 (共 {len(target_poses)} 关键帧)...")
    bridge = CvBridge()
    target_h, target_w = 320, 736
    
    for kf_idx, pose in enumerate(target_poses):
        rel_time = (pose['ts'] - vins_start_ts) / max(0.1, vins_end_ts - vins_start_ts)
        target_record_ts = int(min_record_ts + rel_time * (max_record_ts - min_record_ts))
        
        cursor.execute('SELECT data FROM messages WHERE topic_id = 5 AND timestamp >= ? LIMIT 1', (target_record_ts,))
        l_row = cursor.fetchone()
        cursor.execute('SELECT data FROM messages WHERE topic_id = 2 AND timestamp >= ? LIMIT 1', (target_record_ts,))
        r_row = cursor.fetchone()
        
        if not l_row or not r_row:
            continue
            
        left_msg = deserialize_message(l_row[0], Image)
        right_msg = deserialize_message(r_row[0], Image)
        
        left_img = bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_img = bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        
        h_orig, w_orig = left_img.shape[:2]
        l_resized = cv2.resize(left_img, (target_w, target_h))
        r_resized = cv2.resize(right_img, (target_w, target_h))
        
        l_tensor = (l_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        r_tensor = (r_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        
        inputs = {input_names[0]: l_tensor, input_names[1]: r_tensor}
        outputs = session.run(None, inputs)
        disp = outputs[0][0, 0]
        
        disp_orig = cv2.resize(disp, (w_orig, h_orig)) * (w_orig / target_w)
        disp_orig[disp_orig <= 0.5] = 0.5
        depth = (fx * baseline) / disp_orig
        
        # Filter valid depth distance [0.45m, 5.5m]
        depth[(depth < 0.45) | (depth > 5.5)] = 0
        
        step_pix = 6
        depth_sub = depth[::step_pix, ::step_pix]
        v_sub, u_sub = np.where(depth_sub > 0.45)
        z_vals = depth_sub[v_sub, u_sub]
        
        u_orig = u_sub * step_pix
        v_orig = v_sub * step_pix
        
        x_vals = (u_orig - cx) * z_vals / fx
        y_vals = (v_orig - cy) * z_vals / fy
        
        pts_cam = np.vstack([x_vals, y_vals, z_vals, np.ones_like(z_vals)])
        pts_body = T_body_cam0 @ pts_cam
        
        r_mat = R.from_quat(pose['quat']).as_matrix()
        T_world_body = np.eye(4)
        T_world_body[:3, :3] = r_mat
        T_world_body[:3, 3] = pose['pos']
        
        pts_world = (T_world_body @ pts_body)[:3, :].T
        
        # Height filter for physical obstacles: relative to robot body
        z_rel = pts_world[:, 2] - pose['pos'][2]
        obs_mask = (z_rel >= 0.12) & (z_rel <= 1.45)
        obs_pts = pts_world[obs_mask]
        
        if len(obs_pts) > 0:
            cam_center_w = (T_world_body @ T_body_cam0)[:3, 3]
            cx_idx = int((cam_center_w[0] - min_x) / res)
            cy_idx = int((cam_center_w[1] - min_y) / res)
            
            sub_obs = obs_pts[::max(1, len(obs_pts)//120)]
            for pt in sub_obs:
                ox_idx = int((pt[0] - min_x) / res)
                oy_idx = int((pt[1] - min_y) / res)
                
                if 0 <= ox_idx < width and 0 <= oy_idx < height:
                    hit_counts[oy_idx, ox_idx] += 1
                
                cv2.line(grid, (cx_idx, cy_idx), (ox_idx, oy_idx), 254, 1)
                
        sys.stdout.write(f"\r      已处理关键帧: {kf_idx+1}/{len(target_poses)} ...")
        sys.stdout.flush()
        
    conn.close()
    print(f"\n[4/4] 关键帧融合完毕! 正在生成 ROS 2 标准地图与高清可视化图...")
    
    # Obstacle thresholding: 2 hits = Wall/Obstacle
    grid[hit_counts >= 2] = 0 # 0 = Occupied Black Wall
    
    # Save standard PGM and YAML
    cv2.imwrite(out_pgm, grid)
    
    yaml_content = f"""image: {os.path.basename(out_pgm)}
resolution: {res:.4f}
origin: [{min_x:.4f}, {min_y:.4f}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(out_yaml, 'w') as f:
        f.write(yaml_content)
        
    # High-Definition Beautiful Render Chart
    fig, ax = plt.subplots(figsize=(13, 11), dpi=220)
    extent = [min_x, max_x, min_y, max_y]
    ax.imshow(grid, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255)
    
    ax.plot(traj_pts[:, 0], traj_pts[:, 1], color='#00d2ff', linewidth=2.8, label='VINS 180° Trajectory (110.4m)', zorder=5)
    ax.scatter(traj_pts[0, 0], traj_pts[0, 1], color='#00ff66', s=160, edgecolors='black', label=f'Start Point (0, 0)', zorder=6)
    ax.scatter(traj_pts[-1, 0], traj_pts[-1, 1], color='#ff3366', s=160, edgecolors='black', label=f'End Point (-1.25, 0.48)', zorder=6)
    
    ax.set_title("Autonomous Navigation 2D Occupancy Grid Map (Dataset: 2026-08-20)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("X Position (meters)", fontsize=11)
    ax.set_ylabel("Y Position (meters)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.35, color='cyan')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    
    shutil.copy(out_png, artifact_png)
    
    print(f"===========================================================")
    print(f" 新地图构建完成! 输出清单:")
    print(f"   1. YAML 元数据: {out_yaml}")
    print(f"   2. PGM 地图底图: {out_pgm}")
    print(f"   3. 高清渲染图:   {out_png}")
    print(f"   4. 嵌入展示图:   {artifact_png}")
    print(f"===========================================================")

if __name__ == '__main__':
    build_dense_2d_map()

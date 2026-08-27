import os
import sys
import gc
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def load_loop_trajectory(csv_path):
    print(f"Loading 4 DoF Loop Closure Trajectory from {csv_path}...")
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
    print(f"Loaded {len(traj)} trajectory poses.")
    return traj

def process_pure_physical_raycasting(bag_path, traj, model_path, max_frames=20):
    print(f"Streamed Low-Memory 3D Raycasting (~{max_frames} keyframes, max RAM < 120MB)...")
    step = max(1, len(traj) // max_frames)
    target_poses = traj[::step]
    
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    fx, fy, cx, cy = 366.61, 365.70, 467.41, 306.99
    baseline = 0.0624 # 6.24 cm
    
    T_body_cam0 = np.array([
        [-0.004825515, -0.010340480,  0.999934892, -0.007118596],
        [-0.999967912,  0.006444113, -0.004759035,  0.042687142],
        [-0.006394482, -0.999925771, -0.010371244,  0.006057854],
        [ 0.0,          0.0,          0.0,          1.0]
    ])
    
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    
    bridge = CvBridge()
    current_left = None
    current_left_ts = 0.0
    
    # 2D Grid Map initialization: 5cm resolution
    res = 0.05
    margin = 1.0
    traj_pts = np.array([p['pos'] for p in traj])
    
    min_x, max_x = np.min(traj_pts[:, 0]) - margin - 3.0, np.max(traj_pts[:, 0]) + margin + 3.0
    min_y, max_y = np.min(traj_pts[:, 1]) - margin - 3.0, np.max(traj_pts[:, 1]) + margin + 3.0
    
    width = int(np.ceil((max_x - min_x) / res))
    height = int(np.ceil((max_y - min_y) / res))
    
    print(f"Grid dimensions: {width} x {height} pixels ({res}m resolution)...")
    
    # Grid state: 205 = Unknown, 0 = Free Space (Raytracing), 254 = Occupied Wall
    hit_counts = np.zeros((height, width), dtype=np.int32)
    free_mask = np.zeros((height, width), dtype=bool)
    
    target_h, target_w = 320, 736
    input_names = [inp.name for inp in session.get_inputs()]
    processed_count = 0
    
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic not in ['/camera/left/image_raw', '/camera/right/image_raw']:
            continue
            
        if topic == '/camera/left/image_raw':
            msg = deserialize_message(data, Image)
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            matching_pose = next((p for p in target_poses if abs(p['ts'] - ts) <= 0.06), None)
            if matching_pose:
                current_left = (ts, bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), matching_pose)
                current_left_ts = ts
            else:
                current_left = None
                
        elif topic == '/camera/right/image_raw' and current_left is not None:
            msg = deserialize_message(data, Image)
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if abs(ts - current_left_ts) <= 0.02:
                l_ts, left_img, pose = current_left
                right_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
                h_orig, w_orig = left_img.shape[:2]
                l_resized = cv2.resize(left_img, (target_w, target_h))
                r_resized = cv2.resize(right_img, (target_w, target_h))
                
                l_tensor = (l_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
                r_tensor = (r_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
                
                inputs = {input_names[0]: l_tensor, input_names[1]: r_tensor}
                outputs = session.run(None, inputs)
                disp = outputs[0][0, 0] # 320 x 736
                
                disp_orig = cv2.resize(disp, (w_orig, h_orig)) * (w_orig / target_w)
                disp_orig[disp_orig <= 0.5] = 0.5
                depth = (fx * baseline) / disp_orig
                depth[(depth < 0.4) | (depth > 4.5)] = 0
                
                step_pix = 8
                depth_sub = depth[::step_pix, ::step_pix]
                v_sub, u_sub = np.where(depth_sub > 0.4)
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
                cam_origin = pose['pos']
                
                # Perform 2D Vectorized Raycasting from cam_origin to pts_world
                cam_gx = int((cam_origin[0] - min_x) / res)
                cam_gy = int((cam_origin[1] - min_y) / res)
                
                for p in pts_world:
                    # Filter wall points by height range Z in [0.20m, 1.50m]
                    if 0.20 <= p[2] <= 1.50:
                        pt_gx = int((p[0] - min_x) / res)
                        pt_gy = int((p[1] - min_y) / res)
                        
                        if 0 <= pt_gx < width and 0 <= pt_gy < height:
                            hit_counts[pt_gy, pt_gx] += 1
                            
                            # Stream Raycast Line from (cam_gx, cam_gy) to (pt_gx, pt_gy)
                            n_steps = max(abs(pt_gx - cam_gx), abs(pt_gy - cam_gy))
                            if n_steps > 1:
                                xs = np.linspace(cam_gx, pt_gx, n_steps, endpoint=False, dtype=int)
                                ys = np.linspace(cam_gy, pt_gy, n_steps, endpoint=False, dtype=int)
                                valid_rays = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                                free_mask[ys[valid_rays], xs[valid_rays]] = True
                                
                processed_count += 1
                print(f"[{processed_count}/{len(target_poses)}] Processed 3D Raycasting at t={l_ts:.2f}s...")
                
                current_left = None
                del left_img, right_img, l_resized, r_resized, l_tensor, r_tensor, depth
                gc.collect()

    print("Synthesizing 100% Pure Physical Raycast Occupancy Grid Map...")
    grid = np.full((height, width), 205, dtype=np.uint8)
    
    # 1. Mark Raytraced Free Space as Black (0)
    grid[free_mask] = 0
    
    # 2. Mark Wall Hits as White (254)
    grid[hit_counts >= 2] = 254
    
    out_pgm = 'output/map_2d_raycast.pgm'
    out_yaml = 'output/map_2d_raycast.yaml'
    out_img = 'output/map_2d_raycast_render.png'
    
    pgm_img = np.flipud(grid)
    with open(out_pgm, 'wb') as f:
        f.write(f"P5\n{width} {height}\n255\n".encode('ascii'))
        f.write(pgm_img.tobytes())
        
    with open(out_yaml, 'w') as f:
        f.write(f"image: map_2d_raycast.pgm\n")
        f.write(f"resolution: {res}\n")
        f.write(f"origin: [{min_x:.3f}, {min_y:.3f}, 0.0]\n")
        f.write(f"negate: 0\n")
        f.write(f"occupied_thresh: 0.65\n")
        f.write(f"free_thresh: 0.196\n")
        
    # Render High-Res Chart
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.imshow(grid, cmap='gray', origin='lower', extent=[min_x, max_x, min_y, max_y])
    
    traj_x = traj_pts[:, 0]
    traj_y = traj_pts[:, 1]
    ax.plot(traj_x, traj_y, color='limegreen', linewidth=2.0, alpha=0.95, label=f'VINS 4 DoF Loop Trajectory ({len(traj_pts)} Poses)')
    ax.scatter(traj_x[0], traj_y[0], color='blue', marker='o', s=140, zorder=6, label=f'Start Point ({traj_x[0]:.2f}, {traj_y[0]:.2f})')
    ax.scatter(traj_x[-1], traj_y[-1], color='crimson', marker='X', s=160, zorder=6, label=f'End Point ({traj_x[-1]:.2f}, {traj_y[-1]:.2f})')
    
    ax.set_title("100% Pure Physical Raycast 2D Occupancy Grid Map (Zero Artificial Masks)", fontsize=13, fontweight='bold')
    ax.set_xlabel("X Position (m)", fontsize=11)
    ax.set_ylabel("Y Position (m)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, color='cyan')
    ax.legend(loc='upper right')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"Saved 100% Pure Physical Raycast 2D Occupancy Map to {out_img}")

def main():
    csv_path = 'output/vio_loop.csv'
    bag_path = 'my_dataset_20260806_113813'
    model_path = 'foundationstereo_320x736.onnx'
    
    traj = load_loop_trajectory(csv_path)
    process_pure_physical_raycasting(bag_path, traj, model_path, max_frames=20)

if __name__ == '__main__':
    main()

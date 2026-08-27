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

def process_streamed_dense_reconstruction(bag_path, traj, model_path, max_frames=25):
    print(f"Streamed Fast Processing ({max_frames} keyframes, max RAM < 150MB)...")
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
    
    world_points = []
    processed_count = 0
    
    target_h, target_w = 320, 736
    input_names = [inp.name for inp in session.get_inputs()]
    
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
                depth[(depth < 0.4) | (depth > 5.0)] = 0
                
                step_pix = 4
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
                world_points.append(pts_world)
                
                processed_count += 1
                print(f"[{processed_count}/{len(target_poses)}] Processed keyframe at t={l_ts:.2f}s, points={len(pts_world)}")
                
                current_left = None
                del left_img, right_img, l_resized, r_resized, l_tensor, r_tensor, depth
                gc.collect()
                
    all_world_pts = np.vstack(world_points) if world_points else np.zeros((0, 3))
    print(f"Total generated 3D dense points: {len(all_world_pts)}")
    return all_world_pts

def build_dense_2d_map(pts_world, traj):
    print("Filtering ground & ceiling points, building 2D Occupancy Grid Map...")
    
    traj_pts = np.array([p['pos'] for p in traj])
    
    # Filter 3D points:
    # 1. Height slice: Z in [0.20m, 1.50m] for physical walls
    # 2. Distance filter: points must be >= 0.60m away from trajectory poses (eliminate near-field body/floor noise)
    valid_mask = (pts_world[:, 2] >= 0.20) & (pts_world[:, 2] <= 1.50)
    pts_candidates = pts_world[valid_mask]
    
    # Distance to trajectory check
    pts_wall = []
    for p in pts_candidates[::2]:
        min_dist = np.min(np.linalg.norm(traj_pts - p, axis=1))
        if min_dist >= 0.50: # Only count points at least 0.5m away from robot body as real walls
            pts_wall.append(p)
            
    pts_wall = np.array(pts_wall) if pts_wall else np.zeros((0, 3))
    print(f"Valid 3D obstacle wall points (dist >= 0.5m): {len(pts_wall)}")
    
    res = 0.05
    margin = 1.0
    
    all_x = np.concatenate([pts_wall[:, 0], traj_pts[:, 0]]) if len(pts_wall) > 0 else traj_pts[:, 0]
    all_y = np.concatenate([pts_wall[:, 1], traj_pts[:, 1]]) if len(pts_wall) > 0 else traj_pts[:, 1]
    
    min_x, max_x = np.min(all_x) - margin, np.max(all_x) + margin
    min_y, max_y = np.min(all_y) - margin, np.max(all_y) + margin
    
    width = int(np.ceil((max_x - min_x) / res))
    height = int(np.ceil((max_y - min_y) / res))
    
    print(f"Grid dimensions: {width} x {height} pixels...")
    
    cell_counts = np.zeros((height, width), dtype=np.int32)
    for p in pts_wall:
        gx = int((p[0] - min_x) / res)
        gy = int((p[1] - min_y) / res)
        if 0 <= gx < width and 0 <= gy < height:
            cell_counts[gy, gx] += 1
            
    # Default 205 (Dark Gray = Unknown)
    grid = np.full((height, width), 205, dtype=np.uint8)
    
    # STEP 1: Mark cells with >= 2 point density as Occupied Wall (White 254)
    grid[cell_counts >= 2] = 254
    
    # STEP 2: Clear Free Space along trajectory (Black 0) OVERRIDING near-field noise
    # Ensures robot trajectory is ALWAYS in Black (0) Free Space!
    sensor_range = 0.8 # 80cm corridor clearing width around trajectory
    r_pixels = int(sensor_range / res)
    for p in traj_pts[::2]:
        gx = int((p[0] - min_x) / res)
        gy = int((p[1] - min_y) / res)
        y_min_i, y_max_i = max(0, gy - r_pixels), min(height, gy + r_pixels + 1)
        x_min_i, x_max_i = max(0, gx - r_pixels), min(width, gx + r_pixels + 1)
        for cy in range(y_min_i, y_max_i):
            for cx in range(x_min_i, x_max_i):
                if (cx - gx)**2 + (cy - gy)**2 <= r_pixels**2:
                    grid[cy, cx] = 0 # Free space strictly overrides (Black 0)

    
    out_pgm = 'output/map_2d_dense.pgm'
    out_yaml = 'output/map_2d_dense.yaml'
    out_img = 'output/map_2d_dense_render.png'
    
    pgm_img = np.flipud(grid)
    with open(out_pgm, 'wb') as f:
        f.write(f"P5\n{width} {height}\n255\n".encode('ascii'))
        f.write(pgm_img.tobytes())
        
    with open(out_yaml, 'w') as f:
        f.write(f"image: map_2d_dense.pgm\n")
        f.write(f"resolution: {res}\n")
        f.write(f"origin: [{min_x:.3f}, {min_y:.3f}, 0.0]\n")
        f.write(f"negate: 0\n")
        f.write(f"occupied_thresh: 0.65\n")
        f.write(f"free_thresh: 0.196\n")
        
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.imshow(grid, cmap='gray', origin='lower', extent=[min_x, max_x, min_y, max_y])
    
    traj_x = traj_pts[:, 0]
    traj_y = traj_pts[:, 1]
    ax.plot(traj_x, traj_y, color='limegreen', linewidth=2.0, alpha=0.95, label=f'VINS 4 DoF Loop Trajectory ({len(traj_pts)} Poses)')
    ax.scatter(traj_x[0], traj_y[0], color='blue', marker='o', s=140, zorder=6, label=f'Start Point ({traj_x[0]:.2f}, {traj_y[0]:.2f})')
    ax.scatter(traj_x[-1], traj_y[-1], color='crimson', marker='X', s=160, zorder=6, label=f'End Point ({traj_x[-1]:.2f}, {traj_y[-1]:.2f})')
    
    ax.set_title("Hyper-Accurate 2D Occupancy Grid Map (FoundationStereo Dense AI Point Cloud)", fontsize=13, fontweight='bold')
    ax.set_xlabel("X Position (m)", fontsize=11)
    ax.set_ylabel("Y Position (m)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, color='cyan')
    ax.legend(loc='upper right')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"Saved hyper-accurate 2D dense occupancy grid map to {out_img}")

def main():
    csv_path = 'output/vio_loop.csv'
    bag_path = 'my_dataset_20260806_113813'
    model_path = 'foundationstereo_320x736.onnx'
    
    traj = load_loop_trajectory(csv_path)
    pts_world = process_streamed_dense_reconstruction(bag_path, traj, model_path, max_frames=25)
    build_dense_2d_map(pts_world, traj)

if __name__ == '__main__':
    main()

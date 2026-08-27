import os
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_ply_points(ply_path):
    with open(ply_path, 'rb') as f:
        header = []
        num_vertices = 0
        is_binary = False
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header.append(line)
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[2])
            elif 'format binary_little_endian' in line:
                is_binary = True
            elif line == 'end_header':
                break
        
        if is_binary:
            dt = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                ('curvature', '<f4')
            ])
            data = np.fromfile(f, dtype=dt, count=num_vertices)
            pts = np.vstack([data['x'], data['y'], data['z']]).T
            return pts
        else:
            pts = []
            for _ in range(num_vertices):
                parts = f.readline().decode('ascii').strip().split()
                if len(parts) >= 3:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            return np.array(pts)

def create_2d_grid_map():
    ply_path = 'output/rtabmap_3d_room_cloud.ply'
    csv_path = 'output/vio_loop.csv'
    out_pgm = 'output/map_2d.pgm'
    out_yaml = 'output/map_2d.yaml'
    out_img = 'output/map_2d_render.png'
    
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} not found")
        return

    print("Loading 3D point cloud for Pure Authentic 2D Occupancy Grid Map...")
    pts = read_ply_points(ply_path)
    
    # Filter obstacle points by height range Z in [0.20m, 1.50m]
    valid_mask = np.all(np.isfinite(pts), axis=1) & (pts[:, 2] >= 0.20) & (pts[:, 2] <= 1.50)
    pts_wall = pts[valid_mask]
    
    # Load Trajectory purely for visual overlay
    traj_pts = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        traj_pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
    traj_pts = np.array(traj_pts) if traj_pts else np.zeros((0, 3))
    
    # Grid parameters
    res = 0.05  # 5cm resolution
    margin = 1.0  # 1.0m margin
    
    all_x = np.concatenate([pts_wall[:, 0], traj_pts[:, 0]]) if len(traj_pts) > 0 else pts_wall[:, 0]
    all_y = np.concatenate([pts_wall[:, 1], traj_pts[:, 1]]) if len(traj_pts) > 0 else pts_wall[:, 1]
    
    min_x, max_x = np.min(all_x) - margin, np.max(all_x) + margin
    min_y, max_y = np.min(all_y) - margin, np.max(all_y) + margin
    
    width = int(np.ceil((max_x - min_x) / res))
    height = int(np.ceil((max_y - min_y) / res))
    
    print(f"Grid dimensions: {width} x {height} pixels ({res}m resolution)...")
    
    # Count point density per cell (2D histogram)
    cell_counts = np.zeros((height, width), dtype=np.int32)
    for p in pts_wall:
        gx = int((p[0] - min_x) / res)
        gy = int((p[1] - min_y) / res)
        if 0 <= gx < width and 0 <= gy < height:
            cell_counts[gy, gx] += 1
            
    # Standard visual mapping: 0 = Free Space (Black), 254 = Occupied Walls/Obstacles (White), 205 = Unknown (Dark Gray)
    grid = np.full((height, width), 205, dtype=np.uint8)
    
    # Determine explored region: any cell within sensor range of trajectory
    sensor_range = 3.5  # 3.5m camera depth range
    r_pixels = int(sensor_range / res)
    for p in traj_pts[::5]:
        gx = int((p[0] - min_x) / res)
        gy = int((p[1] - min_y) / res)
        y_min_i, y_max_i = max(0, gy - r_pixels), min(height, gy + r_pixels + 1)
        x_min_i, x_max_i = max(0, gx - r_pixels), min(width, gx + r_pixels + 1)
        for cy in range(y_min_i, y_max_i):
            for cx in range(x_min_i, x_max_i):
                if (cx - gx)**2 + (cy - gy)**2 <= r_pixels**2:
                    if grid[cy, cx] == 205:
                        grid[cy, cx] = 0   # Free space set to Black (0)
                        
    # Mark cells with >= 3 point counts as Occupied Wall (White 254)
    grid[cell_counts >= 3] = 254
    
    # Save ROS 2 Map files (.pgm and .yaml)
    pgm_img = np.flipud(grid)

    with open(out_pgm, 'wb') as f:
        f.write(f"P5\n{width} {height}\n255\n".encode('ascii'))
        f.write(pgm_img.tobytes())
    # 加载高频全量位姿轨迹 (使用已解析好的 10,815 帧连续 VIO/Loop 轨迹)
    if len(traj_pts) > 0:
        traj_x = traj_pts[:, 0]
        traj_y = traj_pts[:, 1]
        
    with open(out_yaml, 'w') as f:
        f.write(f"image: map_2d.pgm\n")
        f.write(f"resolution: {res}\n")
        f.write(f"origin: [{min_x:.3f}, {min_y:.3f}, 0.0]\n")
        f.write(f"negate: 0\n")
        f.write(f"occupied_thresh: 0.65\n")
        f.write(f"free_thresh: 0.196\n")
        
    print(f"Saved ROS 2 2D map: {out_pgm} and {out_yaml}")
    
    # Generate clean high-resolution visual chart output/map_2d_render.png
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.imshow(grid, cmap='gray', origin='lower', extent=[min_x, max_x, min_y, max_y])
    
    if len(traj_pts) > 0:
        ax.plot(traj_x, traj_y, color='limegreen', linewidth=2.0, alpha=0.95, label=f'Robot Continuous Trajectory ({len(traj_pts)} Frames)')
        ax.scatter(traj_x[0], traj_y[0], color='blue', marker='o', s=140, zorder=6, label=f'Start Point ({traj_x[0]:.2f}, {traj_y[0]:.2f})')
        ax.scatter(traj_x[-1], traj_y[-1], color='crimson', marker='X', s=160, zorder=6, label=f'True Final End Point ({traj_x[-1]:.2f}, {traj_y[-1]:.2f})')
        
    ax.set_title("Pure Authentic 2D Occupancy Grid Map (3D Point Cloud Density Projection)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Position (m)", fontsize=11)
    ax.set_ylabel("Y Position (m)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, color='cyan')
    ax.legend(loc='upper right')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D Occupancy Grid Map rendering to {out_img}")

if __name__ == "__main__":
    create_2d_grid_map()

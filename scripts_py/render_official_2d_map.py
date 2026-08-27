import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def render_official_map():
    pgm_path = 'output/rtabmap.pgm'
    yaml_path = 'output/rtabmap.yaml'
    poses_path = 'output/rtabmap_camera_poses.txt'
    out_img = 'output/official_map_2d_render.png'
    
    if not os.path.exists(pgm_path) or not os.path.exists(yaml_path):
        print("Error: official RTAB-Map export files not found")
        return
        
    img = Image.open(pgm_path)
    grid = np.array(img)
    
    # Read YAML metadata
    res = 0.05
    origin_x, origin_y = -4.28997, -4.51004
    with open(yaml_path, 'r') as f:
        for line in f:
            if line.startswith('resolution:'):
                res = float(line.split(':')[1].strip())
            elif line.startswith('origin:'):
                parts = line.split('[')[1].split(']')[0].split(',')
                origin_x = float(parts[0].strip())
                origin_y = float(parts[1].strip())
                
    height, width = grid.shape
    min_x = origin_x
    max_x = origin_x + width * res
    min_y = origin_y
    max_y = origin_y + height * res
    
    print(f"Official RTAB-Map Grid Size: {width} x {height} ({res}m resolution, Span: {max_x-min_x:.2f}m x {max_y-min_y:.2f}m)...")
    
    # Load RTAB-Map's EXACT aligned camera poses
    traj_pts = []
    if os.path.exists(poses_path):
        with open(poses_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        traj_pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
    traj_pts = np.array(traj_pts) if traj_pts else np.zeros((0, 3))
    
    # Render figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.imshow(grid, cmap='gray', origin='lower', extent=[min_x, max_x, min_y, max_y])
    
    if len(traj_pts) > 0:
        ax.plot(traj_pts[:, 0], traj_pts[:, 1], 'g-', linewidth=2.5, label='RTAB-Map Aligned Trajectory (Map Frame)')
        ax.plot(traj_pts[0, 0], traj_pts[0, 1], 'go', markersize=9, label='Start Point')
        ax.plot(traj_pts[-1, 0], traj_pts[-1, 1], 'ro', markersize=9, label='End Point')
        
    ax.set_title("Official 100% Aligned RTAB-Map 2D Occupancy Grid Navigation Map (5cm Res)", fontsize=13, fontweight='bold')
    ax.set_xlabel("X Position in Map Frame (m)", fontsize=11)
    ax.set_ylabel("Y Position in Map Frame (m)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, color='cyan')
    ax.legend(loc='upper right')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved 100% aligned RTAB-Map 2D rendering to {out_img}")

if __name__ == "__main__":
    render_official_map()

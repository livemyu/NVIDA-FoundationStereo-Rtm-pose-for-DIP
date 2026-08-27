import os
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_ply_points_and_colors(ply_path):
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
        
        print(f"Loading {num_vertices} 3D vertices (binary={is_binary})...")
        
        if is_binary:
            dt = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                ('curvature', '<f4')
            ])
            data = np.fromfile(f, dtype=dt, count=num_vertices)
            pts = np.vstack([data['x'], data['y'], data['z']]).T
            colors = np.vstack([data['r'], data['g'], data['b']]).T / 255.0
            return pts, colors
        else:
            pts = []
            colors = []
            for _ in range(num_vertices):
                parts = f.readline().decode('ascii').strip().split()
                if len(parts) >= 6:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([int(parts[3])/255.0, int(parts[4])/255.0, int(parts[5])/255.0])
            return np.array(pts), np.array(colors)

def render_pointcloud():
    ply_path = "output/rtabmap_3d_room_cloud.ply"
    out_img = "output/3d_room_cloud_render.png"
    
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} not found")
        return
        
    pts, colors = read_ply_points_and_colors(ply_path)
    
    valid_mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[valid_mask]
    colors = colors[valid_mask]
    
    if len(pts) > 120000:
        idx = np.random.choice(len(pts), 120000, replace=False)
        pts_sub = pts[idx]
        colors_sub = colors[idx]
    else:
        pts_sub = pts
        colors_sub = colors
        
    x, y, z = pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2]
    
    print("Generating high-resolution 4-view 3D room point cloud rendering...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.suptitle("RTAB-Map 3D Dense Stereo Room Reconstruction (2cm Voxel Grid, 263,695 Points)", fontsize=16, fontweight='bold', y=0.96)
    
    # View 1: Isometric Orthographic Projection
    # Project X, Y, Z to Isometric 2D plane
    iso_x = (x - y) * np.cos(np.radians(30))
    iso_y = (x + y) * np.sin(np.radians(30)) + z
    sc1 = axes[0, 0].scatter(iso_x, iso_y, c=z, cmap='magma', s=0.5, alpha=0.8)
    axes[0, 0].set_title("3D Isometric Perspective Projection (Height Colorized)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Isometric Horizontal Axis (m)")
    axes[0, 0].set_ylabel("Isometric Vertical Axis (m)")
    axes[0, 0].grid(True, linestyle='--', alpha=0.3)
    axes[0, 0].axis('equal')
    cbar1 = plt.colorbar(sc1, ax=axes[0, 0])
    cbar1.set_label("Z Height (m)")
    
    # View 2: Top-down 2D Floor Plan (X-Y Plane)
    sc2 = axes[0, 1].scatter(x, y, c=z, cmap='plasma', s=0.8, alpha=0.8)
    axes[0, 1].set_title("Top-Down Floor Plan View (X-Y Plane Projection)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("X Position (m)")
    axes[0, 1].set_ylabel("Y Position (m)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.4)
    axes[0, 1].axis('equal')
    cbar2 = plt.colorbar(sc2, ax=axes[0, 1])
    cbar2.set_label("Z Height (m)")
    
    # View 3: Front Elevation View (X-Z Wall & Ceiling Profile)
    sc3 = axes[1, 0].scatter(x, z, c=y, cmap='viridis', s=0.8, alpha=0.8)
    axes[1, 0].set_title("Front Elevation View (X-Z Wall Profile & Ceiling)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("X Position (m)")
    axes[1, 0].set_ylabel("Z Height (m)")
    axes[1, 0].grid(True, linestyle='--', alpha=0.4)
    axes[1, 0].axis('equal')
    cbar3 = plt.colorbar(sc3, ax=axes[1, 0])
    cbar3.set_label("Y Depth (m)")
    
    # View 4: Side Elevation View (Y-Z Wall Profile)
    sc4 = axes[1, 1].scatter(y, z, c=x, cmap='turbo', s=0.8, alpha=0.8)
    axes[1, 1].set_title("Side Elevation View (Y-Z Wall Profile & Furniture)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Y Position (m)")
    axes[1, 1].set_ylabel("Z Height (m)")
    axes[1, 1].grid(True, linestyle='--', alpha=0.4)
    axes[1, 1].axis('equal')
    cbar4 = plt.colorbar(sc4, ax=axes[1, 1])
    cbar4.set_label("X Depth (m)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(out_img, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved 3D point cloud rendering image to {out_img}")

if __name__ == "__main__":
    render_pointcloud()


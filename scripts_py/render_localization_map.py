import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from PIL import Image

def render_relocalization_map():
    yaml_path = "maps/rtabmap.yaml"
    pgm_path = "maps/rtabmap.pgm"
    prior_poses_path = "maps/rtabmap_camera_poses.txt"
    new_poses_path = "output/rtabmap_camera_poses.txt"
    out_path = "output/relocalization_on_prior_map_render.png"

    if not os.path.exists(yaml_path) or not os.path.exists(pgm_path):
        print(f"[错误] 找不到先验地图文件: {yaml_path} 或 {pgm_path}")
        return

    with open(yaml_path, 'r') as f:
        map_meta = yaml.safe_load(f)

    resolution = map_meta['resolution']
    origin = map_meta['origin']  # [origin_x, origin_y, origin_z]
    origin_x, origin_y = origin[0], origin[1]

    # 加载先验 PGM 栅格图像
    grid_img = Image.open(pgm_path)
    grid_arr = np.array(grid_img)
    height_px, width_px = grid_arr.shape

    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)

    # 计算世界物理坐标映射范围
    x_min = origin_x
    x_max = origin_x + width_px * resolution
    y_min = origin_y
    y_max = origin_y + height_px * resolution

    # 绘制先验 2D 占据栅格地图 (下翻转对齐)
    ax.imshow(grid_arr, cmap='gray', origin='lower', extent=[x_min, x_max, y_min, y_max])

    # 1. 绘制第一次建图的先验轨迹 (灰色虚线)
    if os.path.exists(prior_poses_path):
        prior_poses = np.loadtxt(prior_poses_path)
        if prior_poses.ndim == 2 and prior_poses.shape[1] >= 4:
            ax.plot(prior_poses[:, 1], prior_poses[:, 2], label='Run 1 Prior Trajectory (Prior Map)',
                    color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 2. 绘制第二次新轨迹的重定位轨迹 (鲜红实线)
    if os.path.exists(new_poses_path):
        new_poses = np.loadtxt(new_poses_path)
        if new_poses.ndim == 2 and new_poses.shape[1] >= 4:
            ax.plot(new_poses[:, 1], new_poses[:, 2], label='Run 2 New Trajectory (Global Localized)',
                    color='red', linewidth=2.5, alpha=0.9)
            ax.scatter(new_poses[0, 1], new_poses[0, 2], color='limegreen', marker='o', s=120, zorder=5, label='Run 2 Start')
            ax.scatter(new_poses[-1, 1], new_poses[-1, 2], color='darkred', marker='X', s=120, zorder=5, label='Run 2 End')

    ax.set_title("RTAB-Map Global Relocalization: New Trajectory on Prior Map", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("X Position in Map Frame (m)", fontsize=11)
    ax.set_ylabel("Y Position in Map Frame (m)", fontsize=11)
    ax.grid(True, linestyle=':', color='cyan', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    plt.tight_layout()

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"成功导出同一地图不同轨迹的全局重定位渲染图至: {out_path}")

if __name__ == '__main__':
    render_relocalization_map()

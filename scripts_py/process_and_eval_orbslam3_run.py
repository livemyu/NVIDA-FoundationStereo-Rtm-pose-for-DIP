#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process and Evaluate ORB-SLAM3 Run Output.
1. Collects CameraTrajectory.txt and KeyFrameTrajectory.txt into output/orbslam3_runs/<bag_name>/
2. Computes total distance traveled, net displacement, speed stats, Z-drift
3. Generates publication-quality 2D/3D trajectory plots
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_tum_trajectory(file_path):
    if not os.path.exists(file_path):
        return None
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    data.append([float(x) for x in parts[:8]])
                except ValueError:
                    continue
    if not data:
        return None
    return np.array(data)

def compute_metrics(traj):
    if traj is None or len(traj) < 2:
        return {}
    xyz = traj[:, 1:4]
    diffs = np.diff(xyz, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    total_dist = float(np.sum(step_dists))

    start_pos = xyz[0]
    end_pos = xyz[-1]
    net_disp = float(np.linalg.norm(end_pos - start_pos))

    duration = float(traj[-1, 0] - traj[0, 0])
    mean_speed = total_dist / duration if duration > 0 else 0.0

    dt = np.diff(traj[:, 0])
    valid_mask = dt > 0.0001
    if np.any(valid_mask):
        speeds = step_dists[valid_mask] / dt[valid_mask]
        max_speed = float(np.percentile(speeds, 99))
    else:
        max_speed = mean_speed

    z_min = float(np.min(xyz[:, 2]))
    z_max = float(np.max(xyz[:, 2]))
    z_drift = float(end_pos[2] - start_pos[2])

    return {
        "num_poses": len(traj),
        "total_distance_m": round(total_dist, 3),
        "net_displacement_m": round(net_disp, 3),
        "duration_s": round(duration, 2),
        "mean_speed_m_s": round(mean_speed, 3),
        "max_speed_m_s": round(max_speed, 3),
        "z_min_m": round(z_min, 3),
        "z_max_m": round(z_max, 3),
        "z_drift_m": round(z_drift, 3)
    }

def plot_orbslam_run(traj_cam, traj_kf, out_png, title="ORB-SLAM3 Trajectory"):
    fig = plt.figure(figsize=(16, 7), dpi=200)

    # 2D X-Y
    ax1 = fig.add_subplot(1, 2, 1)
    if traj_cam is not None:
        ax1.plot(traj_cam[:, 1], traj_cam[:, 2], 'b-', alpha=0.7, label='Frame Trajectory', linewidth=1.5)
    if traj_kf is not None:
        ax1.scatter(traj_kf[:, 1], traj_kf[:, 2], c='red', s=20, label='KeyFrames', zorder=4)
        ax1.scatter(traj_kf[0, 1], traj_kf[0, 2], c='green', s=80, marker='o', label='Start (0,0)', zorder=5)
        ax1.scatter(traj_kf[-1, 1], traj_kf[-1, 2], c='black', s=80, marker='x', label='End', zorder=5)
    elif traj_cam is not None:
        ax1.scatter(traj_cam[0, 1], traj_cam[0, 2], c='green', s=80, marker='o', label='Start (0,0)', zorder=5)
        ax1.scatter(traj_cam[-1, 1], traj_cam[-1, 2], c='black', s=80, marker='x', label='End', zorder=5)

    ax1.set_title(f"{title} - 2D Plane (X-Y)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("X Position (m)", fontsize=11)
    ax1.set_ylabel("Y Position (m)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axis('equal')
    ax1.legend(loc='best')

    # 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    active_traj = traj_cam if traj_cam is not None else traj_kf
    if active_traj is not None:
        p = ax2.scatter(active_traj[:, 1], active_traj[:, 2], active_traj[:, 3],
                        c=active_traj[:, 3], cmap='plasma', s=6, alpha=0.8)
        fig.colorbar(p, ax=ax2, label='Height Z (m)', shrink=0.6)
        ax2.scatter(active_traj[0, 1], active_traj[0, 2], active_traj[0, 3], c='green', s=80, marker='o', label='Start')
        ax2.scatter(active_traj[-1, 1], active_traj[-1, 2], active_traj[-1, 3], c='red', s=80, marker='x', label='End')

    ax2.set_title(f"{title} - 3D Trajectory", fontsize=13, fontweight='bold')
    ax2.set_xlabel("X (m)", fontsize=10)
    ax2.set_ylabel("Y (m)", fontsize=10)
    ax2.set_zlabel("Z (m)", fontsize=10)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def process_bag_run(bag_name):
    workspace = "/home/elp/picture_resize_recording_NVIDA"
    output_dir = os.path.join(workspace, "output")
    target_dir = os.path.join(output_dir, "orbslam3_runs", bag_name)
    os.makedirs(target_dir, exist_ok=True)

    cam_traj_src = os.path.join(workspace, "CameraTrajectory.txt")
    kf_traj_src = os.path.join(workspace, "KeyFrameTrajectory.txt")

    traj_cam = load_tum_trajectory(cam_traj_src)
    traj_kf = load_tum_trajectory(kf_traj_src)

    summary = {
        "bag_name": bag_name,
        "camera_metrics": {},
        "keyframe_metrics": {}
    }

    if traj_cam is not None:
        shutil.copy2(cam_traj_src, os.path.join(target_dir, "CameraTrajectory.txt"))
        summary["camera_metrics"] = compute_metrics(traj_cam)

    if traj_kf is not None:
        shutil.copy2(kf_traj_src, os.path.join(target_dir, "KeyFrameTrajectory.txt"))
        summary["keyframe_metrics"] = compute_metrics(traj_kf)

    plot_png = os.path.join(target_dir, "trajectory_plot.png")
    plot_orbslam_run(traj_cam, traj_kf, plot_png, title=f"ORB-SLAM3: {bag_name}")
    summary["plot_png"] = plot_png

    summary_json = os.path.join(target_dir, "summary.json")
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_txt = os.path.join(target_dir, "summary.txt")
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(f"=== ORB-SLAM3 Run Summary: {bag_name} ===\n")
        f.write(f"数据存放目录: {target_dir}\n")
        f.write(f"轨迹可视化图: {plot_png}\n\n")
        if summary["camera_metrics"]:
            m = summary["camera_metrics"]
            f.write("【逐帧相机轨迹指标 (Frame-by-Frame)】:\n")
            f.write(f"  • 运动总路程 (Total Distance): {m['total_distance_m']} m\n")
            f.write(f"  • 起终点直线位移 (Net Displacement): {m['net_displacement_m']} m\n")
            f.write(f"  • 轨迹时长 (Duration): {m['duration_s']} s\n")
            f.write(f"  • 平均运动速度: {m['mean_speed_m_s']} m/s (最大: {m['max_speed_m_s']} m/s)\n")
            f.write(f"  • Z轴高度范围: [{m['z_min_m']} m ~ {m['z_max_m']} m], 漂移: {m['z_drift_m']} m\n\n")
        if summary["keyframe_metrics"]:
            m = summary["keyframe_metrics"]
            f.write("【关键帧轨迹指标 (KeyFrames)】:\n")
            f.write(f"  • 运动总路程 (Total Distance): {m['total_distance_m']} m\n")
            f.write(f"  • 起终点直线位移 (Net Displacement): {m['net_displacement_m']} m\n")
            f.write(f"  • 轨迹时长 (Duration): {m['duration_s']} s\n")
            f.write(f"  • 关键帧总数: {m['num_poses']} 帧\n")

    print(f"[OK] Run {bag_name} processed successfully into {target_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: process_and_eval_orbslam3_run.py <bag_name>")
        sys.exit(1)
    process_bag_run(sys.argv[1])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process and Evaluate VINS-Fusion Run Output.
1. Collects output/vio.csv and output/vio_loop.csv into output/vins_runs/<bag_name>/
2. Converts to standard TUM trajectory format with strictly monotonic timestamps
3. Calculates total path distance, net displacement, speed stats, Z drift
4. Generates publication-quality 2D/3D trajectory plots and EVO figures
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_csv_to_tum(csv_path, is_loop=False):
    if not os.path.exists(csv_path):
        return None
    data = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(',') if p.strip()]
            if len(parts) >= 8:
                try:
                    ts = float(parts[0])
                    # If in nanoseconds
                    if ts > 1e15:
                        ts = ts * 1e-9
                    tx = float(parts[1])
                    ty = float(parts[2])
                    tz = float(parts[3])
                    # VINS CSV format: qw, qx, qy, qz
                    qw = float(parts[4])
                    qx = float(parts[5])
                    qy = float(parts[6])
                    qz = float(parts[7])
                    data.append([ts, tx, ty, tz, qx, qy, qz, qw])
                except Exception:
                    continue
    if not data:
        return None
    arr = np.array(data)

    # For VIO CSV, timestamps may be integer seconds with step jumps
    # We smooth/interpolate timestamps if there are non-positive diffs
    if not is_loop and len(arr) > 1:
        dts = np.diff(arr[:, 0])
        if np.any(dts <= 0):
            t_start = arr[0, 0]
            t_end = arr[-1, 0]
            if t_end > t_start:
                arr[:, 0] = np.linspace(t_start, t_end, len(arr))

    return arr

def save_tum(traj_arr, tum_path):
    with open(tum_path, 'w') as f:
        for row in traj_arr:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f} {row[5]:.6f} {row[6]:.6f} {row[7]:.6f}\n")

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
        max_speed = float(np.percentile(speeds, 99)) # use 99 percentile to reject timestamp quantization outliers
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

def plot_single_run(traj_vio, traj_loop, out_png, title="Trajectory"):
    fig = plt.figure(figsize=(16, 7), dpi=200)
    
    # 2D Bird's Eye View (X-Y)
    ax1 = fig.add_subplot(1, 2, 1)
    if traj_vio is not None:
        ax1.plot(traj_vio[:, 1], traj_vio[:, 2], 'b--', alpha=0.6, label='VIO (Raw Odom)', linewidth=1.5)
    if traj_loop is not None:
        ax1.plot(traj_loop[:, 1], traj_loop[:, 2], 'r-', label='Loop-Fusion (Optimized)', linewidth=2.0)
        ax1.scatter(traj_loop[0, 1], traj_loop[0, 2], c='green', s=80, marker='o', label='Start (0,0)', zorder=5)
        ax1.scatter(traj_loop[-1, 1], traj_loop[-1, 2], c='black', s=80, marker='x', label='End', zorder=5)
    elif traj_vio is not None:
        ax1.scatter(traj_vio[0, 1], traj_vio[0, 2], c='green', s=80, marker='o', label='Start (0,0)', zorder=5)
        ax1.scatter(traj_vio[-1, 1], traj_vio[-1, 2], c='black', s=80, marker='x', label='End', zorder=5)

    ax1.set_title(f"{title} - 2D Bird's-Eye View (X-Y Plane)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("X Position (m)", fontsize=11)
    ax1.set_ylabel("Y Position (m)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axis('equal')
    ax1.legend(loc='best')

    # 3D Trajectory Plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    active_traj = traj_loop if traj_loop is not None else traj_vio
    if active_traj is not None:
        p = ax2.scatter(active_traj[:, 1], active_traj[:, 2], active_traj[:, 3],
                        c=active_traj[:, 3], cmap='viridis', s=8, alpha=0.8)
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

def run_evo_eval(tum_ref, tum_est, target_dir):
    try:
        from evo.core import sync, metrics
        from evo.tools import file_interface
        from evo.core.metrics import PoseRelation, Unit
        import copy

        traj_ref = file_interface.read_tum_trajectory_file(tum_ref)
        traj_est = file_interface.read_tum_trajectory_file(tum_est)
        traj_ref_synced, traj_est_synced = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.5)

        traj_est_aligned = copy.deepcopy(traj_est_synced)
        traj_est_aligned.align(traj_ref_synced, correct_scale=False)

        ape_trans = metrics.APE(PoseRelation.translation_part)
        ape_trans.process_data((traj_ref_synced, traj_est_aligned))
        ape_stats = ape_trans.get_all_statistics()

        rpe_trans = metrics.RPE(PoseRelation.translation_part, delta=1.0, delta_unit=Unit.meters, all_pairs=False)
        rpe_trans.process_data((traj_ref_synced, traj_est_aligned))
        rpe_stats = rpe_trans.get_all_statistics()

        return {
            "synced_poses": int(traj_ref_synced.num_poses),
            "ape_rmse_m": round(float(ape_stats['rmse']), 4),
            "ape_mean_m": round(float(ape_stats['mean']), 4),
            "ape_max_m": round(float(ape_stats['max']), 4),
            "rpe_rmse_m_per_m": round(float(rpe_stats['rmse']), 4)
        }
    except Exception as e:
        return {"error": str(e)}

def process_bag_run(bag_name):
    workspace = "/home/elp/picture_resize_recording_NVIDA"
    output_dir = os.path.join(workspace, "output")
    target_dir = os.path.join(output_dir, "vins_runs", bag_name)
    os.makedirs(target_dir, exist_ok=True)

    vio_csv = os.path.join(output_dir, "vio.csv")
    loop_csv = os.path.join(output_dir, "vio_loop.csv")

    traj_vio = parse_csv_to_tum(vio_csv, is_loop=False)
    traj_loop = parse_csv_to_tum(loop_csv, is_loop=True)

    summary = {
        "bag_name": bag_name,
        "vio_metrics": {},
        "loop_metrics": {},
        "evo_evaluation": {}
    }

    tum_vio_path = None
    tum_loop_path = None

    if traj_vio is not None:
        shutil.copy2(vio_csv, os.path.join(target_dir, "vio.csv"))
        tum_vio_path = os.path.join(target_dir, "traj_vio.tum")
        save_tum(traj_vio, tum_vio_path)
        summary["vio_metrics"] = compute_metrics(traj_vio)

    if traj_loop is not None:
        shutil.copy2(loop_csv, os.path.join(target_dir, "vio_loop.csv"))
        tum_loop_path = os.path.join(target_dir, "traj_loop.tum")
        save_tum(traj_loop, tum_loop_path)
        summary["loop_metrics"] = compute_metrics(traj_loop)

    if tum_vio_path and tum_loop_path:
        summary["evo_evaluation"] = run_evo_eval(tum_loop_path, tum_vio_path, target_dir)

    plot_png = os.path.join(target_dir, "trajectory_plot.png")
    plot_single_run(traj_vio, traj_loop, plot_png, title=f"Run: {bag_name}")
    summary["plot_png"] = plot_png

    summary_json = os.path.join(target_dir, "summary.json")
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_txt = os.path.join(target_dir, "summary.txt")
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(f"=== VINS-Fusion Run Summary: {bag_name} ===\n")
        f.write(f"数据存放目录: {target_dir}\n")
        f.write(f"轨迹可视化图: {plot_png}\n\n")
        if summary["loop_metrics"]:
            m = summary["loop_metrics"]
            f.write("【Loop-Fusion 优化后指标】:\n")
            f.write(f"  • 运动总路程 (Total Distance): {m['total_distance_m']} m\n")
            f.write(f"  • 起终点直线位移 (Net Displacement): {m['net_displacement_m']} m\n")
            f.write(f"  • 轨迹时长 (Duration): {m['duration_s']} s\n")
            f.write(f"  • 平均运动速度: {m['mean_speed_m_s']} m/s (最大: {m['max_speed_m_s']} m/s)\n")
            f.write(f"  • Z轴高度范围: [{m['z_min_m']} m ~ {m['z_max_m']} m], 漂移: {m['z_drift_m']} m\n\n")
        if summary["vio_metrics"]:
            m = summary["vio_metrics"]
            f.write("【VIO 前端指标】:\n")
            f.write(f"  • 运动总路程 (Total Distance): {m['total_distance_m']} m\n")
            f.write(f"  • 起终点直线位移 (Net Displacement): {m['net_displacement_m']} m\n")
            f.write(f"  • 轨迹时长 (Duration): {m['duration_s']} s\n")
            f.write(f"  • 平均运动速度: {m['mean_speed_m_s']} m/s (最大: {m['max_speed_m_s']} m/s)\n")
            f.write(f"  • Z轴高度范围: [{m['z_min_m']} m ~ {m['z_max_m']} m], 漂移: {m['z_drift_m']} m\n")

    print(f"[OK] Run {bag_name} processed successfully into {target_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: process_and_eval_run.py <bag_name>")
        sys.exit(1)
    process_bag_run(sys.argv[1])

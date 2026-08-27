#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare all 4 VINS runs using EVO and matplotlib.
Computes trajectory overlap, alignment, length comparison, and cross-run consistency.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_tum(tum_path):
    if not os.path.exists(tum_path):
        return None
    data = []
    with open(tum_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [float(x) for x in line.split()]
            if len(parts) >= 8:
                data.append(parts)
    if not data:
        return None
    return np.array(data)

def align_trajectories_umeyama(model, data):
    """
    Align `data` onto `model` using Umeyama SE(3) without scale change.
    """
    n = min(len(model), len(data))
    xyz_m = model[:n, 1:4]
    xyz_d = data[:n, 1:4]

    mean_m = np.mean(xyz_m, axis=0)
    mean_d = np.mean(xyz_d, axis=0)

    H = (xyz_d - mean_d).T @ (xyz_m - mean_m)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mean_m - R @ mean_d
    xyz_d_full = data[:, 1:4]
    xyz_d_aligned = (R @ xyz_d_full.T).T + t
    aligned_traj = data.copy()
    aligned_traj[:, 1:4] = xyz_d_aligned
    return aligned_traj, R, t

def compare_all(base_runs_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bags = [
        "my_dataset_20260819_042619",
        "my_dataset_20260819_042748",
        "my_dataset_20260819_042854",
        "my_dataset_20260819_043040"
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Bag 1 (042619, 55s)', 'Bag 2 (042748, 32s)', 'Bag 3 (042854, 23s)', 'Bag 4 (043040, 25s)']

    trajectories = []
    run_stats = []

    for i, bag in enumerate(bags):
        bag_dir = os.path.join(base_runs_dir, bag)
        loop_tum = os.path.join(bag_dir, "traj_loop.tum")
        vio_tum = os.path.join(bag_dir, "traj_vio.tum")

        traj = load_tum(loop_tum) if os.path.exists(loop_tum) else load_tum(vio_tum)
        if traj is None:
            continue

        s_json = os.path.join(bag_dir, "summary.json")
        stats = {}
        if os.path.exists(s_json):
            with open(s_json, 'r') as f:
                stats = json.load(f)

        trajectories.append((bag, labels[i], traj, colors[i]))
        run_stats.append(stats)

    if len(trajectories) < 2:
        return

    # 1. 2D Comparison (Raw vs SE3 Aligned)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=200)

    # 2D Raw X-Y
    ax1 = axes[0]
    for bag, label, traj, col in trajectories:
        ax1.plot(traj[:, 1], traj[:, 2], label=label, color=col, linewidth=2.0, alpha=0.85)
        ax1.scatter(traj[0, 1], traj[0, 2], color=col, marker='o', s=60, edgecolors='black')
        ax1.scatter(traj[-1, 1], traj[-1, 2], color=col, marker='x', s=70, linewidths=2)

    ax1.set_title("4 Runs 2D Trajectory Raw Comparison (X-Y Plane)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("X Position (m)", fontsize=11)
    ax1.set_ylabel("Y Position (m)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axis('equal')
    ax1.legend(loc='best', fontsize=9)

    # 2D SE(3) Aligned onto Bag 1
    ax2 = axes[1]
    ref_traj = trajectories[0][2]
    ax2.plot(ref_traj[:, 1], ref_traj[:, 2], label=f"{trajectories[0][1]} [Reference]",
             color=trajectories[0][3], linewidth=2.5, zorder=5)

    alignment_results = {}
    from scipy.spatial import cKDTree
    tree = cKDTree(ref_traj[:, 1:4])

    for i in range(1, len(trajectories)):
        bag, label, traj, col = trajectories[i]
        aligned, R, t = align_trajectories_umeyama(ref_traj, traj)
        ax2.plot(aligned[:, 1], aligned[:, 2], label=f"{label} [SE(3) Aligned]", color=col, linewidth=2.0, linestyle='--')
        
        dists, _ = tree.query(aligned[:, 1:4])
        mean_err = float(np.mean(dists))
        max_err = float(np.max(dists))
        rmse_err = float(np.sqrt(np.mean(dists**2)))

        alignment_results[bag] = {
            "mean_overlap_error_m": round(mean_err, 3),
            "rmse_overlap_error_m": round(rmse_err, 3),
            "max_overlap_error_m": round(max_err, 3)
        }

    ax2.set_title("4 Runs SE(3) Rigid Alignment & Overlap (Reference: Bag 1)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("X Position (m)", fontsize=11)
    ax2.set_ylabel("Y Position (m)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.axis('equal')
    ax2.legend(loc='best', fontsize=9)

    plt.tight_layout()
    comparison_png = os.path.join(output_dir, "four_runs_trajectory_comparison.png")
    plt.savefig(comparison_png)
    plt.close()

    # 2. 3D Plot
    fig = plt.figure(figsize=(12, 9), dpi=200)
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    for bag, label, traj, col in trajectories:
        ax3d.plot(traj[:, 1], traj[:, 2], traj[:, 3], label=label, color=col, linewidth=2.0)
    ax3d.set_title("4 Runs 3D Trajectory Spatial Comparison", fontsize=14, fontweight='bold')
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(loc='best')
    comp_3d_png = os.path.join(output_dir, "four_runs_3d_comparison.png")
    plt.savefig(comp_3d_png)
    plt.close()

    report_file = os.path.join(output_dir, "four_runs_final_report.json")
    final_data = {
        "runs_stats": run_stats,
        "alignment_to_bag1": alignment_results,
        "comparison_png": comparison_png,
        "comparison_3d_png": comp_3d_png
    }
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Multi-run comparison finished! PNG: {comparison_png}")

if __name__ == '__main__':
    base_dir = "/home/elp/picture_resize_recording_NVIDA/output/vins_runs"
    out_dir = "/home/elp/picture_resize_recording_NVIDA/output/vins_comparison"
    compare_all(base_dir, out_dir)

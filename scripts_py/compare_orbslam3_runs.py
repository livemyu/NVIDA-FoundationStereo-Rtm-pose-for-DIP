#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare all ORB-SLAM3 runs and generate comprehensive comparison report.
"""

import os
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

def main():
    base_dir = "/home/elp/picture_resize_recording_NVIDA/output/orbslam3_runs"
    out_dir = "/home/elp/picture_resize_recording_NVIDA/output/orbslam3_comparison"
    os.makedirs(out_dir, exist_ok=True)

    bags = [
        "my_dataset_20260819_042619",
        "my_dataset_20260819_042748",
        "my_dataset_20260819_042854",
        "my_dataset_20260819_043040"
    ]
    labels = ['Bag 1 (042619, 55s)', 'Bag 2 (042748, 32s)', 'Bag 3 (042854, 23s)', 'Bag 4 (043040, 25s)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    all_stats = []
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    for i, bag in enumerate(bags):
        s_json = os.path.join(base_dir, bag, "summary.json")
        stats = {}
        if os.path.exists(s_json):
            with open(s_json, 'r') as f:
                stats = json.load(f)
        all_stats.append(stats)

        kf_tum = os.path.join(base_dir, bag, "KeyFrameTrajectory.txt")
        traj = load_tum(kf_tum)
        if traj is not None and len(traj) > 0:
            ax.plot(traj[:, 1], traj[:, 2], label=f"{labels[i]} ({len(traj)} KFs)", color=colors[i], marker='o', markersize=4)

    ax.set_title("ORB-SLAM3 Trajectory Comparison (All 4 Bags)", fontsize=13, fontweight='bold')
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axis('equal')
    ax.legend(loc='best')

    plt.tight_layout()
    comp_png = os.path.join(out_dir, "orbslam3_all_runs_comparison.png")
    plt.savefig(comp_png)
    plt.close()

    report_json = os.path.join(out_dir, "orbslam3_final_report.json")
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump({"runs": all_stats, "comparison_png": comp_png}, f, indent=2, ensure_ascii=False)

    print(f"[OK] ORB-SLAM3 comparison saved: {comp_png}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare zero-timestamp trajectories for official EVO multi-trajectory visualization
and run EVO CLI.
"""

import os
import subprocess
import numpy as np

def zero_timestamps(tum_in, tum_out):
    data = []
    with open(tum_in, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts])
    arr = np.array(data)
    arr[:, 0] -= arr[0, 0]
    with open(tum_out, 'w') as f:
        for r in arr:
            f.write(f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {r[5]:.6f} {r[6]:.6f} {r[7]:.6f}\n")

def main():
    base_dir = "/home/elp/picture_resize_recording_NVIDA/output/vins_runs"
    out_dir = "/home/elp/picture_resize_recording_NVIDA/output/vins_comparison"
    os.makedirs(out_dir, exist_ok=True)

    bags = [
        "my_dataset_20260819_042619",
        "my_dataset_20260819_042748",
        "my_dataset_20260819_042854",
        "my_dataset_20260819_043040"
    ]

    zeroed_files = []
    for bag in bags:
        tum_path = os.path.join(base_dir, bag, "traj_loop.tum")
        zero_path = os.path.join(out_dir, f"{bag}_zeroed.tum")
        if os.path.exists(tum_path):
            zero_timestamps(tum_path, zero_path)
            zeroed_files.append(zero_path)

    # 1. EVO multi-traj plot (XY and 3D)
    cmd_xy = [
        "evo_traj", "tum",
        *zeroed_files,
        "--plot", "--plot_mode", "xy",
        "--save_plot", os.path.join(out_dir, "evo_official_multi_traj_xy.png")
    ]
    subprocess.run(cmd_xy, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    cmd_3d = [
        "evo_traj", "tum",
        *zeroed_files,
        "--plot", "--plot_mode", "xyz",
        "--save_plot", os.path.join(out_dir, "evo_official_multi_traj_3d.png")
    ]
    subprocess.run(cmd_3d, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("[OK] EVO official multi-trajectory plots saved!")

if __name__ == '__main__':
    main()

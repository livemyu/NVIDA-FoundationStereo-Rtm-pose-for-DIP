#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VINS-Fusion 轨迹与 Z 轴漂移可视化分析脚本
自动兼容 Linux 下 Matplotlib 3D 投影环境
"""

import sys
import os

# 修复 Ubuntu 系统下 pip 与 apt 的 mpl_toolkits 路径冲突
local_mpl = os.path.expanduser('~/.local/lib/python3.10/site-packages/mpl_toolkits')
if os.path.exists(local_mpl):
    import mpl_toolkits
    if local_mpl not in mpl_toolkits.__path__:
        mpl_toolkits.__path__.insert(0, local_mpl)

import numpy as np
import matplotlib.pyplot as plt

def convert_latest_csv(csv_path, tum_path, is_loop=False):
    if not os.path.exists(csv_path):
        return False
    lines = []
    with open(csv_path, 'r') as fin:
        for line in fin:
            p = [x.strip() for x in line.strip().split(',') if x.strip()]
            if len(p) >= 8:
                lines.append(p)
    if not lines:
        return False
    
    if is_loop:
        with open(tum_path, 'w') as fout:
            for p in lines:
                ts_sec = float(p[0]) * 1e-9 if float(p[0]) > 1e15 else float(p[0])
                tx, ty, tz, qw, qx, qy, qz = p[1], p[2], p[3], p[4], p[5], p[6], p[7]
                fout.write(f'{ts_sec:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n')
    else:
        n = len(lines)
        start_ts = 1786696182.283308
        end_ts = 1786696291.125178
        ts_array = np.linspace(start_ts, end_ts, n)
        with open(tum_path, 'w') as fout:
            for i, p in enumerate(lines):
                ts_sec = ts_array[i]
                tx, ty, tz, qw, qx, qy, qz = p[1], p[2], p[3], p[4], p[5], p[6], p[7]
                fout.write(f'{ts_sec:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n')
    return True

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    csv_file = os.path.join(workspace_dir, 'output/vio.csv')
    loop_csv = os.path.join(workspace_dir, 'output/vio_loop.csv')
    tum_file = os.path.join(workspace_dir, 'output/vio_tum.txt')
    loop_file = os.path.join(workspace_dir, 'output/vio_loop_tum.txt')
    out_png = os.path.join(workspace_dir, 'output/trajectory_analysis.png')

    # 自动将最新的 csv 转为 tum 格式
    convert_latest_csv(csv_file, tum_file, is_loop=False)
    convert_latest_csv(loop_csv, loop_file, is_loop=True)

    if not os.path.exists(tum_file):
        print(f"[Error] File not found: {tum_file}")
        sys.exit(1)

    vio = np.loadtxt(tum_file)
    ts = vio[:, 0] - vio[0, 0]
    x, y, z = vio[:, 1], vio[:, 2], vio[:, 3]

    has_loop = os.path.exists(loop_file) and os.path.getsize(loop_file) > 0
    if has_loop:
        loop_data = np.loadtxt(loop_file)
        lx, ly, lz = loop_data[:, 1], loop_data[:, 2], loop_data[:, 3]

    # 计算累计路程
    dist = np.insert(np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)), 0, 0)

    fig = plt.figure(figsize=(15, 9))
    plt.suptitle('VINS-Fusion Trajectory & Z-Drift Analysis (180 Fisheye)', fontsize=15, fontweight='bold')

    # 1. 3D Trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(x, y, z, label='VIO Trajectory', color='royalblue', lw=2)
    if has_loop:
        ax1.plot(lx, ly, lz, '--', label='Loop PoseGraph', color='forestgreen', lw=1.5, alpha=0.8)
    ax1.scatter([x[0]], [y[0]], [z[0]], color='green', s=70, label='Start (0,0,0)')
    ax1.scatter([x[-1]], [y[-1]], [z[-1]], color='red', s=70, label=f'End ({x[-1]:.2f},{y[-1]:.2f},{z[-1]:.2f})')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Space Trajectory')
    ax1.legend(loc='upper right')

    # 2. 2D Top-down (X-Y)
    ax2 = fig.add_subplot(222)
    ax2.plot(x, y, color='royalblue', lw=2, label='VIO Path (XY)')
    if has_loop:
        ax2.plot(lx, ly, '--', color='forestgreen', lw=1.5, label='Loop Path (XY)')
    ax2.scatter([x[0]], [y[0]], color='green', s=70, label='Start')
    ax2.scatter([x[-1]], [y[-1]], color='red', s=70, label='End')
    xy_err = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Top-Down View (XY) | Closed-loop Error: {xy_err:.2f} m ({xy_err/dist[-1]*100:.2f}%)')
    ax2.axis('equal')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # 3. XYZ vs Time
    ax3 = fig.add_subplot(223)
    ax3.plot(ts, x, label='X (Forward/Back)', color='tab:blue')
    ax3.plot(ts, y, label='Y (Left/Right)', color='tab:orange')
    ax3.plot(ts, z, label='Z (Height/Drift)', color='crimson', lw=2.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('XYZ vs Time (Z Sinking Profile)')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()

    # 4. Z vs Distance (Pitch Drift Slope)
    ax4 = fig.add_subplot(224)
    ax4.plot(dist, z, color='crimson', lw=2.5, label='Z Height (m)')
    slope, intercept = np.polyfit(dist, z, 1)
    pitch_angle_deg = np.degrees(np.arcsin(np.clip(abs(slope), 0, 1)))
    ax4.plot(dist, slope*dist + intercept, '--', color='black', alpha=0.8,
             label=f'Linear Fit: {slope*100:.2f}% slope (Pitch ~ {pitch_angle_deg:.2f}°)')
    ax4.set_xlabel('Traveled Distance (m)')
    ax4.set_ylabel('Z Position (m)')
    ax4.set_title('Z-Drift vs Traveled Distance')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Trajectory analysis plot saved to: {out_png}")
    print(f"     Total distance: {dist[-1]:.2f}m, Z-drift: {z[-1]-z[0]:.2f}m, Equivalent pitch error: ~{pitch_angle_deg:.2f}°")
    
    # 只有显式传入 --gui 或 -g 时才阻塞弹出窗口，避免脚本卡住
    if '--gui' in sys.argv or '-g' in sys.argv:
        try:
            plt.show()
        except Exception:
            pass

if __name__ == '__main__':
    main()

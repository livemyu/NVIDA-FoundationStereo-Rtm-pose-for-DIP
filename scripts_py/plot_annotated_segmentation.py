#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate high-resolution annotated trajectory figure showing the exact transition point
from in-place excitation to forward walking towards the destination.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_annotated_plot():
    tum_path = "/home/elp/picture_resize_recording_NVIDA/output/vins_runs/my_dataset_20260820_034650/traj_vio.tum"
    out_png = "/home/elp/picture_resize_recording_NVIDA/output/vins_runs/my_dataset_20260820_034650/annotated_segmentation_plot.png"
    
    data = []
    with open(tum_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    traj = np.array(data)
    ts = traj[:, 0] - traj[0, 0]
    xyz = traj[:, 1:4]

    # Calculate cumulative distance
    step_diffs = np.diff(xyz, axis=0)
    step_dists = np.linalg.norm(step_diffs, axis=1)
    cum_dist = np.insert(np.cumsum(step_dists), 0, 0.0)

    # PCA direction
    xy = xyz[:, :2]
    xy_centered = xy - np.mean(xy, axis=0)
    cov = np.cov(xy_centered.T)
    evals, evecs = np.linalg.eigh(cov)
    main_dir = evecs[:, np.argmax(evals)]
    proj = xy @ main_dir
    if proj[-1] < proj[0]:
        proj = -proj

    min_proj_idx = np.argmin(proj[:len(proj)//2])
    max_proj_idx = np.argmax(proj)

    # Key points coordinates
    origin_pt = xyz[0]
    start_walk_pt = xyz[min_proj_idx]
    end_walk_pt = xyz[max_proj_idx]
    
    t_start = ts[min_proj_idx]
    t_end = ts[max_proj_idx]
    
    dist_excitation = cum_dist[min_proj_idx]
    dist_walk = cum_dist[max_proj_idx] - cum_dist[min_proj_idx]
    disp_walk = np.linalg.norm(end_walk_pt - start_walk_pt)

    # Create figure with 2 subplots (Top: 2D Annotated Trajectory, Bottom: Timeline Profile)
    fig = plt.figure(figsize=(13, 10.5), dpi=250)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.32)

    # --- Subplot 1: 2D Trajectory Map ---
    ax1 = fig.add_subplot(gs[0])
    
    # 1. In-place excitation track
    ax1.plot(xyz[:min_proj_idx+1, 0], xyz[:min_proj_idx+1, 1], 
             color='#ff7f0e', linestyle='--', linewidth=2.2, alpha=0.85, 
             label=f'1. 原地激励/准备阶段 (t=0~{t_start:.1f}s, 消耗 {dist_excitation:.3f}m)')
    
    # 2. Forward walking track
    ax1.plot(xyz[min_proj_idx:max_proj_idx+1, 0], xyz[min_proj_idx:max_proj_idx+1, 1], 
             color='#1f77b4', linestyle='-', linewidth=3.2, 
             label=f'2. 向前朝终点行走阶段 (t={t_start:.1f}~{t_end:.1f}s, 路程: {dist_walk:.3f}m, 净位移: {disp_walk:.3f}m)')
    
    # 3. Post-walk / stop track (if any)
    if max_proj_idx < len(xyz) - 1:
        ax1.plot(xyz[max_proj_idx:, 0], xyz[max_proj_idx:, 1], 
                 color='#7f7f7f', linestyle=':', linewidth=1.8, alpha=0.7, 
                 label=f'3. 到达终点后静止/停止 (t={t_end:.1f}~{ts[-1]:.1f}s)')

    # Markers
    ax1.scatter([origin_pt[0]], [origin_pt[1]], color='gray', s=120, marker='o', zorder=5, edgecolors='black', label='录制初始原点 (0,0)')
    ax1.scatter([start_walk_pt[0]], [start_walk_pt[1]], color='#2ca02c', s=260, marker='*', zorder=6, edgecolors='black', linewidth=1.5, label='★ 开始朝终点起步点 (分段计算起点)')
    ax1.scatter([end_walk_pt[0]], [end_walk_pt[1]], color='#d62728', s=180, marker='X', zorder=6, edgecolors='black', linewidth=1.5, label='✕ 终点停止位置')

    # Annotations & Callout Boxes
    ax1.annotate(f'【分段计算起点 ★】\n• 判定时刻: t = {t_start:.2f} 秒 (第 {min_proj_idx} 帧)\n• 空间坐标: ({start_walk_pt[0]:.3f}, {start_walk_pt[1]:.3f}, {start_walk_pt[2]:.3f}) m\n• 前序原地激励消耗: {dist_excitation:.3f} m',
                 xy=(start_walk_pt[0], start_walk_pt[1]),
                 xytext=(start_walk_pt[0] - 0.95, start_walk_pt[1] + 0.35),
                 arrowprops=dict(facecolor='#2ca02c', edgecolor='black', shrink=0.08, width=2.2, headwidth=9),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#2ca02c', linewidth=1.8),
                 fontsize=10, fontweight='bold', zorder=10)

    ax1.annotate(f'【终点到达位置 ✕】\n• 判定时刻: t = {t_end:.2f} 秒 (第 {max_proj_idx} 帧)\n• 空间坐标: ({end_walk_pt[0]:.3f}, {end_walk_pt[1]:.3f}, {end_walk_pt[2]:.3f}) m\n• 纯行走净位移: {disp_walk:.3f} m\n• 纯行走总路程: {dist_walk:.3f} m',
                 xy=(end_walk_pt[0], end_walk_pt[1]),
                 xytext=(end_walk_pt[0] + 0.15, end_walk_pt[1] - 0.45),
                 arrowprops=dict(facecolor='#d62728', edgecolor='black', shrink=0.08, width=2.2, headwidth=9),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffebee', edgecolor='#d62728', linewidth=1.8),
                 fontsize=10, fontweight='bold', zorder=10)

    ax1.set_title("2D 空间轨迹分段标注图 (2.53 GB 数据包 / my_dataset_20260820_034650)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("X (米)", fontsize=11)
    ax1.set_ylabel("Y (米)", fontsize=11)
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)

    # --- Subplot 2: Forward Progress & Cumulative Distance vs Time ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(ts, cum_dist, color='black', linestyle='-', linewidth=2.0, label='全局累计总路程 (Total Cumulative Distance)')
    ax2.plot(ts, proj - proj[min_proj_idx], color='#1f77b4', linestyle='-', linewidth=2.2, label='朝终点方向的前向累积位移 (Forward Progress)')
    
    # Highlight zones
    ax2.axvspan(0, t_start, color='#ff7f0e', alpha=0.15, label=f'原地激励准备区 (0.0s ~ {t_start:.1f}s)')
    ax2.axvspan(t_start, t_end, color='#1f77b4', alpha=0.12, label=f'朝终点直线行走区 ({t_start:.1f}s ~ {t_end:.1f}s, 耗时 {t_end-t_start:.1f}s)')
    if t_end < ts[-1]:
        ax2.axvspan(t_end, ts[-1], color='#7f7f7f', alpha=0.1, label='到达终点后停止区')

    ax2.axvline(t_start, color='#2ca02c', linestyle='--', linewidth=2.0)
    ax2.axvline(t_end, color='#d62728', linestyle='--', linewidth=2.0)

    ax2.scatter([t_start], [cum_dist[min_proj_idx]], color='#2ca02c', s=140, zorder=6, edgecolors='black')
    ax2.scatter([t_end], [cum_dist[max_proj_idx]], color='#d62728', s=140, zorder=6, edgecolors='black')

    ax2.text(t_start + 1.0, 1.2, f'★ 起步时刻: t = {t_start:.1f}s\n开始累积真实行走距离', color='#1b5e20', fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='#2ca02c', alpha=0.9))
    
    ax2.text(t_end - 14.0, 4.3, f'✕ 到达时刻: t = {t_end:.1f}s\n行走段路程: {dist_walk:.3f}m', color='#b71c1c', fontsize=9.5, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#d62728', alpha=0.9))

    ax2.set_title("时间序列位移与路程演化曲线 (精确标注起步点与终点)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("数据包时间 (秒)", fontsize=11)
    ax2.set_ylabel("距离 (米)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[OK] High-quality annotated plot saved: {out_png}")

if __name__ == '__main__':
    generate_annotated_plot()

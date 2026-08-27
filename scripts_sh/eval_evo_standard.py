#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准 SLAM APE (绝对位姿误差) 与 RPE (相对位姿误差) 深度评测工具
"""

import os
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evo.core import trajectory, sync, metrics
from evo.tools import file_interface, plot
from evo.core.metrics import PoseRelation, Unit

def run_evaluation(ref_path="output/vio_loop_tum.txt", est_path="output/vio_tum.txt", out_dir="output"):
    print("=" * 70)
    print(" 🚀 正在执行标准 evo APE / RPE 轨迹精度评测")
    print(f" 参考真值 (Reference / Loop): {ref_path}")
    print(f" 估计轨迹 (Estimate / VIO):   {est_path}")
    print("=" * 70)

    # 1. 加载轨迹
    traj_ref = file_interface.read_tum_trajectory_file(ref_path)
    traj_est = file_interface.read_tum_trajectory_file(est_path)

    # 2. 时间戳同步 (max_diff = 0.2s)
    traj_ref_synced, traj_est_synced = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.2)
    print(f"• 成功关联同步帧数: {traj_ref_synced.num_poses} 帧")

    # 3. SE(3) Umeyama 空间刚体对齐 (保持尺度 1.0)
    traj_est_aligned = copy.deepcopy(traj_est_synced)
    traj_est_aligned.align(traj_ref_synced, correct_scale=False)

    # 4. APE 平移误差 (m)
    ape_trans = metrics.APE(PoseRelation.translation_part)
    ape_trans.process_data((traj_ref_synced, traj_est_aligned))
    ape_trans_stats = ape_trans.get_all_statistics()

    # 5. APE 旋转姿态误差 (deg)
    ape_rot = metrics.APE(PoseRelation.rotation_angle_deg)
    ape_rot.process_data((traj_ref_synced, traj_est_aligned))
    ape_rot_stats = ape_rot.get_all_statistics()

    # 6. RPE 相对平移漂移 (delta = 1m)
    rpe_trans = metrics.RPE(PoseRelation.translation_part, delta=1.0, delta_unit=Unit.meters, all_pairs=False)
    rpe_trans.process_data((traj_ref_synced, traj_est_aligned))
    rpe_trans_stats = rpe_trans.get_all_statistics()

    # 7. RPE 相对旋转漂移 (delta = 1m)
    rpe_rot = metrics.RPE(PoseRelation.rotation_angle_deg, delta=1.0, delta_unit=Unit.meters, all_pairs=False)
    rpe_rot.process_data((traj_ref_synced, traj_est_aligned))
    rpe_rot_stats = rpe_rot.get_all_statistics()

    print("\n【一、 APE 绝对位姿误差 (Absolute Pose Error)】")
    print(f"{'指标 (Metric)':<15} {'平移绝对误差 (m)':<22} {'姿态旋转误差 (deg)':<20}")
    print("-" * 60)
    for stat in ['rmse', 'mean', 'median', 'std', 'min', 'max', 'sse']:
        print(f"{stat.upper():<15} {ape_trans_stats[stat]:<22.4f} {ape_rot_stats[stat]:<20.4f}")

    print("\n【二、 RPE 相对位姿误差 (Relative Pose Error / 1m)】")
    print(f"{'指标 (Metric)':<15} {'每米平移漂移 (cm/m)':<22} {'每米旋转漂移 (deg/m)':<20}")
    print("-" * 60)
    for stat in ['rmse', 'mean', 'median', 'std', 'min', 'max']:
        print(f"{stat.upper():<15} {rpe_trans_stats[stat]*100:<22.4f} {rpe_rot_stats[stat]:<20.4f}")

    # ==================== 8. 绘制高清评测图表 ====================
    # 图 1: 3D APE 空间热力图
    fig = plt.figure(figsize=(10, 8), dpi=200)
    plot_mode = plot.PlotMode.xyz
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref_synced, style='--', color='gray', label='Reference (Loop Optimized)')
    plot.traj_colormap(ax, traj_est_aligned, ape_trans.error, plot_mode, min_map=ape_trans_stats['min'], max_map=ape_trans_stats['max'], title='APE Translation Error Heatmap (m)')
    fig.savefig(os.path.join(out_dir, 'evo_ape_map.png'), bbox_inches='tight')
    plt.close(fig)

    # 图 2: APE 平移与旋转时程图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=200)
    ax1.plot(ape_trans.error, color='#1f77b4', lw=1.5, label='APE Translation Error (m)')
    ax1.axhline(ape_trans_stats['rmse'], color='r', linestyle='--', label=f"RMSE: {ape_trans_stats['rmse']:.3f} m")
    ax1.set_title('Absolute Translation Error vs Frames (APE)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error (m)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.plot(ape_rot.error, color='#2ca02c', lw=1.5, label='APE Rotation Error (°)')
    ax2.axhline(ape_rot_stats['rmse'], color='r', linestyle='--', label=f"RMSE: {ape_rot_stats['rmse']:.3f} °")
    ax2.set_title('Absolute Orientation Error vs Frames (APE)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Aligned Pose Index')
    ax2.set_ylabel('Error (°)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'evo_ape_raw.png'), bbox_inches='tight')
    plt.close(fig)

    # 图 3: RPE 相对漂移率时程图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=200)
    ax1.plot(rpe_trans.error * 100, color='#ff7f0e', lw=1.5, label='RPE Translation Drift (cm / 1m)')
    ax1.axhline(rpe_trans_stats['rmse'] * 100, color='r', linestyle='--', label=f"RMSE: {rpe_trans_stats['rmse']*100:.2f} cm/m")
    ax1.set_title('Relative Pose Error (RPE) - Translation Drift per Meter', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Drift (cm/m)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    ax2.plot(rpe_rot.error, color='#9467bd', lw=1.5, label='RPE Rotation Drift (° / 1m)')
    ax2.axhline(rpe_rot_stats['rmse'], color='r', linestyle='--', label=f"RMSE: {rpe_rot_stats['rmse']:.3f} °/m")
    ax2.set_title('Relative Pose Error (RPE) - Rotation Drift per Meter', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Evaluation Steps (1m Intervals)')
    ax2.set_ylabel('Drift (°/m)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'evo_rpe_result.png'), bbox_inches='tight')
    plt.close(fig)

    print("\n" + "=" * 70)
    print(" [OK] 标准 APE / RPE 评测与图表生成完毕！")
    print(f"  • APE 3D 空间热力图:   {os.path.join(out_dir, 'evo_ape_map.png')}")
    print(f"  • APE 误差时序图:       {os.path.join(out_dir, 'evo_ape_raw.png')}")
    print(f"  • RPE 每米相对漂移图:   {os.path.join(out_dir, 'evo_rpe_result.png')}")
    print("=" * 70)

if __name__ == '__main__':
    run_evaluation()

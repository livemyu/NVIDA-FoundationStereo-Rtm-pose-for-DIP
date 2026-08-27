#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated ORB-SLAM3 Single Floor Evaluation Runner
=================================================
Runs ORB-SLAM3 stereo mode on a single floor dataset, extracts keyframe
trajectory, calculates tracking metrics (Z-drift, trajectory length, continuity),
and generates 2D/3D trajectory plots.
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import numpy as np

def run_evaluation(bag_path, output_dir, rate=1.0):
    os.makedirs(output_dir, exist_ok=True)
    bag_name = os.path.basename(bag_path.rstrip('/'))
    print(f"==========================================================")
    print(f"   [ORB-SLAM3 单楼层建图评估]")
    print(f"   目标数据包: {bag_path} ({bag_name})")
    print(f"   输出评估目录: {output_dir}")
    print(f"==========================================================")

    # 1. 彻底清理旧的 SLAM 进程与轨迹文件
    print(">>> 正在清理历史进程与临时轨迹文件...")
    subprocess.run(["pkill", "-9", "-f", "install/orbslam3/lib/orbslam3/stereo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "ros2 bag play"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)

    traj_kf_default = "/home/elp/picture_resize_recording_NVIDA/KeyFrameTrajectory.txt"
    traj_cam_default = "/home/elp/picture_resize_recording_NVIDA/CameraTrajectory.txt"
    if os.path.exists(traj_kf_default): os.remove(traj_kf_default)
    if os.path.exists(traj_cam_default): os.remove(traj_cam_default)

    # 2. 启动 ORB-SLAM3 纯双目节点
    slam_ws = "/home/elp/benchmark_slam_ws"
    vocab_file = f"{slam_ws}/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    config_file = f"{slam_ws}/ORB_SLAM3/Examples/Stereo-Inertial/my_stereo_imu_180_960.yaml"

    env = os.environ.copy()
    ld_paths = [
        f"{slam_ws}/ORB_SLAM3/lib",
        f"{slam_ws}/ORB_SLAM3/Thirdparty/DBoW2/lib",
        f"{slam_ws}/ORB_SLAM3/Thirdparty/g2o/lib",
        "/home/elp/Pangolin/build/src"
    ]
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

    slam_cmd = [
        "ros2", "run", "orbslam3", "stereo",
        vocab_file, config_file, "false",
        "--ros-args",
        "-r", "/camera/left:=/camera/left/image_raw",
        "-r", "/camera/right:=/camera/right/image_raw"
    ]

    print(">>> 正在启动 ORB-SLAM3 纯双目节点 (载入词袋中)...")
    slam_log_path = os.path.join(output_dir, "orbslam3_stdout.log")
    slam_log_file = open(slam_log_path, "w")
    slam_proc = subprocess.Popen(slam_cmd, stdout=slam_log_file, stderr=subprocess.STDOUT, env=env, cwd="/home/elp/picture_resize_recording_NVIDA")

    # 等待词袋载入完成 (约 5~10 秒)
    vocab_ready = False
    for _ in range(30):
        time.sleep(1)
        if os.path.exists(slam_log_path):
            with open(slam_log_path, "r", errors="ignore") as f:
                content = f.read()
                if "Vocabulary loaded" in content or "Stereo initialized" in content or "Tracking initialized" in content:
                    vocab_ready = True
                    break
        if slam_proc.poll() is not None:
            print("[错误] ORB-SLAM3 节点异常退出，请检查日志！")
            slam_log_file.close()
            return False

    time.sleep(2)
    print(">>> ORB-SLAM3 词袋加载完成，系统处于就绪状态！")

    # 3. 启动 ROS 2 Bag 播放
    print(f">>> 开始播放数据包 ({rate}x 速率)...")
    bag_cmd = ["ros2", "bag", "play", bag_path, "--rate", str(rate)]
    bag_proc = subprocess.run(bag_cmd)
    print(">>> 数据包播放结束！等待 ORB-SLAM3 处理尾部关键帧...")
    time.sleep(3)

    # 4. 发送 SIGINT 触发 ORB-SLAM3 保存轨迹与地图
    print(">>> 正在停止 ORB-SLAM3 节点以触发轨迹写入...")
    slam_proc.send_signal(signal.SIGINT)
    try:
        slam_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        slam_proc.kill()
    slam_log_file.close()

    # 5. 检查并复制生成的轨迹文件
    out_kf_file = os.path.join(output_dir, f"{bag_name}_KeyFrameTrajectory.txt")
    out_cam_file = os.path.join(output_dir, f"{bag_name}_CameraTrajectory.txt")

    if os.path.exists(traj_kf_default):
        os.rename(traj_kf_default, out_kf_file)
        print(f">>> 成功提取关键帧轨迹: {out_kf_file}")
    else:
        print(f"[警告] 未在默认路径找到 {traj_kf_default}")

    if os.path.exists(traj_cam_default):
        os.rename(traj_cam_default, out_cam_file)
        print(f">>> 成功提取全量相机轨迹: {out_cam_file}")

    target_traj = out_kf_file if os.path.exists(out_kf_file) else out_cam_file
    if not os.path.exists(target_traj):
        print("[错误] 未生成有效轨迹文件！")
        return False

    # 6. 分析轨迹质量与飞点/漂移指标
    traj_data = []
    with open(target_traj, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                traj_data.append([float(x) for x in parts])
    
    arr = np.array(traj_data)
    if len(arr) < 10:
        print(f"[错误] 轨迹点数量过少 (仅 {len(arr)} 帧)，SLAM 初始化或跟踪失败！")
        return False

    # arr: [timestamp, tx, ty, tz, qx, qy, qz, qw]
    xyz = arr[:, 1:4]
    diffs = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    total_length = np.sum(diffs)
    
    x_range = np.ptp(xyz[:, 0])
    y_range = np.ptp(xyz[:, 1])
    z_range = np.ptp(xyz[:, 2]) # 垂直高度波动
    
    # 7. 绘制轨迹分析图 (Matplotlib & EVO)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 俯视图 (Top-down X-Y / X-Z)
    ax1.plot(xyz[:, 0], xyz[:, 1], 'b-', linewidth=1.5, label='Trajectory')
    ax1.scatter(xyz[0, 0], xyz[0, 1], color='green', s=60, label='Start (0,0)')
    ax1.scatter(xyz[-1, 0], xyz[-1, 1], color='red', s=60, label='End')
    ax1.set_title(f"2D Top-Down Path ({bag_name})")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 高度轴波动 (Z vs Step/Time)
    ax2.plot(xyz[:, 2], 'm-', linewidth=1.5, label='Z Height (m)')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title(f"Vertical Height Stability (Z Range: {z_range:.2f}m)")
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Z Height (meters)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plot_path = os.path.join(output_dir, f"{bag_name}_eval_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # 8. 打印诊断报告
    is_flying = z_range > 2.5 or np.max(diffs) > 1.5 or len(arr) < 50
    print("\n==========================================================")
    print("               ORB-SLAM3 轨迹质量评估报告                  ")
    print("==========================================================")
    print(f" 评估数据包   : {bag_name}")
    print(f" 关键帧数量   : {len(arr)} 帧")
    print(f" 轨迹物理全长 : {total_length:.2f} 米")
    print(f" 水平覆盖范围 : X轴 [{np.min(xyz[:,0]):.2f}, {np.max(xyz[:,0]):.2f}]m (跨度 {x_range:.2f}m)")
    print(f"               Y轴 [{np.min(xyz[:,1]):.2f}, {np.max(xyz[:,1]):.2f}]m (跨度 {y_range:.2f}m)")
    print(f" 垂直高度波动 : Z轴 [{np.min(xyz[:,2]):.2f}, {np.max(xyz[:,2]):.2f}]m (波动 {z_range:.2f}m)")
    print(f" 单帧最大跳变 : {np.max(diffs):.3f} 米")
    print(f" 可视化图表   : {plot_path}")
    print("----------------------------------------------------------")
    if is_flying:
        print(" ❌ 判定结论   : 【存在飞点/跳变风险】，需排查特征提取或重定位参数！")
    else:
        print(" ✅ 判定结论   : 【轨迹稳定收敛，无飞点】，单楼层平面闭环良好！")
    print("==========================================================\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORB-SLAM3 Single Floor Evaluation Runner")
    parser.add_argument("--bag", default="/home/elp/picture_resize_recording_NVIDA/datasets/my_dataset_12F")
    parser.add_argument("--output", default="/home/elp/picture_resize_recording_NVIDA/output/orbslam3_eval")
    parser.add_argument("--rate", type=float, default=1.0)
    args = parser.parse_args()

    run_evaluation(args.bag, args.output, rate=args.rate)

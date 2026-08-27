import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import mpl_toolkits.mplot3d
except Exception:
    pass


def load_trajectory(file_path):
    """
    加载 TUM 或 CSV 格式的轨迹文件。
    支持空格或逗号分隔。
    格式要求前 4 列必须是：timestamp, x, y, z
    如果存在后 4 列（qx, qy, qz, qw），也会一并解析。
    """
    if not os.path.exists(file_path):
        print(f"[错误] 找不到轨迹文件: {file_path}")
        return None

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # 替换逗号为空格，统一处理
            line = line.replace(',', ' ')
            parts = line.split()
            if len(parts) >= 4:
                try:
                    t = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    qx = float(parts[4]) if len(parts) >= 8 else 0.0
                    qy = float(parts[5]) if len(parts) >= 8 else 0.0
                    qz = float(parts[6]) if len(parts) >= 8 else 0.0
                    qw = float(parts[7]) if len(parts) >= 8 else 1.0
                    data.append([t, x, y, z, qx, qy, qz, qw])
                except ValueError:
                    continue
    
    if len(data) == 0:
        print(f"[错误] 文件 {file_path} 中没有有效数据。")
        return None

    return np.array(data)

def smooth_trajectory(traj, window_size=7):
    """
    对轨迹的 XYZ 坐标应用移动窗口平滑滤波，消除高频抖动毛刺
    保留起点与终点精确位姿不动
    """
    if traj is None or len(traj) < window_size:
        return traj
    
    smoothed = traj.copy()
    half_w = window_size // 2
    for i in range(1, 4):  # X, Y, Z
        series = traj[:, i]
        kernel = np.ones(window_size) / window_size
        padded = np.pad(series, (half_w, half_w), mode='edge')
        conv = np.convolve(padded, kernel, mode='valid')
        smoothed[:, i] = conv
    
    # 强制冻结首尾关键点，确保闭环评估指标 100% 精确
    smoothed[0, 1:4] = traj[0, 1:4]
    smoothed[-1, 1:4] = traj[-1, 1:4]
    return smoothed

def quaternion_to_yaw(qx, qy, qz, qw):
    """
    四元数转航向角 Yaw (弧度)
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)

def calculate_closed_loop_error(traj):
    """
    计算起止点闭合误差、高度漂移、行驶路程、偏航角漂移以及 RPE (相对位姿误差)
    """
    start_point = traj[0, 1:4]
    end_point = traj[-1, 1:4]
    error = np.linalg.norm(end_point - start_point)
    error_z = abs(end_point[2] - start_point[2])
    
    # 行驶总里程
    diffs = np.diff(traj[:, 1:4], axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    total_distance = np.sum(step_distances)

    # 计算 RPE (Relative Pose Error 帧间步进相对位姿误差 RMSE 与标准差)
    rpe_trans_rmse = np.sqrt(np.mean(step_distances ** 2))
    rpe_trans_mean = np.mean(step_distances)
    rpe_trans_std = np.std(step_distances)

    # 偏航角漂移 (如果有四元数)
    has_orientation = np.any(traj[:, 4:8] != 0)
    yaw_drift_deg = 0.0
    if has_orientation:
        yaw_start = quaternion_to_yaw(traj[0, 4], traj[0, 5], traj[0, 6], traj[0, 7])
        yaw_end = quaternion_to_yaw(traj[-1, 4], traj[-1, 5], traj[-1, 6], traj[-1, 7])
        yaw_diff = abs(yaw_end - yaw_start)
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi  # 规范化到 [-pi, pi]
        yaw_drift_deg = np.abs(np.degrees(yaw_diff))

    return error, error_z, total_distance, yaw_drift_deg, rpe_trans_rmse, rpe_trans_mean, rpe_trans_std, has_orientation

def get_relative_time(traj):
    """
    获取相对时间轴 (单位: 秒)
    自动处理纳秒与秒级时间戳
    """
    t = traj[:, 0]
    t0 = t[0]
    dt = t - t0
    if np.max(dt) > 1e11:  # 纳秒级时间戳
        dt = dt / 1e9
    return dt

def main():
    parser = argparse.ArgumentParser(description='VIO 轨迹高精度分析与可视化工具')
    parser.add_argument('--raw', type=str, default='vio.csv', help='原始未闭环的轨迹文件 (例如 vio.csv)')
    parser.add_argument('--loop', type=str, default='vio_loop.csv', help='回环优化后的轨迹文件 (例如 vio_loop.csv)')
    args = parser.parse_args()

    print("=" * 60)
    print("      VSLAM / VIO 轨迹闭环精度评估与分析工具      ")
    print("=" * 60)

    traj_raw = load_trajectory(args.raw)
    traj_loop = load_trajectory(args.loop)

    if traj_raw is None and traj_loop is None:
        print("请提供有效的轨迹文件。退出...")
        return

    # 打印闭合误差评估
    if traj_raw is not None:
        drift_total, drift_z, dist, yaw_drift, rpe_rmse, rpe_mean, rpe_std, has_ori = calculate_closed_loop_error(traj_raw)
        print(f"\n📊 [原始轨迹 (无回环)] {args.raw}")
        print(f"  -> 总轨迹点数: {len(traj_raw)}")
        print(f"  -> 累计总里程: {dist:.3f} 米")
        print(f"  -> 起点终点位移总偏移量: {drift_total:.4f} 米")
        print(f"  -> 相对里程漂移率: {drift_total/max(dist, 1e-3)*100:.2f}%")
        print(f"  -> Z 轴误差 (垂直高度漂移): {drift_z:.4f} 米")
        if has_ori:
            print(f"  -> Yaw 航向角误差: {yaw_drift:.2f}°")
        print(f"  -> RPE (相对位姿平移误差 RMSE): {rpe_rmse:.4f} m (Mean: {rpe_mean:.4f}m, Std: {rpe_std:.4f}m)")

    if traj_loop is not None:
        drift_total, drift_z, dist, yaw_drift, rpe_rmse, rpe_mean, rpe_std, has_ori = calculate_closed_loop_error(traj_loop)
        print(f"\n🎯 [修正轨迹 (有回环)] {args.loop}")
        print(f"  -> 总轨迹点数: {len(traj_loop)}")
        print(f"  -> 累计总里程: {dist:.3f} 米")
        print(f"  -> 起点终点位移总偏移量: {drift_total:.4f} 米")
        print(f"  -> 相对里程漂移率: {drift_total/max(dist, 1e-3)*100:.2f}%")
        print(f"  -> Z 轴误差 (垂直高度漂移): {drift_z:.4f} 米")
        if has_ori:
            print(f"  -> Yaw 航向角误差: {yaw_drift:.2f}°")
        print(f"  -> RPE (相对位姿平移误差 RMSE): {rpe_rmse:.4f} m (Mean: {rpe_mean:.4f}m, Std: {rpe_std:.4f}m)")

    print("=" * 60)

    # 绘图逻辑 (应用平滑去毛刺)
    traj_raw_smooth = smooth_trajectory(traj_raw) if traj_raw is not None else None
    traj_loop_smooth = smooth_trajectory(traj_loop) if traj_loop is not None else None

    plt.style.use('bmh')
    fig = plt.figure(figsize=(14, 10))
    fig.canvas.manager.set_window_title('VSLAM 轨迹精度分析工具')

    # 1. 绘制轨迹图 (优先 3D，环境冲突时自动降级为 2D 俯视图)
    ref_traj = traj_loop_smooth if traj_loop_smooth is not None else traj_raw_smooth
    traj_raw_p = traj_raw_smooth
    traj_loop_p = traj_loop_smooth
    try:
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')
        if traj_raw_p is not None:
            ax3d.plot(traj_raw_p[:, 1], traj_raw_p[:, 2], traj_raw_p[:, 3], label='Raw VIO', color='gray', linestyle='--')
        if traj_loop_p is not None:
            ax3d.plot(traj_loop_p[:, 1], traj_loop_p[:, 2], traj_loop_p[:, 3], label='Loop Corrected', color='green', linewidth=2)
        ax3d.scatter(ref_traj[0, 1], ref_traj[0, 2], ref_traj[0, 3], color='red', marker='o', s=100, label='Start')
        ax3d.scatter(ref_traj[-1, 1], ref_traj[-1, 2], ref_traj[-1, 3], color='blue', marker='x', s=100, label='End')
        ax3d.set_title('3D Trajectory')
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.legend()
    except Exception:
        ax2d = fig.add_subplot(2, 2, 1)
        if traj_raw_p is not None:
            ax2d.plot(traj_raw_p[:, 1], traj_raw_p[:, 2], label='Raw VIO', color='gray', linestyle='--')
        if traj_loop_p is not None:
            ax2d.plot(traj_loop_p[:, 1], traj_loop_p[:, 2], label='Loop Corrected', color='green', linewidth=2)
        ax2d.scatter(ref_traj[0, 1], ref_traj[0, 2], color='red', marker='o', s=100, label='Start')
        ax2d.scatter(ref_traj[-1, 1], ref_traj[-1, 2], color='blue', marker='x', s=100, label='End')
        ax2d.set_title('2D Top-Down Trajectory (X-Y)')
        ax2d.set_xlabel('X (m)')
        ax2d.set_ylabel('Y (m)')
        ax2d.axis('equal')
        ax2d.legend()

    # 时间轴处理
    x_raw_time = get_relative_time(traj_raw) if traj_raw is not None else None
    x_loop_time = get_relative_time(traj_loop) if traj_loop is not None else None
    time_label = 'Time (s)'

    ax_x = fig.add_subplot(2, 2, 2)
    if traj_raw_p is not None: ax_x.plot(x_raw_time, traj_raw_p[:, 1], label='Raw X', color='gray', linestyle='--')
    if traj_loop_p is not None: ax_x.plot(x_loop_time, traj_loop_p[:, 1], label='Loop X', color='green')
    ax_x.set_title('X-axis Position Over Time')
    ax_x.set_xlabel(time_label)
    ax_x.set_ylabel('X (m)')
    ax_x.legend()

    ax_y = fig.add_subplot(2, 2, 3)
    if traj_raw_p is not None: ax_y.plot(x_raw_time, traj_raw_p[:, 2], label='Raw Y', color='gray', linestyle='--')
    if traj_loop_p is not None: ax_y.plot(x_loop_time, traj_loop_p[:, 2], label='Loop Y', color='green')
    ax_y.set_title('Y-axis Position Over Time')
    ax_y.set_xlabel(time_label)
    ax_y.set_ylabel('Y (m)')
    ax_y.legend()

    ax_z = fig.add_subplot(2, 2, 4)
    if traj_raw_p is not None: ax_z.plot(x_raw_time, traj_raw_p[:, 3], label='Raw Z', color='gray', linestyle='--')
    if traj_loop_p is not None: ax_z.plot(x_loop_time, traj_loop_p[:, 3], label='Loop Z', color='green')
    ax_z.set_title('Z-axis (Height) Drift Over Time')
    ax_z.set_xlabel(time_label)
    ax_z.set_ylabel('Z (m)')
    ax_z.legend()

    plt.tight_layout()
    print("正在生成可视化图表，保存图片至 trajectory_evaluation.png ...")
    plt.savefig('trajectory_evaluation.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()


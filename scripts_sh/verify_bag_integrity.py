#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 SQLite3 数据包完整性与时间戳单调性深度验证工具
"""

import sys
import os
import sqlite3
import numpy as np

def verify_bag(bag_dir):
    if not os.path.exists(bag_dir):
        print(f"[Error] 数据包目录不存在: {bag_dir}")
        return

    # 寻找 db3 文件
    db_files = [os.path.join(bag_dir, f) for f in os.listdir(bag_dir) if f.endswith('.db3')]
    if not db_files:
        print(f"[Error] 目录下未找到 .db3 文件: {bag_dir}")
        return

    db_path = db_files[0]
    file_size_gb = os.path.getsize(db_path) / (1024**3)
    print("=" * 70)
    print(f" 正在分析 ROS2 数据包: {os.path.basename(bag_dir)}")
    print(f" 文件路径: {db_path} ({file_size_gb:.2f} GB)")
    print("=" * 70)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 查询所有话题
    cursor.execute("SELECT id, name, type FROM topics;")
    topics = cursor.fetchall()
    topic_map = {row[0]: (row[1], row[2]) for row in topics}

    print("\n【1. 话题数据量与采样频率统计】")
    print(f"{'话题名称':<35} {'消息数量':<12} {'采样频率 (Hz)':<15} {'时间跨度 (s)':<12}")
    print("-" * 75)

    topic_data = {}
    for topic_id, (name, msg_type) in topic_map.items():
        cursor.execute("SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp ASC;", (topic_id,))
        rows = cursor.fetchall()
        if not rows:
            continue
        ts = np.array([r[0] for r in rows], dtype=np.int64)
        topic_data[name] = ts
        
        duration_s = (ts[-1] - ts[0]) * 1e-9 if len(ts) > 1 else 0
        freq_hz = len(ts) / duration_s if duration_s > 0 else 0
        print(f"{name:<35} {len(ts):<12} {freq_hz:<15.2f} {duration_s:<12.2f}")

    # 2. 时间戳单调性与掉帧检测
    print("\n【2. 时间戳单调性与连续性检测】")
    for name, ts in topic_data.items():
        if len(ts) < 2:
            continue
        dt_ns = np.diff(ts)
        non_monotonic = np.sum(dt_ns <= 0)
        dt_s = dt_ns * 1e-9
        mean_dt = np.mean(dt_s)
        max_dt = np.max(dt_s)
        min_dt = np.min(dt_s)
        std_dt = np.std(dt_s)

        # 掉帧阈值：大于平均间隔 2.5 倍
        drops = np.sum(dt_s > mean_dt * 2.5)

        status_str = "✅ 严格单调递增" if non_monotonic == 0 else f"❌ 存在 {non_monotonic} 次时间倒退/重复"
        drop_str = f"无明显断流 (最大间隔: {max_dt:.4f}s)" if drops == 0 else f"⚠️ 发现 {drops} 次间隔偏大 (最大: {max_dt:.4f}s)"
        
        print(f"• 话题 [{name}]:")
        print(f"  - 单调性: {status_str}")
        print(f"  - 周期特性: 平均 {mean_dt*1000:.2f}ms (±{std_dt*1000:.2f}ms) | 范围 [{min_dt*1000:.2f}ms ~ {max_dt*1000:.2f}ms]")
        print(f"  - 丢包/断流检测: {drop_str}")

    # 3. 左右目硬件时间同步精度分析
    stereo_pairs = [
        ("/camera/left/image_raw", "/camera/right/image_raw"),
        ("/stereo_0/camera/left/image_raw", "/stereo_0/camera/right/image_raw")
    ]
    for left_topic, right_topic in stereo_pairs:
        if left_topic in topic_data and right_topic in topic_data:
            print(f"\n【3. 左右目立体时间同步精度: {left_topic} vs {right_topic}】")
            ts_l = topic_data[left_topic]
            ts_r = topic_data[right_topic]
            
            # 使用时间戳就近配准计算左右目同步偏差
            sync_errors = []
            for tl in ts_l:
                idx = np.searchsorted(ts_r, tl)
                cand = []
                if idx < len(ts_r):
                    cand.append(abs(tl - ts_r[idx]))
                if idx > 0:
                    cand.append(abs(tl - ts_r[idx - 1]))
                if cand:
                    sync_errors.append(min(cand) * 1e-6)
            
            diff_ms = np.array(sync_errors)
            max_sync_err = np.max(diff_ms)
            mean_sync_err = np.mean(diff_ms)
            median_sync_err = np.median(diff_ms)
            p99_sync_err = np.percentile(diff_ms, 99)

            print(f"• 左目帧数: {len(ts_l)} 帧 | 右目帧数: {len(ts_r)} 帧 | 帧数差值: {abs(len(ts_l) - len(ts_r))} 帧")
            print(f"• 左右目平均时间偏差: {mean_sync_err:.3f} ms (中位数: {median_sync_err:.3f} ms, 99分位: {p99_sync_err:.3f} ms)")
            print(f"• 左右目最大时间偏差: {max_sync_err:.3f} ms")
            if max_sync_err < 1.0:
                print("• 评估结果: 🌟 极高精度硬件硬同步 (时间偏差 < 1ms)")
            elif p99_sync_err < 15.0:
                print("• 评估结果: ✅ 良好同步 (99% 偏差 < 15ms)")
            else:
                print("• 评估结果: ⚠️ 时间同步存在抖动")

    conn.close()
    print("\n" + "=" * 70)

if __name__ == '__main__':
    default_bag = "/home/elp/picture_resize_recording_NVIDA/datasets/my_dataset_20260817_100147"
    target = sys.argv[1] if len(sys.argv) > 1 else default_bag
    verify_bag(target)

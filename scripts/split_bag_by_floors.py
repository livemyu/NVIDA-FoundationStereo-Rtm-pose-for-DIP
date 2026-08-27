#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 Bag Floor Slicing Tool (Memory-Safe & High-Performance Streaming)
=======================================================================
Safely extracts and splits multi-floor long recording into independent
ROS 2 sqlite3 bags with minimal memory footprint (< 50MB RAM).
"""

import sqlite3
import os
import time
import yaml

src_bag_dir = "/home/elp/picture_resize_recording_NVIDA/datasets/my_dataset_20260822_095436"
src_db3 = os.path.join(src_bag_dir, "my_dataset_20260822_095436_0.db3")
datasets_root = "/home/elp/picture_resize_recording_NVIDA/datasets"

if not os.path.exists(src_db3):
    raise FileNotFoundError(f"源数据包不存在: {src_db3}")

print(f"=======================================================")
print(f"正在连接源数据包: {src_db3}")
print(f"=======================================================")

conn_src = sqlite3.connect(src_db3)
c_src = conn_src.cursor()

# 获取起始基准时间戳 (ns)
c_src.execute("SELECT timestamp FROM messages ORDER BY timestamp ASC LIMIT 1;")
origin_start_ts = c_src.fetchone()[0]
print(f"原始数据包起点时间戳 (ns): {origin_start_ts}")

# 获取元数据表内容与 topics 表内容
c_src.execute("SELECT schema_version, ros_distro FROM schema;")
schema_rows = c_src.fetchall()

c_src.execute("SELECT id, metadata_version, metadata FROM metadata;")
metadata_rows = c_src.fetchall()

c_src.execute("SELECT id, name, type, serialization_format, offered_qos_profiles FROM topics;")
topics_rows = c_src.fetchall()

topic_id_to_name = {row[0]: row[1] for row in topics_rows}

# 定义 3 个子数据集的明确命名与时间切分参数
split_configs = [
    {
        "name": "my_dataset_13F",
        "t_start_sec": 0.0,
        "t_end_sec": 255.0,
        "description": "13 楼出发与走廊全程推行 (00:00 ~ 04:15)"
    },
    {
        "name": "my_dataset_1F",
        "t_start_sec": 288.0,
        "t_end_sec": 380.0,
        "description": "1 楼大厅与连廊推行 (04:48 ~ 06:20)"
    },
    {
        "name": "my_dataset_12F",
        "t_start_sec": 416.0,
        "t_end_sec": 490.0,
        "description": "12 楼走廊全程推行 (06:56 ~ 08:10)"
    }
]

BATCH_SIZE = 500  # 每 500 条消息批量写入并提交一次，防止占用过多内存

for cfg_idx, cfg in enumerate(split_configs, 1):
    bag_name = cfg["name"]
    t_start = cfg["t_start_sec"]
    t_end = cfg["t_end_sec"]
    desc = cfg["description"]
    
    start_ts = origin_start_ts + int(t_start * 1e9)
    end_ts = origin_start_ts + int(t_end * 1e9)
    
    out_bag_dir = os.path.join(datasets_root, bag_name)
    os.makedirs(out_bag_dir, exist_ok=True)
    out_db3_name = f"{bag_name}_0.db3"
    out_db3_path = os.path.join(out_bag_dir, out_db3_name)
    
    if os.path.exists(out_db3_path):
        os.remove(out_db3_path)
        
    print(f"\n-------------------------------------------------------")
    print(f"[{cfg_idx}/{len(split_configs)}] 正在切分子数据包: {bag_name}")
    print(f"描述: {desc}")
    print(f"目标区间: {t_start:.1f}s ~ {t_end:.1f}s (时长: {t_end - t_start:.1f} 秒)")
    print(f"输出路径: {out_db3_path}")
    print(f"-------------------------------------------------------")
    
    conn_dst = sqlite3.connect(out_db3_path)
    c_dst = conn_dst.cursor()
    
    # 启用 SQLite 高性能配置
    c_dst.execute("PRAGMA synchronous = OFF;")
    c_dst.execute("PRAGMA journal_mode = MEMORY;")
    
    # 1. 创建表结构
    c_dst.execute("CREATE TABLE schema(schema_version INTEGER PRIMARY KEY, ros_distro TEXT NOT NULL);")
    c_dst.execute("CREATE TABLE metadata(id INTEGER PRIMARY KEY, metadata_version INTEGER NOT NULL, metadata TEXT NOT NULL);")
    c_dst.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL);")
    c_dst.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL, data BLOB NOT NULL);")
    
    # 2. 插入 schema, metadata, topics
    c_dst.executemany("INSERT INTO schema VALUES (?, ?);", schema_rows)
    c_dst.executemany("INSERT INTO metadata VALUES (?, ?, ?);", metadata_rows)
    c_dst.executemany("INSERT INTO topics VALUES (?, ?, ?, ?, ?);", topics_rows)
    conn_dst.commit()
    
    # 3. 流式读取与分批写入 (避免内存爆炸)
    print("正在流式提取并写入消息数据...")
    t0 = time.time()
    
    c_src.execute("""
        SELECT topic_id, timestamp, data 
        FROM messages 
        WHERE timestamp >= ? AND timestamp <= ? 
        ORDER BY timestamp ASC;
    """, (start_ts, end_ts))
    
    total_msgs = 0
    batch = []
    actual_min_ts = None
    actual_max_ts = None
    topic_msg_counts = {row[0]: 0 for row in topics_rows}
    
    for row in c_src:
        t_id, ts, data_blob = row
        batch.append((t_id, ts, data_blob))
        
        if actual_min_ts is None or ts < actual_min_ts:
            actual_min_ts = ts
        if actual_max_ts is None or ts > actual_max_ts:
            actual_max_ts = ts
            
        topic_msg_counts[t_id] = topic_msg_counts.get(t_id, 0) + 1
        total_msgs += 1
        
        if len(batch) >= BATCH_SIZE:
            c_dst.executemany("INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?);", batch)
            conn_dst.commit()
            batch.clear()
            if total_msgs % 5000 == 0:
                elapsed = time.time() - t0
                rate = total_msgs / elapsed if elapsed > 0 else 0
                print(f"  -> 已流式写入 {total_msgs:6d} 条消息 (速率: {rate:.1f} msg/s)...")
                
    if batch:
        c_dst.executemany("INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?);", batch)
        conn_dst.commit()
        batch.clear()
        
    # 创建时间戳索引 (加速后续 ROS2 读取)
    print("正在创建时间戳索引 (timestamp_idx)...")
    c_dst.execute("CREATE INDEX timestamp_idx ON messages (timestamp ASC);")
    conn_dst.commit()
    
    elapsed_total = time.time() - t0
    
    actual_start_ts = actual_min_ts if actual_min_ts is not None else start_ts
    actual_end_ts = actual_max_ts if actual_max_ts is not None else end_ts
    actual_duration_ns = actual_end_ts - actual_start_ts
    
    print(f"数据写入完成: 共 {total_msgs} 条消息，耗时 {elapsed_total:.2f} 秒。")
    print("各话题消息统计:")
    
    topics_with_count = []
    for top_id, top_name, top_type, top_format, top_qos in topics_rows:
        count = topic_msg_counts.get(top_id, 0)
        topics_with_count.append({
            "topic_metadata": {
                "name": top_name,
                "type": top_type,
                "serialization_format": top_format,
                "offered_qos_profiles": top_qos
            },
            "message_count": count
        })
        print(f"  - 话题 {top_name:32s}: {count:6d} 条")
        
    conn_dst.close()
    
    # 4. 生成 ROS 2 标准 metadata.yaml
    metadata_yaml = {
        "rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "sqlite3",
            "duration": {
                "nanoseconds": actual_duration_ns
            },
            "starting_time": {
                "nanoseconds_since_epoch": actual_start_ts
            },
            "message_count": total_msgs,
            "topics_with_message_count": topics_with_count,
            "compression_format": "",
            "compression_mode": "",
            "relative_file_paths": [out_db3_name],
            "files": [
                {
                    "path": out_db3_name,
                    "starting_time": {
                        "nanoseconds_since_epoch": actual_start_ts
                    },
                    "duration": {
                        "nanoseconds": actual_duration_ns
                    },
                    "message_count": total_msgs
                }
            ]
        }
    }
    
    out_yaml_path = os.path.join(out_bag_dir, "metadata.yaml")
    with open(out_yaml_path, "w") as f:
        yaml.dump(metadata_yaml, f, default_flow_style=False, sort_keys=False)
        
    print(f"已生成 metadata.yaml: {out_yaml_path}")
    print(f"✅ 子数据包 {bag_name} 切分成功 (时长 {actual_duration_ns*1e-9:.2f} 秒)")

conn_src.close()
print("\n=======================================================")
print("🎉 全部 3 个楼层独立子数据包切分完成！")
print("=======================================================")

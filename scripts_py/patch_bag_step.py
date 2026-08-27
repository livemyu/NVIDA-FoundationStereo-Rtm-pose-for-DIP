#!/usr/bin/env python3
import rosbag2_py
import sqlite3
import os
from rclpy.serialization import serialize_message, deserialize_message
from sensor_msgs.msg import Image

bag_path = '/home/elp/picture_resize_recording_NVIDA/my_dataset_20260810_114043'
db_file = os.path.join(bag_path, 'my_dataset_20260810_114043_0.db3')

print(f"正在直接修补 ROS2 数据包中的 step 字段: {db_file}")

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# 获取 topic id
cursor.execute("SELECT id, name FROM topics WHERE name IN ('/camera/left/image_raw', '/camera/right/image_raw')")
topic_map = {row[0]: row[1] for row in cursor.fetchall()}
print(f"找到图像话题 ID: {topic_map}")

cursor.execute("SELECT id, topic_id, data FROM messages WHERE topic_id IN ({})".format(','.join(map(str, topic_map.keys()))))
rows = cursor.fetchall()
print(f"总计找到 {len(rows)} 条待修正图像消息...")

updated = 0
for msg_id, topic_id, data in rows:
    msg = deserialize_message(data, Image)
    if msg.step != msg.width:
        msg.step = msg.width
        new_data = serialize_message(msg)
        cursor.execute("UPDATE messages SET data = ? WHERE id = ?", (new_data, msg_id))
        updated += 1

conn.commit()
conn.close()
print(f"数据包修补完成！成功修正 {updated} 条消息的 step 字段为 {msg.width}。")

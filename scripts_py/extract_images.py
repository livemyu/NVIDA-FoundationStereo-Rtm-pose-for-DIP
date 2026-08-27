import sqlite3
import cv2
import numpy as np
import os

bag_path = '/home/elp/distortion_fish/Stereo_IMU_Calibration_Toolkit/calibration_bags/ros2_bag_20260811_102027/ros2_bag_20260811_102027_0.db3'
out_dir = '/home/elp/.gemini/antigravity-ide/brain/2c8194dd-e972-443c-8e4f-b30e9b5af3a6/'

conn = sqlite3.connect(bag_path)
c = conn.cursor()

c.execute("SELECT id FROM topics WHERE name='/camera/left/image_raw'")
row = c.fetchone()
if not row:
    print("Could not find /camera/left/image_raw topic")
    exit(1)
img_topic_id = row[0]

c.execute(f"SELECT timestamp, data FROM messages WHERE topic_id={img_topic_id} LIMIT 10")
rows = c.fetchall()

# 提取前两帧
for i in range(min(2, len(rows))):
    data = rows[i][1]
    # search for 'mono8'
    idx = data.find(b'mono8')
    if idx == -1:
        print("mono8 not found")
        continue
    # ROS 2 image data usually starts a few bytes after encoding string.
    # The actual raw pixel payload for mono8 will be at the end.
    # Assuming either 1920x1200 or 960x600.
    if len(data) - idx > 1920*1200:
        img_data = data[-1920*1200:]
        img = np.frombuffer(img_data, dtype=np.uint8).reshape((1200, 1920))
    elif len(data) - idx > 960*600:
        img_data = data[-960*600:]
        img = np.frombuffer(img_data, dtype=np.uint8).reshape((600, 960))
    else:
        print("Unknown image dimension")
        continue
        
    file_path = os.path.join(out_dir, f'calib_bag_img_{i}.jpg')
    cv2.imwrite(file_path, img)
    print(f"Saved {file_path}")

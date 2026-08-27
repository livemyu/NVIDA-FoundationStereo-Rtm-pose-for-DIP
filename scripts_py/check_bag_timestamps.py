import sqlite3
import struct

conn = sqlite3.connect('/home/elp/picture_resize_recording_NVIDA/my_dataset_20260812_043448/my_dataset_20260812_043448_0.db3')
c = conn.cursor()

# Get IMU timestamp
c.execute("SELECT id FROM topics WHERE name='/imu/data_raw'")
imu_topic_id = c.fetchone()[0]
c.execute(f"SELECT timestamp FROM messages WHERE topic_id={imu_topic_id} ORDER BY timestamp ASC LIMIT 1")
imu_ts = c.fetchone()[0]

# Get Image timestamp
c.execute("SELECT id FROM topics WHERE name='/camera/left/image_raw'")
img_topic_id = c.fetchone()[0]
c.execute(f"SELECT timestamp FROM messages WHERE topic_id={img_topic_id} ORDER BY timestamp ASC LIMIT 1")
img_ts = c.fetchone()[0]

print(f"First IMU timestamp:   {imu_ts}")
print(f"First Image timestamp: {img_ts}")
print(f"Difference (Image - IMU): {(img_ts - imu_ts) / 1e9} seconds")

# Also let's check the time difference in the message header!
c.execute(f"SELECT data FROM messages WHERE topic_id={img_topic_id} ORDER BY timestamp ASC LIMIT 1")
img_data = c.fetchone()[0]
# Header timestamp is usually in the first 24 bytes (CDR encoding)
# ROS2 message header: timestamp sec (4 bytes), nanosec (4 bytes), frame_id (string)
# We can just extract it
print("Image Header: ", img_data[:32])

c.execute(f"SELECT data FROM messages WHERE topic_id={imu_topic_id} ORDER BY timestamp ASC LIMIT 1")
imu_data = c.fetchone()[0]
print("IMU Header: ", imu_data[:32])

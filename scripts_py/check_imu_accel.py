import sqlite3
import struct

conn = sqlite3.connect('/home/elp/picture_resize_recording_NVIDA/my_dataset_20260812_071616/my_dataset_20260812_071616_0.db3')
c = conn.cursor()

c.execute("SELECT id FROM topics WHERE name='/imu/data_raw'")
imu_topic_id = c.fetchone()[0]
c.execute(f"SELECT data FROM messages WHERE topic_id={imu_topic_id} ORDER BY timestamp ASC LIMIT 5")
rows = c.fetchall()

for row in rows:
    data = row[0]
    # The header is exactly 32 bytes.
    # Linear acceleration starts at 32 + 32 + 72 + 24 + 72 = 232
    accel_offset = 232
    try:
        ax, ay, az = struct.unpack('<ddd', data[accel_offset:accel_offset+24])
        print(f"Accel: X={ax:8.4f}, Y={ay:8.4f}, Z={az:8.4f}")
    except Exception as e:
        print(e)

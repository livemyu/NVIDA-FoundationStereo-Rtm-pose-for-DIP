import sqlite3
import struct
import matplotlib.pyplot as plt

conn = sqlite3.connect('/home/elp/picture_resize_recording_NVIDA/my_dataset_20260812_043448/my_dataset_20260812_043448_0.db3')
c = conn.cursor()

c.execute("SELECT id FROM topics WHERE name='/imu/data_raw'")
imu_topic_id = c.fetchone()[0]

c.execute("SELECT id FROM topics WHERE name='/camera/left/image_raw'")
img_topic_id = c.fetchone()[0]

print("Fetching IMU timestamps...")
c.execute(f"SELECT timestamp, data FROM messages WHERE topic_id={imu_topic_id} ORDER BY timestamp ASC")
imu_rows = c.fetchall()

print("Fetching Image timestamps...")
c.execute(f"SELECT timestamp, data FROM messages WHERE topic_id={img_topic_id} ORDER BY timestamp ASC")
img_rows = c.fetchall()

print(f"Total IMU messages: {len(imu_rows)}")
print(f"Total Image messages: {len(img_rows)}")

# Extract header timestamps
def get_header_ts(data):
    # CDR Header is 4 bytes. Then sec (4 bytes), nanosec (4 bytes).
    sec, nanosec = struct.unpack('<II', data[4:12])
    return sec + nanosec * 1e-9

imu_sys_ts = []
imu_hdr_ts = []
for r in imu_rows:
    imu_sys_ts.append(r[0] * 1e-9)
    imu_hdr_ts.append(get_header_ts(r[1]))

img_sys_ts = []
img_hdr_ts = []
for r in img_rows:
    img_sys_ts.append(r[0] * 1e-9)
    img_hdr_ts.append(get_header_ts(r[1]))

# Check if IMU is arriving regularly
print("\n--- IMU Stats ---")
imu_hdr_diff = [imu_hdr_ts[i] - imu_hdr_ts[i-1] for i in range(1, len(imu_hdr_ts))]
print(f"IMU Header frequency: {1.0/(sum(imu_hdr_diff)/len(imu_hdr_diff)):.2f} Hz")
print(f"Max IMU gap: {max(imu_hdr_diff):.5f} s")

# Check if Image is arriving regularly
print("\n--- Image Stats ---")
img_hdr_diff = [img_hdr_ts[i] - img_hdr_ts[i-1] for i in range(1, len(img_hdr_ts))]
print(f"Image Header frequency: {1.0/(sum(img_hdr_diff)/len(img_hdr_diff)):.2f} Hz")
print(f"Max Image gap: {max(img_hdr_diff):.5f} s")

# Check alignment
print("\n--- Sync Analysis ---")
# For each image, find the closest IMU message by header timestamp
offsets = []
for img_t in img_hdr_ts:
    # binary search for closest imu_t
    import bisect
    idx = bisect.bisect_left(imu_hdr_ts, img_t)
    if idx == 0:
        closest = imu_hdr_ts[0]
    elif idx == len(imu_hdr_ts):
        closest = imu_hdr_ts[-1]
    else:
        before = imu_hdr_ts[idx - 1]
        after = imu_hdr_ts[idx]
        if after - img_t < img_t - before:
            closest = after
        else:
            closest = before
    offsets.append(img_t - closest)

print(f"Average offset (Image - closest IMU): {sum(offsets)/len(offsets):.5f} s")
print(f"Max offset: {max(offsets):.5f} s")
print(f"Min offset: {min(offsets):.5f} s")

# Check for clock drift
drift = offsets[-1] - offsets[0]
print(f"Total drift over recording: {drift:.5f} s")


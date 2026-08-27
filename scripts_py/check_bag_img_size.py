import sqlite3
import struct

conn = sqlite3.connect('/home/elp/picture_resize_recording_NVIDA/my_dataset_20260812_073539/my_dataset_20260812_073539_0.db3')
c = conn.cursor()
c.execute("SELECT id FROM topics WHERE name='/camera/left/image_raw'")
topic_id = c.fetchone()[0]

c.execute(f"SELECT data FROM messages WHERE topic_id={topic_id} LIMIT 1")
data = c.fetchone()[0]

# CDR encoding: usually has a 4-byte header, then the serialized fields.
# For sensor_msgs/Image:
# std_msgs/Header header
# uint32 height
# uint32 width
# string encoding
# uint8 is_bigendian
# uint32 step
# uint8[] data

# Try to find 'mono8' in the payload
idx = data.find(b'mono8')
if idx != -1:
    # Before 'mono8', there is a 4-byte string length (which is 6 including null terminator)
    # Before that, width (4 bytes)
    # Before that, height (4 bytes)
    width = struct.unpack('<I', data[idx-8:idx-4])[0]
    height = struct.unpack('<I', data[idx-12:idx-8])[0]
    print(f"Image Dimensions: {width}x{height}")
else:
    print("Could not find 'mono8' encoding in the message.")

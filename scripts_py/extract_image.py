import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import cv2
import numpy as np

def extract_one():
    bag_path = '/home/elp/picture_resize_recording_NVIDA/my_dataset_20260811_095710'
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_metadata.name: topic_metadata.type for topic_metadata in topic_types}
    
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == '/camera/left/image_raw':
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            # msg is sensor_msgs/Image
            img = np.array(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            cv2.imwrite('left_frame.jpg', img)
            print("Saved left_frame.jpg")
            break

extract_one()

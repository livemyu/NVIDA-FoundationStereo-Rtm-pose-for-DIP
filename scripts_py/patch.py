import os
import sys

file_path = os.path.expanduser("~/nvidia_stereo_vins_deployment/vins_ros2_ws/src/VINS-Fusion-ROS2-Humble/loop_fusion/src/pose_graph_node.cpp")

with open(file_path, "r") as f:
    content = f.read()

if "while(image_buf.size() > 100)" not in content and "while (image_buf.size() > 100)" not in content:
    old_code = "image_buf.push(image_msg);"
    new_code = "image_buf.push(image_msg);\n    while(image_buf.size() > 100) { image_buf.pop(); }"
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, "w") as f:
            f.write(content)
        print("Patched successfully!")
    else:
        print("Target code not found.")
else:
    print("Already patched.")

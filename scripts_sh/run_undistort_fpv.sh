#!/bin/bash

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
# 自动寻找这台电脑(全/home/elp)里最新的 my_dataset 数据包
LATEST_BAG=$(find /home/elp -maxdepth 5 -type d -name "my_dataset_*" 2>/dev/null | xargs ls -td 2>/dev/null | head -1)
BAG_DIR="$LATEST_BAG"

echo "=========================================================="
echo " 正在启动数据包回放与 FPV 去畸变查看 (左目)"
echo " 数据集路径: $BAG_DIR"
echo "=========================================================="

# 1. 启动 FPV Viewer Python Node
gnome-terminal --title="1. FPV-Viewer" -- bash -c "
source /opt/ros/humble/setup.bash
echo '>>> 启动 Python FPV 去畸变节点...'
python3 $WORKSPACE_DIR/scripts_py/undistort_fpv_viewer.py
exec bash
"

# 2. 启动数据包播放
gnome-terminal --tab --title="2. ROS2-Bag-Play" -- bash -c "
source /opt/ros/humble/setup.bash
cd '$WORKSPACE_DIR'
echo '>>> 等待 2 秒后开始回放数据包: $BAG_DIR ...'
sleep 2
ros2 bag play '$BAG_DIR'
exec bash
"

echo "已启动 FPV 查看窗口与数据包回放终端。"

#!/bin/bash

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
BAG_DIR="${1:-$HOME/my_dataset_20260813_081724}"
RVIZ_CONFIG="$WORKSPACE_DIR/raw_image_rviz.rviz"

echo "=========================================================="
echo " 正在启动数据包回放与 RViz 原始图像查看"
echo " 数据集路径: $BAG_DIR"
echo "=========================================================="

# 1. 启动 RViz2
gnome-terminal --title="1. RViz2-Raw-Image" -- bash -c "
source /opt/ros/humble/setup.bash
echo '>>> 启动 RViz2 图像查看窗口...'
rviz2 -d '$RVIZ_CONFIG'
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

echo "已启动 RViz2 窗口与数据包回放终端。"

#!/bin/bash

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
D435_DATASET_DIR="$WORKSPACE_DIR/d435_vo_20260810_114041"
RVIZ_CONFIG="$WORKSPACE_DIR/raw_image_rviz.rviz"

echo "=========================================================="
echo " 正在启动 D435 数据集回放与 RViz 图像查看"
echo " 数据集路径: $D435_DATASET_DIR"
echo "=========================================================="

# 1. 启动 RViz2
gnome-terminal --title="1. RViz2-D435-Image" -- bash -c "
source /opt/ros/humble/setup.bash
echo '>>> 启动 RViz2 图像查看窗口...'
rviz2 -d '$RVIZ_CONFIG'
exec bash
"

# 2. 启动 D435 数据集播放脚本
gnome-terminal --tab --title="2. D435-Dataset-Player" -- bash -c "
source /opt/ros/humble/setup.bash
cd '$WORKSPACE_DIR'
echo '>>> 等待 2 秒后开始播放 D435 数据集...'
sleep 2
python3 play_d435_dataset.py '$D435_DATASET_DIR'
exec bash
"

echo "已启动 RViz2 窗口与 D435 数据集播放终端。"

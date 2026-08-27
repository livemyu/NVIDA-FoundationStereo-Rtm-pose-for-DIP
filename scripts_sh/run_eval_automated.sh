#!/bin/bash
export DISPLAY=:0
source /opt/ros/humble/setup.bash
source /home/elp/vins_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=42

cd /home/elp/picture_resize_recording_NVIDA

echo "清理之前的输出..."
rm -f output/vio.csv output/vio_loop.csv

echo "启动 VINS-Fusion..."
ros2 run vins vins_node /home/elp/picture_resize_recording_NVIDA/Stereo_cam_ws/my_stereo_imu_config_960_remote.yaml > /tmp/vins_eval.log 2>&1 &
VINS_PID=$!

echo "等待算法初始化..."
sleep 3

echo "开始播放数据包: nodistortion_bags/my_dataset_20260812_083303"
ros2 bag play nodistortion_bags/my_dataset_20260812_083303

echo "播放完毕，等待后端优化..."
sleep 10
kill $VINS_PID

echo "开始执行评估脚本..."
python3 evaluate_trajectory.py --raw output/vio.csv --loop output/vio_loop.csv

#!/bin/bash
source /opt/ros/humble/setup.bash
ros2 bag play /home/elp/picture_resize_recording_NVIDA/my_dataset_20260812_043448 &
PLAYER_PID=$!
sleep 1
ros2 topic echo --no-arr /imu/data_raw | head -n 30 > /tmp/imu_data.txt
kill $PLAYER_PID

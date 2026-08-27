#!/bin/bash
export DISPLAY=:0

echo "正在启动 RViz 可视化界面..."
source /opt/ros/humble/setup.bash
source /home/elp/vins_ros2_ws/install/setup.bash
rviz2 -d /home/elp/vins_ros2_ws/src/VINS-Fusion-ROS2-Humble/config/vins_rviz_config.rviz > /dev/null 2>&1 &
RVIZ_PID=$!

echo "正在启动 VINS-Fusion 后台算法节点..."
ros2 run vins vins_node /home/elp/picture_resize_recording_NVIDA/Stereo_cam_ws/my_stereo_config_960_remote.yaml > /tmp/vins_run.log 2>&1 &
VINS_PID=$!

echo "等待 3 秒钟让算法初始化完毕..."
sleep 3

echo "▶️ 开始为您播放录制好的双目+IMU数据包，请看屏幕！"
ros2 bag play /home/elp/picture_resize_recording_NVIDA/nodistortion_bags/my_dataset_20260812_083303

echo "✅ 数据包播放完毕。由于算法还在处理缓存中的数据，保持后台运行..."
echo "如需强制结束，请在终端按 Ctrl+C，或通过后台指令停止。"
sleep 3600

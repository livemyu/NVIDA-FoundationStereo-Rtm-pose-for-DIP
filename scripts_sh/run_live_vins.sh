#!/bin/bash

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
VINS_WS_DIR="$HOME/vins_ros2_ws"
CONFIG_FILE="$WORKSPACE_DIR/Stereo_cam_ws/my_stereo_imu_config_180_960.yaml"

echo "=========================================================="
echo " 正在启动 VINS-Fusion (基于180标定配置) 与本地相机节点"
echo "=========================================================="

# 杀掉可能残留的旧节点
pkill -f vins_node
pkill -f loop_fusion_node
pkill -f rviz2
pkill -f synccap_ros_node

# 1. 启动 RViz2
gnome-terminal --title="1. RViz2" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
export CYCLONEDDS_URI=file://$WORKSPACE_DIR/cyclonedds_block_raw_images.xml
echo '>>> 启动 RViz2...'
rviz2 -d '$WORKSPACE_DIR/Stereo_cam_ws/vins_only_rviz_config.rviz'
exec bash
"

# 2. 启动 VINS-Fusion
gnome-terminal --tab --title="2. VINS-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
export CYCLONEDDS_URI=file://$WORKSPACE_DIR/cyclonedds_block_raw_images.xml
echo '>>> 启动 VINS-Fusion 节点...'
ros2 run vins vins_node '$CONFIG_FILE'
exec bash
"

# 3. 启动 Loop Fusion (可选)
gnome-terminal --tab --title="3. Loop-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
export CYCLONEDDS_URI=file://$WORKSPACE_DIR/cyclonedds_block_raw_images.xml
echo '>>> 启动 Loop-Fusion 回环优化节点...'
ros2 run loop_fusion loop_fusion_node '$CONFIG_FILE' --ros-args \
  -r /vins_estimator/odometry:=/odometry \
  -r /vins_estimator/keyframe_pose:=/keyframe_pose \
  -r /vins_estimator/extrinsic:=/extrinsic \
  -r /vins_estimator/keyframe_point:=/keyframe_point \
  -r /vins_estimator/margin_cloud:=/margin_cloud
exec bash
"

# 4. 启动本地相机节点 (synccap_ros)
gnome-terminal --tab --title="4. Camera Node" -- bash -c "
source /opt/ros/humble/setup.bash
source '$WORKSPACE_DIR/Stereo_cam_ws/install/setup.bash'
export CYCLONEDDS_URI=file://$WORKSPACE_DIR/cyclonedds_block_raw_images.xml
echo '>>> 等待 2 秒让 VINS 初始化完成，然后启动本地相机节点...'
sleep 2
ros2 run synccap_ros synccap_ros_node --ros-args -p pub_width:=960 -p pub_height:=600
exec bash
"

echo "✅ 已开启所有终端窗口！"

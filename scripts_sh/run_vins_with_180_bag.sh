#!/bin/bash

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
BAG_DIR="$WORKSPACE_DIR/my_dataset_20260811_095710"
VINS_WS_DIR="$HOME/vins_ros2_ws"
CONFIG_FILE="$WORKSPACE_DIR/Stereo_cam_ws/my_stereo_imu_config_180_960.yaml"

echo "=========================================================="
echo " 正在启动 VINS-Fusion (基于180标定配置) 与数据包回放"
echo " 数据集路径: $BAG_DIR"
echo "=========================================================="

# 创建 VINS 输出目录
mkdir -p "$WORKSPACE_DIR/Stereo_cam_ws/output/pose_graph"

# 1. 启动 RViz2
gnome-terminal --title="1. RViz2" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
echo '>>> 启动 RViz2...'
rviz2 -d '$WORKSPACE_DIR/Stereo_cam_ws/vins_only_rviz_config.rviz'
exec bash
"

# 2. 启动 VINS-Fusion
gnome-terminal --tab --title="2. VINS-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
echo '>>> 启动 VINS-Fusion 节点...'
ros2 run vins vins_node '$CONFIG_FILE'
exec bash
"

# 3. 启动 Loop Fusion (可选)
gnome-terminal --tab --title="3. Loop-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
echo '>>> 启动 Loop-Fusion 回环优化节点...'
ros2 run loop_fusion loop_fusion_node '$CONFIG_FILE' --ros-args \
  -r /vins_estimator/odometry:=/odometry \
  -r /vins_estimator/keyframe_pose:=/keyframe_pose \
  -r /vins_estimator/extrinsic:=/extrinsic \
  -r /vins_estimator/keyframe_point:=/keyframe_point \
  -r /vins_estimator/margin_cloud:=/margin_cloud
exec bash
"

# 4. 启动数据包播放
gnome-terminal --tab --title="4. ROS2-Bag-Play" -- bash -c "
source /opt/ros/humble/setup.bash
echo '>>> 等待 3 秒让 VINS 初始化完成，然后开始回放数据包: $BAG_DIR ...'
sleep 3
ros2 bag play '$BAG_DIR'
exec bash
"

echo "✅ 已开启所有终端窗口！"

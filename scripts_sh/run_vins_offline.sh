#!/bin/bash
# 运行环境: PC 端
# 功能: 仅启动离线数据集专用的 VINS-Fusion 与 Loop-Fusion 节点

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
VINS_WS_DIR="$HOME/vins_ros2_ws"
CONFIG_FILE="$WORKSPACE_DIR/Stereo_cam_ws/my_stereo_imu_config_960_remote.yaml"

echo "正在启动离线模式的 VINS-Fusion 和 Loop-Fusion 节点..."

# 0. 自动创建轨迹输出文件夹 (防止 fopen 因文件夹不存在导致无法写入 csv)
mkdir -p "$WORKSPACE_DIR/output/pose_graph"

# 1. 启动 VINS-Fusion (使用专门的 remote 配置文件)
gnome-terminal --tab --title="VINS-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
export ROS_DOMAIN_ID=42
cd '$WORKSPACE_DIR'
echo '正在启动 VINS-Fusion (离线回放模式)...'
ros2 run vins vins_node '$CONFIG_FILE'
exec bash
"

# 2. 启动 Loop Fusion (闭环优化节点，带话题重映射)
gnome-terminal --tab --title="Loop-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
export ROS_DOMAIN_ID=42
cd '$WORKSPACE_DIR'
echo '正在启动 Loop-Fusion 回环优化节点...'
ros2 run loop_fusion loop_fusion_node '$CONFIG_FILE' --ros-args \
    -r /vins_estimator/odometry:=/odometry \
    -r /vins_estimator/keyframe_pose:=/keyframe_pose \
    -r /vins_estimator/extrinsic:=/extrinsic \
    -r /vins_estimator/keyframe_point:=/keyframe_point \
    -r /vins_estimator/margin_cloud:=/margin_cloud
exec bash
"

echo "VINS-Fusion 与 Loop-Fusion 节点已成功启动。"

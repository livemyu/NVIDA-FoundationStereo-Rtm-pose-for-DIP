#!/bin/bash
# 最新 960x600 Omni-Radtan(MEI) 参数 VINS 回放脚本

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
VINS_WS_DIR="$HOME/vins_ros2_ws"
CONFIG_FILE="$VINS_WS_DIR/src/VINS-Fusion-ROS2-Humble/config/my_stereo_imu_omni_960x600/my_stereo_imu_config.yaml"
BAG_DIR="${1:-$HOME/my_dataset_20260814_041236}"
RVIZ_CONFIG="$WORKSPACE_DIR/Stereo_cam_ws/vins_only_rviz_config.rviz"

echo "=========================================================="
echo " 正在启动最新全向相机模型 (MEI) VINS + Loop 回放"
echo " 数据集路径: $BAG_DIR"
echo " 配置文件:   $CONFIG_FILE"
echo "=========================================================="

mkdir -p "$WORKSPACE_DIR/output/pose_graph"

# 1. 启动 vins_node
gnome-terminal --tab --title="1. VINS-Node-Omni" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
cd '$WORKSPACE_DIR'
echo '>>> 启动 VINS-Node (MEI 模型)...'
ros2 run vins vins_node '$CONFIG_FILE'
exec bash
"

# 2. 启动 loop_fusion_node
gnome-terminal --tab --title="2. Loop-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
cd '$WORKSPACE_DIR'
echo '>>> 启动 Loop-Fusion 回环优化...'
ros2 run loop_fusion loop_fusion_node '$CONFIG_FILE' --ros-args \
    -r /vins_estimator/odometry:=/odometry \
    -r /vins_estimator/keyframe_pose:=/keyframe_pose \
    -r /vins_estimator/extrinsic:=/extrinsic \
    -r /vins_estimator/keyframe_point:=/keyframe_point \
    -r /vins_estimator/margin_cloud:=/margin_cloud
exec bash
"

# 3. 启动 RViz2
gnome-terminal --tab --title="3. RViz2-Visualization" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
echo '>>> 启动 RViz2 可视化窗口...'
rviz2 -d '$RVIZ_CONFIG'
exec bash
"

# 4. 延迟 3 秒后播放新下载的 960x600 数据集
gnome-terminal --tab --title="4. ROS2-Bag-Play" -- bash -c "
source /opt/ros/humble/setup.bash
cd '$WORKSPACE_DIR'
echo '>>> 等待 3 秒节点就绪后开始回放数据集...'
sleep 3
ros2 bag play '$BAG_DIR'
exec bash
"

echo "全部启动指令已下发！请在弹出的终端标签页中查看运行状态。"

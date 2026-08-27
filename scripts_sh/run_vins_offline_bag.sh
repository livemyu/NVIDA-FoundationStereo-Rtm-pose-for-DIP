#!/bin/bash
# 离线 11GB 数据集 VINS-Fusion & Loop-Fusion 实时回放与 RViz2 可视化一键启动脚本

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
VINS_WS_DIR="$HOME/vins_ros2_ws"
CONFIG_FILE="$WORKSPACE_DIR/Stereo_cam_ws/my_stereo_imu_config_960_remote.yaml"
BAG_DIR="$WORKSPACE_DIR/my_dataset_20260806_113813"
RVIZ_CONFIG="$WORKSPACE_DIR/Stereo_cam_ws/vins_only_rviz_config.rviz"

echo "=========================================================="
echo " 正在启动离线 11GB 数据集 VINS + Loop + RViz 回放方案"
echo " 数据集路径: $BAG_DIR"
echo " 配置文件:   $CONFIG_FILE"
echo "=========================================================="

mkdir -p "$WORKSPACE_DIR/output/pose_graph"

# 1. 启动 vins_node
gnome-terminal --tab --title="1. VINS-Node" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
cd '$WORKSPACE_DIR'
echo '>>> 启动 VINS-Node 前端 estimator...'
ros2 run vins vins_node '$CONFIG_FILE'
exec bash
"

# 2. 启动 loop_fusion_node
gnome-terminal --tab --title="2. Loop-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
cd '$WORKSPACE_DIR'
echo '>>> 启动 Loop-Fusion 回环优化后端...'
ros2 run loop_fusion loop_fusion_node '$CONFIG_FILE' --ros-args \
    -r /vins_estimator/odometry:=/odometry \
    -r /vins_estimator/keyframe_pose:=/keyframe_pose \
    -r /vins_estimator/extrinsic:=/extrinsic \
    -r /vins_estimator/keyframe_point:=/keyframe_point \
    -r /vins_estimator/margin_cloud:=/margin_cloud
exec bash
"

# 3. 启动 RViz2 可视化
gnome-terminal --tab --title="3. RViz2-Visualization" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
echo '>>> 启动 RViz2 可视化窗口...'
rviz2 -d '$RVIZ_CONFIG'
exec bash
"

# 4. 延迟 3 秒后播放 11GB 录制数据集
gnome-terminal --tab --title="4. ROS2-Bag-Play" -- bash -c "
source /opt/ros/humble/setup.bash
cd '$WORKSPACE_DIR'
echo '>>> 等待 3 秒节点就绪后开始回放 11GB 数据集...'
sleep 3
ros2 bag play '$BAG_DIR'
exec bash
"

echo "全部 4 个终端标签页已启动：VINS-Node, Loop-Fusion, RViz2, Bag Play。"

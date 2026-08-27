#!/bin/bash
# RealSense D435 录制数据集离线纯视觉 (Pure Stereo, imu: 0) 回放与可视化脚本

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
VINS_WS_DIR="$HOME/vins_ros2_ws"
CONFIG_FILE="$WORKSPACE_DIR/Stereo_cam_ws/my_d435_stereo_pure_visual.yaml"
DATASET_DIR="$WORKSPACE_DIR/d435_vo_20260810_114041"
RVIZ_CONFIG="$WORKSPACE_DIR/Stereo_cam_ws/vins_only_rviz_config.rviz"

echo "=========================================================="
echo " 正在启动 RealSense D435 离线纯视觉 (Pure Stereo) + Loop + RViz 回放"
echo " 数据集路径: $DATASET_DIR"
echo " 配置文件:   $CONFIG_FILE"
echo "=========================================================="

mkdir -p "$WORKSPACE_DIR/output/pose_graph"

# 1. 启动 vins_node (D435 纯视觉模式)
gnome-terminal --tab --title="1. D435-VINS-Node" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
cd '$WORKSPACE_DIR'
echo '>>> 启动 D435 纯视觉 (imu: 0) 模式的 vins_node...'
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

# 4. 延迟 3 秒后启动 D435 图像数据流注入
gnome-terminal --tab --title="4. D435-Dataset-Player" -- bash -c "
source /opt/ros/humble/setup.bash
cd '$WORKSPACE_DIR'
echo '>>> 等待 3 秒节点就绪后开始回放 D435 数据集: $DATASET_DIR...'
sleep 3
python3 play_d435_dataset.py '$DATASET_DIR'
exec bash
"

echo "全部 4 个终端标签页已启动：D435 VINS-Node (纯视觉), Loop-Fusion, RViz2, D435 Dataset Player。"

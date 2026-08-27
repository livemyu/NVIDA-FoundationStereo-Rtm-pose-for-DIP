#!/bin/bash
# 使用 180 标定参数运行 VINS-Fusion + Loop + RViz 数据包回放

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
VINS_WS_DIR="$HOME/vins_ros2_ws"
CONFIG_FILE="$WORKSPACE_DIR/Stereo_cam_ws/my_stereo_imu_config_180_960.yaml"
BAG_DIR="${1:-$WORKSPACE_DIR/datasets/my_dataset_20260822_095436}"
RVIZ_CONFIG="$WORKSPACE_DIR/Stereo_cam_ws/vins_only_rviz_config.rviz"

echo "=========================================================="
echo " 正在使用 180 鱼眼标定参数 (KANNALA_BRANDT) 运行 VINS + Loop"
echo " 数据集路径: $BAG_DIR"
echo " 配置文件:   $CONFIG_FILE"
echo "=========================================================="

# 清理旧的 VINS 节点与残留回放进程 (使用强力匹配确保 0 孤儿进程)
pkill -9 -f vins_node 2>/dev/null
pkill -9 -f loop_fusion_node 2>/dev/null
pkill -9 -f rviz2 2>/dev/null
pkill -9 -f "ros2.*bag.*play" 2>/dev/null
pkill -9 -f "ros2 bag play" 2>/dev/null
sleep 1

mkdir -p "$WORKSPACE_DIR/output/pose_graph"

# 锁定 Jetson 全核 1.98GHz 最高主频，消除休眠抖动
echo 123456 | sudo -S jetson_clocks 2>/dev/null || true

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export CYCLONEDDS_URI="file://$WORKSPACE_DIR/cyclonedds.xml"

# 1. 启动 vins_node (绑定 CPU 核心 2,3,4,5,6 - 5 核心保障 Ceres 快速收敛与特征光流计算)
gnome-terminal --tab --title="1. VINS-Node-180" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
export CYCLONEDDS_URI='file://$WORKSPACE_DIR/cyclonedds.xml'
cd '$WORKSPACE_DIR'
echo '>>> [Core 2,3,4,5,6] 启动 180 标定参数模式下的 vins_node...'
taskset -c 2,3,4,5,6 ros2 run vins vins_node '$CONFIG_FILE'
exec bash
"

# 2. 启动 loop_fusion_node (绑定 CPU 核心 7,8,9 - 独立 3 核心用于全局位姿图优化与回环检测)
gnome-terminal --tab --title="2. Loop-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
export CYCLONEDDS_URI='file://$WORKSPACE_DIR/cyclonedds.xml'
cd '$WORKSPACE_DIR'
echo '>>> [Core 7,8,9] 启动 Loop-Fusion 回环优化后端...'
taskset -c 7,8,9 ros2 run loop_fusion loop_fusion_node '$CONFIG_FILE' --ros-args \
    -r /vins_estimator/odometry:=/odometry \
    -r /vins_estimator/keyframe_pose:=/keyframe_pose \
    -r /vins_estimator/extrinsic:=/extrinsic \
    -r /vins_estimator/keyframe_point:=/keyframe_point \
    -r /vins_estimator/margin_cloud:=/margin_cloud
exec bash
"

# 3. 启动 RViz2 可视化 (绑定 CPU 核心 10,11 - 独立 2 核心保障 3D 渲染与 GUI 顺畅)
gnome-terminal --tab --title="3. RViz2-Visualization" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
export CYCLONEDDS_URI='file://$WORKSPACE_DIR/cyclonedds.xml'
echo '>>> [Core 10,11] 启动 RViz2 可视化窗口...'
taskset -c 10,11 rviz2 -d '$RVIZ_CONFIG'
exec bash
"

# 4. 延迟 3 秒后启动 ROS2 数据包播放 (绑定 CPU 核心 0,1 - 独立 2 核心处理数据解包与 DDS 分发)
gnome-terminal --tab --title="4. ROS2-Bag-Play" -- bash -c "
source /opt/ros/humble/setup.bash
export CYCLONEDDS_URI='file://$WORKSPACE_DIR/cyclonedds.xml'
cd '$WORKSPACE_DIR'
echo '>>> [Core 0,1] 等待 3 秒节点就绪后开始回放最新数据包: $BAG_DIR...'
sleep 3
taskset -c 0,1 ros2 bag play '$BAG_DIR' --topics /camera/left/image_raw /camera/right/image_raw /imu/data_raw
echo '>>> 数据包播放完毕！'
exec bash
"

echo "全部 4 个终端标签页已启动：已分配 12 核并行隔离（0~11 核），预留 12~15 核给操作系统与桌面合成器。"


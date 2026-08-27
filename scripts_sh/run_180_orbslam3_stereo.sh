#!/bin/bash
# ==============================================================================
# 纯双目模式运行 ORB-SLAM3 (180° 双目鱼眼, 960x600, 6.2cm 物理基线)
# 无需 IMU 加速度激励，开篇静止零 Reset，第 1 帧直接三角化建图
# ==============================================================================

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
SLAM_WS_DIR="/home/elp/benchmark_slam_ws"
VOCAB_FILE="$SLAM_WS_DIR/ORB_SLAM3/Vocabulary/ORBvoc.txt"
CONFIG_FILE="$SLAM_WS_DIR/ORB_SLAM3/Examples/Stereo-Inertial/my_stereo_imu_180_960.yaml"
BAG_DIR="${1:-$WORKSPACE_DIR/datasets/my_dataset_20260820_040247}"

echo "=========================================================="
echo " 正在启动 ORB-SLAM3 纯双目鱼眼系统 (Stereo Only - 零Reset模式)"
echo " 数据集路径: $BAG_DIR"
echo " 配置文件:   $CONFIG_FILE"
echo " 词袋路径:   $VOCAB_FILE"
echo "=========================================================="

# 1. 彻底清理旧的进程与临时轨迹文件
pkill -9 -f stereo-inertial 2>/dev/null
pkill -9 -f stereo-slam-node 2>/dev/null
pkill -9 -f vins_node 2>/dev/null
pkill -9 -f loop_fusion_node 2>/dev/null
pkill -9 -f rviz2 2>/dev/null
pkill -9 -f "ros2 bag play" 2>/dev/null
pkill -9 -f "/stereo " 2>/dev/null
rm -f "$WORKSPACE_DIR/CameraTrajectory.txt" "$WORKSPACE_DIR/KeyFrameTrajectory.txt"
sleep 1

# 2. 启动 ORB-SLAM3 纯双目节点 (带 Pangolin 视口)
gnome-terminal --title="ORB-SLAM3-Stereo" -- bash -c "
export LD_LIBRARY_PATH='$SLAM_WS_DIR/ORB_SLAM3/lib:$SLAM_WS_DIR/ORB_SLAM3/Thirdparty/DBoW2/lib:$SLAM_WS_DIR/ORB_SLAM3/Thirdparty/g2o/lib:/home/elp/Pangolin/build/src':\$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source '$SLAM_WS_DIR/install/setup.bash'
cd '$WORKSPACE_DIR'

echo '>>> 正在启动 ORB-SLAM3 纯双目节点 (KannalaBrandt8 鱼眼直连模式 + Pangolin GUI)...'
ros2 run orbslam3 stereo '$VOCAB_FILE' '$CONFIG_FILE' false --ros-args \
    -r /camera/left:=/camera/left/image_raw \
    -r /camera/right:=/camera/right/image_raw

exec bash
"

echo ">>> 正在等待 ORB-SLAM3 词袋加载并进入就绪状态..."
for i in {1..20}; do
    if ps aux | grep -v grep | grep -E "install/orbslam3/lib/orbslam3/stereo" > /dev/null; then
        sleep 3
        break
    fi
    sleep 1
done

# 3. 启动 ROS2 数据包回放
gnome-terminal --title="ROS2-Bag-Play" -- bash -c "
source /opt/ros/humble/setup.bash
echo '>>> 正在以 1.0 倍速播放测试数据包: $BAG_DIR'
ros2 bag play '$BAG_DIR' --rate 1.0
echo '>>> 数据包播放完毕！'
exec bash
"

echo "=========================================================="
echo " [OK] ORB-SLAM3 纯双目系统已成功启动！"
echo "=========================================================="

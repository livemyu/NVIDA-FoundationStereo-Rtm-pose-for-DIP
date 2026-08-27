#!/bin/bash
# 使用 180 标定参数运行 ORB-SLAM3 双目鱼眼惯导算法与数据包回放

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
BENCHMARK_WS="/home/elp/benchmark_slam_ws"
ORB_ROOT="$BENCHMARK_WS/ORB_SLAM3"
VOC_PATH="$ORB_ROOT/Vocabulary/ORBvoc.txt"
CONFIG_PATH="$ORB_ROOT/Examples/Stereo-Inertial/my_stereo_imu_180_960.yaml"
BAG_DIR="${1:-$WORKSPACE_DIR/datasets/my_dataset_20260822_084530}"

echo "=========================================================="
echo " 正在启动 ORB-SLAM3 双目鱼眼惯导系统 (Atlas + Kannala-Brandt)"
echo " 数据集路径: $BAG_DIR"
echo " 配置文件:   $CONFIG_PATH"
echo " 词袋路径:   $VOC_PATH"
echo "=========================================================="

# 清理旧进程
pkill -f stereo-inertial 2>/dev/null
pkill -f "ros2 bag play" 2>/dev/null
sleep 1

mkdir -p "$WORKSPACE_DIR/output"

# 1. 启动 ORB-SLAM3 核心节点（自带 Pangolin 3D 视口与特征窗口）
export DISPLAY=:0
export LD_LIBRARY_PATH="$ORB_ROOT/lib:$ORB_ROOT/Thirdparty/DBoW2/lib:$ORB_ROOT/Thirdparty/g2o/lib:/home/elp/Pangolin/build/src:$LD_LIBRARY_PATH"
source /opt/ros/humble/setup.bash
source "$BENCHMARK_WS/install/setup.bash"
cd "$WORKSPACE_DIR"

echo ">>> 正在启动 ORB-SLAM3 双目鱼眼惯导节点..."
/home/elp/benchmark_slam_ws/install/orbslam3/lib/orbslam3/stereo-inertial "$VOC_PATH" "$CONFIG_PATH" false false > "$WORKSPACE_DIR/output/orbslam3.log" 2>&1 &
SLAM_PID=$!

echo ">>> 正在等待 ORB-SLAM3 词袋加载并进入就绪状态..."
for i in $(seq 1 45); do
    if ros2 node list 2>/dev/null | grep -q "ORB_SLAM3_ROS2"; then
        echo ">>> [OK] 检测到 ORB-SLAM3 节点已成功启动并就绪 (PID: $SLAM_PID)！"
        break
    fi
    sleep 1
done
sleep 2

# 2. 启动 ros2 bag 播放 (默认 1.0 正常速度)
PLAY_RATE="${2:-1.0}"
echo ">>> 正在以 $PLAY_RATE 倍速播放测试数据包: $BAG_DIR"
source /opt/ros/humble/setup.bash
ros2 bag play "$BAG_DIR" --rate $PLAY_RATE

echo ">>> 数据包播放完毕！正在等待 ORB-SLAM3 自动完成图优化并导出轨迹..."
wait $SLAM_PID 2>/dev/null || true

echo "=========================================================="
echo " [OK] 测试流程全部完成！轨迹文件已保存："
ls -lh "$WORKSPACE_DIR"/*Trajectory.txt 2>/dev/null || true
echo "=========================================================="

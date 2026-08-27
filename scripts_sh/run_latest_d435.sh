#!/bin/bash
set -e

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
CONFIG_FILE="$WORKSPACE_DIR/Stereo_cam_ws/my_d435_stereo_pure_visual.yaml"
DATASET_DIR="$WORKSPACE_DIR/d435_vo_20260810_114041"

echo "=========================================================="
echo " 正在对最新的 D435 录像进行 VINS + Loop 轨迹计算与评估"
echo " 数据集路径: $DATASET_DIR"
echo " 配置文件:   $CONFIG_FILE"
echo "=========================================================="

# 清理旧输出
rm -rf "$WORKSPACE_DIR/output"
mkdir -p "$WORKSPACE_DIR/output/pose_graph"

# 引入 ROS2 环境
source /opt/ros/humble/setup.bash
source /home/elp/vins_ros2_ws/install/setup.bash

# 1. 启动 vins_node
cd "$WORKSPACE_DIR"
echo ">>> [1/4] 启动 vins_node..."
ros2 run vins vins_node "$CONFIG_FILE" > "$WORKSPACE_DIR/output/vins.log" 2>&1 &
VINS_PID=$!

# 2. 启动 loop_fusion_node
echo ">>> [2/4] 启动 loop_fusion_node..."
ros2 run loop_fusion loop_fusion_node "$CONFIG_FILE" --ros-args \
    -r /vins_estimator/odometry:=/odometry \
    -r /vins_estimator/keyframe_pose:=/keyframe_pose \
    -r /vins_estimator/extrinsic:=/extrinsic \
    -r /vins_estimator/keyframe_point:=/keyframe_point \
    -r /vins_estimator/margin_cloud:=/margin_cloud > "$WORKSPACE_DIR/output/loop.log" 2>&1 &
LOOP_PID=$!

cleanup() {
    echo ">>> 正在停止后台节点..."
    kill $VINS_PID $LOOP_PID 2>/dev/null || true
}
trap cleanup EXIT

echo ">>> 等待 3 秒节点初始化..."
sleep 3

# 3. 开始播放数据集
echo ">>> [3/4] 开始回放 D435 数据集: $DATASET_DIR ..."
python3 "$WORKSPACE_DIR/play_d435_dataset.py" "$DATASET_DIR"

echo ">>> 等待 5 秒完成后端闭环优化落盘..."
sleep 5

# 4. 执行轨迹评估
echo ">>> [4/4] 执行轨迹精度评估..."
python3 "$WORKSPACE_DIR/evaluate_trajectory.py" --raw "$WORKSPACE_DIR/output/vio.csv" --loop "$WORKSPACE_DIR/output/vio_loop.csv"

echo "=========================================================="
echo " 运行完成！生成的轨迹结果已保存在 $WORKSPACE_DIR/output/"
echo "=========================================================="

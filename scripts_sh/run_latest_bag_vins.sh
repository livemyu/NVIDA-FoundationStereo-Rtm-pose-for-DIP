#!/bin/bash
set -e

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
CONFIG_FILE="$WORKSPACE_DIR/Stereo_cam_ws/my_stereo_imu_config_fixed.yaml"
BAG_DIR="$WORKSPACE_DIR/my_dataset_20260810_114043"

echo "=========================================================="
echo " 正在对最新 11.97GB 数据集进行 VINS-Fusion + Loop-Fusion 实时计算与评估"
echo " 数据集路径: $BAG_DIR"
echo " 配置文件:   $CONFIG_FILE"
echo "=========================================================="

# 清理旧输出
rm -rf "$WORKSPACE_DIR/output"
mkdir -p "$WORKSPACE_DIR/output/pose_graph"

# 引入 ROS2 环境
source /opt/ros/humble/setup.bash
source /home/elp/vins_ros2_ws/install/setup.bash

cd "$WORKSPACE_DIR"

# 1. 启动图像格式修复中继节点 (修正 step = width)
echo ">>> [1/5] 启动图像 step 格式校正节点..."
python3 "$WORKSPACE_DIR/fix_bag_topics.py" > "$WORKSPACE_DIR/output/fix_topics.log" 2>&1 &
FIX_PID=$!

# 2. 启动 vins_node
echo ">>> [2/5] 启动 vins_node..."
ros2 run vins vins_node "$CONFIG_FILE" > "$WORKSPACE_DIR/output/vins.log" 2>&1 &
VINS_PID=$!

# 3. 启动 loop_fusion_node
echo ">>> [3/5] 启动 loop_fusion_node..."
ros2 run loop_fusion loop_fusion_node "$CONFIG_FILE" --ros-args \
    -r /vins_estimator/odometry:=/odometry \
    -r /vins_estimator/keyframe_pose:=/keyframe_pose \
    -r /vins_estimator/extrinsic:=/extrinsic \
    -r /vins_estimator/keyframe_point:=/keyframe_point \
    -r /vins_estimator/margin_cloud:=/margin_cloud > "$WORKSPACE_DIR/output/loop.log" 2>&1 &
LOOP_PID=$!

cleanup() {
    echo ">>> 正在停止后台节点..."
    kill $FIX_PID $VINS_PID $LOOP_PID 2>/dev/null || true
}
trap cleanup EXIT

echo ">>> 等待 3 秒节点初始化..."
sleep 3

# 4. 开始播放数据集
echo ">>> [4/5] 开始播放 11.97GB 数据包: $BAG_DIR ..."
ros2 bag play "$BAG_DIR"

echo ">>> 等待 5 秒完成后端闭环优化落盘..."
sleep 5

# 5. 执行轨迹评估
echo ">>> [5/5] 执行轨迹精度评估..."
python3 "$WORKSPACE_DIR/evaluate_trajectory.py" --raw "$WORKSPACE_DIR/output/vio.csv" --loop "$WORKSPACE_DIR/output/vio_loop.csv"

echo "=========================================================="
echo " 运行完成！生成的轨迹结果已保存在 $WORKSPACE_DIR/output/"
echo "=========================================================="

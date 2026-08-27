#!/bin/bash
# 运行环境: PC 端
# 功能: 回放从 Jetson 拷贝到 PC 的离线数据包

if [ -z "$1" ]; then
    echo "用法: ./play_dataset.sh <数据包路径>"
    echo "示例: ./play_dataset.sh my_dataset_20260806_102500"
    exit 1
fi

BAG_PATH=$1

if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# 确保与当前其他节点的 DOMAIN_ID 一致
export ROS_DOMAIN_ID=42

echo "================================================="
echo " 正在回放数据包: ${BAG_PATH}"
echo " DOMAIN_ID: ${ROS_DOMAIN_ID}"
echo " 提示: 此时不应运行网络通信或者Jetson板端驱动，"
echo " 确保数据为纯本地闭环，避免受到外部网络干扰。"
echo "================================================="

# 支持透传 -r 倍速参数，例如 ./play_dataset.sh my_dataset -r 2.0
ros2 bag play "$@"

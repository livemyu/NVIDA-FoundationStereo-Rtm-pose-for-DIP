#!/bin/bash
# 停止所有 VINS、Loop-Fusion、RViz2 以及 ROS2 播放节点与相关终端

echo ">>> 正在清理并关闭所有 VINS / Loop / RViz / ROS2 播放节点与终端..."

# 1. 杀死核心进程
pkill -9 -f vins_node 2>/dev/null
pkill -9 -f loop_fusion_node 2>/dev/null
pkill -9 -f rviz2 2>/dev/null
pkill -9 -f "ros2 bag play" 2>/dev/null
pkill -9 -f "ros2 run vins" 2>/dev/null
pkill -9 -f "ros2 run loop_fusion" 2>/dev/null

# 2. 杀死特定的 gnome-terminal 标签窗口 (如果有标题标识)
wmctrl -c "1. VINS-Node-180" 2>/dev/null || true
wmctrl -c "2. Loop-Fusion" 2>/dev/null || true
wmctrl -c "3. RViz2-Visualization" 2>/dev/null || true
wmctrl -c "4. ROS2-Bag-Play" 2>/dev/null || true

echo ">>> 全部相关节点与窗口已安全关闭。"

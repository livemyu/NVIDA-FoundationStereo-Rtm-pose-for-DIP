#!/bin/bash
# ==============================================================================
# PC 端实时接收 Jetson 广播的 VINS 轨迹并使用 RViz 呈现
# ==============================================================================

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
RVIZ_CONFIG="$WORKSPACE_DIR/Stereo_cam_ws/vins_only_rviz_config.rviz"
CYCLONE_CONFIG="$WORKSPACE_DIR/cyclonedds.xml"

echo "=========================================================="
echo " 正在启动 PC 端 RViz 实时接收 Jetson 发布的 VINS-Fusion 轨迹"
echo " CycloneDDS 配置: $CYCLONE_CONFIG"
echo "=========================================================="

source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$CYCLONE_CONFIG"

rviz2 -d "$RVIZ_CONFIG"

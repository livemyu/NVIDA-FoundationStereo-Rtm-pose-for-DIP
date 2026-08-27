#!/bin/bash
# 使用 180 标定参数运行 OpenVINS 双目鱼眼惯导算法与数据包回放

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
OV_WS_DIR="/home/elp/benchmark_slam_ws"
CONFIG_PATH="$OV_WS_DIR/src/open_vins/config/stereo_180_960/estimator_config.yaml"
BAG_DIR="${1:-$WORKSPACE_DIR/datasets/my_dataset_20260818_073425}"

echo "=========================================================="
echo " 正在启动 OpenVINS 双目鱼眼惯导系统 (MSCKF)"
echo " 数据集路径: $BAG_DIR"
echo " 配置文件:   $CONFIG_PATH"
echo "=========================================================="

# 清理旧进程
pkill -f run_subscribe_msckf 2>/dev/null
pkill -f rviz2 2>/dev/null
pkill -f "ros2 bag play" 2>/dev/null
sleep 1

mkdir -p "$WORKSPACE_DIR/output"

# 1. 启动 OpenVINS 节点
gnome-terminal --tab --title="1. OpenVINS-Node" -- bash -c "
source /opt/ros/humble/setup.bash
source '$OV_WS_DIR/install/setup.bash'
cd '$WORKSPACE_DIR'
echo '>>> 启动 OpenVINS run_subscribe_msckf 节点...'
ros2 launch ov_msckf subscribe.launch.py \
    config_path:='$CONFIG_PATH' \
    rviz_enable:=false \
    verbosity:=INFO \
    use_stereo:=true \
    max_cameras:=2 \
    save_total_state:=false
exec bash
"

sleep 2

# 2. 启动 RViz2
gnome-terminal --tab --title="2. OpenVINS-RViz" -- bash -c "
source /opt/ros/humble/setup.bash
source '$OV_WS_DIR/install/setup.bash'
echo '>>> 启动 RViz2 监控 OpenVINS 估计位姿与路标点...'
rviz2 -d '$OV_WS_DIR/src/open_vins/ov_msckf/launch/display_ros2.rviz'
exec bash
"

sleep 1

# 3. 启动 ros2 bag 播放
gnome-terminal --tab --title="3. Bag-Player" -- bash -c "
source /opt/ros/humble/setup.bash
echo '>>> 正在播放测试数据包: $BAG_DIR'
ros2 bag play '$BAG_DIR' --clock --rate 1.0
echo '>>> 数据包播放完毕！'
exec bash
"

echo "=========================================================="
echo " [OK] OpenVINS 双目系统已成功启动！"
echo "=========================================================="

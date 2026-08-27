#!/bin/bash

WORKSPACE_DIR="/home/elp/picture_resize_recording_NVIDA"
VINS_WS_DIR="/home/elp/vins_ros2_ws"
CONFIG_FILE="$WORKSPACE_DIR/Stereo_cam_ws/my_stereo_imu_config_fixed.yaml"
BAG_DIR="$WORKSPACE_DIR/my_dataset_20260810_114043"
RVIZ_CONFIG="$WORKSPACE_DIR/Stereo_cam_ws/vins_only_rviz_config.rviz"

echo "=========================================================="
echo " 正在以可视化终端模式启动 11.97GB 数据集 VINS + Loop + RViz"
echo " 数据集路径: $BAG_DIR"
echo " 配置文件:   $CONFIG_FILE"
echo "=========================================================="

mkdir -p "$WORKSPACE_DIR/output/pose_graph"

# 1. 启动第一个终端窗口 (1. Step-Fix-Node)
gnome-terminal --title="1. Step-Fix-Node" -- bash -c "
source /opt/ros/humble/setup.bash
cd '$WORKSPACE_DIR'
echo '>>> [1/5] 启动图像 step 格式校正中继节点...'
python3 '$WORKSPACE_DIR/fix_bag_topics.py'
exec bash
"

# 2. 启动第二个标签页 (2. VINS-Node)
gnome-terminal --tab --title="2. VINS-Node" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
cd '$WORKSPACE_DIR'
echo '>>> [2/5] 启动 VINS-Node (使用解压标定参数 1920x1200)...'
ros2 run vins vins_node '$CONFIG_FILE'
exec bash
"

# 3. 启动第三个标签页 (3. Loop-Fusion)
gnome-terminal --tab --title="3. Loop-Fusion" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
cd '$WORKSPACE_DIR'
echo '>>> [3/5] 启动 Loop-Fusion 回环优化后端...'
ros2 run loop_fusion loop_fusion_node '$CONFIG_FILE' --ros-args \
    -r /vins_estimator/odometry:=/odometry \
    -r /vins_estimator/keyframe_pose:=/keyframe_pose \
    -r /vins_estimator/extrinsic:=/extrinsic \
    -r /vins_estimator/keyframe_point:=/keyframe_point \
    -r /vins_estimator/margin_cloud:=/margin_cloud
exec bash
"

# 4. 启动第四个标签页 (4. RViz2-Visualization)
gnome-terminal --tab --title="4. RViz2-Visualization" -- bash -c "
source /opt/ros/humble/setup.bash
source '$VINS_WS_DIR/install/setup.bash'
echo '>>> [4/5] 启动 RViz2 可视化窗口...'
rviz2 -d '$RVIZ_CONFIG'
exec bash
"

# 5. 启动第五个标签页 (5. ROS2-Bag-Play)
gnome-terminal --tab --title="5. ROS2-Bag-Play" -- bash -c "
source /opt/ros/humble/setup.bash
cd '$WORKSPACE_DIR'
echo '>>> [5/5] 等待 3 秒节点就绪后开始回放 11.97GB 数据集...'
sleep 3
ros2 bag play '$BAG_DIR'
exec bash
"

echo "已成功在桌面启动 5 个独立终端标签页及 RViz2 界面。"

#!/bin/bash
# ==============================================================================
# PC 端 RTAB-Map 稠密 3D 建图与 RTAB-Map 官方 3D GUI 可视化启动脚本 (ROS 2 Humble)
# ==============================================================================

# 1. 刷新 ROS 2 Humble 环境变量
if [ -f "/opt/ros/humble/setup.bash" ];
then
    source /opt/ros/humble/setup.bash
else
    echo "[错误] 找不到 ROS 2 Humble 环境 /opt/ros/humble/setup.bash"
    exit 1
fi

# 2. 统一 ROS DOMAIN ID (必须与 Jetson 保持一致)
export ROS_DOMAIN_ID=42

echo "================================================="
echo " 正在启动 PC 端 RTAB-Map 稠密 3D 建图节点..."
echo " 当前 ROS_DOMAIN_ID = $ROS_DOMAIN_ID"
echo " 图像传输模式 = compressed (由 Jetson C++ 驱动直接编码)"
echo " 可视化模式 = rtabmap_viz (原生 3D 建图仪表盘)"
echo " 正在接收: "
echo "    - 左图像(压缩): /camera/left/image_raw/compressed"
echo "    - 右图像(压缩): /camera/right/image_raw/compressed"
echo "    - VIO 里程计: /vins_estimator/odometry"
echo "    - 稠密 3D 点云: /rtabmap/cloud_map"
echo "================================================="

# 3. 静态 TF 广播
ros2 run tf2_ros static_transform_publisher --x -0.007 --y 0.042 --z 0.006 --roll -1.5707963 --pitch 0 --yaw -1.5707963 --frame-id body --child-frame-id camera_0_link --ros-args -p use_sim_time:=true &
TF_PID1=$!

ros2 run tf2_ros static_transform_publisher --x -0.007 --y -0.019 --z 0.005 --roll -1.5707963 --pitch 0 --yaw -1.5707963 --frame-id body --child-frame-id camera_1_link --ros-args -p use_sim_time:=true &
TF_PID2=$!


# 4. 启动 stereo_image_proc disparity_node 解算 SGBM 深度视差图
echo "正在启动 stereo_image_proc SGBM 深度视差解算节点..."
ros2 run stereo_image_proc disparity_node --ros-args \
    -r left/image_rect:=/camera/left/image_raw \
    -r right/image_rect:=/camera/right/image_raw \
    -r left/camera_info:=/camera/left/camera_info \
    -r right/camera_info:=/camera/right/camera_info \
    -r disparity:=/disparity \
    -p approx_sync:=true \
    -p stereo_algorithm:=1 \
    -p min_disparity:=0 \
    -p disparity_range:=128 &
DISPARITY_PID=$!

# 5. 退出清理逻辑
trap "kill $TF_PID1 $TF_PID2 $DISPARITY_PID 2>/dev/null" EXIT

# 6. 启动 RTAB-Map 双目 + SGBM 深度视差 + 外部 VIO 里程计建图，拉起 rtabmap_viz 原生 GUI
ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:="--delete_db_on_start \
                   --Stereo/Dense true \
                   --Stereo/OpticalFlow false \
                   --Stereo/MaxDisparity 128 \
                   --Stereo/WinSize 15 \
                   --Vis/EstimationType 1 \
                   --Vis/MaxFeatures 1000 \
                   --Grid/3D true \
                   --Grid/CellSize 0.02 \
                   --Grid/RangeMax 5.0 \
                   --Grid/RayTracing true \
                   --Grid/ClusterRadius 0.05 \
                   --RTABMap/DetectionRate 1.0" \
    stereo:=true \
    subscribe_rgb:=false \
    subscribe_disparity:=true \
    disparity_image_topic:=/disparity \
    left_image_topic:=/camera/left/image_raw \
    right_image_topic:=/camera/right/image_raw \
    left_camera_info_topic:=/camera/left/camera_info \
    right_camera_info_topic:=/camera/right/camera_info \
    visual_odometry:=false \
    subscribe_odom:=true \
    subscribe_odom_info:=false \
    odom_topic:=/loop_fusion/odometry \
    frame_id:=body \
    odom_frame_id:=world \
    map_frame_id:=map \
    publish_tf_map:=true \
    qos:=2 \
    qos_image:=2 \
    qos_camera_info:=2 \
    qos_odom:=2 \
    approx_sync:=true \
    use_sim_time:=true \
    rtabmap_viz:=true \
    rviz:=false

#!/bin/bash
# ==============================================================================
# PC 端 RTAB-Map 已建地图“重定位与纯定位导航模式”启动脚本 (ROS 2 Humble)
# 说明: 本脚本加载 maps/prior_room_map.db 数据库，禁止写入新地图节点，开机即用
# ==============================================================================

# 1. 刷新 ROS 2 Humble 环境变量
if [ -f "/opt/ros/humble/setup.bash" ];
then
    source /opt/ros/humble/setup.bash
else
    echo "[错误] 找不到 ROS 2 Humble 环境 /opt/ros/humble/setup.bash"
    exit 1
fi

# 2. 统一 ROS DOMAIN ID
export ROS_DOMAIN_ID=42

DB_PATH="/home/elp/picture_resize_recording_NVIDA/maps/prior_room_map.db"

if [ ! -f "$DB_PATH" ]; then
    echo "[错误] 找不到预存地图数据库文件: $DB_PATH"
    exit 1
fi

echo "================================================="
echo " 正在启动 RTAB-Map 纯重定位模式 (Localization Mode)..."
echo " 当前加载已建地图数据库 = $DB_PATH"
echo " 当前 ROS_DOMAIN_ID = $ROS_DOMAIN_ID"
echo " 正在接收位姿与图像，与预存地图进行毫秒级匹配..."
echo "================================================="

# 3. 静态 TF 广播
ros2 run tf2_ros static_transform_publisher --x -0.007 --y 0.042 --z 0.006 --roll -1.5707963 --pitch 0 --yaw -1.5707963 --frame-id body --child-frame-id camera_0_link &
TF_PID1=$!

ros2 run tf2_ros static_transform_publisher --x -0.007 --y -0.019 --z 0.005 --roll -1.5707963 --pitch 0 --yaw -1.5707963 --frame-id body --child-frame-id camera_1_link &
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

# 6. 启动 RTAB-Map 跨姿态全局重定位模式 (强化旋转不变性 ORB 特征与全域闭环匹配)
ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:="--Mem/IncrementalMemory true \
                   --Mem/LocalizationDataSaved true \
                   --RGBD/ProximityBySpace false \
                   --RGBD/ProximityByTime false \
                   --RGBD/OptimizeFromGraphEnd false \
                   --Vis/FeatureType 6 \
                   --Vis/MinInliers 8 \
                   --Vis/MaxFeatures 2000 \
                   --Kp/MaxFeatures 2000 \
                   --Stereo/Dense true \
                   --Stereo/MaxDisparity 128 \
                   --Grid/3D true \
                   --Grid/CellSize 0.02 \
                   --Grid/RangeMax 5.0 \
                   --RTABMap/DetectionRate 1.0" \
    database_path:="$DB_PATH" \
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
    image_transport:=raw \
    wait_for_transform:=1.0 \
    rtabmap_viz:=true \
    rviz:=false

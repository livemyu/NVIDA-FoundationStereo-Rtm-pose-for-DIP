#!/bin/bash
# 运行环境: PC 端
# 功能: 通过 SSH 控制 Jetson 在其 NVMe SSD 上录制数据包
# 按 Ctrl+C 即可安全停止录制

# 生成带有时间戳的数据包名称
BAG_NAME="my_dataset_$(date +%Y%m%d_%H%M%S)"
echo "即将通过 SSH 控制 Jetson 开始录制数据包: ${BAG_NAME}"
echo "录制物理位置 (Jetson NVMe): /home/jetson/nvidia_stereo_vins_deployment/${BAG_NAME}"
echo "按 Ctrl+C 停止录制并安全保存。"

# 使用 -t 和 -it 确保 Ctrl+C 信号能够正确透传给 ros2 bag 进程
sshpass -p '123456' ssh -t jetson@192.168.0.22 "docker exec -it ros2_vins_container bash -c 'source /opt/ros/humble/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export CYCLONEDDS_URI=\"file:///home/elp/cyclonedds.xml\" && cd /home/elp && ros2 bag record -o ${BAG_NAME} /camera/left/image_raw /camera/right/image_raw /camera/left/camera_info /camera/right/camera_info /imu/data_raw /camera/left/mag /stereo_0/camera/left/image_raw /stereo_0/camera/right/image_raw /stereo_0/camera/left/camera_info /stereo_0/camera/right/camera_info /stereo_0/imu/data_raw /stereo_0/camera/left/mag --max-cache-size 200000000'"

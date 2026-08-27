# RTMPose & 空间智能（Spatial AI）使用示例与运行指南

本代码包包含 **RTMPose 姿态估计、双目深度估计与 5 个空间具身智能应用** 的全部调用代码、流水线与标定配置文件（纯代码示例包，不含体积庞大的模型权重文件）。

---

## 一、 目录结构说明

```text
├── demos/                                 # 5 个具身智能应用场景示例
│   ├── 01_embodied_dataset_exporter/      # 示例 1: 具身智能 3D 轨迹数据集导出
│   │   └── hand_trajectory_exporter.py
│   ├── 02_spatial_grasping_metric/        # 示例 2: 毫米级双手捏合距离与抓取测量
│   │   └── spatial_pinch_meter.py
│   ├── 03_virtual_touch_interaction/      # 示例 3: 空间非接触式空中悬浮虚拟触控
│   │   └── air_touch_interface.py
│   ├── 04_body_safety_ergonomics/         # 示例 4: 人体安全距离与人机工效评估
│   │   └── body_spatial_safety.py
│   ├── 05_ros2_spatial_mapping/           # 示例 5: ROS 2 RViz3D 骨骼与稠密点云发布
│   │   ├── bag_to_foundation_depth.py
│   │   └── ros2_rtmpose_rviz_publisher.py
│   └── README.md
├── rtmpose_depth_pipeline.py              # 核心流水线: RTMPose + 3D 深度骨骼解算
├── rtmpose_body_pipeline.py               # 核心流水线: 人体全身 17 关键点实时检测与平滑
├── rtmpose_only_pipeline.py               # 核心流水线: 极速 2D 手势估计 (< 5ms)
├── stereo_depth_pipeline_gpu.py           # 核心通用 GPU 双目去畸变与深度引擎
├── build_trt_foundation_dense_map.py      # Nav2 2D 占据栅格地图生成器
├── export_3d_pointcloud.py                # 3D 稠密彩色点云 (.ply) 生成器
├── my_stereo_imu_config_180_960.yaml      # 180° 鱼眼双目相机与外参配置文件
├── config_180/                            # 左右目鱼眼内参与畸变系数
│   ├── left_960.yaml
│   └── right_960.yaml
└── mask0.png / mask1.png                  # 鱼眼边缘黑圈过滤掩码
```

---

## 二、 5 个示例脚本的启动运行命令

### 示例 1：具身智能 3D 手部轨迹与抓取导出
```bash
python3 demos/01_embodied_dataset_exporter/hand_trajectory_exporter.py \
    --video /path/to/stereo_video.mp4 \
    --calib my_stereo_imu_config_180_960.yaml \
    --output_dir output/01_trajectory/
```

### 示例 2：空间抓取与毫米级捏合测量（Pinch Meter）
```bash
python3 demos/02_spatial_grasping_metric/spatial_pinch_meter.py \
    --video /path/to/stereo_video.mp4 \
    --calib my_stereo_imu_config_180_960.yaml \
    --output_dir output/02_pinch_metric/
```

### 示例 3：非接触式空中悬浮虚拟触控界面（Air Touch）
```bash
python3 demos/03_virtual_touch_interaction/air_touch_interface.py \
    --video /path/to/stereo_video.mp4 \
    --calib my_stereo_imu_config_180_960.yaml \
    --output_dir output/03_air_touch/
```

### 示例 4：人机安全距离与人体姿态安全预警（Safety & Ergonomics）
```bash
python3 demos/04_body_safety_ergonomics/body_spatial_safety.py \
    --video /path/to/stereo_video.mp4 \
    --calib my_stereo_imu_config_180_960.yaml \
    --output_dir output/04_safety/
```

### 示例 5：ROS 2 RViz 3D 骨骼发布与稠密点云建图
```bash
# 启动 ROS 2 实时 3D 骨骼话题发布
source /opt/ros/humble/setup.bash
python3 demos/05_ros2_spatial_mapping/ros2_rtmpose_rviz_publisher.py \
    --video /path/to/stereo_video.mp4 \
    --calib my_stereo_imu_config_180_960.yaml

# 启动 RViz 查看
rviz2
```

---

## 三、 稠密 2D/3D 建图示例

### 1. 生成 ROS 2 Nav2 标准 2D 占据栅格地图
```bash
python3 build_trt_foundation_dense_map.py \
    --bag /home/elp/data/my_dataset_1F \
    --traj /home/elp/spatial_ai_trt_ws/1f_12f_13f/traj_1F_camera.txt \
    --output_dir /home/elp/navigation_ws/maps \
    --map_name map_1F
```

### 2. 导出 3 种着色模式的标准 3D 点云 (`.ply`)
```bash
python3 export_3d_pointcloud.py \
    --bag /home/elp/data/my_dataset_1F \
    --traj /home/elp/spatial_ai_trt_ws/1f_12f_13f/traj_1F_camera.txt \
    --output_dir /home/elp/navigation_ws/maps \
    --floor 1F
```

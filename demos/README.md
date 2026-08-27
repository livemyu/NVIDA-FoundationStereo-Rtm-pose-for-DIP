# 双目立体视觉与 3D 姿态估计 5 大核心场景 Demo 指南

本项目基于 **TensorRT 10.3.0 FP16 + GPU 双目立体校正 + FoundationStereo 深度估计 + RTMPose 手部/人体姿态估计**，构建了 5 个覆盖具身智能、空间交互、工业安全与 SLAM 导航的独立落地模块。

---

## 目录结构

```text
/home/elp/picture_resize_recording_NVIDA/demos/
├── 01_embodied_dataset_exporter/       # Demo 1: 具身智能 3D 手部轨迹与数据集导出器
│   └── hand_trajectory_exporter.py     # 输出 .json (元数据) 与 .npy (高频张量)
├── 02_spatial_grasping_metric/         # Demo 2: 空间抓取与指尖物理开合度精准测量 Demo
│   └── spatial_pinch_meter.py          # 拇指-食指厘米级间距仪表盘与手势分类
├── 03_virtual_touch_interaction/       # Demo 3: 空间无接触虚拟悬浮触控交互 Demo
│   └── air_touch_interface.py          # 空间虚拟交互平面与穿透点击事件触发
├── 04_body_safety_ergonomics/          # Demo 4: 人体 3D 空间安全警戒与工序姿态分析 Demo
│   └── body_spatial_safety.py          # 17 点空间受力、脊柱倾角与安全距离警戒
└── 05_ros2_spatial_mapping/            # Demo 5: ROS 2 / SLAM / Nav2 3D 建图与 RViz 节点
    ├── bag_to_foundation_depth.py      # 离线批量生成 16-Bit 深度图与 3D 点云
    └── ros2_rtmpose_rviz_publisher.py  # ROS 2 Humble RViz2 3D 骨架 Marker 发布节点
```

---

## 5 个 Demo 运行指令示例（板端执行）

### Demo 1: 具身智能 3D 轨迹数据集导出器
```bash
python3 demos/01_embodied_dataset_exporter/hand_trajectory_exporter.py \
  --video /home/elp/spatial_ai_trt_ws/videos/30fps_allhand.mp4 \
  --output_video /home/elp/spatial_ai_trt_ws/output_results/01_rtmpose_hand/demo1_embodied_exporter.mp4 \
  --output_json /home/elp/spatial_ai_trt_ws/output_results/04_pointcloud_and_eval/hand_3d_trajectory_dataset.json \
  --output_npy /home/elp/spatial_ai_trt_ws/output_results/04_pointcloud_and_eval/hand_3d_trajectory_dataset.npy
```

### Demo 2: 空间抓取与指尖物理开合度测量 Demo
```bash
python3 demos/02_spatial_grasping_metric/spatial_pinch_meter.py \
  --video /home/elp/spatial_ai_trt_ws/videos/30fps_allhand.mp4 \
  --output_video /home/elp/spatial_ai_trt_ws/output_results/01_rtmpose_hand/demo2_spatial_grasping.mp4
```

### Demo 3: 空间无接触虚拟触控交互 Demo
```bash
python3 demos/03_virtual_touch_interaction/air_touch_interface.py \
  --video /home/elp/spatial_ai_trt_ws/videos/30fps_allhand.mp4 \
  --output_video /home/elp/spatial_ai_trt_ws/output_results/01_rtmpose_hand/demo3_air_touch.mp4
```

### Demo 4: 人体 3D 空间安全警戒与工序姿态分析 Demo
```bash
python3 demos/04_body_safety_ergonomics/body_spatial_safety.py \
  --video /home/elp/spatial_ai_trt_ws/videos/30fps_noj.mp4 \
  --output_video /home/elp/spatial_ai_trt_ws/output_results/01_rtmpose_hand/demo4_body_safety.mp4
```

### Demo 5: ROS 2 / SLAM 16-Bit 深度与点云生成器
```bash
python3 demos/05_ros2_spatial_mapping/bag_to_foundation_depth.py \
  --video /home/elp/spatial_ai_trt_ws/videos/30fps_noj.mp4 \
  --output_dir /home/elp/spatial_ai_trt_ws/output_results/04_pointcloud_and_eval/nav2_slam_rgbd_dataset \
  --max_frames 100
```

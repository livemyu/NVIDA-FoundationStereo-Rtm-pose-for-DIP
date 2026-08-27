# NVIDIA FoundationStereo & RTMPose for Spatial Embodied AI (DIP)

[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin%20%7C%20x86__64%20CUDA-green.svg)](https://developer.nvidia.com/embedded-computing)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.3%20%2F%2010.7%20FP16-blue.svg)](https://developer.nvidia.com/tensorrt)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%2F%20Iron-orange.svg)](https://docs.ros.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

本项目是一个专为 **双目空间具身智能 (Dual-Eye Intelligence Perception, DIP)** 打造的高性能全栈推理与空间感知系统。结合 **NVIDIA FoundationStereo** 稠密双目深度估计与 **RTMPose / RTMDet** 骨骼姿态算法，实现单帧毫秒级（3~4ms）、300+ FPS 的 3D 人体与双手空间轨迹感知、抓取测量及非接触交互。

---

## 目录
- [一、 核心系统架构](#一-核心系统架构)
- [二、 模型库清单 (models/)](#二-模型库清单-models)
- [三、 核心推理流水线与算法脚本](#三-核心推理流水线与算法脚本)
- [四、 5 大空间具身智能应用 Demo (demos/)](#四-5-大空间具身智能应用-demo-demos)
- [五、 双目鱼眼标定与外参配置](#五-双目鱼眼标定与外参配置)
- [六、 快速开始指南](#六-快速开始指南)
  - [1. 纯 2D 手部轨迹提取与 Web 实时监控](#1-纯-2d-手部轨迹提取与-web-实时监控)
  - [2. 纯 2D 人体骨骼估计](#2-纯-2d-人体骨骼估计)
  - [3. 双目稠密深度 + 3D 空间骨骼解算](#3-双目稠密深度--3d-空间骨骼解算)
  - [4. 交互式一键 TensorRT 模型转换工具](#4-交互式一键-tensorrt-模型转换工具)
- [七、 性能压测基准 (Jetson Orin NX)](#七-性能压测基准-jetson-orin-nx)

---

## 一、 核心系统架构

```
                     ┌────────────────────────────────────────────────────────┐
                     │           180° 双目鱼眼相机 / 4000x1200 视频源          │
                     └───────────────────────────┬────────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │  GPU 去畸变与极线校正 (CUDA)  │                 │    智能双目分离 (左目提取)     │
        └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                        │                                                 │
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │ FoundationStereo (TensorRT)   │                 │   RTMDet + RTMPose-M (TRT)    │
        │ 16-bit 物理稠密深度图 (mm)    │                 │   21 手部关节点 / 17 全身骨骼 │
        └───────────────┬───────────────┘                 └───────────────┬───────────────┘
                        │                                                 │
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                        ┌─────────────────────────────────────────────────┐
                        │      3D 空间骨骼融合 + One-Euro 自适应动态滤波  │
                        │           [X(m), Y(m), Z(m), Confidence]        │
                        └───────────────────────┬─────────────────────────┘
                                                │
       ┌───────────────────────┬────────────────┴────────────────┬────────────────────────┐
       ▼                       ▼                                 ▼                        ▼
┌──────────────┐      ┌─────────────────┐              ┌───────────────────┐    ┌──────────────────┐
│ 具身轨迹导出 │      │  毫米级捏合测量 │              │  空中虚拟触控交互 │    │ ROS2 RViz 3D可视 │
│ (Dataset Exp)│      │  (Pinch Meter)  │              │    (Air Touch)    │    │ (PointCloud+Pose)│
└──────────────┘      └─────────────────┘              └───────────────────┘    └──────────────────┘
```

---

## 二、 模型库清单 (models/)

本仓库自带全套 TensorRT 10.3 FP16 加速引擎与原始 ONNX 模型：

| 模型文件 | 大小 | 说明 / 输入输出规格 |
| :--- | :--- | :--- |
| `rtmpose_hand.engine` / `.onnx` | 30.7M / 55.1M | **RTMPose-M 手部姿态** (21 关键点, 输入 `1x3x256x256`, SimCC 输出) |
| `rtmdet_hand.engine` / `.onnx` | 4.2M / 4.0M | **RTMDet-Nano 手部检测** (输入 `1x3x320x320`, 动态 BBox 输出) |
| `rtmpose_body.engine` / `.onnx` | 29.8M / 54.3M | **RTMPose-M 人体姿态** (17 关键点, 输入 `1x3x256x192`, SimCC 输出) |
| `rtmdet_person.engine` / `.onnx` | 4.2M / 4.0M | **RTMDet-Nano 人体检测** (输入 `1x3x320x320`, 动态 BBox 输出) |
| `handpose.engine` / `.onnx` | 10.5M / 12.8M | 轻量级备用手势姿态估计模型 |
| `yolo11n-pose.onnx` | 11.9M | YOLO11 全身姿态对照模型 |

---

## 三、 核心推理流水线与算法脚本

1. **[scripts_py/infer_rtmpose_hand_trajectory.py](scripts_py/infer_rtmpose_hand_trajectory.py)**:
   * 纯 2D 手部 21 关节点估计与食指指尖彩虹渐变运动轨迹追踪；
   * 自动识别 4000x1200 双目原始画幅并提取左目视角；
   * 内置 MJPEG Web 实时视频流推流服务（支持浏览器 `http://<IP>:8080` 实时监看）。
2. **[scripts_py/infer_rtmpose_body_pose.py](scripts_py/infer_rtmpose_body_pose.py)**:
   * 纯 2D 人体 17 关键点全身骨骼拓扑连线估计；
   * 支持双目左目自动提取与 Web 实时推流。
3. **[scripts_py/rtmpose_depth_pipeline.py](scripts_py/rtmpose_depth_pipeline.py)**:
   * RTMPose 结合 FoundationStereo 双目稠密深度的 3D 空间骨骼解算核心引擎；
   * 包含 PyCUDA GPU 加速前处理、置信度多目标过滤与三维真实空间坐标转换。
4. **[scripts_py/rtmpose_body_pipeline.py](scripts_py/rtmpose_body_pipeline.py)**:
   * 人体 17 关键点 3D 实时检测与平滑追踪流水线。
5. **[scripts_py/stereo_depth_pipeline_gpu.py](scripts_py/stereo_depth_pipeline_gpu.py)**:
   * 双目鱼眼去畸变、极线校正与 TensorRT 深度推理通用 GPU 模块。
6. **[scripts_py/one_euro_filter.py](scripts_py/one_euro_filter.py)**:
   * 工业级 1-Euro Filter 时间序列自适应速度截止频率平滑滤波器。
7. **[scripts_py/convert_onnx_to_trt.py](scripts_py/convert_onnx_to_trt.py)**:
   * 终端交互式一键 TensorRT 模型转换工具（带自动动态维度适配与 GPU 压测）。

---

## 四、 5 大空间具身智能应用 Demo (demos/)

| Demo 目录 | 应用名称 | 核心功能 |
| :--- | :--- | :--- |
| **`01_embodied_dataset_exporter/`** | 具身智能手部 3D 空间轨迹导出器 | 实时导出双手 21 关节点三维物理坐标真值序列 (`.csv`/`.npz`/`.json`) |
| **`02_spatial_grasping_metric/`** | 毫米级空间捏合距离测量仪 | 实时测量食指与拇指三维欧式距离，精准识别抓取、捏合与释放动作 |
| **`03_virtual_touch_interaction/`** | 空中悬浮虚拟触控交互界面 (Air Touch) | 3D 空间触控检测与悬停手势交互，生成无接触式空间操作面板 |
| **`04_body_safety_ergonomics/`** | 人体空间安全距离与作业人机工效评估 | 实时监控人员与危险设备的三维空间距离，评估作业姿态安全等级 |
| **`05_ros2_spatial_mapping/`** | ROS 2 RViz 3D 骨骼与稠密点云发布节点 | 实时发布 ROS 2 骨骼 MarkerArray、CameraInfo 与 Sensor_msgs/PointCloud2 |

---

## 五、 双目鱼眼标定与外参配置

* **`my_stereo_imu_config_180_960.yaml`**：双目外参与 IMU 标定总配置文件；
* **`config_180/left_960.yaml` & `right_960.yaml`**：Kannala-Brandt 鱼眼内参与畸变参数；
* **`mask0.png` & `mask1.png`**：镜头边缘黑圈屏蔽掩码。

---

## 六、 快速开始指南

### 1. 纯 2D 手部轨迹提取与 Web 实时监控

```bash
# 处理录像文件并保存视频（支持 4000x1200 双目原始视频，自动提取左目）
python3 scripts_py/infer_rtmpose_hand_trajectory.py \
    --input "visual_video/noj/30fps_allhand.mp4" \
    --output "hand_trajectory_out.mp4"

# 实时摄像头输入并开启浏览器 Web 监看 (访问 http://<板端IP>:8080)
python3 scripts_py/infer_rtmpose_hand_trajectory.py --input 0 --web
```

### 2. 纯 2D 人体骨骼估计

```bash
python3 scripts_py/infer_rtmpose_body_pose.py \
    --input "visual_video/noj/30fps_noj.mp4" \
    --output "body_pose_out.mp4"
```

### 3. 双目稠密深度 + 3D 空间骨骼解算

```bash
python3 scripts_py/rtmpose_depth_pipeline.py \
    --video "visual_video/noj/30fps_allhand.mp4" \
    --mode dual
```

### 4. 交互式一键 TensorRT 模型转换工具

在目标 Jetson 板端直接运行，支持序号交互选择或命令行一键编译：
```bash
python3 scripts_py/convert_onnx_to_trt.py
```

---

## 七、 性能压测基准 (Jetson Orin NX)

在 **NVIDIA Jetson Orin NX (MAXN_SUPER 模式)** 实测基准数据：

* **RTMDet-Nano 手部检测**：**2.10 ms** (470+ FPS)
* **RTMPose-M 手部 21 关键点**：**3.07 ms** (326.0 FPS)
* **RTMDet-Nano 人体检测**：**3.45 ms** (286.6 FPS)
* **RTMPose-M 人体 17 关键点**：**3.36 ms** (297.2 FPS)
* **端到端 3D 空间骨骼融合流水线**：**25 ~ 35 ms** (30~40 FPS 全实时)

---

## 许可证
本项目遵循 Apache 2.0 开源许可证。

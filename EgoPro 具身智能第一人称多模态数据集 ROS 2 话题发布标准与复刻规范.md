
---

## 1. 数据集总体架构与坐标系定义


### 1.1 硬件配置与传感器拓扑

EgoPro 采集系统由 **4 路硬件同步相机** 与 **6-DoF 头部/手部时序位姿** 组成：

- **头戴式双目相机（Head Stereo）**：`head_left` 与 `head_right`，分辨率 1920x1456 @ 30 FPS，构成第一人称双目立体视觉。

- **双腕部局部相机（Wrist Cameras）**：`left_wrist` 与 `right_wrist`，分辨率 1920x1536 @ 30 FPS，提供近距离操作视角。

- **6-DoF 空间位姿流**：头部主位姿、头戴相机位姿、双眼相机位姿、双手 21 关节空间位姿。

  

### 1.2 坐标系定义（Coordinate Frames）

- **世界全局坐标系 (`world`)**：固定在作业空间中（如桌面或操作台基准）。

- **相机光学坐标系 (`head_left_camera`, `head_right_camera`, `left_wrist_camera`, `right_wrist_camera`)**：

- $+X$：图像右侧

- $+Y$：图像下方

- $+Z$：光轴前方（深度方向）

- **手部关节参考系 (`left_hand_wrist`, `right_hand_wrist`)**：以手腕关节（Joint 0）为基准，各指节沿运动学链向下延伸。

  

---

  

## 2. ROS 2 话题清单与发布标准规范

  

| 话题名称 (Topic) | Protobuf Schema / ROS 2 对应类型 | 频率 (Hz) | 数据流向 / 内容描述 |

| :--- | :--- | :--- | :--- |

| **`/sensor/camera/head_left/video`** | `foxglove.CompressedVideo` / `sensor_msgs/msg/CompressedImage` | 30 | 头戴左目第一人称 H.264/H.265 压缩视频流（1920x1456） |

| **`/sensor/camera/head_right/video`** | `foxglove.CompressedVideo` / `sensor_msgs/msg/CompressedImage` | 30 | 头戴右目第一人称 H.264/H.265 压缩视频流（1920x1456） |

| **`/sensor/camera/left_wrist/video`** | `foxglove.CompressedVideo` / `sensor_msgs/msg/CompressedImage` | 30 | 左手腕视角操作压缩视频流（1920x1536） |

| **`/sensor/camera/right_wrist/video`** | `foxglove.CompressedVideo` / `sensor_msgs/msg/CompressedImage` | 30 | 右手腕视角操作压缩视频流（1920x1536） |

| **`/sensor/camera/head_left/intrinsic`** | `foxglove.CameraCalibration` / `sensor_msgs/msg/CameraInfo` | 30 | 左目相机内参矩阵 $K$、畸变参数 $D$ 与分辨率 |

| **`/sensor/camera/head_right/intrinsic`** | `foxglove.CameraCalibration` / `sensor_msgs/msg/CameraInfo` | 30 | 右目相机内参矩阵 $K$、畸变参数 $D$ 与分辨率 |

| **`/sensor/camera/left_wrist/intrinsic`** | `foxglove.CameraCalibration` / `sensor_msgs/msg/CameraInfo` | 30 | 左手腕相机内参矩阵与畸变参数 |

| **`/sensor/camera/right_wrist/intrinsic`** | `foxglove.CameraCalibration` / `sensor_msgs/msg/CameraInfo` | 30 | 右手腕相机内参矩阵与畸变参数 |

| **`/sensor/camera/head_left/extrinsic`** | `foxglove.FrameTransforms` / `geometry_msgs/msg/TransformStamped` | 30 | 头左相机在 `world` 坐标系下的 6-DoF 外参位姿（平移+四元数） |

| **`/sensor/camera/head_right/extrinsic`** | `foxglove.FrameTransforms` / `geometry_msgs/msg/TransformStamped` | 30 | 头右相机在 `world` 坐标系下的 6-DoF 外参位姿（平移+四元数） |

| **`/pose/left_hand`** | `pose.LeftHandFrame` / 自定义 `PoseArray` | 30 | 左手 **21 个关节** 完整 3D 空间坐标与四元数姿态 |

| **`/pose/right_hand`** | `pose.RightHandFrame` / 自定义 `PoseArray` | 30 | 右手 **21 个关节** 完整 3D 空间坐标与四元数姿态 |

| **`/pose/head`** | `pose.HeadFrame` / `geometry_msgs/msg/PoseStamped` | 30 | 人体头部在世界坐标系下的 6-DoF 姿态真值 |

| **`/pose/headcam`** | `pose.HeadCamFrame` / `geometry_msgs/msg/TransformStamped` | 30 | 头部相机主视点刚体位姿 |

| **`/pose/right_eye_cam`** | `pose.RightEyeCamFrame` / `geometry_msgs/msg/TransformStamped` | 30 | 右眼相机视点刚体位姿 |

| **`/annotation/semantic_segments`** | `annotation.SemanticSegment` / `std_msgs/msg/String` (JSON) | 事件触发 | 全局任务描述（Task）与各子任务动作时间轴分割点 |

| **`/annotation/bad_frame/pose/hand`** | `annotation.BadFrameHand` / `std_msgs/msg/Header` | 30 | 手部姿态遮挡/跟踪异常标记位（用于过滤劣质数据） |

| **`/session/metadata`** | `session.SessionMetadata` / `std_msgs/msg/String` (JSON) | 会话级 (1次) | 采集设备型号、固件版本、模态与启动时间戳元数据 |

---
## 3. 详细消息字段与数据结构标准

### 3.1 手部 21 关节位姿标准 (`/pose/left_hand` 与 `/pose/right_hand`)

每个消息包含 `transforms` 数组，严格按以下 **21 关节层级顺序（0 - 20）** 排列：

```

0: Wrist (手腕)

├── 1: Thumb_CMC -> 2: Thumb_MCP -> 3: Thumb_IP -> 4: Thumb_TIP (大拇指)

├── 5: Index_MCP -> 6: Index_PIP -> 7: Index_DIP -> 8: Index_TIP (食指)

├── 9: Middle_MCP -> 10: Middle_PIP -> 11: Middle_DIP -> 12: Middle_TIP (中指)

├── 13: Ring_MCP -> 14: Ring_PIP -> 15: Ring_DIP -> 16: Ring_TIP (无名指)

└── 17: Pinky_MCP -> 18: Pinky_PIP -> 19: Pinky_DIP -> 20: Pinky_TIP (小指)

```

每个关节包含：

- **`pos`**：`{ "x": float, "y": float, "z": float }`（单位：米，相对于基准坐标系）

- **`quat`**：`{ "w": float, "x": float, "y": float, "z": float }`（标准归一化四元数）

### 3.2 相机内参与标定标准 (`/sensor/camera/*/intrinsic`)

遵循 ROS `sensor_msgs/CameraInfo` 与 Foxglove 规范：

- `width`: 1920

- `height`: 1456（头戴）/ 1536（腕部）

- `K`: 3x3 标定矩阵 `[fx, 0, cx, 0, fy, cy, 0, 0, 1]`

- `R`: 3x3 旋转校正矩阵（通常为单位阵）

- `P`: 3x4 投影矩阵

  

### 3.3 语义标注数据结构 (`/annotation/semantic_segments`)

- **`task_description`**：长文本自然语言总体意图（例如："整理桌面上的托盘与纸巾盒"）。

- **`segment`**：

- `subtask_description`：当前阶段子动作描述（例如："抓取托盘移至右上角"）。

- `end_time`：子阶段截止时间点（秒，float）。

---
## 4. 复刻实现路线图（在现有代码库基础上的落地指南）

当前代码库已具备复刻该数据集 80% 以上的基础算力与算法模块，复刻对接路径如下：

```
[双目硬件采集] -> [StereoCalibrator 双目标定]

↓

[FoundationStereo 稠密深度 (16-bit 真实物理深度)]

↓

[RTMPose-Hand (21关键点) + HandPoseSmoother3D (手性防抖)]

↓

[ROS 2 话题发布节点 (ros2_egopro_publisher)] -> [MCAP / Rosbag2 录制容器]

```

### 4.1 核心对接模块映射表

1. **视频流**：从现有双目相机驱动（如 4000x1200 / 3840x1080）裁剪分离出左右目，通过 `image_transport` 发布 H.264 CompressedImage。

2. **内参标定**：直接从 [StereoCalibrator](file:///home/elp/picture_resize_recording_NVIDA/scripts_py/stereo_depth_pipeline_gpu.py) 读取 `fx, fy, cx, cy, baseline` 并填入 `CameraInfo` 话题。

3. **手部 21 关节位姿真值**：

- 现已通过 [demos/01_embodied_dataset_exporter/hand_trajectory_exporter.py](file:///home/elp/picture_resize_recording_NVIDA/demos/01_embodied_dataset_exporter/hand_trajectory_exporter.py) 和 [scripts_py/one_euro_filter.py](file:///home/elp/picture_resize_recording_NVIDA/scripts_py/one_euro_filter.py) 计算出左右手完整的 21 关节 `[X(m), Y(m), Z(m), Conf]`。

- 仅需增加各指节骨骼朝向的四元数（`Quaternion`）拟合，即可 100% 对齐 `/pose/left_hand` 与 `/pose/right_hand`。

4. **头部位姿**：已部署的 ORB-SLAM3 双目视觉里程计或机载 IMU 数据，可直接输出 `/pose/head` 6-DoF 位姿真值。

---

## 5. 后续改进与演进方向

在完全复刻 EgoPro 标准的基础上，可针对机器人具身操作与移动导航引入以下增强项：

1. **增加高频 IMU 惯导话题**：

- 发布 `/sensor/imu/head`（200 Hz，角速度 + 线性加速度），为 VIO 融合与快速晃动场景提供高频姿态支撑。

2. **增加稠密深度与点云话题**：

- 发布 `/sensor/camera/head_left/depth`（16-bit 毫米深度）与 `/sensor/camera/head/pointcloud`（3D 点云），直接服务于 Nav2 避障与 3D 场景重建。

3. **增加机械臂末端执行器（Gripper）控制反馈**：

- 发布 `/robot/gripper/state` 与 `/robot/action`，打通从人类双手演示到机械臂执行的 Policy 闭环。
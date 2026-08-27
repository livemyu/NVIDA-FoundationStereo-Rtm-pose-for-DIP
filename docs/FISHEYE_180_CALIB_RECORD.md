# 180° 鱼眼双目+IMU 标定参数基准记录表 (960x600)

> **记录时间**: 2026-08-19  
> **标定源文件**: `/home/elp/picture_resize_recording_NVIDA/Stereo_cam_ws/my_stereo_imu_config_180_960.yaml`  
> **适用传感器**: 双目 180° 超广角鱼眼镜头 + 同步 IMU (300Hz)  
> **适用框架**: VINS-Fusion / OpenVINS / ORB-SLAM3  

---

## 1. 相机内参及 Kannala-Brandt (KB4) 畸变模型 (960x600)

### 左目相机 (cam0 / Left)
- **相机模型**: `KannalaBrandt8` / `KANNALA_BRANDT`
- **图像分辨率**: 960 x 600
- **投影内参 (Intrinsics)**:
  - `fx / mu`: `220.0762756`
  - `fy / mv`: `219.9819675`
  - `cx / u0`: `472.6256002`
  - `cy / v0`: `306.0109961`
- **畸变系数 (Distortion - KB4: $k_1, k_2, k_3, k_4$ / $k_2, k_3, k_4, k_5$)**:
  - `k1 / k2`: `0.01925902`
  - `k2 / k3`: `-0.01790589`
  - `k3 / k4`: `0.00107010`
  - `k4 / k5`: `-0.00079444`

### 右目相机 (cam1 / Right)
- **相机模型**: `KannalaBrandt8` / `KANNALA_BRANDT`
- **图像分辨率**: 960 x 600
- **投影内参 (Intrinsics)**:
  - `fx / mu`: `219.8418225`
  - `fy / mv`: `219.8238973`
  - `cx / u0`: `481.1262635`
  - `cy / v0`: `298.1493852`
- **畸变系数 (Distortion - KB4: $k_1, k_2, k_3, k_4$ / $k_2, k_3, k_4, k_5$)**:
  - `k1 / k2`: `0.01831141`
  - `k2 / k3`: `-0.01676312`
  - `k3 / k4`: `0.00072503`
  - `k4 / k5`: `-0.00073737`

---

## 2. IMU-相机外参 ($T_{\text{imu}\leftarrow\text{cam}}$ / $T_{b\_c}$)

> **坐标系约定说明**：
> - $T_{b\_c}$ 表示将点从相机坐标系转换到 IMU/Body 坐标系。
> - 在 VINS-Fusion 中为 `body_T_cam`。
> - 在 ORB-SLAM3 中为 `IMU.T_b_c1`（对应左目 cam0）。

### 左目到 IMU (`body_T_cam0` / `IMU.T_b_c1`)
```yaml
body_T_cam0: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [  0.25091278,  0.00548910,  0.96799413, -0.10491511,
           -0.96797581,  0.00979053,  0.25085251,  0.09482819,
           -0.00810022, -0.99993701,  0.00776989,  0.00509889,
            0.0,         0.0,         0.0,         1.0 ]
```

### 右目到 IMU (`body_T_cam1`)
```yaml
body_T_cam1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [  0.23686719, -0.00674644,  0.97151862, -0.09048009,
           -0.97151445,  0.00589150,  0.23690709,  0.04275609,
           -0.00732198, -0.99995989, -0.00515877,  0.00263698,
            0.0,         0.0,         0.0,         1.0 ]
```

---

## 3. 双目相机相对外参 ($T_{c0\leftarrow c1}$ / $T_{\text{left}\leftarrow\text{right}}$)

> **公式**: $T_{c0\leftarrow c1} = (T_{\text{body}\leftarrow\text{cam0}})^{-1} \cdot T_{\text{body}\leftarrow\text{cam1}}$  
> **基线说明**: 水平基线约 **5.4 cm** ($X \approx 0.054\text{ m}$)。用于 ORB-SLAM3 双目鱼眼对极曲线搜索与双目深度恢复。

### ORB-SLAM3 双目外参 (`Stereo.T_c1_c2`)
```yaml
Stereo.T_c1_c2: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [ 0.99989462,  0.00070432,  0.01448789,  0.05404642,
         -0.00088991,  0.99991763,  0.01281061,  0.00203121,
         -0.01447758, -0.01282218,  0.99981297,  0.00089153,
          0.0, 0.0, 0.0, 1.0 ]
```

---

## 4. IMU 噪声模型与时间同步参数

- **陀螺仪白噪声 (`IMU.NoiseGyro` / `gyr_n`)**: `0.010 rad/s^0.5`
- **加速度计白噪声 (`IMU.NoiseAcc` / `acc_n`)**: `0.100 m/s^1.5`
- **陀螺仪零偏随机游走 (`IMU.GyroWalk` / `gyr_w`)**: `0.0001 rad/s^1.5`
- **加速度计零偏随机游走 (`IMU.AccWalk` / `acc_w`)**: `0.001 m/s^2.5`
- **IMU 频率 (`IMU.Frequency`)**: `300.0 Hz`
- **时间同步偏差 (`IMU.TimeOffset` / `td`)**: `0.01128 s`
- **重力加速度标量 (`g_norm`)**: `9.81007 m/s^2`

---

## 5. 各算法配置文件对应路径索引

| 算法框架 | 配置文件路径 |
| :--- | :--- |
| **VINS-Fusion** | [`/home/elp/picture_resize_recording_NVIDA/Stereo_cam_ws/my_stereo_imu_config_180_960.yaml`](file:///home/elp/picture_resize_recording_NVIDA/Stereo_cam_ws/my_stereo_imu_config_180_960.yaml) |
| **ORB-SLAM3** | [`/home/elp/benchmark_slam_ws/ORB_SLAM3/Examples/Stereo-Inertial/my_stereo_imu_180_960.yaml`](file:///home/elp/benchmark_slam_ws/ORB_SLAM3/Examples/Stereo-Inertial/my_stereo_imu_180_960.yaml) |
| **OpenVINS** | [`/home/elp/benchmark_slam_ws/src/open_vins/config/stereo_180_960/kalibr_imucam_chain.yaml`](file:///home/elp/benchmark_slam_ws/src/open_vins/config/stereo_180_960/kalibr_imucam_chain.yaml) |

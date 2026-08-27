# NVIDIA Jetson Orin NX 平台 Spatial AI 双目深度部署与多模型基准测试文档

## 1. 硬件与软件环境规格

- **部署设备**：Seeed reComputer Super J401 (搭载 NVIDIA Jetson Orin NX 16GB)
- **设备 IP**：`192.168.0.145`
- **工作空间路径**：`/home/elp/spatial_ai_trt_ws`
- **GPU 架构**：NVIDIA Ampere (1024 CUDA Cores + 32 Tensor Cores, 100 TOPS INT8)
- **内存规格**：16GB LPDDR5 统一内存 (显存共享，带宽 102.4 GB/s)
- **底层系统**：Ubuntu 22.04 LTS (aarch64) / L4T R36.4.3 (JetPack 6.2)
- **AI 推理栈**：CUDA 12.6, cuDNN 9.3.0, TensorRT 10.3.0.30, PyCUDA

---

## 2. 空间目录与模型库组织

```
/home/elp/spatial_ai_trt_ws/
├── foundationstereo_320x736_fp16.engine   # FoundationStereo 高精度建图引擎 (35 MB)
├── ess_fp16.engine                        # NVIDIA Standard ESS 576x960 实时引擎 (34 MB)
├── light_ess_fp16.engine                  # NVIDIA Light-ESS 288x480 极速引擎 (34 MB)
├── stereo_depth_pipeline_gpu.py           # 全 GPU (All-CUDA) 零 CPU 阻塞双目流水线 (推荐)
├── stereo_depth_pipeline.py               # 原生多模型 CPU/TRT 基础双目流水线
├── test_foundationstereo_trt.py           # FoundationStereo 纯推理基准测试脚本
├── test_ess_trt.py                        # ESS / Light-ESS 纯推理基准测试脚本
└── dnn_stereo_disparity_v4.1.0_onnx/      # 官方 NGC 资产与 aarch64 算子插件 (ess_plugins.so)
```

---

## 3. 全链路端到端性能实测对比 (Jetson Orin NX 16GB)

所有测试均以 **960x600 原始双目相机分辨率** 作为基准输入输出：

| 模型方案 | 模型输入尺寸 | 纯 GPU 核心推理时延 | CPU 插值端到端帧率 | **All-CUDA 全 GPU 端到端帧率** | **All-CUDA 单帧总时延** | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA Light-ESS** | 288 x 480 | **5.23 ms** | 26.27 FPS (38.1 ms) | **124.48 FPS** | **8.03 ms** | **超低时延极速避障** |
| **NVIDIA Standard ESS** | 576 x 960 | **18.35 ms** | 15.31 FPS (65.3 ms) | **51.01 FPS** | **19.61 ms** | **高清实时深度 (30-60 FPS)** |
| **FoundationStereo** | 320 x 736 | **186.10 ms** | 4.52 FPS (221.0 ms) | **5.25 FPS** | **190.54 ms** | **高精度稠密建图** |

---

## 4. 全 GPU (All-CUDA) 流水线使用指南

### 4.1 命令行运行模式

#### (1) 单张高分辨率静态图推理
```bash
# 1. Standard ESS (高清推荐：51 FPS / 19.6 ms)
python3 /home/elp/spatial_ai_trt_ws/stereo_depth_pipeline_gpu.py \
  --model ess \
  --image /home/elp/spatial_ai_trt_ws/2026-08-25_11-55-38.jpg

# 2. Light-ESS (极速模式：124 FPS / 8.0 ms)
python3 /home/elp/spatial_ai_trt_ws/stereo_depth_pipeline_gpu.py \
  --model light_ess \
  --image /home/elp/spatial_ai_trt_ws/2026-08-25_11-55-38.jpg

# 3. FoundationStereo (稠密建图：5.2 FPS / 190 ms)
python3 /home/elp/spatial_ai_trt_ws/stereo_depth_pipeline_gpu.py \
  --model foundation \
  --image /home/elp/spatial_ai_trt_ws/2026-08-25_11-55-38.jpg
```

#### (2) 4-5倍极速双目视频深度流生成流水线 (推荐：31.0 FPS 实时同速)
```bash
# 1. 使用 Standard ESS 模型生成全量视频 (Turbo 鲜艳热力图，近红远蓝，彻底告别黑影)
python3 /home/elp/spatial_ai_trt_ws/fast_video_depth_pipeline.py \
  --video /home/elp/spatial_ai_trt_ws/videos/30fps_noj.mp4 \
  --model ess \
  --max_depth 8.0

# 2. 使用 Light-ESS 模型 (极速模式：32.2 FPS)
python3 /home/elp/spatial_ai_trt_ws/fast_video_depth_pipeline.py \
  --video /home/elp/spatial_ai_trt_ws/videos/30fps_noj.mp4 \
  --model light_ess

# 3. 前 200 帧快速测试验证 (仅需 9.6 秒)
python3 /home/elp/spatial_ai_trt_ws/fast_video_depth_pipeline.py \
  --video /home/elp/spatial_ai_trt_ws/videos/30fps_noj.mp4 \
  --model ess \
  --max_frames 200
```

### 4.2 离线视频生成端到端处理速度对比 (Orin NX 16GB)

| 运行模式 | 极线校正耗时 | 画面渲染拼版 | 综合端到端帧率 | 3658 帧 (121秒视频) 总生成耗时 | 性能评价 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **基础版 (`video_depth_pipeline.py`)** | CPU 单核 68 ms | CPU 单核 30 ms | **5.58 FPS** | **10.5 分钟 (630 秒)** | 偏慢，CPU 单核饱和 |
| **极速版 (`fast_video_depth_pipeline.py`)** | **CUDA 显存 0.9 ms** | **CUDA 显存 0.2 ms** | **31.0 FPS** | **~1.8 分钟 (110 秒)** | **提速 4.5 倍，1:1 实时同速！** |
| **极速版 Light-ESS** | **CUDA 显存 0.9 ms** | **CUDA 显存 0.2 ms** | **32.2 FPS** | **~1.8 分钟 (108 秒)** | **极速秒出** |


### 4.2 生成产物清单 (`output_results/`)

执行推理后，系统自动在 `output_results/` 生成全套数据产物：
1. `sdk_depth_30fps_noj_ess.mp4`：SDK 硬件解码生成的左目真彩 + 物理深度热力图并排 1920x600 深度视频。
2. `sdk_imu_30fps_noj.csv`：与视频帧微秒级时间戳严格对齐的 11 点 ICM42688 IMU 高频加速度与陀螺仪轨迹表。
3. `rectified_left.jpg`：去畸变并水平行对齐后的 1920x1200 左目基准图。
4. `disparity_<model>.jpg`：Turbo 色彩映射的高分辨率视差热力图。
5. `depth_<model>.jpg`：带公制物理距离刻度的 Inferno 深度距离图。
6. `comparison_<model>.jpg`：四视图综合对比图 (左目原图 + 右目原图 + 视差图 + 深度图)。
7. `pointcloud_<model>.ply`：标准 3D 稠密彩色点云文件（可用 Meshlab / CloudCompare 直接拖入查看）。




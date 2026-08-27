# RTMPose 极简独立运行代码包

本代码包仅包含 **手部运动轨迹** 与 **人体全身姿态** 两个极简独立推理脚本，**零深度图依赖、零复杂 Demo 逻辑**，开箱即用。

---

## 包含文件清单

1. **`infer_rtmpose_hand_trajectory.py`**：
   * 手部 21 关键点骨骼估计；
   * 食指指尖实时运动轨迹（彩虹渐变拖尾）追踪。
2. **`infer_rtmpose_body_pose.py`**：
   * 人体全身 17 关键点姿态估计与分肢骨骼绘制。

---

## 运行命令

### 1. 运行手部姿态与运动轨迹追踪
```bash
# 方式 A: 处理本地视频文件
python3 infer_rtmpose_hand_trajectory.py \
    --input test_video.mp4 \
    --output hand_trajectory_result.mp4

# 方式 B: 打开实时摄像头 (编号 0) 并在窗口实时显示
python3 infer_rtmpose_hand_trajectory.py \
    --input 0 \
    --show
```

### 2. 运行人体全身 17 关键点姿态估计
```bash
# 方式 A: 处理本地视频文件
python3 infer_rtmpose_body_pose.py \
    --input test_video.mp4 \
    --output body_pose_result.mp4

# 方式 B: 打开实时摄像头并在窗口实时显示
python3 infer_rtmpose_body_pose.py \
    --input 0 \
    --show
```

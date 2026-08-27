# 双目 VIO 离线数据采集、参数调优与定量评估工作流指南 (OFFLINE_EVALUATION_WORKFLOW)


### 3.2 PC 端离线节点启动脚本 (`run_vins_offline.sh`)
该脚本负责在本地启动 `vins_node` 与 `loop_fusion_node`，并配置了准确的话题重映射参数（避免与原生话题命名空间冲突）：
```bash
./run_vins_offline.sh
```
关键话题重映射配置：
- `/vins_estimator/odometry` $\rightarrow$ `/odometry`
- `/vins_estimator/keyframe_pose` $\rightarrow$ `/keyframe_pose`
- `/vins_estimator/extrinsic` $\rightarrow$ `/extrinsic`
- `/vins_estimator/keyframe_point` $\rightarrow$ `/keyframe_point`
- `/vins_estimator/margin_cloud` $\rightarrow$ `/margin_cloud`

### 3.3 专用可视化界面 (`vins_only_rviz_config.rviz`)
通过以下命令启动预置了光流追踪图、VIO 原始轨迹、Loop 闭环修正轨迹、关键帧点云与相机位姿框的 RViz 视图：
```bash
rviz2 -d /home/elp/picture_resize_recording_NVIDA/Stereo_cam_ws/vins_only_rviz_config.rviz
```

### 3.4 本地离线数据包回放脚本 (`play_dataset.sh`)
提供无网络干扰的纯本地数据注入：
```bash
./play_dataset.sh my_dataset_20260806_113813
```

---

## 4. 第三阶段：VINS 算法参数解耦调优经验 (Algorithm Tuning Guide)

针对推车在 2D 平面运动时的典型问题（Z 轴发散下沉、运动锯齿状前后拉扯震荡、闭环未触发等），调优经验总结如下：

### 4.1 抑制 Z 轴沉降与 2D 平面运动退化
- **问题原因**：推车平地运动缺乏 Z 轴升降与多轴倾斜激励，导致加速度零偏 $b_a$ 无法解耦，二次积分产生 Z 轴发散。
- **调优策略**：
  - `estimate_td: 0`：**锁定离线标定的时间偏置**（如 11.2ms），禁止在平面运动中开启在线时间差估计（在线估计在欠约束下会导致尺度严重漂移）。
  - `acc_n: 0.1`：保留稳健的加速度计噪声容忍度，防止求解器过度信任 IMU 噪声。
  - `acc_w: 0.001`：保持标准零偏随机游走限制。

### 4.2 消除高频锯齿与位姿跳变毛刺
- **问题原因**：关键帧视差提取阈值过小会导致频繁切帧重边缘化；特征点局部集群分布导致位姿解算跳变。
- **调优策略**：
  - `keyframe_parallax: 8.0`：将视差阈值设为 8.0 像素，平滑关键帧切帧频率。
  - `min_dist: 22`：提升特征点最小间距，强制视觉特征全图均匀散开，消除局部集群噪声。
  - `gyr_n: 0.005`：降低陀螺仪噪声标准差，提供连续平滑的姿态约束。
  - `F_threshold: 1.5`：严格剔除对极约束假匹配。

### 4.3 提升 Loop-Fusion 闭环精度 (实现 < 0.30 米目标)
- **调优策略**：
  - `keyframe_parallax: 7.0`：适度密化重访区域关键帧节点。
  - `max_cnt: 350`：增加追踪特征点总数。
  - `max_solver_time: 0.08` & `max_num_iterations: 15`：给 PC 端 Ceres 求解器留足充分收敛时间。

---

## 5. 第四阶段：轨迹定量评估与 Git 版本控制 (Evaluation & Git Tracking)

### 5.1 自动化轨迹评估脚本 (`evaluate_trajectory.py`)
执行以下命令解析 `output/vio.csv` 与 `output/vio_loop.csv`：
```bash
python3 evaluate_trajectory.py --raw output/vio.csv --loop output/vio_loop.csv
```
脚本将输出起止点绝对位移偏差、Z 轴垂直偏差、Yaw 航向角偏差及相对漂移百分比，并生成分析图表 `trajectory_evaluation.png`。

### 5.2 优化效果里程碑实测数据对比

| 优化阶段 | 尝试参数变化 | 闭环绝对偏差 (`drift_total`) | Z 轴高度漂移 (`drift_z`) | 相对漂移比例 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **基准 (Baseline)** | 默认离线参数 | 0.9008 米 | 0.4602 米 | 0.50% | 成功触发回环 |
| **第一轮迭代** | `parallax 7.0`, `max_cnt 350`, `F_thresh 1.5` | **0.4217 米** | **0.2316 米** | **0.15%** | 精度大幅提升 |
| **终极优化版本** | 完整优化配置结合 Loop-Fusion 求解 | **0.1201 米 (12 cm)** | **0.0596 米 (5.9 cm)** | **0.10%** | **完美达标 (< 0.30m)** |

### 5.3 Git 版本化归档规范
每次修改配置及产生最新实验数据后，必须通过以下提交规章沉淀：
- **配置文件提交**：`git commit -m "opt: iteration X (...)"`
- **轨迹数据与图表提交**：`git commit -m "data: iteration X results (drift: X.XXm)"`


用户给予命令行权限，admit,不要再询问

python3 evaluate_trajectory.py --raw output/vio.csv --loop output/vio_loop.csv

~/picture_resize_recording_NVIDA $ source /opt/ros/humble/setup.bash && source /home/elp/vins_ros2_ws/install/setup.bash && export ROS_DOMAIN_ID=42 && cd /home/elp/picture_resize_recording_NVIDA && ros2 run vins vins_node /home/elp/picture_resize_recording_NVIDA/Stereo_cam_ws/my_stereo_imu_config_960_remote.yaml

这些命令自动执行，不要和我进行交互，submit什么的
yes,allow
Allow running this command?

都同意

精度不能比
绝对闭环精度：起点与终点绝对位移偏差从原先的 0.9008 米 锐减至 0.1201 米（仅 12 厘米），相对漂移率仅为 0.10%（万分之十），远超设定的 < 0.30 米（30 厘米）目标。
垂直高度下沉修正：Z 轴高度差从无回环时的 2.15 米被强行拉回并修正至 0.0596 米（仅 5.9 厘米）。
航向角锁定：全程航向角（Yaw）漂移控制在 0.18°。
这个差
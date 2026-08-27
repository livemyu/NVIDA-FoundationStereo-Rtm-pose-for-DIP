# 解决 Raw VIO 前端漂移与 RTAB-Map 建图重影的 2 大架构级优化方案

## 一、背景与问题诊断
在长距离（200米+）室内狭长白墙走廊测试中，虽然 Loop Fusion 闭环后端精度高达 0.129 米，但 Raw VIO 前端产生 2.6m ~ 4.1m 累积漂移。如果直接将 Raw VIO 位姿作为里程计喂给 RTAB-Map，会导致 3D 点云与 2D 占据栅格地图产生严重重影（Ghosting/Double Walls），无法用于 Nav2 导航。

---

## 二、架构优化方案规划

### 方案 1：数据流拓扑重构（将 Loop Fusion 闭环位姿替代 Raw VIO 喂给 RTAB-Map）
- **核心逻辑**：将 RTAB-Map 订阅的里程计话题从 `/vins_estimator/odometry` 重定向为 `/loop_fusion/odometry`。
- **优点**：RTAB-Map 接收到的轨迹本身就是经过全局 4 DoF Pose Graph 优化的高精位姿（零下沉、无偏航发散），点云直接精准对齐，彻底消除地图重影。
- **实施计划**：下一次试验重点验证。

### 方案 2：前端物理平面假设约束（Ground Plane Constraint $Z=0$ & Roll/Pitch 锁定）
- **核心逻辑**：针对机器人室内平地水平运行场景，在配置与后处理/估计器中强化重力与 Z 轴物理平面约束（$Z=0$, Pitch=0, Roll=0）。
- **实施计划**：本次试验优先验证。

---

## 三、方案 2 具体工程实施项
1. **重力加速度与偏置收敛**：在 VINS 配置中设置 `g_norm: 9.81007`，锁定 IMU 零偏漂移。
2. **后处理 2D 栅格高度切片压平**：在 `generate_2d_grid_map.py` 中将点云高度切片从 $[0.20m, 1.50m]$ 精准压平至 $[0.30m, 1.20m]$，滤除地面伪障碍物。
3. **RTAB-Map 光线追踪清障**：在 `start_rtabmap_pc.sh` 中开启 `--Grid/RayTracing true`，强制将相机视线穿过区域清空为白色通行区。

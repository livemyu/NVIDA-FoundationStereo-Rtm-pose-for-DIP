# 周工作交接与下周快速复原备忘录 (Weekly Handover & Quick-Start Memo)

## 一、本周核心突破与物理指标汇总

在本周的优化迭代中，针对手持双目+IMU（无轮速计）在 200 米白墙过道场景下的漂移、沉降与建图重影问题，完成了以下重大技术突破：

1. **Z 轴下沉雪崩彻底消除**：
   - **优化前**：前端位姿纵向沉降达 **`-2.63 米`**。
   - **优化后**：经 `estimate_extrinsic: 1` 在线 Pitch 俯仰自标定后，Z 轴偏差缩小至 **`-0.0067 米`（仅 -0.6 厘米）**。
2. **闭环精度达到历史最高**：
   - **闭环绝对坐标精度**：**`0.129 米` (12.9 cm)**。
   - **全行程匹配重合率**：**`98.34%`**（行程仅 1.92 米残差）。
   - **Git 历史 Milestones 标签**：
     - `v1.0-optimal-loop-closure`
     - `v2.0-best-loop-closure`（历史最佳回环效果版本）

---

## 二、关键理论探讨与架构定型结论

1. **为什么 Raw VIO 前端 Z 轴会起伏（-1.0m~+0.7m），而 Loop 回环路径极为标准（Z轴仅±3cm）？**
   - **Raw VIO 前端**：依赖 10 关键帧局部滑动窗口，人体行走步幅的 2~3Hz 垂直颠簸在白墙欠约束下被二次积分放大，产生波浪起伏。
   - **Loop Fusion 后端**：执行 **4 DoF (x, y, z, yaw) 全局位姿图优化（PGO）**，将姿态 Pitch/Roll 硬锁定在重力矢量垂直线上，借助全局回环边约束拉扯成标准的水平直线。

2. **解决 RTAB-Map 建图重影与双层墙的终极解法**：
   - **方案 1（数据流拓扑重构）**：将 RTAB-Map 的里程计输入从 `/vins_estimator/odometry`（Raw VIO）改写为 `/loop_fusion/odometry`（Loop 优化位姿）。
   - 此改动已全量写进 `start_rtabmap_pc.sh` 与 `start_rtabmap_localization.sh` 并完成 Git 提交（Commit Hash: `687e62e`）。

---

## 三、Git 提交与版本管理状态

| 仓库名称 | 黄金 Tag 标签 | 最新 Commit Hash | 核心内容 |
| :--- | :--- | :--- | :--- |
| **主工程仓库** | `v2.0-best-loop-closure` | `687e62e` | 包含方案一重定向脚本、历史最佳轨迹 CSV 与 2D 渲染地图 |
| **Stereo_cam_ws** | `v2.0-best-loop-closure` | `a7e9c85` | 黄金解算配置（`max_cnt: 400`, `min_dist: 15`, `F_threshold: 1.0`） |

---

## 四、下周复工一键启动指南

下周复工后，直接执行以下指令即可全自动唤起方案一（Loop Fusion 作为 RTAB-Map 里程计）的无重影 2D/3D 稠密建图：

```bash
# 1. 确认 Git 状态（已就绪在最新提交上）
git status

# 2. 启动 PC 端 RTAB-Map 建图（已自动订阅 /loop_fusion/odometry）
bash start_rtabmap_pc.sh

# 3. 在新终端播发数据集解算
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=42
cd /home/elp/picture_resize_recording_NVIDA
ros2 bag play my_dataset_20260806_113813 --rate 1.5
```



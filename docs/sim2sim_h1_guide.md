# Isaac Lab → MuJoCo Sim2Sim 迁移说明书（Unitree H1）

> 记录 2026-05 完成的 H1 机器人从 Isaac Lab 到 MuJoCo 的策略迁移过程，包括关键发现、实现细节和验证结果。

---

## 一、为什么"突然成功了"

在此之前，迁移一直在失败：机器人在 2–4 秒内倒地，无论怎么调物理参数（armature、damping、timestep）都无济于事。

**真正的原因只有一个：用错了 MuJoCo 机器人模型。**

我们之前用的是 Google DeepMind 的 `mujoco_menagerie` 仓库里的 H1 模型。通过编写批量诊断脚本，系统对比测试了两种模型，结论如下：

| MuJoCo 模型来源 | 所有场景结果 | 原因 |
|---|---|---|
| `mujoco_menagerie/unitree_h1/` | 全部 2–4s 倒地 | 关节坐标轴方向与 Isaac Lab 不一致 |
| `unitree_mujoco/unitree_robots/h1/` | 全部 30s+ 稳定存活 | 与 Isaac Lab USD 模型坐标轴一致 |

menagerie 模型在测试中表现出一致的**向后漂移**（`pitch` 持续负增长），说明策略输出的力矩方向在该模型里存在系统性偏差，属于模型几何定义层面的问题，不是参数调整能解决的。

---

## 二、整体工作流程

```
Isaac Lab 训练 (RSL-RL / PPO)
       ↓  scripts/rsl_rl/play.py 导出
policy.onnx  +  deploy.yaml
       ↓  scripts/sim2sim/sim2sim_h1.py 加载
MuJoCo 仿真 (unitree_mujoco H1 模型)
       ↓  scripts/sim2sim/record_sim2sim.py 离屏录制
sim2sim_*.mp4
```

---

## 三、关键文件

| 文件 | 用途 |
|---|---|
| `scripts/rsl_rl/play.py` | 从 checkpoint 导出 ONNX 策略 |
| `scripts/sim2sim/sim2sim_h1.py` | MuJoCo 可视化运行（交互式 viewer） |
| `scripts/sim2sim/record_sim2sim.py` | 离屏渲染录像（无需 X11 窗口） |
| `scripts/sim2sim/diagnose_mujoco.py` | 批量诊断脚本（无 viewer，对比参数/模型组合） |
| `logs/.../exported/policy.onnx` | 导出的策略网络 |
| `logs/.../params/deploy.yaml` | 部署参数（关节顺序、PD 增益、观测缩放等） |

---

## 四、分步操作指南

### 4.1 导出策略

```bash
conda activate env_isaaclab
cd /home/hazzzard/unitree_rl_lab

python scripts/rsl_rl/play.py \
  --task Unitree-H1-Velocity \
  --checkpoint logs/rsl_rl/unitree_h1_velocity/<run_dir>/model_<iter>.pt
```

执行后会在 checkpoint 同目录的 `exported/` 下生成：
- `policy.onnx`：ONNX 格式，供 MuJoCo 端使用
- `policy.pt`：TorchScript 格式备用

同时 `params/deploy.yaml` 记录了所有部署所需参数。

### 4.2 准备 MuJoCo 模型

**必须使用 `unitree_mujoco` 仓库的 H1 模型，不能用 `mujoco_menagerie`。**

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco
cd unitree_mujoco/simulate
mkdir build && cd build
cmake .. && make -j$(nproc)
```

H1 场景文件路径：`unitree_mujoco/unitree_robots/h1/scene.xml`

> ⚠ 若克隆到 `/tmp`，系统重启后会被清空。建议克隆到 `/home` 下持久保存。

### 4.3 运行可视化

```bash
conda activate env_isaaclab
cd /home/hazzzard/unitree_rl_lab
DISPLAY=:1 python scripts/sim2sim/sim2sim_h1.py --vx 0.5

# 常用参数
# --vx 0.0     站立不动
# --vx 1.0     快速前进
# --vy 0.3     侧向移动
# --wz 0.5     旋转
# --headless   不开窗口，仅打印状态日志
```

### 4.4 录制视频

```bash
python scripts/sim2sim/record_sim2sim.py \
  --vx 0.5 --duration 30 --out sim2sim_vx05_30s.mp4
# 视频保存在 logs/.../exported/ 同级目录
```

### 4.5 批量诊断（排查问题用）

```bash
python scripts/sim2sim/diagnose_mujoco.py --duration 15
# 自动测试 9 种参数/模型组合，输出结构化报告

# 只测某个场景前缀（如只测 unitree 模型）
python scripts/sim2sim/diagnose_mujoco.py --duration 30 --scene H
```

---

## 五、核心技术细节

### 5.1 关节顺序映射

Isaac Lab 策略输出的 19 维动作按**策略顺序**排列，与 MuJoCo 模型内部的关节数组顺序不同。映射关系由 `deploy.yaml` 中的 `joint_ids_map` 推导：

```yaml
joint_ids_map: [7, 8, 6, 3, 0, 16, 12, 4, 1, 17, 13, 5, 2, 18, 14, 10, 11, 19, 15]
```

**策略顺序 → 关节名（unitree_mujoco 命名，含 `_joint` 后缀）：**

| policy idx | 关节名 | policy idx | 关节名 |
|---|---|---|---|
| 0 | `left_hip_yaw_joint` | 10 | `right_shoulder_roll_joint` |
| 1 | `right_hip_yaw_joint` | 11 | `left_knee_joint` |
| 2 | `torso_joint` | 12 | `right_knee_joint` |
| 3 | `left_hip_roll_joint` | 13 | `left_shoulder_yaw_joint` |
| 4 | `right_hip_roll_joint` | 14 | `right_shoulder_yaw_joint` |
| 5 | `left_shoulder_pitch_joint` | 15 | `left_ankle_joint` |
| 6 | `right_shoulder_pitch_joint` | 16 | `right_ankle_joint` |
| 7 | `left_hip_pitch_joint` | 17 | `left_elbow_joint` |
| 8 | `right_hip_pitch_joint` | 18 | `right_elbow_joint` |
| 9 | `left_shoulder_roll_joint` | | |

> ⚠ menagerie 里的关节名**没有** `_joint` 后缀（如 `left_hip_yaw`），unitree_mujoco **有**（`left_hip_yaw_joint`）。

### 5.2 观测向量（68 维）

观测向量必须与 Isaac Lab `PolicyCfg` 完全一致：

```
维度      内容                           缩放
[0:3]    躯干角速度（体坐标系）           × 0.2
[3:6]    投影重力方向（体坐标系）          × 1.0
[6:9]    速度指令 (vx, vy, wz)           × 1.0
[9:28]   关节位置 − 默认值（策略顺序）    × 1.0
[28:47]  关节速度（策略顺序）             × 0.05
[47:66]  上一步动作（策略顺序）           × 1.0
[66:68]  步态相位 (sin, cos)，周期 0.6s  × 1.0
```

**坐标系转换（最容易遗漏）**：MuJoCo 3.x 自由关节 `qvel[3:6]` 是**全局坐标系**角速度，Isaac Lab 的 `base_ang_vel` 是**体坐标系**，必须用根节点四元数做逆旋转：

```python
def quat_rotate_inverse(q_wxyz, v):
    w, x, y, z = q_wxyz
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    return R.T @ v  # R.T = 世界系 → 体坐标系
```

### 5.3 动作执行

策略输出是关节位置**偏移量**（未缩放），转换为 PD 控制力矩：

```python
ACTION_SCALE = 0.25
q_target = DEFAULT_JOINT_POS + ACTION_SCALE * actions
torques  = KP * (q_target - q_current) - KD * dq_current
data.ctrl[pol2ctrl] = torques
```

**PD 增益（按策略顺序，来自 `deploy.yaml`）：**

| 关节组 | 关节名 | Kp | Kd |
|---|---|---|---|
| hip_yaw | left/right_hip_yaw | 150 | 2 |
| torso | torso | 300 | 6 |
| hip_roll | left/right_hip_roll | 150 | 2 |
| shoulder_pitch | left/right_shoulder_pitch | 100 | 2 |
| hip_pitch | left/right_hip_pitch | 150 | 2 |
| shoulder_roll | left/right_shoulder_roll | 100 | 2 |
| knee | left/right_knee | 200 | 4 |
| shoulder_yaw | left/right_shoulder_yaw | 50 | 2 |
| ankle | left/right_ankle | 40 | 2 |
| elbow | left/right_elbow | 50 | 2 |

**默认关节位置（策略顺序）：**

```python
DEFAULT_JOINT_POS = [
    0.0,  0.0,  0.0,   # left_hip_yaw, right_hip_yaw, torso
    0.0,  0.0,         # left/right_hip_roll
    0.2,  0.2,         # left/right_shoulder_pitch
   -0.1, -0.1,         # left/right_hip_pitch
    0.0,  0.0,         # left/right_shoulder_roll
    0.3,  0.3,         # left/right_knee
    0.0,  0.0,         # left/right_shoulder_yaw
   -0.2, -0.2,         # left/right_ankle
    0.32, 0.32,        # left/right_elbow
]
```

### 5.4 仿真时序

```python
POLICY_DT  = 0.02    # 策略控制频率 50 Hz
PHYSICS_DT = 0.005   # 物理仿真频率 200 Hz（与 Isaac Lab sim.dt 一致）
N_SUBSTEPS = 4       # 每次策略推理之间推进 4 步物理
```

unitree_mujoco 的 `h1.xml` 默认 `timestep=0.002`，必须在代码中覆盖：

```python
model.opt.timestep = PHYSICS_DT  # 强制设为 0.005
```

---

## 六、物理参数影响的实验对比

通过 `diagnose_mujoco.py` 批量测试，每场景运行 15s：

| 场景 | 模型 | armature | damping | 存活 |
|---|---|---|---|---|
| A | menagerie | 0.1（原始） | 1.0（原始） | 3.0s ❌ |
| B | menagerie | 0.01（修正） | 0.0（修正） | 2.9s ❌ |
| C | menagerie + 策略 vx=0 | 0.1 | 1.0 | 2.3s ❌ |
| D | menagerie + 策略 vx=0 | 0.01 | 0.0 | 3.5s ❌ |
| E | menagerie + 策略 vx=0 | 0.01 | 1.0 | 4.1s ❌ |
| F | menagerie + 策略 vx=0 | 0.1 | 0.0 | 2.6s ❌ |
| **H** | **unitree + 策略 vx=0** | **0.1（原始）** | **1.0（原始）** | **15s+ ✅** |
| **I** | **unitree + 策略 vx=0** | **0.01（修正）** | **0.0（修正）** | **15s+ ✅** |

**结论：模型来源是决定性因素，不是物理参数。** 对 unitree_mujoco 模型，原始参数和修正参数都能成功，但修正参数（`armature=0.01, damping=0`）更接近 Isaac Lab `IdealPDActuator` 的物理假设，建议保留。

---

## 七、已验证性能（导出的 velocity tracking 策略）

| vx 指令 | 存活时间 | 实际 vx（稳定后） |
|---|---|---|
| 0.0 | 30s+ | ~0 m/s（稳定站立） |
| 0.3 | 30s+ | ~0.25 m/s |
| 0.5 | 30s+ | ~0.35 m/s |
| 0.8 | 30s+ | ~0.6 m/s |
| 1.0 | 30s+ | ~0.75 m/s |

---

## 八、已知问题与后续改进方向

1. **速度追踪稳态误差**：实际速度约为指令的 70–80%，后期有衰减。原因是当前导出策略的训练收敛程度仍有限，泛化能力不足。在 DR（Domain Randomization）强化环境中继续训练后可改善。

2. **unitree_mujoco 存放在 `/tmp`**：系统重启后会被清空，需要重新克隆编译。建议移到 `/home` 下的持久目录。

3. **步态稳定性**：机器人行走时有轻微左右漂移（vy 分量），说明策略未完全收敛到直线行走，追加训练或 DR 训练后可改善。

#!/usr/bin/env python3
"""
Sim2Sim: 在 MuJoCo 中运行 Isaac Lab 训练的 H1 速度追踪策略

观测向量 (68 维) 与 Isaac Lab PolicyCfg 完全一致:
  [3]  躯干角速度（体坐标系）× 0.2
  [3]  投影重力方向（体坐标系）
  [3]  速度指令 (vx, vy, wz)
  [19] 关节位置相对默认值
  [19] 关节速度 × 0.05
  [19] 上一步动作
  [2]  步态相位 (sin, cos)，周期 0.6s

动作: 19 维关节位置偏移 → 目标位置 = default + scale × action
力矩: τ = Kp × (q_target − q) − Kd × dq

用法:
  python sim2sim_h1.py --vx 0.5 --vy 0.0 --wz 0.0
"""

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# ── 文件路径 ──────────────────────────────────────────────────────────────────
# unitree_mujoco 的关节坐标轴定义与 Isaac Lab USD 模型一致，menagerie 不一致
REPO_ROOT = Path(__file__).resolve().parents[1]
UNITREE_MUJOCO_DIR = Path(os.environ.get("UNITREE_MUJOCO_DIR", "/tmp/unitree_mujoco"))
SCENE_XML = str(UNITREE_MUJOCO_DIR / "unitree_robots/h1/scene.xml")
POLICY_ONNX = str(REPO_ROOT / "policies/weight-velocity-track/weight-velocity-track.onnx")

# ── 仿真时序 ──────────────────────────────────────────────────────────────────
# 与 Isaac Lab 配置完全一致: sim.dt=0.005, decimation=4 → policy_dt=0.02s
POLICY_DT = 0.02    # 50 Hz 策略控制频率
PHYSICS_DT = 0.005  # 200 Hz 物理仿真频率（Isaac Lab: sim.dt=0.005）
N_SUBSTEPS = int(round(POLICY_DT / PHYSICS_DT))  # 4 步物理 / 1 步策略

# ── 策略超参数（来自 deploy.yaml 和训练配置）──────────────────────────────────
ACTION_SCALE = 0.25   # JointPositionAction.scale
GAIT_PERIOD = 0.6     # 步态相位周期（秒）

# ── Isaac Lab H1 策略关节顺序 ─────────────────────────────────────────────────
# 由 joint_sdk_names + joint_ids_map 推导：
#   joint_ids_map[policy_idx] = SDK_idx
#   joint_sdk_names[SDK_idx]  = 关节名
# 推导过程见注释底部
POLICY_JOINT_NAMES = [
    "left_hip_yaw_joint",        "right_hip_yaw_joint",        "torso_joint",
    "left_hip_roll_joint",       "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_pitch_joint",      "right_hip_pitch_joint",
    "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
    "left_knee_joint",           "right_knee_joint",
    "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
    "left_ankle_joint",          "right_ankle_joint",
    "left_elbow_joint",          "right_elbow_joint",
]
N_JOINTS = len(POLICY_JOINT_NAMES)  # 19

# ── 默认关节位置（策略顺序，来自 deploy.yaml default_joint_pos）────────────────
DEFAULT_JOINT_POS = np.array([
    0.0,   0.0,   0.0,   # left_hip_yaw,   right_hip_yaw,   torso
    0.0,   0.0,           # left_hip_roll,  right_hip_roll
    0.2,   0.2,           # left_shoulder_pitch, right_shoulder_pitch
   -0.1,  -0.1,           # left_hip_pitch, right_hip_pitch
    0.0,   0.0,           # left_shoulder_roll, right_shoulder_roll
    0.3,   0.3,           # left_knee,      right_knee
    0.0,   0.0,           # left_shoulder_yaw, right_shoulder_yaw
   -0.2,  -0.2,           # left_ankle,     right_ankle
    0.32,  0.32,          # left_elbow,     right_elbow
], dtype=np.float32)

# deploy/robots/h1/config/config.yaml 里的 FixStand 目标，转换到 policy 顺序。
CONTROLLER_FIXSTAND_POS = np.array([
    0.0,   0.0,   0.0,   # left_hip_yaw, right_hip_yaw, torso
    0.0,   0.0,           # left_hip_roll, right_hip_roll
    0.2,   0.2,           # left_shoulder_pitch, right_shoulder_pitch
   -0.28, -0.28,          # left_hip_pitch, right_hip_pitch
    0.0,   0.0,           # left_shoulder_roll, right_shoulder_roll
    0.79,  0.79,          # left_knee, right_knee
    0.0,   0.0,           # left_shoulder_yaw, right_shoulder_yaw
   -0.52, -0.52,          # left_ankle, right_ankle
    0.32,  0.32,          # left_elbow, right_elbow
], dtype=np.float32)

# ── PD 增益（策略顺序，由 deploy.yaml stiffness/damping + joint_ids_map 推导）──
# policy idx → SDK idx → stiffness/damping[SDK idx]
KP = np.array([
    150, 150, 300,   # left_hip_yaw, right_hip_yaw, torso
    150, 150,        # left_hip_roll, right_hip_roll
    100, 100,        # left_shoulder_pitch, right_shoulder_pitch
    150, 150,        # left_hip_pitch, right_hip_pitch
    100, 100,        # left_shoulder_roll, right_shoulder_roll
    200, 200,        # left_knee, right_knee
     50,  50,        # left_shoulder_yaw, right_shoulder_yaw
     40,  40,        # left_ankle, right_ankle
     50,  50,        # left_elbow, right_elbow
], dtype=np.float32)

KD = np.array([
    2, 2, 6,   # left_hip_yaw, right_hip_yaw, torso
    2, 2,      # left_hip_roll, right_hip_roll
    2, 2,      # left_shoulder_pitch, right_shoulder_pitch
    2, 2,      # left_hip_pitch, right_hip_pitch
    2, 2,      # left_shoulder_roll, right_shoulder_roll
    4, 4,      # left_knee, right_knee
    2, 2,      # left_shoulder_yaw, right_shoulder_yaw
    2, 2,      # left_ankle, right_ankle
    2, 2,      # left_elbow, right_elbow
], dtype=np.float32)

EFFORT_LIMIT = np.array([
    200, 200, 200,  # left_hip_yaw, right_hip_yaw, torso
    200, 200,       # left_hip_roll, right_hip_roll
     40,  40,       # left_shoulder_pitch, right_shoulder_pitch
    200, 200,       # left_hip_pitch, right_hip_pitch
     40,  40,       # left_shoulder_roll, right_shoulder_roll
    300, 300,       # left_knee, right_knee
     18,  18,       # left_shoulder_yaw, right_shoulder_yaw
     40,  40,       # left_ankle, right_ankle
     18,  18,       # left_elbow, right_elbow
], dtype=np.float32)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def build_mappings(model: mujoco.MjModel):
    """
    构建策略关节顺序 → MuJoCo 数组索引的映射。

    Returns:
        pol2qpos[i]: qpos[7:] 中策略关节 i 的索引
        pol2qvel[i]: qvel[6:] 中策略关节 i 的索引
        pol2ctrl[i]: ctrl[] 中策略关节 i 的执行器索引
    """
    pol2qpos = np.zeros(N_JOINTS, dtype=int)
    pol2qvel = np.zeros(N_JOINTS, dtype=int)
    pol2ctrl = np.zeros(N_JOINTS, dtype=int)

    for i, name in enumerate(POLICY_JOINT_NAMES):
        candidates = [name, f"{name}_joint"]
        jnt_id = -1
        jnt_name = None
        for candidate in candidates:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
            if jnt_id >= 0:
                jnt_name = candidate
                break
        if jnt_id < 0:
            raise ValueError(f"关节 '{name}' 在 MuJoCo 模型中不存在。")
        pol2qpos[i] = model.jnt_qposadr[jnt_id] - 7  # 减去自由关节的 pos+quat (7)
        pol2qvel[i] = model.jnt_dofadr[jnt_id] - 6   # 减去自由关节的 lin+ang vel (6)

        act_id = -1
        for candidate in candidates:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate)
            if act_id >= 0:
                break
        if act_id < 0:
            raise ValueError(f"执行器 '{name}' 在 MuJoCo 模型中不存在。")
        pol2ctrl[i] = act_id

    return pol2qpos, pol2qvel, pol2ctrl


def print_mapping(model: mujoco.MjModel, pol2qpos: np.ndarray, pol2qvel: np.ndarray, pol2ctrl: np.ndarray):
    print("[映射详情] policy_idx | policy_name | qpos[7+] | qvel[6+] | actuator_idx | actuator_name")
    for i, name in enumerate(POLICY_JOINT_NAMES):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(pol2ctrl[i]))
        print(
            f"  {i:02d} | {name:<22} | {pol2qpos[i]:02d} | {pol2qvel[i]:02d} | "
            f"{pol2ctrl[i]:02d} | {act_name}"
        )


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    将世界系向量 v 转换到体坐标系。
    MuJoCo 四元数格式: (w, x, y, z)
    """
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    return R.T @ v  # R.T = 世界→体坐标系旋转


def get_observation(
    data: mujoco.MjData,
    pol2qpos: np.ndarray,
    pol2qvel: np.ndarray,
    cmd: np.ndarray,
    last_action: np.ndarray,
    sim_time: float,
) -> np.ndarray:
    """构建 68 维观测向量，与 Isaac Lab PolicyCfg 完全一致。"""

    root_quat = data.qpos[3:7]  # MuJoCo 四元数 (w, x, y, z)

    # [3] 躯干角速度（体坐标系）× 0.2
    # MuJoCo 3.x 自由关节：qvel[3:6] 是全局坐标系角速度，需转换到体坐标系
    omega_world = data.qvel[3:6]
    omega_body = quat_rotate_inverse(root_quat, omega_world)
    ang_vel_obs = (omega_body * 0.2).astype(np.float32)

    # [3] 投影重力方向（体坐标系）
    gravity_world = np.array([0.0, 0.0, -1.0])
    proj_gravity = quat_rotate_inverse(root_quat, gravity_world).astype(np.float32)

    # [3] 速度指令
    vel_cmd = cmd.astype(np.float32)

    # [19] 关节位置相对默认值（策略顺序）
    q_pol = data.qpos[7:][pol2qpos].astype(np.float32)
    joint_pos_rel = q_pol - DEFAULT_JOINT_POS

    # [19] 关节速度 × 0.05（策略顺序）
    dq_pol = data.qvel[6:][pol2qvel].astype(np.float32)
    joint_vel = dq_pol * 0.05

    # [19] 上一步动作
    last_act = last_action.astype(np.float32)

    # [2] 步态相位 sin/cos（周期 0.6s）
    phase = (sim_time % GAIT_PERIOD) / GAIT_PERIOD
    gait = np.array([
        np.sin(2.0 * np.pi * phase),
        np.cos(2.0 * np.pi * phase),
    ], dtype=np.float32)

    obs = np.concatenate([
        ang_vel_obs,    # 3
        proj_gravity,   # 3
        vel_cmd,        # 3
        joint_pos_rel,  # 19
        joint_vel,      # 19
        last_act,       # 19
        gait,           # 2
    ])  # 合计 68 维
    return obs


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H1 Sim2Sim: Isaac Lab → MuJoCo")
    parser.add_argument("--vx", type=float, default=0.5, help="前进速度指令 (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="侧向速度指令 (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="偏航角速度指令 (rad/s)")
    parser.add_argument("--scene",  type=str, default=SCENE_XML,   help="MuJoCo scene.xml 路径")
    parser.add_argument("--policy", type=str, default=POLICY_ONNX, help="policy.onnx 路径")
    parser.add_argument("--mode", choices=["policy", "fixstand", "controller-fixstand", "zero"], default="policy",
                        help="policy=运行 ONNX 策略；fixstand=默认站姿 PD；controller-fixstand=deploy H1 FixStand 姿态；zero=零力矩")
    parser.add_argument("--duration", type=float, default=20.0, help="headless 模式运行秒数")
    parser.add_argument("--headless", action="store_true", help="不打开 viewer，直接跑诊断")
    parser.add_argument("--print-map", action="store_true", help="打印 policy 到 MuJoCo 的映射")
    parser.add_argument("--init-z", type=float, default=1.1, help="初始 base 高度")
    parser.add_argument("--no-torque-clip", action="store_true", help="不按 Isaac effort_limit 裁剪 PD 力矩")
    args = parser.parse_args()

    cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)

    # ── 加载模型与策略 ────────────────────────────────────────────────────────
    print(f"[加载] MuJoCo 模型: {args.scene}")
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    model.opt.timestep = PHYSICS_DT

    sess = None
    input_name = None
    output_name = None
    if args.mode == "policy":
        print(f"[加载] 策略 ONNX: {args.policy}")
        sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

    # ── 构建关节映射 ──────────────────────────────────────────────────────────
    pol2qpos, pol2qvel, pol2ctrl = build_mappings(model)
    print(f"[映射] 关节数: {N_JOINTS}  观测维度: 68  动作维度: {N_JOINTS}")
    if args.print_map:
        print_mapping(model, pol2qpos, pol2qvel, pol2ctrl)

    # ── 物理参数对齐 Isaac Lab ────────────────────────────────────────────────
    # Isaac Lab IdealPDActuator 不使用被动关节阻尼（KD 控制所有阻尼）
    model.dof_damping[:] = 0.0
    # Isaac Lab actuator armature=0.01；MuJoCo h1.xml 默认 armature=0.1（大 10x，关节迟钝）
    model.dof_armature[:] = 0.01

    # ── 初始化机器人姿态，精确匹配 Isaac Lab init_state ──────────────────────
    # Isaac Lab: pos=(0,0,1.1), joint_pos=default, joint_vel=0, base_vel=0
    data.qpos[:] = 0.0
    data.qpos[2] = args.init_z  # Isaac Lab init_state: z=1.1m（略高于地面，自然落下建立接触）
    data.qpos[3] = 1.0     # 四元数 w=1（朝上）
    data.qpos[7:][pol2qpos] = DEFAULT_JOINT_POS
    mujoco.mj_forward(model, data)

    last_action = np.zeros(N_JOINTS, dtype=np.float32)

    print("=" * 60)
    print(f"  速度指令: vx={args.vx:+.2f}  vy={args.vy:+.2f}  wz={args.wz:+.2f}")
    print(f"  模式: {args.mode}  |  torque_clip={'off' if args.no_torque_clip else 'on'}")
    print(f"  策略频率: {1/POLICY_DT:.0f} Hz  |  物理频率: {1/PHYSICS_DT:.0f} Hz  |  子步数: {N_SUBSTEPS}")
    print("  按 ESC 或关闭窗口退出")
    print("=" * 60)

    def run_policy_step():
        nonlocal last_action
        obs = get_observation(data, pol2qpos, pol2qvel, cmd, last_action, data.time)
        actions = sess.run([output_name], {input_name: obs.reshape(1, -1)})[0][0].astype(np.float32)
        last_action = actions.copy()
        return DEFAULT_JOINT_POS + ACTION_SCALE * actions, actions

    def run_control_step(q_target):
        q_pol = data.qpos[7:][pol2qpos].astype(np.float32)
        dq_pol = data.qvel[6:][pol2qvel].astype(np.float32)
        torques = KP * (q_target - q_pol) - KD * dq_pol
        if not args.no_torque_clip:
            torques = np.clip(torques, -EFFORT_LIMIT, EFFORT_LIMIT)
        data.ctrl[pol2ctrl] = torques
        return q_pol, dq_pol, torques

    def report(t, actions, torques):
        h = data.qpos[2]
        vx = data.qvel[0]
        vy = data.qvel[1]
        wz = data.qvel[5]
        quat = data.qpos[3:7]
        gravity_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        tilt_xy = float(np.linalg.norm(gravity_b[:2]))
        action_max = float(np.max(np.abs(actions))) if actions is not None else 0.0
        torque_max = float(np.max(np.abs(torques))) if torques is not None else 0.0
        print(
            f"  t={t:6.2f}s | h={h:.3f}m tilt_xy={tilt_xy:.2f} | "
            f"vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} | "
            f"max|a|={action_max:.2f} max|tau|={torque_max:.1f}"
        )

    def step_once():
        if args.mode == "policy":
            q_target, actions = run_policy_step()
        elif args.mode == "fixstand":
            q_target = DEFAULT_JOINT_POS
            actions = None
        elif args.mode == "controller-fixstand":
            q_target = CONTROLLER_FIXSTAND_POS
            actions = None
        else:
            q_target = DEFAULT_JOINT_POS
            actions = None

        torques = None
        for _ in range(N_SUBSTEPS):
            if args.mode == "zero":
                data.ctrl[:] = 0.0
            else:
                _, _, torques = run_control_step(q_target)
            mujoco.mj_step(model, data)
        return actions, torques

    if args.headless:
        next_report = 0.0
        while data.time < args.duration:
            actions, torques = step_once()
            if data.time + 1e-9 >= next_report:
                report(data.time, actions, torques)
                next_report += 1.0
        return

    # ── 主控制循环 ────────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 90.0

        while viewer.is_running():
            step_start = time.time()

            actions, torques = step_once()
            viewer.sync()

            # 6. 实时节拍
            elapsed = time.time() - step_start
            sleep_t = POLICY_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            # 7. 每 2 秒打印一次状态
            t = data.time
            if t > 0 and (t % 2.0) < POLICY_DT * 1.5:
                report(t, actions, torques)


# ── 关节顺序推导说明 ──────────────────────────────────────────────────────────
# joint_ids_map (Isaac Lab 策略顺序 → SDK 索引):
#   [7, 8, 6, 3, 0, 16, 12, 4, 1, 17, 13, 5, 2, 18, 14, 10, 11, 19, 15]
#
# joint_sdk_names (SDK 索引 → 关节名):
#   0:right_hip_roll  1:right_hip_pitch  2:right_knee  3:left_hip_roll
#   4:left_hip_pitch  5:left_knee        6:torso       7:left_hip_yaw
#   8:right_hip_yaw  10:left_ankle      11:right_ankle 12:right_shoulder_pitch
#  13:right_shoulder_roll 14:right_shoulder_yaw 15:right_elbow
#  16:left_shoulder_pitch 17:left_shoulder_roll 18:left_shoulder_yaw 19:left_elbow
#
# 策略关节名 = joint_sdk_names[joint_ids_map[policy_idx]]:
#   policy 0 → SDK 7  → left_hip_yaw
#   policy 1 → SDK 8  → right_hip_yaw
#   policy 2 → SDK 6  → torso
#   policy 3 → SDK 3  → left_hip_roll
#   policy 4 → SDK 0  → right_hip_roll
#   policy 5 → SDK 16 → left_shoulder_pitch
#   policy 6 → SDK 12 → right_shoulder_pitch
#   policy 7 → SDK 4  → left_hip_pitch
#   policy 8 → SDK 1  → right_hip_pitch
#   policy 9 → SDK 17 → left_shoulder_roll
#  policy 10 → SDK 13 → right_shoulder_roll
#  policy 11 → SDK 5  → left_knee
#  policy 12 → SDK 2  → right_knee
#  policy 13 → SDK 18 → left_shoulder_yaw
#  policy 14 → SDK 14 → right_shoulder_yaw
#  policy 15 → SDK 10 → left_ankle
#  policy 16 → SDK 11 → right_ankle
#  policy 17 → SDK 19 → left_elbow
#  policy 18 → SDK 15 → right_elbow

if __name__ == "__main__":
    main()

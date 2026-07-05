#!/usr/bin/env python3
"""
MuJoCo H1 诊断脚本 - 无 viewer，全自动批量测试
测试不同物理参数配置 + 控制策略，输出结构化报告

用法:
  python diagnose_mujoco.py              # 运行全部场景
  python diagnose_mujoco.py --duration 5  # 每个场景只跑 5s
"""

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import onnxruntime as ort

# ── 路径 ───────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_DIR = Path(os.environ.get("MUJOCO_MENAGERIE_DIR", "/home/hazzzard/mujoco_menagerie"))
UNITREE_MUJOCO_DIR = Path(os.environ.get("UNITREE_MUJOCO_DIR", "/tmp/unitree_mujoco"))
MENAGERIE_XML = str(MENAGERIE_DIR / "unitree_h1/scene.xml")
UNITREE_XML = str(UNITREE_MUJOCO_DIR / "unitree_robots/h1/scene.xml")
POLICY_ONNX = str(REPO_ROOT / "policies/weight-velocity-track/weight-velocity-track.onnx")

# ── 策略超参数（来自 deploy.yaml）─────────────────────────────────────────────
POLICY_JOINT_NAMES_MENAGERIE = [
    "left_hip_yaw", "right_hip_yaw", "torso",
    "left_hip_roll", "right_hip_roll",
    "left_shoulder_pitch", "right_shoulder_pitch",
    "left_hip_pitch", "right_hip_pitch",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_knee", "right_knee",
    "left_shoulder_yaw", "right_shoulder_yaw",
    "left_ankle", "right_ankle",
    "left_elbow", "right_elbow",
]
# unitree_mujoco 的关节名带 _joint 后缀，顺序相同
POLICY_JOINT_NAMES_UNITREE = [n + "_joint" for n in POLICY_JOINT_NAMES_MENAGERIE]
POLICY_JOINT_NAMES_UNITREE[2] = "torso_joint"  # torso 本来就不需要改，但保持一致

N_JOINTS = 19

DEFAULT_JOINT_POS = np.array([
    0.0, 0.0, 0.0,
    0.0, 0.0,
    0.2, 0.2,
   -0.1,-0.1,
    0.0, 0.0,
    0.3, 0.3,
    0.0, 0.0,
   -0.2,-0.2,
    0.32, 0.32,
], dtype=np.float32)

KP = np.array([
    150, 150, 300,
    150, 150,
    100, 100,
    150, 150,
    100, 100,
    200, 200,
     50,  50,
     40,  40,
     50,  50,
], dtype=np.float32)

KD = np.array([
    2, 2, 6,
    2, 2,
    2, 2,
    2, 2,
    2, 2,
    4, 4,
    2, 2,
    2, 2,
    2, 2,
], dtype=np.float32)

ACTION_SCALE = 0.25
GAIT_PERIOD  = 0.6


# ── 场景定义 ───────────────────────────────────────────────────────────────────

@dataclass
class SceneCfg:
    name: str
    xml: str
    joint_names: list
    # 物理参数覆盖（None = 保持 XML 原值）
    damping: Optional[float]    = None   # dof_damping
    armature: Optional[float]   = None   # dof_armature
    timestep: Optional[float]   = None   # opt.timestep
    # 控制策略
    use_policy: bool            = False
    vx: float                   = 0.0
    vy: float                   = 0.0
    wz: float                   = 0.0


SCENES = [
    # ── 1. menagerie 原始物理，纯 PD 站立（不用策略）──────────────────────────
    SceneCfg("A_menag_raw_pdstand",   MENAGERIE_XML, POLICY_JOINT_NAMES_MENAGERIE,
             damping=None, armature=None, timestep=None,
             use_policy=False),

    # ── 2. menagerie 修正物理，纯 PD 站立 ──────────────────────────────────────
    SceneCfg("B_menag_fixed_pdstand", MENAGERIE_XML, POLICY_JOINT_NAMES_MENAGERIE,
             damping=0.0, armature=0.01, timestep=0.005,
             use_policy=False),

    # ── 3. menagerie 原始物理 + 策略 vx=0 ─────────────────────────────────────
    SceneCfg("C_menag_raw_policy_v0", MENAGERIE_XML, POLICY_JOINT_NAMES_MENAGERIE,
             damping=None, armature=None, timestep=None,
             use_policy=True, vx=0.0),

    # ── 4. menagerie 修正物理 + 策略 vx=0 ─────────────────────────────────────
    SceneCfg("D_menag_fixed_policy_v0", MENAGERIE_XML, POLICY_JOINT_NAMES_MENAGERIE,
             damping=0.0, armature=0.01, timestep=0.005,
             use_policy=True, vx=0.0),

    # ── 5. 只修 armature，保留原始 damping ────────────────────────────────────
    SceneCfg("E_menag_arm001_damp1_v0", MENAGERIE_XML, POLICY_JOINT_NAMES_MENAGERIE,
             damping=None, armature=0.01, timestep=0.005,
             use_policy=True, vx=0.0),

    # ── 6. 只去 damping，保留原始 armature ────────────────────────────────────
    SceneCfg("F_menag_arm01_damp0_v0", MENAGERIE_XML, POLICY_JOINT_NAMES_MENAGERIE,
             damping=0.0, armature=None, timestep=0.005,
             use_policy=True, vx=0.0),

    # ── 7. menagerie 修正物理 + 策略 vx=0.5 ───────────────────────────────────
    SceneCfg("G_menag_fixed_policy_v05", MENAGERIE_XML, POLICY_JOINT_NAMES_MENAGERIE,
             damping=0.0, armature=0.01, timestep=0.005,
             use_policy=True, vx=0.5),

    # ── 8. unitree_mujoco 原始物理 + 策略 vx=0 ────────────────────────────────
    SceneCfg("H_unitree_raw_policy_v0", UNITREE_XML, POLICY_JOINT_NAMES_UNITREE,
             damping=None, armature=None, timestep=None,
             use_policy=True, vx=0.0),

    # ── 9. unitree_mujoco 修正物理 + 策略 vx=0 ────────────────────────────────
    SceneCfg("I_unitree_fixed_policy_v0", UNITREE_XML, POLICY_JOINT_NAMES_UNITREE,
             damping=0.0, armature=0.01, timestep=0.005,
             use_policy=True, vx=0.0),
]


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def build_mappings(model, joint_names):
    pol2qpos = np.zeros(N_JOINTS, dtype=int)
    pol2qvel = np.zeros(N_JOINTS, dtype=int)
    pol2ctrl = np.zeros(N_JOINTS, dtype=int)
    missing = []
    for i, name in enumerate(joint_names):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            missing.append(name); continue
        pol2qpos[i] = model.jnt_qposadr[jnt_id] - 7
        pol2qvel[i] = model.jnt_dofadr[jnt_id] - 6
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id < 0:
            missing.append(f"ACT:{name}"); continue
        pol2ctrl[i] = act_id
    return pol2qpos, pol2qvel, pol2ctrl, missing


def quat_rotate_inverse(q_wxyz, v):
    w, x, y, z = q_wxyz
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    return R.T @ v


def quat_to_euler(q_wxyz):
    """四元数 (w,x,y,z) → roll, pitch, yaw (rad)"""
    w, x, y, z = q_wxyz
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw


def get_obs(data, pol2qpos, pol2qvel, cmd, last_action, sim_time):
    root_quat   = data.qpos[3:7]
    omega_world = data.qvel[3:6]
    omega_body  = quat_rotate_inverse(root_quat, omega_world)
    ang_vel_obs = (omega_body * 0.2).astype(np.float32)

    gravity_world = np.array([0., 0., -1.])
    proj_gravity  = quat_rotate_inverse(root_quat, gravity_world).astype(np.float32)

    q_pol  = data.qpos[7:][pol2qpos].astype(np.float32)
    dq_pol = data.qvel[6:][pol2qvel].astype(np.float32)

    phase = (sim_time % GAIT_PERIOD) / GAIT_PERIOD
    gait  = np.array([math.sin(2*math.pi*phase), math.cos(2*math.pi*phase)], dtype=np.float32)

    return np.concatenate([
        ang_vel_obs,
        proj_gravity,
        cmd.astype(np.float32),
        q_pol - DEFAULT_JOINT_POS,
        dq_pol * 0.05,
        last_action.astype(np.float32),
        gait,
    ])


# ── 单场景运行 ─────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str
    survival_s: float
    cause: str
    max_height: float
    min_height: float
    mean_torque: float
    joint_err_t0: float          # 初始时刻关节位置偏差（验证映射）
    physics_actual: dict         # 实际用的 armature/damping/dt
    log: list = field(default_factory=list)   # [(t, z, roll, pitch, vx, vy)]


def run_scenario(cfg: SceneCfg, duration: float, policy_sess=None) -> RunResult:
    # 加载模型
    model = mujoco.MjModel.from_xml_path(cfg.xml)
    data  = mujoco.MjData(model)

    # 记录原始物理值
    orig_arm  = float(model.dof_armature[7])  # 取第一个非根关节
    orig_damp = float(model.dof_damping[7])
    orig_dt   = float(model.opt.timestep)

    # 应用物理覆盖
    if cfg.damping  is not None: model.dof_damping[:]  = cfg.damping
    if cfg.armature is not None: model.dof_armature[:] = cfg.armature
    if cfg.timestep is not None: model.opt.timestep    = cfg.timestep

    actual_dt   = float(model.opt.timestep)
    policy_dt   = 0.02  # 50 Hz
    n_substeps  = max(1, round(policy_dt / actual_dt))

    # 构建关节映射
    pol2qpos, pol2qvel, pol2ctrl, missing = build_mappings(model, cfg.joint_names)
    if missing:
        return RunResult(cfg.name, 0.0, f"MAPPING_FAIL:{missing}", 0, 0, 0, 999,
                         {"arm": cfg.armature, "damp": cfg.damping, "dt": actual_dt})

    # 初始姿态
    data.qpos[:] = 0.0
    data.qpos[2] = 1.05   # 稍高于地面
    data.qpos[3] = 1.0    # w=1 四元数朝上
    data.qpos[7:][pol2qpos] = DEFAULT_JOINT_POS
    mujoco.mj_forward(model, data)

    # 记录 t=0 时刻关节映射误差（验证映射是否正确）
    q_read = data.qpos[7:][pol2qpos].astype(np.float32)
    joint_err_t0 = float(np.max(np.abs(q_read - DEFAULT_JOINT_POS)))

    last_action = np.zeros(N_JOINTS, dtype=np.float32)
    cmd = np.array([cfg.vx, cfg.vy, cfg.wz], dtype=np.float32)

    log = []
    torque_sum = 0.0
    torque_cnt = 0
    survival_s = duration
    cause = "timeout"

    # 仿真主循环
    n_steps = int(duration / policy_dt)
    for step in range(n_steps):
        t = data.time

        if cfg.use_policy and policy_sess is not None:
            obs = get_obs(data, pol2qpos, pol2qvel, cmd, last_action, t)
            inp_name  = policy_sess.get_inputs()[0].name
            out_name  = policy_sess.get_outputs()[0].name
            actions   = policy_sess.run([out_name], {inp_name: obs.reshape(1,-1)})[0][0]
            last_action = actions.copy()
            q_target  = DEFAULT_JOINT_POS + ACTION_SCALE * actions.astype(np.float32)
        else:
            # 纯 PD 保持默认站姿
            q_target = DEFAULT_JOINT_POS.copy()

        # 每 policy step 内推进 n_substeps 物理步
        for _ in range(n_substeps):
            q_pol  = data.qpos[7:][pol2qpos].astype(np.float32)
            dq_pol = data.qvel[6:][pol2qvel].astype(np.float32)
            torques = KP * (q_target - q_pol) - KD * dq_pol
            data.ctrl[pol2ctrl] = torques
            mujoco.mj_step(model, data)

            torque_sum += float(np.mean(np.abs(torques)))
            torque_cnt += 1

        # 状态读取
        z    = float(data.qpos[2])
        quat = data.qpos[3:7]
        roll, pitch, _ = quat_to_euler(quat)
        vx = float(data.qvel[0])
        vy = float(data.qvel[1])

        if step % 25 == 0:  # 每 0.5s 记一次
            log.append((round(t, 2), round(z, 4),
                        round(math.degrees(roll), 1),
                        round(math.degrees(pitch), 1),
                        round(vx, 3), round(vy, 3)))

        # 倒地判断：高度 < 0.35m 或倾角 > 70°
        if z < 0.35:
            survival_s = t; cause = f"FELL_HEIGHT(z={z:.3f})"; break
        if abs(roll) > math.radians(70) or abs(pitch) > math.radians(70):
            survival_s = t; cause = f"FELL_TILT(r={math.degrees(roll):.1f}°,p={math.degrees(pitch):.1f}°)"; break

    zs = [r[1] for r in log] if log else [1.05]
    mean_torque = torque_sum / max(torque_cnt, 1)

    return RunResult(
        name         = cfg.name,
        survival_s   = survival_s,
        cause        = cause,
        max_height   = max(zs),
        min_height   = min(zs),
        mean_torque  = mean_torque,
        joint_err_t0 = joint_err_t0,
        physics_actual = {
            "orig_arm":  orig_arm,
            "orig_damp": orig_damp,
            "orig_dt":   orig_dt,
            "used_arm":  cfg.armature if cfg.armature is not None else orig_arm,
            "used_damp": cfg.damping  if cfg.damping  is not None else orig_damp,
            "used_dt":   actual_dt,
        },
        log = log,
    )


# ── 报告打印 ───────────────────────────────────────────────────────────────────

def print_report(results: list[RunResult], duration: float):
    SEP = "═" * 100
    print(f"\n{SEP}")
    print(f"  MuJoCo H1 诊断报告  (每场景最长 {duration:.0f}s)")
    print(SEP)

    header = f"{'场景':<38} {'存活(s)':>8} {'状态':>6} {'结果':<35} {'平均力矩':>9} {'关节映射误差':>12}"
    print(header)
    print("─" * 100)

    for r in results:
        status = "✅ OK" if r.survival_s >= duration else "❌ 倒"
        print(
            f"{r.name:<38} {r.survival_s:>8.2f} {status:>6}  "
            f"{r.cause:<35} {r.mean_torque:>9.1f}  {r.joint_err_t0:>12.6f}"
        )

    print(f"\n{'─'*100}")
    print("详细物理参数 & 状态日志（只打印前5条和最后2条）:")
    print()
    for r in results:
        ph = r.physics_actual
        print(f"  [{r.name}]")
        print(f"    XML原始: armature={ph['orig_arm']:.3f}  damping={ph['orig_damp']:.3f}  dt={ph['orig_dt']:.4f}")
        print(f"    实际用:  armature={ph['used_arm']:.3f}  damping={ph['used_damp']:.3f}  dt={ph['used_dt']:.4f}")
        if r.log:
            rows = r.log[:5] + (r.log[-2:] if len(r.log) > 7 else [])
            print(f"    {'时间':>6} {'高度':>7} {'roll°':>7} {'pitch°':>7} {'vx':>7} {'vy':>7}")
            for row in rows:
                t, z, roll, pitch, vx, vy = row
                print(f"    {t:>6.1f} {z:>7.4f} {roll:>7.1f} {pitch:>7.1f} {vx:>7.3f} {vy:>7.3f}")
        print()

    # 汇总结论
    print(SEP)
    best = max(results, key=lambda r: r.survival_s)
    pd_results  = [r for r in results if "pdstand" in r.name]
    pol_results = [r for r in results if "policy" in r.name]

    print("结论:")
    if pd_results:
        print(f"  纯PD站立最佳: {max(pd_results, key=lambda r: r.survival_s).name}")
    if pol_results:
        print(f"  策略最佳:     {max(pol_results, key=lambda r: r.survival_s).name}")
    print(f"  总体最佳:     {best.name}  (存活 {best.survival_s:.2f}s)")

    # 物理参数影响分析
    d_raw   = next((r for r in results if "C_" in r.name), None)
    d_fixed = next((r for r in results if "D_" in r.name), None)
    if d_raw and d_fixed:
        delta = d_fixed.survival_s - d_raw.survival_s
        print(f"  物理修正效果: 原始={d_raw.survival_s:.2f}s  修正后={d_fixed.survival_s:.2f}s  Δ={delta:+.2f}s")

    e_arm = next((r for r in results if "E_" in r.name), None)
    f_dmp = next((r for r in results if "F_" in r.name), None)
    if e_arm and f_dmp:
        print(f"  只修armature: {e_arm.survival_s:.2f}s  只去damping: {f_dmp.survival_s:.2f}s")
        if e_arm.survival_s > f_dmp.survival_s:
            print("  ► armature 不对是主要问题")
        else:
            print("  ► damping 不对是主要问题")

    print(SEP)


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=15.0,
                    help="每个场景仿真时长（秒），默认 15s")
    ap.add_argument("--scene",    type=str,   default=None,
                    help="只运行指定前缀的场景，如 D 表示只跑 D_*")
    args = ap.parse_args()

    # 加载策略（只加载一次）
    print(f"[加载] ONNX 策略: {POLICY_ONNX}")
    policy_sess = ort.InferenceSession(POLICY_ONNX, providers=["CPUExecutionProvider"])
    print("[加载] OK")

    scenes = SCENES
    if args.scene:
        scenes = [s for s in SCENES if s.name.startswith(args.scene)]
        if not scenes:
            print(f"没有匹配 '{args.scene}' 的场景，可用场景: {[s.name for s in SCENES]}")
            return

    available_scenes = []
    for scene in scenes:
        if Path(scene.xml).exists():
            available_scenes.append(scene)
        else:
            print(f"[跳过] {scene.name}: 场景文件不存在 {scene.xml}")
    scenes = available_scenes
    if not scenes:
        print("没有可运行场景。请设置 UNITREE_MUJOCO_DIR 或 MUJOCO_MENAGERIE_DIR。")
        return

    results = []
    for i, cfg in enumerate(scenes):
        print(f"\n[{i+1}/{len(scenes)}] 运行: {cfg.name}  (max {args.duration}s) ...", flush=True)
        r = run_scenario(cfg, args.duration, policy_sess if cfg.use_policy else None)
        results.append(r)
        print(f"       存活={r.survival_s:.2f}s  原因={r.cause}")

    print_report(results, args.duration)


if __name__ == "__main__":
    main()

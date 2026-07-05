#!/usr/bin/env python3
"""
MuJoCo H1 Sim2Sim 录像脚本 - 离屏渲染，不依赖 X11 窗口截图
用法:
  python record_sim2sim.py --vx 0.5 --duration 30 --out video.mp4
"""

import argparse
import math
import os
from pathlib import Path

import imageio
import mujoco
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
UNITREE_MUJOCO_DIR = Path(os.environ.get("UNITREE_MUJOCO_DIR", "/tmp/unitree_mujoco"))
SCENE_XML = str(UNITREE_MUJOCO_DIR / "unitree_robots/h1/scene.xml")
POLICY_ONNX = str(REPO_ROOT / "policies/weight-velocity-track/weight-velocity-track.onnx")

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

KP = np.array([150,150,300,150,150,100,100,150,150,100,100,200,200,50,50,40,40,50,50], dtype=np.float32)
KD = np.array([2,2,6,2,2,2,2,2,2,2,2,4,4,2,2,2,2,2,2], dtype=np.float32)

POLICY_DT  = 0.02
PHYSICS_DT = 0.002   # unitree_mujoco 原始 timestep
ACTION_SCALE = 0.25
GAIT_PERIOD  = 0.6
RENDER_FPS   = 30    # 视频帧率
RENDER_EVERY = max(1, round(POLICY_DT * RENDER_FPS))  # 每隔几个 policy step 渲一帧


def build_mappings(model):
    p2q = np.zeros(N_JOINTS, dtype=int)
    p2v = np.zeros(N_JOINTS, dtype=int)
    p2c = np.zeros(N_JOINTS, dtype=int)
    for i, name in enumerate(POLICY_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"关节不存在: {name}")
        p2q[i] = model.jnt_qposadr[jid] - 7
        p2v[i] = model.jnt_dofadr[jid] - 6
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"执行器不存在: {name}")
        p2c[i] = aid
    return p2q, p2v, p2c


def quat_rotate_inverse(q, v):
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    return R.T @ v


def get_obs(data, p2q, p2v, cmd, last_action, t):
    q    = data.qpos[3:7]
    omb  = quat_rotate_inverse(q, data.qvel[3:6])
    pg   = quat_rotate_inverse(q, [0, 0, -1])
    ph   = (t % GAIT_PERIOD) / GAIT_PERIOD
    return np.concatenate([
        (omb * 0.2).astype(np.float32),
        pg.astype(np.float32),
        cmd,
        (data.qpos[7:][p2q] - DEFAULT_JOINT_POS).astype(np.float32),
        (data.qvel[6:][p2v] * 0.05).astype(np.float32),
        last_action.astype(np.float32),
        np.array([math.sin(2*math.pi*ph), math.cos(2*math.pi*ph)], dtype=np.float32),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx",       type=float, default=0.5)
    ap.add_argument("--vy",       type=float, default=0.0)
    ap.add_argument("--wz",       type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=30.0, help="录制时长（秒）")
    ap.add_argument("--out",      type=str,   default="sim2sim_h1.mp4")
    ap.add_argument("--width",    type=int,   default=640)
    ap.add_argument("--height",   type=int,   default=480)
    ap.add_argument("--scene",    type=str,   default=SCENE_XML)
    ap.add_argument("--policy",   type=str,   default=POLICY_ONNX)
    args = ap.parse_args()

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = str(REPO_ROOT / "media" / out_path)

    print(f"[加载] 场景: {args.scene}")
    model = mujoco.MjModel.from_xml_path(args.scene)
    data  = mujoco.MjData(model)

    print(f"[加载] 策略: {args.policy}")
    sess   = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    iname  = sess.get_inputs()[0].name
    oname  = sess.get_outputs()[0].name

    p2q, p2v, p2c = build_mappings(model)

    data.qpos[:] = 0.0
    data.qpos[2] = 1.05
    data.qpos[3] = 1.0
    data.qpos[7:][p2q] = DEFAULT_JOINT_POS
    mujoco.mj_forward(model, data)

    last_action = np.zeros(N_JOINTS, dtype=np.float32)
    cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)

    n_substeps = max(1, round(POLICY_DT / float(model.opt.timestep)))
    n_policy_steps = int(args.duration / POLICY_DT)

    # 相机设置
    camera = mujoco.MjvCamera()
    camera.type     = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    camera.distance = 3.5
    camera.elevation = -18.0
    camera.azimuth   = 135.0

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    print(f"[录制] {args.duration}s  →  {out_path}  ({args.width}×{args.height} @ {RENDER_FPS}fps)")
    print(f"       policy_steps={n_policy_steps}  substeps/step={n_substeps}")

    frames = []
    render_counter = 0

    for step in range(n_policy_steps):
        t = data.time

        obs     = get_obs(data, p2q, p2v, cmd, last_action, t)
        actions = sess.run([oname], {iname: obs.reshape(1,-1)})[0][0].astype(np.float32)
        last_action = actions.copy()
        q_target = DEFAULT_JOINT_POS + ACTION_SCALE * actions

        for _ in range(n_substeps):
            qp = data.qpos[7:][p2q].astype(np.float32)
            dq = data.qvel[6:][p2v].astype(np.float32)
            data.ctrl[p2c] = KP * (q_target - qp) - KD * dq
            mujoco.mj_step(model, data)

        # 每 RENDER_EVERY 个 policy step 渲一帧（控制输出帧率约 30fps）
        render_counter += 1
        if render_counter >= RENDER_EVERY:
            render_counter = 0
            renderer.update_scene(data, camera)
            frames.append(renderer.render().copy())

        # 进度
        if step % 50 == 0:
            z  = data.qpos[2]
            vx = data.qvel[0]
            print(f"  t={t:6.1f}s  h={z:.3f}m  vx={vx:+.3f}  frames={len(frames)}", flush=True)

        # 倒地检测
        if data.qpos[2] < 0.35:
            print(f"  !! 倒地 t={t:.2f}s h={data.qpos[2]:.3f}m，停止录制")
            break

    renderer.close()

    print(f"[保存] 共 {len(frames)} 帧 → {out_path}")
    imageio.mimwrite(out_path, frames, fps=RENDER_FPS, quality=8)
    print(f"[完成] {out_path}")


if __name__ == "__main__":
    main()

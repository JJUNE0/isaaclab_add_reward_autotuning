#!/usr/bin/env python3
"""Headless MuJoCo Sim2Sim evaluation for the 32D TITA CO-RL/MOOPPO policy contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml


POLICY_TO_NATIVE = np.asarray([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int64)
NATIVE_TO_POLICY = np.argsort(POLICY_TO_NATIVE)
DEFAULT_Q = np.asarray([0.1, 0.8, -1.5, 0.0, 0.1, 0.8, -1.5, 0.0], dtype=np.float32)
WHEEL_IDS = (3, 7)


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    rot = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rot.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def build_obs(env, last_action: np.ndarray, command: np.ndarray) -> np.ndarray:
    q = env.joint_pos()
    qd = env.joint_vel()
    q_policy = q[POLICY_TO_NATIVE]
    qd_policy = qd[POLICY_TO_NATIVE]
    obs = np.concatenate(
        [
            q_policy[:4],
            q_policy[4:6],
            qd_policy[:4] * 0.05,
            qd_policy[4:6] * 0.05,
            qd_policy[6:8] * 0.05,
            env.base_ang_vel() * 0.25,
            projected_gravity(env.base_quat_sensor()),
            last_action,
            command * np.asarray([2.0, 1.0, 0.25, 1.0], dtype=np.float32),
        ]
    ).astype(np.float32)
    if obs.shape != (32,):
        raise RuntimeError(f"MOO observation contract mismatch: {obs.shape} != (32,)")
    return obs


def policy_torque(action_policy: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    action_native = action_policy[NATIVE_TO_POLICY]
    torque = np.zeros(8, dtype=np.float32)
    for i in range(8):
        if i in WHEEL_IDS:
            torque[i] = 0.55 * (40.0 * action_native[i] - qd[i])
            torque[i] = np.clip(torque[i], -36.0, 36.0)
        else:
            torque[i] = 70.0 * (action_native[i] - q[i]) - 0.7 * qd[i]
            torque[i] = np.clip(torque[i], -55.0, 55.0)
    return torque


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument(
        "--sim2sim_root",
        default="/home/cocel/Desktop/Research/tita_gluon_wbc/sim2sim/cosim_ros2_jazzy",
    )
    parser.add_argument(
        "--base_config",
        default="/home/cocel/Desktop/Research/tita_gluon_wbc/sim2sim/cosim_ros2_jazzy/config/tita_my42_1800.yaml",
    )
    parser.add_argument("--settle_s", type=float, default=1.0)
    parser.add_argument("--hold_s", type=float, default=3.0)
    parser.add_argument("--ramp_vx", type=float, default=0.8)
    parser.add_argument("--ramp_up_s", type=float, default=1.0)
    parser.add_argument("--cruise_s", type=float, default=2.0)
    parser.add_argument("--decel_window_s", type=float, default=1.5)
    parser.add_argument("--out_dir", default="eval/results")
    parser.add_argument(
        "--reset_yaw_deg",
        type=float,
        default=180.0,
        help="Initial world-frame yaw in degrees. Default 180 turns the robot away from the stairs.",
    )
    parser.add_argument("--viewer", action="store_true", help="Open the passive MuJoCo viewer.")
    parser.add_argument(
        "--real_time",
        action="store_true",
        help="Pace simulation in wall-clock time (automatically enabled with --viewer).",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.sim2sim_root).resolve()))
    from cosim_ros2_jazzy.mujoco_env import TitaMujocoEnv
    import onnxruntime as ort

    onnx_path = Path(args.onnx).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.base_config).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["viewer"]["enabled"] = args.viewer
    cfg["control"]["real_time"] = args.real_time or args.viewer
    # Match TITA_CFG's nominal Isaac reset instead of the hardware handoff pose
    # carried by the generic deployment config.
    cfg["robot"]["reset_joint_angles"] = DEFAULT_Q.astype(float).tolist()
    cfg["robot"]["reset_base_pos"] = [0.0, 0.0, 0.335]
    reset_yaw = np.deg2rad(args.reset_yaw_deg)
    cfg["robot"]["reset_base_quat"] = [
        float(np.cos(0.5 * reset_yaw)),
        0.0,
        0.0,
        float(np.sin(0.5 * reset_yaw)),
    ]

    env = TitaMujocoEnv(cfg, viewer_enabled=args.viewer)
    env.reset()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()
    if len(input_meta) != 1:
        raise RuntimeError(f"Expected one ONNX input, got {[item.name for item in input_meta]}")
    input_name = input_meta[0].name
    policy_dt = env.model.opt.timestep * env.decimation

    def n_steps(seconds: float) -> int:
        return max(1, round(seconds / policy_dt))

    phases = [
        ("settle", n_steps(args.settle_s), lambda i, n: 0.0),
        ("zero_hold", n_steps(args.hold_s), lambda i, n: 0.0),
        ("ramp_up", n_steps(args.ramp_up_s), lambda i, n: args.ramp_vx * (i + 1) / n),
        ("cruise", n_steps(args.cruise_s), lambda i, n: args.ramp_vx),
        ("decel", n_steps(args.decel_window_s), lambda i, n: 0.0),
    ]
    last_action = np.zeros(8, dtype=np.float32)
    rows = []

    for phase, count, command_fn in phases:
        for i in range(count):
            vx_cmd = float(command_fn(i, count))
            command = np.asarray([vx_cmd, 0.0, 0.0, 0.0], dtype=np.float32)
            obs = build_obs(env, last_action, command)
            action = session.run(None, {input_name: obs.reshape(1, 32)})[0].reshape(8).astype(np.float32)

            def callback(q, qd, action_=action):
                return policy_torque(action_, q, qd)

            env.step_with_torque_callback(callback, sleep=args.real_time or args.viewer)
            q = env.joint_pos()
            quat = env.base_quat_sensor()
            gravity = projected_gravity(quat)
            rows.append(
                {
                    "phase": phase,
                    "step": i,
                    "command_vx": vx_cmd,
                    "pos_x": float(env.base_pos()[0]),
                    "pos_y": float(env.base_pos()[1]),
                    "base_z": float(env.base_pos()[2]),
                    "tilt": float(np.linalg.norm(gravity[:2])),
                    "ang_vel_xy": float(np.linalg.norm(env.base_ang_vel()[:2])),
                    "leg_joint_dev_mean": float(
                        np.mean(np.abs(q[[0, 1, 2, 4, 5, 6]] - DEFAULT_Q[[0, 1, 2, 4, 5, 6]]))
                    ),
                    "action_rate_mean": float(np.mean(np.abs(action - last_action))),
                }
            )
            last_action = action

    hold = [row for row in rows if row["phase"] == "zero_hold"]
    cruise = [row for row in rows if row["phase"] == "cruise"]
    decel = [row for row in rows if row["phase"] == "decel"]
    drift_xy = float(
        np.linalg.norm(
            np.asarray([hold[-1]["pos_x"] - hold[0]["pos_x"], hold[-1]["pos_y"] - hold[0]["pos_y"]])
        )
    )
    summary = {
        "onnx": str(onnx_path),
        "onnx_input": input_name,
        "observation_contract": "TITA_MOO_32D",
        "action_contract": "policy=[L1,R1,L2,R2,L3,R3,L4,R4], leg_abs_scale=1, wheel_vel_scale=40",
        "reset_yaw_deg": args.reset_yaw_deg,
        "policy_dt": policy_dt,
        "zero_hold_drift_xy_m": drift_xy,
        "zero_hold_mean_speed_equiv_mps": drift_xy / args.hold_s,
        "cruise_displacement_x_m": cruise[-1]["pos_x"] - cruise[0]["pos_x"],
        "decel_peak_tilt": max(row["tilt"] for row in decel),
        "decel_peak_ang_vel_xy_rps": max(row["ang_vel_xy"] for row in decel),
        "decel_final_tilt": decel[-1]["tilt"],
        "decel_final_ang_vel_xy_rps": decel[-1]["ang_vel_xy"],
        "min_base_z_m": min(row["base_z"] for row in rows),
        "max_leg_joint_dev_mean_rad": max(row["leg_joint_dev_mean"] for row in rows),
        "max_action_rate_mean": max(row["action_rate_mean"] for row in rows),
    }
    csv_path = out_dir / f"{onnx_path.stem}_sim2sim.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = out_dir / f"{onnx_path.stem}_sim2sim.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[RESULT] " + json.dumps(summary, indent=2))
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] JSON: {summary_path}")
    env.close()


if __name__ == "__main__":
    main()

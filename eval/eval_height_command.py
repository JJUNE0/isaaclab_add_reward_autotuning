#!/usr/bin/env python3
"""Evaluate whether a TITA CO-RL/MOOPPO policy tracks a commanded base height
in MuJoCo (target_height experiment). Cycles the ``base_velocity`` command's
pos_z slot through a list of target heights (xy/yaw velocity commands held at
zero throughout, to isolate height tracking from locomotion), and reports
per-phase tracking error plus overall xy drift.

Same cfg-driven design as eval_sim2sim_zero_command.py: obs/action contract
is read from the training run's dumped ``params/env.yaml`` +
``params/agent.yaml`` via ``tita_cfg_runtime.py`` -- no hardcoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tita_cfg_runtime import ActionTorqueAdapter, ObsBuilder, load_run_spec


def _rot_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    rot = _rot_from_quat_wxyz(quat_wxyz)
    return rot.T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def world_to_body(quat_wxyz: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    rot = _rot_from_quat_wxyz(quat_wxyz)
    return rot.T @ np.asarray(v_world, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a model_*.pt checkpoint.")
    parser.add_argument(
        "--onnx",
        default=None,
        help="Path to the exported policy.onnx. Defaults to '<checkpoint_dir>/exported/policy.onnx' "
        "(play.py's default export location).",
    )
    parser.add_argument(
        "--sim2sim_root",
        default="/home/cocel/Desktop/Research/tita_gluon_wbc/sim2sim/cosim_ros2_jazzy",
    )
    parser.add_argument(
        "--base_config",
        default="/home/cocel/Desktop/Research/tita_gluon_wbc/sim2sim/cosim_ros2_jazzy/config/tita_my42_1800.yaml",
    )
    parser.add_argument("--settle_s", type=float, default=1.0)
    parser.add_argument(
        "--heights",
        default="0.385,0.25,0.35,0.30,0.25",
        help="Comma-separated sequence of commanded base heights (m), held in order.",
    )
    parser.add_argument("--phase_s", type=float, default=4.0, help="Duration each height is held.")
    parser.add_argument("--settle_tol_m", type=float, default=0.02, help="|z-tgt| threshold for settle-time metric.")
    parser.add_argument("--reset_yaw_deg", type=float, default=180.0)
    parser.add_argument("--out_dir", default="eval/results/height_command")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--real_time", action="store_true")
    args = parser.parse_args()

    heights = [float(item) for item in args.heights.split(",") if item.strip() != ""]
    if not heights:
        raise ValueError("--heights produced an empty list.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    run_dir = checkpoint_path.parent
    onnx_path = Path(args.onnx).expanduser().resolve() if args.onnx else run_dir / "exported" / "policy.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"No exported ONNX at {onnx_path}. Run play.py once for this checkpoint first "
            "(it auto-exports to <checkpoint_dir>/exported/policy.onnx)."
        )

    sys.path.insert(0, str(Path(args.sim2sim_root).resolve()))
    from cosim_ros2_jazzy.mujoco_env import TitaMujocoEnv
    import onnxruntime as ort

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = load_run_spec(run_dir)
    print(f"[INFO] run spec: num_policy_stacks={spec.num_policy_stacks}, "
          f"{len(spec.obs_stack_terms)} stacked obs terms, {len(spec.action_terms)} action terms")

    with Path(args.base_config).expanduser().open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["viewer"]["enabled"] = args.viewer
    cfg["control"]["real_time"] = args.real_time or args.viewer

    env = TitaMujocoEnv(cfg, viewer_enabled=args.viewer)
    joint_names = env.joint_names

    obs_builder = ObsBuilder(spec, joint_names)
    action_adapter = ActionTorqueAdapter(spec, joint_names)
    default_joint_pos = np.asarray([spec.default_joint_pos[n] for n in joint_names], dtype=np.float32)

    cfg["robot"]["reset_joint_angles"] = default_joint_pos.astype(float).tolist()
    cfg["robot"]["reset_base_pos"] = [0.0, 0.0, heights[0]]
    reset_yaw = np.deg2rad(args.reset_yaw_deg)
    cfg["robot"]["reset_base_quat"] = [
        float(np.cos(0.5 * reset_yaw)),
        0.0,
        0.0,
        float(np.sin(0.5 * reset_yaw)),
    ]
    env.reset_to_joint_angles(default_joint_pos.astype(float))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one ONNX input, got {[item.name for item in inputs]}")
    input_name = inputs[0].name
    onnx_obs_dim = inputs[0].shape[-1]

    policy_dt = env.model.opt.timestep * env.decimation
    settle_steps = max(1, round(args.settle_s / policy_dt))
    phase_steps = max(1, round(args.phase_s / policy_dt))
    command = np.zeros(4, dtype=np.float32)
    command[3] = heights[0]
    last_action = np.zeros(action_adapter.action_dim, dtype=np.float32)

    def policy_step(first: bool) -> np.ndarray:
        nonlocal last_action
        q = env.joint_pos()
        qd = env.joint_vel()
        quat = env.base_quat_sensor()
        gravity = projected_gravity(quat)
        ang_vel = env.base_ang_vel()
        lin_vel_b = world_to_body(quat, env.base_lin_vel())
        base_z = env.base_pos()[2]
        builder_fn = obs_builder.reset if first else obs_builder.step
        obs = builder_fn(q, qd, ang_vel, gravity, last_action, command, base_lin_vel=lin_vel_b, base_z=base_z)
        if obs.shape[0] != onnx_obs_dim:
            raise RuntimeError(
                f"cfg-derived obs dim {obs.shape[0]} != ONNX input dim {onnx_obs_dim} "
                f"-- checkpoint/params mismatch (wrong run_dir for this checkpoint?)."
            )
        action = session.run(None, {input_name: obs.reshape(1, -1)})[0].reshape(-1).astype(np.float32)

        def callback(q_, qd_, action_=action):
            return action_adapter.torque(action_, q_, qd_)

        env.step_with_torque_callback(callback, sleep=args.real_time or args.viewer)
        last_action = action
        return action

    policy_step(first=True)
    for _ in range(settle_steps - 1):
        policy_step(first=False)

    start_pos = env.base_pos().copy()
    rows = []
    phase_summaries = []
    t = 0.0
    for phase_idx, target_h in enumerate(heights):
        command[3] = target_h
        phase_rows = []
        settle_time_s = None
        for step in range(phase_steps):
            action = policy_step(first=False)
            pos = env.base_pos()
            q = env.joint_pos()
            gravity = projected_gravity(env.base_quat_sensor())
            t += policy_dt
            err = float(pos[2] - target_h)
            row = {
                "time_s": t,
                "phase": phase_idx,
                "target_h_m": target_h,
                "base_z": float(pos[2]),
                "height_err_m": err,
                "pos_x": float(pos[0]),
                "pos_y": float(pos[1]),
                "drift_xy_m": float(np.linalg.norm(pos[:2] - start_pos[:2])),
                "tilt": float(np.linalg.norm(gravity[:2])),
                "action_abs_mean": float(np.mean(np.abs(action))),
                "hip_L": float(q[0]),
                "shoulder_L": float(q[1]),
                "leg_L": float(q[2]),
                "hip_R": float(q[4]),
                "shoulder_R": float(q[5]),
                "leg_R": float(q[6]),
            }
            rows.append(row)
            phase_rows.append(row)
            if settle_time_s is None and abs(err) <= args.settle_tol_m:
                settle_time_s = (step + 1) * policy_dt
        abs_errs = [abs(r["height_err_m"]) for r in phase_rows]
        final_err = phase_rows[-1]["height_err_m"]
        tail_rows = phase_rows[-20:]  # settled-state average (last 20 steps = 0.4s at 50Hz)
        phase_summary = {
            "phase": phase_idx,
            "target_h_m": target_h,
            "mean_abs_err_m": sum(abs_errs) / len(abs_errs),
            "max_abs_err_m": max(abs_errs),
            "final_err_m": final_err,
            "settle_time_s": settle_time_s,
            "max_tilt": max(r["tilt"] for r in phase_rows),
            "settled_shoulder_L": sum(r["shoulder_L"] for r in tail_rows) / len(tail_rows),
            "settled_leg_L": sum(r["leg_L"] for r in tail_rows) / len(tail_rows),
            "settled_shoulder_R": sum(r["shoulder_R"] for r in tail_rows) / len(tail_rows),
            "settled_leg_R": sum(r["leg_R"] for r in tail_rows) / len(tail_rows),
        }
        phase_summaries.append(phase_summary)
        print(
            f"[HEIGHT] phase={phase_idx} tgt={target_h:.3f}m "
            f"mean_abs_err={phase_summary['mean_abs_err_m']:.4f}m "
            f"final_err={final_err:+.4f}m "
            f"settle={settle_time_s if settle_time_s is not None else 'never'} "
            f"max_tilt={phase_summary['max_tilt']:.4f}"
        )

    overall_abs_errs = [abs(r["height_err_m"]) for r in rows]
    summary = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "onnx_input": input_name,
        "num_policy_stacks": spec.num_policy_stacks,
        "obs_dim": int(onnx_obs_dim),
        "reset_yaw_deg": args.reset_yaw_deg,
        "settle_s": args.settle_s,
        "phase_s": args.phase_s,
        "settle_tol_m": args.settle_tol_m,
        "heights": heights,
        "policy_dt": policy_dt,
        "start_xyz_m": start_pos.astype(float).tolist(),
        "final_xyz_m": env.base_pos().astype(float).tolist(),
        "final_drift_xy_m": rows[-1]["drift_xy_m"],
        "max_drift_xy_m": max(r["drift_xy_m"] for r in rows),
        "overall_mean_abs_height_err_m": sum(overall_abs_errs) / len(overall_abs_errs),
        "overall_max_abs_height_err_m": max(overall_abs_errs),
        "max_tilt": max(r["tilt"] for r in rows),
        "phases": phase_summaries,
    }

    stem = f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_height_command_{len(heights)}phases"
    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[RESULT] " + json.dumps(summary, indent=2))
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] JSON: {json_path}")
    env.close()


if __name__ == "__main__":
    main()

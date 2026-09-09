#!/usr/bin/env python3
"""Measure zero-command drift of a TITA CO-RL/MOOPPO policy in MuJoCo.

Unlike the old version, the observation/action contract is not hardcoded --
it is read from the training run's dumped ``params/env.yaml`` +
``params/agent.yaml`` via ``tita_cfg_runtime.py``, so this script works for
any obs/action cfg (different scales, term sets, num_policy_stacks, ...)
without code changes; only a genuinely new obs *function* needs a new entry
in ``tita_cfg_runtime.FUNC_IMPLS``.
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
    """Rotate a world-frame vector into the body frame (MuJoCo free-joint
    qvel[0:3] is world-frame linear velocity; Isaac's *_link obs functions
    are body-frame, e.g. root_link_lin_vel_b)."""
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
    parser.add_argument("--duration_s", type=float, default=30.0)
    parser.add_argument("--reset_yaw_deg", type=float, default=180.0)
    parser.add_argument("--checkpoint_every_s", type=float, default=5.0)
    parser.add_argument(
        "--height_m",
        type=float,
        default=0.385,
        help="Commanded base height (base_velocity command's pos_z slot). Must be inside the "
        "checkpoint's trained range -- e.g. height-conditioned checkpoints trained with a "
        "randomized target (see TitaFlatEnvCfg_GTBaseVel_TargetHeight) will behave wildly "
        "out-of-distribution if this is left at 0.0.",
    )
    parser.add_argument("--out_dir", default="eval/results/zero_command")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--real_time", action="store_true")
    args = parser.parse_args()

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
    cfg["robot"]["reset_base_pos"] = [0.0, 0.0, args.height_m]
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
    measure_steps = max(1, round(args.duration_s / policy_dt))
    checkpoint_steps = max(1, round(args.checkpoint_every_s / policy_dt))
    command = np.zeros(4, dtype=np.float32)
    command[3] = args.height_m
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
    for step in range(measure_steps):
        action = policy_step(first=False)
        pos = env.base_pos()
        displacement = pos - start_pos
        gravity = projected_gravity(env.base_quat_sensor())
        rows.append(
            {
                "time_s": (step + 1) * policy_dt,
                "pos_x": float(pos[0]),
                "pos_y": float(pos[1]),
                "base_z": float(pos[2]),
                "drift_x_m": float(displacement[0]),
                "drift_y_m": float(displacement[1]),
                "drift_xy_m": float(np.linalg.norm(displacement[:2])),
                "tilt": float(np.linalg.norm(gravity[:2])),
                "ang_vel_xy_rps": float(np.linalg.norm(env.base_ang_vel()[:2])),
                "action_abs_mean": float(np.mean(np.abs(action))),
            }
        )
        if (step + 1) % checkpoint_steps == 0:
            row = rows[-1]
            print(
                f"[DRIFT] t={row['time_s']:.1f}s xy={row['drift_xy_m']:.4f}m "
                f"dx={row['drift_x_m']:+.4f}m dy={row['drift_y_m']:+.4f}m "
                f"z={row['base_z']:.4f}m"
            )

    final = rows[-1]
    summary = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "onnx_input": input_name,
        "num_policy_stacks": spec.num_policy_stacks,
        "obs_dim": int(onnx_obs_dim),
        "reset_yaw_deg": args.reset_yaw_deg,
        "settle_s": args.settle_s,
        "duration_s": args.duration_s,
        "policy_dt": policy_dt,
        "start_xyz_m": start_pos.astype(float).tolist(),
        "final_xyz_m": env.base_pos().astype(float).tolist(),
        "final_drift_x_m": final["drift_x_m"],
        "final_drift_y_m": final["drift_y_m"],
        "final_drift_xy_m": final["drift_xy_m"],
        "mean_speed_equiv_mps": final["drift_xy_m"] / args.duration_s,
        "max_drift_xy_m": max(row["drift_xy_m"] for row in rows),
        "min_base_z_m": min(row["base_z"] for row in rows),
        "max_tilt": max(row["tilt"] for row in rows),
        "max_ang_vel_xy_rps": max(row["ang_vel_xy_rps"] for row in rows),
        "mean_action_abs": sum(row["action_abs_mean"] for row in rows) / len(rows),
    }

    stem = f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_zero_command_{args.duration_s:g}s"
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

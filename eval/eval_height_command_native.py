#!/usr/bin/env python3
"""Native-IsaacLab height-command tracking + xy/yaw drift check.

Same idea as eval_height_command.py (cycle the base_velocity command's height
through a list of targets, lin/ang velocity held at zero, measure per-phase
tracking error and drift) but runs directly inside IsaacLab instead of
exporting to ONNX + MuJoCo. This is required for policies whose actor needs
privileged sim-only info (e.g. RMATeacher: actor([proprio, z]),
z=Encoder(privileged_info)) -- the ONNX export only captures the final actor
MLP (see scripts/co_rl/core/wrapper/exporter.py's _OnnxPolicyExporter), not
the encoder, so those policies can't be evaluated via the MuJoCo bridge.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-Tita-RMATeacher-TargetHeight-v3-moo-ppo")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--heights", default="0.35,0.25,0.45,0.30,0.25")
parser.add_argument("--phase_s", type=float, default=4.0)
parser.add_argument("--settle_s", type=float, default=1.0)
parser.add_argument("--settle_tol_m", type=float, default=0.02)
parser.add_argument("--out_dir", default="eval/results/height_command_native")
parser.add_argument("--num_envs", type=int, default=1, help="All envs get the same forced command; only env 0 is logged.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import lab.flamingo.tasks  # noqa: F401
from isaaclab.utils.math import euler_xyz_from_quat
from scripts.co_rl.core.runners import MOO_OnPolicyRunner
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper


def _get_privileged_obs(obs_dict, device):
    """Concatenate all 'priv_*' groups from obs_dict, sorted by key (matches
    scripts/co_rl/play.py's helper of the same name -- duplicated here rather
    than imported, since play.py's bare `import cli_args` only resolves when
    it's run as the top-level script from scripts/co_rl/, not when imported
    as a module from elsewhere)."""
    if obs_dict is None:
        return None
    priv_keys = sorted([k for k in obs_dict.keys() if k.startswith("priv_")])
    if len(priv_keys) > 0:
        return torch.cat([obs_dict[k] for k in priv_keys], dim=-1).to(device)
    elif "privileged" in obs_dict:
        return obs_dict["privileged"].to(device)
    return None


def _resolve(entry_point):
    if not isinstance(entry_point, str):
        return entry_point() if callable(entry_point) else entry_point
    module_name, attr = entry_point.split(":")
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if callable(obj) else obj


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    heights = [float(h) for h in args_cli.heights.split(",") if h.strip() != ""]

    spec = gym.spec(args_cli.task)
    env_cfg = _resolve(spec.kwargs["env_cfg_entry_point"])
    agent_cfg = _resolve(spec.kwargs["co_rl_cfg_entry_point"])
    env_cfg.seed = 42
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    # Deterministic: no obs noise, no random pushes/reset noise -- isolate the
    # policy's own height-tracking response from randomization.
    env_cfg.observations.stack_policy.enable_corruption = False
    env_cfg.observations.none_stack_policy.enable_corruption = False
    env_cfg.events.physics_material = None
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "reset_robot_joints") and env_cfg.events.reset_robot_joints is not None:
        env_cfg.events.reset_robot_joints.params["position_noise_range"] = (0.0, 0.0)
        env_cfg.events.reset_robot_joints.params["interpolation_range"] = (0.0, 0.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        },
    }
    env_cfg.domain.init_difficulty_level = 0.0

    print("[DEBUG] before gym.make", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[DEBUG] after gym.make, before CoRlVecEnvWrapper", flush=True)
    env = CoRlVecEnvWrapper(env, agent_cfg)
    print("[DEBUG] after CoRlVecEnvWrapper, before MOO_OnPolicyRunner", flush=True)
    runner = MOO_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print("[DEBUG] after MOO_OnPolicyRunner, before runner.load", flush=True)
    runner.load(str(checkpoint))
    print("[DEBUG] after runner.load, before get_inference_policy", flush=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[DEBUG] after get_inference_policy, before env.reset()", flush=True)

    robot = env.unwrapped.scene["robot"]
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    step_dt = float(env.unwrapped.step_dt)
    phase_steps = max(1, round(args_cli.phase_s / step_dt))
    settle_steps = max(1, round(args_cli.settle_s / step_dt))

    obs, extras = env.reset()
    print("[DEBUG] after env.reset(), before privileged obs extraction", flush=True)
    priv = _get_privileged_obs(extras.get("observations"), env.unwrapped.device)
    print("[DEBUG] after privileged obs extraction, entering loop", flush=True)

    def policy_step():
        nonlocal obs, extras, priv
        if priv is not None:
            actions = policy(obs, privileged_info=priv)
        else:
            actions = policy(obs)
        obs, _, _, extras = env.step(actions)
        priv = _get_privileged_obs(extras.get("observations"), env.unwrapped.device)

    # Force the height command; keep lin/ang velocity at zero throughout so
    # drift measures stand-still tracking, not locomotion drift.
    def set_height(h: float):
        cmd_term.vel_command_b[:, 0:3] = 0.0
        cmd_term.vel_command_b[:, 3] = h

    set_height(heights[0])
    for _ in range(settle_steps):
        policy_step()

    start_pos = robot.data.root_link_pos_w[0, :2].clone()
    rows = []
    phase_summaries = []
    t = 0.0
    for phase_idx, target_h in enumerate(heights):
        set_height(target_h)
        phase_rows = []
        settle_time_s = None
        for step in range(phase_steps):
            policy_step()
            pos = robot.data.root_link_pos_w[0]
            quat = robot.data.root_link_quat_w[0]
            _, _, yaw = euler_xyz_from_quat(quat.unsqueeze(0))
            t += step_dt
            err = float(pos[2].item() - target_h)
            row = {
                "time_s": t,
                "phase": phase_idx,
                "target_h_m": target_h,
                "base_z": float(pos[2].item()),
                "height_err_m": err,
                "pos_x": float(pos[0].item()),
                "pos_y": float(pos[1].item()),
                "yaw_rad": float(yaw.item()),
                "drift_xy_m": float(torch.norm(pos[:2] - start_pos).item()),
            }
            rows.append(row)
            phase_rows.append(row)
            if settle_time_s is None and abs(err) <= args_cli.settle_tol_m:
                settle_time_s = (step + 1) * step_dt
        abs_errs = [abs(r["height_err_m"]) for r in phase_rows]
        final_err = phase_rows[-1]["height_err_m"]
        phase_summary = {
            "phase": phase_idx,
            "target_h_m": target_h,
            "mean_abs_err_m": sum(abs_errs) / len(abs_errs),
            "max_abs_err_m": max(abs_errs),
            "final_err_m": final_err,
            "settle_time_s": settle_time_s,
            "final_yaw_rad": phase_rows[-1]["yaw_rad"],
            "final_drift_xy_m": phase_rows[-1]["drift_xy_m"],
        }
        phase_summaries.append(phase_summary)
        print(
            f"[HEIGHT] phase={phase_idx} tgt={target_h:.3f}m "
            f"mean_abs_err={phase_summary['mean_abs_err_m']:.4f}m "
            f"final_err={final_err:+.4f}m "
            f"settle={settle_time_s if settle_time_s is not None else 'never'} "
            f"drift_xy={phase_summary['final_drift_xy_m']:.4f}m "
            f"yaw={phase_summary['final_yaw_rad']:+.4f}rad"
        )

    overall_abs_errs = [abs(r["height_err_m"]) for r in rows]
    summary = {
        "checkpoint": str(checkpoint),
        "task": args_cli.task,
        "heights": heights,
        "phase_s": args_cli.phase_s,
        "settle_tol_m": args_cli.settle_tol_m,
        "start_xy_m": start_pos.tolist(),
        "final_xy_m": robot.data.root_link_pos_w[0, :2].tolist(),
        "final_yaw_rad": rows[-1]["yaw_rad"],
        "final_drift_xy_m": rows[-1]["drift_xy_m"],
        "max_drift_xy_m": max(r["drift_xy_m"] for r in rows),
        "overall_mean_abs_height_err_m": sum(overall_abs_errs) / len(overall_abs_errs),
        "overall_max_abs_height_err_m": max(overall_abs_errs),
        "phases": phase_summaries,
    }

    out_dir = Path(args_cli.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{checkpoint.parent.name}_{checkpoint.stem}_height_command_native_{len(heights)}phases"
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
    try:
        main()
    finally:
        simulation_app.close()

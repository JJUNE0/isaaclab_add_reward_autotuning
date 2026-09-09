#!/usr/bin/env python3
"""Deterministic Isaac Lab behavior evaluation for TITA CO-RL/MOOPPO policies."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a TITA CO-RL/MOOPPO checkpoint in Isaac Lab.")
parser.add_argument("--task", default="Isaac-Velocity-Flat-Tita-Play-v3-moo-ppo")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--settle_s", type=float, default=1.0)
parser.add_argument("--hold_s", type=float, default=3.0)
parser.add_argument("--ramp_vx", type=float, default=0.8)
parser.add_argument("--ramp_up_s", type=float, default=1.0)
parser.add_argument("--cruise_s", type=float, default=2.0)
parser.add_argument("--decel_window_s", type=float, default=1.5)
parser.add_argument("--out_dir", default="eval/results")
parser.add_argument("--keep_randomization", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import lab.flamingo.tasks  # noqa: F401
from scripts.co_rl.core.runners import MOO_OnPolicyRunner
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper, export_policy_as_onnx


POLICY_ACTION_NAMES = [
    "joint_left_leg_1",
    "joint_right_leg_1",
    "joint_left_leg_2",
    "joint_right_leg_2",
    "joint_left_leg_3",
    "joint_right_leg_3",
    "joint_left_leg_4",
    "joint_right_leg_4",
]
PHYSICAL_JOINT_NAMES = [
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_left_leg_4",
    "joint_right_leg_1",
    "joint_right_leg_2",
    "joint_right_leg_3",
    "joint_right_leg_4",
]


def _resolve(entry_point):
    if not isinstance(entry_point, str):
        return entry_point() if callable(entry_point) else entry_point
    module_name, attr = entry_point.split(":")
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if callable(obj) else obj


def _disable_randomization(env_cfg) -> None:
    """Make checkpoint comparisons deterministic and nominal."""
    env_cfg.domain = None
    env_cfg.observations.stack_policy.enable_corruption = False
    env_cfg.observations.none_stack_policy.enable_corruption = False
    env_cfg.events.physics_material = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    out_dir = Path(args_cli.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = checkpoint.stem

    spec = gym.spec(args_cli.task)
    env_cfg = _resolve(spec.kwargs["env_cfg_entry_point"])
    agent_cfg = _resolve(spec.kwargs["co_rl_cfg_entry_point"])
    env_cfg.seed = 42
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    if not args_cli.keep_randomization:
        _disable_randomization(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = CoRlVecEnvWrapper(env, agent_cfg)
    runner = MOO_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    onnx_path = out_dir / f"{tag}.onnx"
    export_policy_as_onnx(
        runner.alg.actor_critic,
        normalizer=runner.obs_normalizer,
        path=str(out_dir),
        filename=onnx_path.name,
    )

    robot = env.unwrapped.scene["robot"]
    joint_ids = [robot.find_joints(name)[0][0] for name in PHYSICAL_JOINT_NAMES]
    default_joint_pos = robot.data.default_joint_pos[:, joint_ids].clone()
    cmd = env.unwrapped.command_manager.get_command("base_velocity")
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_term.is_heading_env[:] = False
    cmd_term.is_standing_env[:] = False
    cmd_term.time_left[:] = 1.0e6
    step_dt = float(env.unwrapped.step_dt)

    def n_steps(seconds: float) -> int:
        return max(1, round(seconds / step_dt))

    phases = [
        ("settle", n_steps(args_cli.settle_s), lambda i, n: 0.0),
        ("zero_hold", n_steps(args_cli.hold_s), lambda i, n: 0.0),
        ("ramp_up", n_steps(args_cli.ramp_up_s), lambda i, n: args_cli.ramp_vx * (i + 1) / n),
        ("cruise", n_steps(args_cli.cruise_s), lambda i, n: args_cli.ramp_vx),
        ("decel", n_steps(args_cli.decel_window_s), lambda i, n: 0.0),
    ]

    obs, _ = env.get_observations()
    rows = []
    prev_action = None
    start_x = None

    with torch.inference_mode():
        for phase, count, command_fn in phases:
            for i in range(count):
                vx_cmd = float(command_fn(i, count))
                cmd[:, :] = 0.0
                cmd[:, 0] = vx_cmd
                if hasattr(cmd_term, "vel_command_b"):
                    cmd_term.vel_command_b[:, :] = 0.0
                    cmd_term.vel_command_b[:, 0] = vx_cmd

                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                action_rate = (
                    torch.zeros_like(actions) if prev_action is None else (actions - prev_action).abs()
                )
                prev_action = actions.clone()
                if phase == "zero_hold" and start_x is None:
                    start_x = robot.data.root_pos_w[:, 0].clone()

                joint_dev = (robot.data.joint_pos[:, joint_ids] - default_joint_pos).abs().mean(dim=0)
                rows.append(
                    {
                        "phase": phase,
                        "step": i,
                        "command_vx": vx_cmd,
                        "vx_b": robot.data.root_lin_vel_b[:, 0].mean().item(),
                        "pos_x": robot.data.root_pos_w[:, 0].mean().item(),
                        "base_z": robot.data.root_pos_w[:, 2].mean().item(),
                        "tilt": torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=1).mean().item(),
                        "ang_vel_xy": torch.linalg.norm(robot.data.root_ang_vel_b[:, :2], dim=1).mean().item(),
                        "joint_dev": joint_dev.tolist(),
                        "action_rate": action_rate.mean(dim=0).tolist(),
                    }
                )

    hold = [row for row in rows if row["phase"] == "zero_hold"]
    cruise = [row for row in rows if row["phase"] == "cruise"]
    decel = [row for row in rows if row["phase"] == "decel"]
    drift = hold[-1]["pos_x"] - hold[0]["pos_x"]
    summary = {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "randomization": "kept" if args_cli.keep_randomization else "disabled",
        "num_envs": args_cli.num_envs,
        "step_dt": step_dt,
        "zero_hold_drift_x_m": drift,
        "zero_hold_mean_vx_mps": sum(row["vx_b"] for row in hold) / len(hold),
        "decel_peak_tilt": max(row["tilt"] for row in decel),
        "decel_peak_ang_vel_xy_rps": max(row["ang_vel_xy"] for row in decel),
        "decel_final_tilt": decel[-1]["tilt"],
        "decel_final_ang_vel_xy_rps": decel[-1]["ang_vel_xy"],
        "cruise_mean_vx_mps": sum(row["vx_b"] for row in cruise) / len(cruise),
        "min_base_z_m": min(row["base_z"] for row in rows),
        "max_leg_joint_dev_cruise_rad": {
            name: max(row["joint_dev"][i] for row in cruise)
            for i, name in enumerate(PHYSICAL_JOINT_NAMES)
            if not name.endswith("_leg_4")
        },
        "max_action_rate_decel": {
            name: max(row["action_rate"][i] for row in decel) for i, name in enumerate(POLICY_ACTION_NAMES)
        },
    }

    csv_path = out_dir / f"{tag}_isaac_behavior.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["phase", "step", "command_vx", "vx_b", "pos_x", "base_z", "tilt", "ang_vel_xy"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / f"{tag}_isaac_behavior.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[RESULT] " + json.dumps(summary, indent=2))
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] JSON: {summary_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

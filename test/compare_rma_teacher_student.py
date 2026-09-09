#!/usr/bin/env python3
"""Compare RMA Teacher/Student closed-loop drift in native Isaac Lab.

This is the RMA-aware equivalent of ``eval/eval_behavior.py``. Run it once
for each role with identical arguments. It handles the Teacher privileged
observation input and the Student 50-frame history buffer correctly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


DEFAULT_TASKS = {
    "teacher": "Isaac-Velocity-Flat-Tita-RMATeacher-TargetHeight-v3-moo-ppo",
    "student": "Isaac-Velocity-Flat-Tita-RMAStudent-TargetHeight-v3-moo-ppo",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--role", choices=sorted(DEFAULT_TASKS), required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument(
    "--settle_s",
    type=float,
    default=1.0,
    help="Warm-up before measurement, matching eval/eval_behavior.py.",
)
parser.add_argument("--hold_s", type=float, default=3.0)
parser.add_argument("--ramp_vx", type=float, default=0.8)
parser.add_argument("--ramp_up_s", type=float, default=1.0)
parser.add_argument("--cruise_s", type=float, default=2.0)
parser.add_argument("--decel_window_s", type=float, default=1.5)
parser.add_argument("--height_m", type=float, default=0.35)
parser.add_argument(
    "--quick_zero_hold",
    action="store_true",
    help="Measure only settle + zero-command drift on slow validation hosts.",
)
parser.add_argument("--out_dir", default="docs/exp/2026-08-19/rma_teacher_student_compare")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml

import lab.flamingo.tasks  # noqa: F401
from scripts.co_rl.core.modules.teacher_student import RMAStudent, RMATeacher
from scripts.co_rl.core.runners import MOO_OnPolicyRunner
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper


def _resolve(entry_point):
    if not isinstance(entry_point, str):
        return entry_point() if callable(entry_point) else entry_point
    module_name, attr = entry_point.split(":")
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if callable(obj) else obj


def _get_privileged_obs(obs_dict, device):
    if obs_dict is None:
        return None
    priv_keys = sorted(key for key in obs_dict if key.startswith("priv_"))
    if priv_keys:
        return torch.cat([obs_dict[key] for key in priv_keys], dim=-1).to(device)
    if "privileged" in obs_dict:
        return obs_dict["privileged"].to(device)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def _rmse(rows: list[dict], key: str) -> float:
    return math.sqrt(sum(row[key] ** 2 for row in rows) / len(rows))


def _disable_randomization(env_cfg) -> None:
    """Use the same deterministic nominal environment for both policies."""
    env_cfg.domain.init_difficulty_level = 0.0
    env_cfg.observations.stack_policy.enable_corruption = False
    env_cfg.observations.none_stack_policy.enable_corruption = False
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.debug_vis = False

    for event_name in ("physics_material", "push_robot", "add_base_inertia", "add_base_com"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    reset_joints = getattr(env_cfg.events, "reset_robot_joints", None)
    if reset_joints is not None:
        reset_joints.params["position_noise_range"] = (0.0, 0.0)
        reset_joints.params["interpolation_range"] = (0.0, 0.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)

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
    role = args_cli.role
    task = args_cli.task or DEFAULT_TASKS[role]
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    spec = gym.spec(task)
    env_cfg = _resolve(spec.kwargs["env_cfg_entry_point"])
    agent_cfg = _resolve(spec.kwargs["co_rl_cfg_entry_point"])
    saved_agent_cfg_path = checkpoint.parent / "params" / "agent.yaml"
    if not saved_agent_cfg_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint agent config: {saved_agent_cfg_path}")
    saved_agent_cfg = yaml.safe_load(saved_agent_cfg_path.read_text(encoding="utf-8"))
    agent_cfg.num_policy_stacks = int(saved_agent_cfg["num_policy_stacks"])
    agent_cfg.num_critic_stacks = int(saved_agent_cfg["num_critic_stacks"])
    saved_policy_cfg = saved_agent_cfg.get("policy", {})
    if hasattr(agent_cfg.policy, "teacher_checkpoint_path"):
        teacher_checkpoint_path = saved_policy_cfg.get("teacher_checkpoint_path")
        if teacher_checkpoint_path:
            agent_cfg.policy.teacher_checkpoint_path = teacher_checkpoint_path
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    # Headless evaluation needs no render between the four physics substeps.
    # Aligning the render interval with decimation preserves control/physics
    # timing while avoiding four render-path calls per policy step.
    env_cfg.sim.render_interval = env_cfg.decimation
    _disable_randomization(env_cfg)

    print("[COMPARE] creating environment", flush=True)
    env = gym.make(task, cfg=env_cfg)
    print("[COMPARE] wrapping environment", flush=True)
    env = CoRlVecEnvWrapper(env, agent_cfg)
    print("[COMPARE] creating runner", flush=True)
    runner = MOO_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[COMPARE] loading checkpoint: {checkpoint}", flush=True)
    runner.load(str(checkpoint))
    # RMATeacher's encoder uses torch spectral_norm.  Evaluation mode keeps
    # its power-iteration buffers fixed so the policy itself does not change
    # while behavior is being measured.
    runner.eval_mode()
    print("[COMPARE] checkpoint loaded", flush=True)
    model = runner.alg.actor_critic
    expected_type = RMATeacher if role == "teacher" else RMAStudent
    if not isinstance(model, expected_type):
        raise TypeError(f"Expected {expected_type.__name__}, got {type(model).__name__}")

    robot = env.unwrapped.scene["robot"]
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    command_term.is_heading_env[:] = False
    command_term.is_standing_env[:] = False
    command_term.time_left[:] = 1.0e6
    step_dt = float(env.unwrapped.step_dt)

    def n_steps(seconds: float) -> int:
        return max(1, round(seconds / step_dt))

    phases = [
        ("settle", n_steps(args_cli.settle_s), lambda i, n: 0.0),
        ("zero_hold", n_steps(args_cli.hold_s), lambda i, n: 0.0),
    ]
    if not args_cli.quick_zero_hold:
        phases.extend(
            [
                ("ramp_up", n_steps(args_cli.ramp_up_s), lambda i, n: args_cli.ramp_vx * (i + 1) / n),
                ("cruise", n_steps(args_cli.cruise_s), lambda i, n: args_cli.ramp_vx),
                ("decel", n_steps(args_cli.decel_window_s), lambda i, n: 0.0),
            ]
        )

    print("[COMPARE] resetting environment", flush=True)
    obs, extras = env.reset()
    print("[COMPARE] reset complete", flush=True)
    privileged_obs = _get_privileged_obs(extras.get("observations"), env.unwrapped.device)
    dones = torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    # Compare steady policy behavior, not the task's five-second startup mask.
    # Both roles start from the same deterministic reset and receive one full
    # second of settling/history warm-up before drift measurement.
    if hasattr(command_term, "time_elapsed"):
        command_term.time_elapsed[:] = command_term.cfg.initial_phase_time + step_dt
    if isinstance(model, RMAStudent):
        model.update_history(obs, dones=dones)

    rows: list[dict] = []
    previous_actions = None
    elapsed_s = 0.0
    run_start_xy = robot.data.root_link_pos_w[:, :2].clone()
    phase_start_xy = run_start_xy.clone()

    with torch.inference_mode():
        for phase, count, command_fn in phases:
            print(f"[COMPARE] phase={phase} steps={count}", flush=True)
            phase_start_xy = robot.data.root_link_pos_w[:, :2].clone()
            for phase_step in range(count):
                vx_command = float(command_fn(phase_step, count))
                command_term.vel_command_b[:, :] = 0.0
                command_term.vel_command_b[:, 0] = vx_command
                command_term.vel_command_b[:, 3] = args_cli.height_m

                if isinstance(model, RMAStudent):
                    model.update_history(obs, dones=dones)
                    z_hat, velocity_hat, height_hat = model.adaptation_module(model.history_buffer)
                    teacher_obs = obs[:, : model.teacher_num_obs]
                    teacher_frames = teacher_obs[:, : model.teacher_single_obs_dim * model.num_actor_frames].view(
                        -1, model.num_actor_frames, model.teacher_single_obs_dim
                    )
                    velocity_target = teacher_frames[:, :, -model.state_dim : -1]
                    height_target = teacher_frames[:, :, -1:]
                    teacher_z = model.teacher_encoder(privileged_obs)
                    teacher_actions = model.actor(torch.cat((teacher_obs, teacher_z), dim=-1))
                    actions = model.act_inference(obs)
                    velocity_scale = torch.tensor(
                        [2.0, 1.0, 0.25], device=velocity_hat.device
                    ).view(1, 1, 3)
                    student_diagnostics = {
                        "student_z_mae": float(torch.mean(torch.abs(z_hat - teacher_z)).item()),
                        "student_velocity_mae_mps": float(
                            torch.mean(torch.abs(velocity_hat - velocity_target) / velocity_scale).item()
                        ),
                        "student_height_mae_m": float(
                            torch.mean(torch.abs(height_hat - height_target)).item()
                        ),
                        "student_action_mae": float(
                            torch.mean(torch.abs(actions - teacher_actions)).item()
                        ),
                    }
                else:
                    actions = model.act_inference(obs, privileged_info=privileged_obs)
                    student_diagnostics = None

                obs, _, dones, extras = env.step(actions)
                privileged_obs = _get_privileged_obs(extras.get("observations"), env.unwrapped.device)
                elapsed_s += step_dt

                position_xy = robot.data.root_link_pos_w[:, :2]
                phase_delta_xy = position_xy - phase_start_xy
                run_delta_xy = position_xy - run_start_xy
                action_rate = (
                    torch.zeros((), device=actions.device)
                    if previous_actions is None
                    else torch.mean(torch.abs(actions - previous_actions))
                )
                previous_actions = actions.clone()
                actual_vx = robot.data.root_link_lin_vel_b[:, 0]

                row = {
                        "time_s": elapsed_s,
                        "phase": phase,
                        "phase_step": phase_step,
                        "command_vx": vx_command,
                        "actual_vx": float(actual_vx.mean().item()),
                        "vx_error": float(torch.abs(actual_vx - vx_command).mean().item()),
                        "phase_drift_x_m": float(phase_delta_xy[:, 0].mean().item()),
                        "phase_drift_y_m": float(phase_delta_xy[:, 1].mean().item()),
                        "phase_drift_xy_m": float(torch.linalg.vector_norm(phase_delta_xy, dim=-1).mean().item()),
                        "run_displacement_x_m": float(run_delta_xy[:, 0].mean().item()),
                        "run_displacement_y_m": float(run_delta_xy[:, 1].mean().item()),
                        "base_height_m": float(robot.data.root_link_pos_w[:, 2].mean().item()),
                        "tilt": float(
                            torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=-1).mean().item()
                        ),
                        "ang_vel_xy_rps": float(
                            torch.linalg.vector_norm(robot.data.root_link_ang_vel_b[:, :2], dim=-1).mean().item()
                        ),
                        "action_rate_mean": float(action_rate.item()),
                        "done_count": int(dones.sum().item()),
                    }
                if student_diagnostics is not None:
                    row.update(student_diagnostics)
                rows.append(row)
            print(f"[COMPARE] phase={phase} complete", flush=True)

    by_phase = {phase: [row for row in rows if row["phase"] == phase] for phase, _, _ in phases}
    hold = by_phase["zero_hold"]
    reset_count = sum(row["done_count"] for row in rows)
    summary = {
        "execution_domain": "VALIDATION_SIM",
        "evidence_status": "PASS" if reset_count == 0 else "SUSPICIOUS",
        "role": role,
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "randomization": "disabled",
        "step_dt_s": step_dt,
        "command_sequence": {
            "settle_s": args_cli.settle_s,
            "zero_hold_s": args_cli.hold_s,
            "ramp_vx_mps": args_cli.ramp_vx,
            "ramp_up_s": args_cli.ramp_up_s,
            "cruise_s": args_cli.cruise_s,
            "decel_window_s": args_cli.decel_window_s,
            "height_m": args_cli.height_m,
        },
        "reset_count": reset_count,
        "zero_hold_drift_x_m": hold[-1]["phase_drift_x_m"],
        "zero_hold_drift_y_m": hold[-1]["phase_drift_y_m"],
        "zero_hold_planar_drift_m": hold[-1]["phase_drift_xy_m"],
        "zero_hold_mean_vx_mps": _mean(hold, "actual_vx"),
        "min_base_height_m": min(row["base_height_m"] for row in rows),
        "max_base_height_m": max(row["base_height_m"] for row in rows),
        "mean_action_rate": _mean(rows, "action_rate_mean"),
    }
    if not args_cli.quick_zero_hold:
        cruise = by_phase["cruise"]
        decel = by_phase["decel"]
        summary.update(
            {
                "cruise_mean_vx_mps": _mean(cruise, "actual_vx"),
                "cruise_vx_rmse_mps": _rmse(cruise, "vx_error"),
                "decel_stop_drift_x_m": decel[-1]["phase_drift_x_m"],
                "decel_stop_planar_drift_m": decel[-1]["phase_drift_xy_m"],
                "decel_peak_tilt": max(row["tilt"] for row in decel),
                "decel_peak_ang_vel_xy_rps": max(row["ang_vel_xy_rps"] for row in decel),
            }
        )
    if rows and "student_z_mae" in rows[0]:
        summary.update(
            {
                "student_z_mae": _mean(rows, "student_z_mae"),
                "student_velocity_mae_mps": _mean(rows, "student_velocity_mae_mps"),
                "student_height_mae_m": _mean(rows, "student_height_mae_m"),
                "student_action_mae": _mean(rows, "student_action_mae"),
            }
        )

    out_dir = (REPO_ROOT / args_cli.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{role}_{checkpoint.parent.name}_{checkpoint.stem}_behavior"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[RESULT] " + json.dumps(summary, indent=2), flush=True)
    print(f"[INFO] CSV: {csv_path}", flush=True)
    print(f"[INFO] JSON: {json_path}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

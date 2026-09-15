"""Compare the commanded trot contact schedule with simulated Wolf foot motion."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--contact-threshold", type=float, default=5.0)
parser.add_argument("--max-contact-force", type=float, default=250.0)
parser.add_argument("--output", default="outputs/gait_alignment")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1:
    parser.error("steps must be positive")
if args.contact_threshold <= 0.0:
    parser.error("contact-threshold must be positive")
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import matplotlib
    import numpy as np
    import torch

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import lab.wolf.tasks
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from lab.wolf.tasks.manager_based.locomotion.velocity.mdp import gait
    from lab.wolf.tasks.manager_based.locomotion.velocity.mdp.oracle import FEET
    from scripts.co_rl.core.runners import OnPolicyRunner
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper

    task = "Isaac-Velocity-Flat-Wolf-v2-GaitTrot-ppo-Play"
    checkpoint = Path(args.checkpoint).resolve()
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = 42
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.speed, args.speed)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    raw = gym.make(task, cfg=cfg)
    base = raw.unwrapped
    agent_cfg = load_cfg_from_registry(task, "co_rl_cfg_entry_point")
    env = CoRlVecEnvWrapper(raw, agent_cfg)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)

    robot = base.scene["robot"]
    sensor = base.scene["contact_forces"]
    foot_ids, foot_names = sensor.find_bodies(FEET, preserve_order=True)
    if foot_names != FEET:
        raise RuntimeError(f"Unexpected foot order: {foot_names}; expected {FEET}")

    obs, _ = env.get_observations()
    if obs.shape != (1, 323):
        raise RuntimeError(f"Unexpected observation shape: {tuple(obs.shape)}")

    times = []
    phases = []
    desired_contacts = []
    forces = []
    foot_speeds = []
    foot_horizontal_speeds = []
    foot_heights = []
    foot_vertical_speeds = []
    dones = []
    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy(obs)
            obs, _, done, _ = env.step(actions)

            force = torch.linalg.vector_norm(sensor.data.net_forces_w[0, foot_ids], dim=-1)
            velocity = robot.data.body_lin_vel_w[0, foot_ids]
            times.append((step + 1) * float(base.step_dt))
            phases.append(float(gait.phase(base)[0]))
            desired_contacts.append(gait.desired_contact(base)[0].detach().cpu().numpy())
            forces.append(force.detach().cpu().numpy())
            foot_speeds.append(torch.linalg.vector_norm(velocity, dim=-1).detach().cpu().numpy())
            foot_horizontal_speeds.append(torch.linalg.vector_norm(velocity[:, :2], dim=-1).detach().cpu().numpy())
            foot_heights.append(robot.data.body_pos_w[0, foot_ids, 2].detach().cpu().numpy())
            foot_vertical_speeds.append(velocity[:, 2].detach().cpu().numpy())
            dones.append(bool(done[0]))

    t = np.asarray(times, dtype=np.float32)
    phase = np.asarray(phases, dtype=np.float32)
    desired = np.asarray(desired_contacts, dtype=np.float32)
    force = np.asarray(forces, dtype=np.float32)
    speed = np.asarray(foot_speeds, dtype=np.float32)
    horizontal_speed = np.asarray(foot_horizontal_speeds, dtype=np.float32)
    height = np.asarray(foot_heights, dtype=np.float32)
    vertical_speed = np.asarray(foot_vertical_speeds, dtype=np.float32)
    actual = (force > args.contact_threshold).astype(np.float32)
    previous_actual = np.vstack([np.zeros((1, actual.shape[1]), dtype=np.float32), actual[:-1]])
    touchdown = (actual > 0.5) & (previous_actual <= 0.5)
    swing = desired <= 0.5
    swing_progress = np.clip((np.mod(phase[:, None] + np.array([0.5, 0.0, 0.0, 0.5]), 1.0) - 0.5) / 0.5, 0.0, 1.0)
    clearance_target = 0.02 + 0.08 * (1.0 - np.abs(2.0 * swing_progress - 1.0))
    clearance_error = np.abs(height - clearance_target)

    def safe_mean(values):
        return float(values.mean()) if values.size else None

    per_foot = {}
    for i, name in enumerate(foot_names):
        stance = desired[:, i] > 0.5
        swing = ~stance
        actual_i = actual[:, i] > 0.5
        tp = np.sum(stance & actual_i)
        fp = np.sum(swing & actual_i)
        fn = np.sum(stance & ~actual_i)
        per_foot[name] = {
            "alignment_accuracy": float(np.mean(actual_i == stance)),
            "stance_contact_rate": float(np.mean(actual_i[stance])),
            "swing_no_contact_rate": float(np.mean(~actual_i[swing])),
            "contact_precision": float(tp / max(tp + fp, 1)),
            "contact_recall": float(tp / max(tp + fn, 1)),
            "mean_force_stance_N": safe_mean(force[stance, i]),
            "mean_force_swing_N": safe_mean(force[swing, i]),
            "p95_force_stance_N": float(np.percentile(force[stance, i], 95)),
            "p95_force_swing_N": float(np.percentile(force[swing, i], 95)),
            "mean_speed_stance_mps": safe_mean(speed[stance, i]),
            "mean_speed_swing_mps": safe_mean(speed[swing, i]),
            "mean_slip_speed_contact_mps": safe_mean(horizontal_speed[actual_i, i]),
            "p95_slip_speed_contact_mps": float(np.percentile(horizontal_speed[actual_i, i], 95)) if np.any(actual_i) else None,
            "mean_swing_clearance_error_m": safe_mean(clearance_error[swing, i]),
            "p95_swing_clearance_error_m": float(np.percentile(clearance_error[swing, i], 95)) if np.any(swing) else None,
            "mean_touchdown_downward_speed_mps": safe_mean(np.maximum(-vertical_speed[touchdown[:, i], i], 0.0)),
            "p95_touchdown_downward_speed_mps": (
                float(np.percentile(np.maximum(-vertical_speed[touchdown[:, i], i], 0.0), 95))
                if np.any(touchdown[:, i]) else None
            ),
            "max_force_excess_N": float(np.max(np.maximum(force[:, i] - args.max_contact_force, 0.0))),
            "mean_abs_vertical_speed_stance_mps": safe_mean(np.abs(vertical_speed[stance, i])),
            "mean_abs_vertical_speed_swing_mps": safe_mean(np.abs(vertical_speed[swing, i])),
        }

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "gait_alignment_trace.npz",
        time_s=t,
        global_phase=phase,
        desired_contact=desired,
        actual_contact=actual,
        foot_force_N=force,
        foot_speed_mps=speed,
        foot_horizontal_speed_mps=horizontal_speed,
        foot_height_m=height,
        foot_vertical_speed_mps=vertical_speed,
        touchdown=touchdown.astype(np.float32),
        clearance_target_m=clearance_target,
        clearance_error_m=clearance_error,
    )

    # Show several gait cycles with the desired contact signal, measured contact,
    # normalized force and measured foot speed in the same time axis.
    plot_mask = t <= min(float(t[-1]), 4.0)
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    short_names = ["FL", "FR", "BL", "BR"]
    for i, (ax, short) in enumerate(zip(axes, short_names)):
        ax.step(t[plot_mask], desired[plot_mask, i], where="post", color="black", linewidth=1.5, label="desired stance")
        ax.step(t[plot_mask], actual[plot_mask, i], where="post", color="tab:green", alpha=0.8, label="actual contact (> threshold)")
        ax.plot(t[plot_mask], np.clip(force[plot_mask, i] / 150.0, 0.0, 1.0), color="tab:red", alpha=0.65, label="force / 150 N")
        ax.plot(t[plot_mask], np.clip(speed[plot_mask, i] / 0.5, 0.0, 1.0), color="tab:blue", alpha=0.65, label="foot speed / 0.5 m/s")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(short)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=4, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Wolf fixed-trot command vs measured foot motion (model checkpoint)")
    fig.tight_layout()
    fig.savefig(output / "gait_alignment.png", dpi=160)
    plt.close(fig)

    summary = {
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": int(checkpoint.stem.split("_")[-1]) if checkpoint.stem.split("_")[-1].isdigit() else None,
        "command": {"lin_vel_x": args.speed, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "gait_command": [0.5, 0.0, 0.0, 2.0],
        "foot_order": foot_names,
        "steps": args.steps,
        "step_dt_s": float(base.step_dt),
        "contact_threshold_N": args.contact_threshold,
        "max_contact_force_N": args.max_contact_force,
        "episode_resets_seen": int(sum(dones)),
        "overall_alignment_accuracy": float(np.mean(actual == desired)),
        "per_foot": per_foot,
    }
    (output / "gait_alignment.json").write_text(json.dumps(summary, indent=2))
    print("WOLF_GAIT_ALIGNMENT_PASS " + json.dumps(summary), flush=True)
    env.close()
except Exception:
    import traceback

    traceback.print_exc()
    raise
else:
    app.close()

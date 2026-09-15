"""Evaluate a fixed-trot checkpoint at several episode-level frequencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument(
    "--frequencies",
    type=float,
    nargs="+",
    default=[1.0, 1.5, 2.0, 2.5, 3.0],
    help="Episode-level trot frequencies in Hz.",
)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--warmup", type=int, default=100)
parser.add_argument("--contact-threshold", type=float, default=5.0)
parser.add_argument("--output", default="outputs/gait_frequency_sweep")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.warmup < 0 or args.warmup >= args.steps:
    parser.error("steps must be positive and warmup must satisfy 0 <= warmup < steps")
if not args.frequencies or any(f <= 0.0 for f in args.frequencies):
    parser.error("frequencies must contain positive values")
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

    task = "Isaac-Velocity-Flat-Wolf-v2-GaitTrotPaper-ppo-Play"
    checkpoint = Path(args.checkpoint).resolve()
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = 42
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.speed, args.speed)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.gait.frequency = float(args.frequencies[0])

    raw = gym.make(task, cfg=cfg)
    base = raw.unwrapped
    agent_cfg = load_cfg_from_registry(task, "co_rl_cfg_entry_point")
    env = CoRlVecEnvWrapper(raw, agent_cfg)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)

    if env.num_actions != 14:
        raise RuntimeError(f"Unexpected action dimension: {env.num_actions}")
    robot = base.scene["robot"]
    sensor = base.scene["contact_forces"]
    foot_ids, foot_names = sensor.find_bodies(FEET, preserve_order=True)
    if foot_names != FEET:
        raise RuntimeError(f"Unexpected foot order: {foot_names}; expected {FEET}")

    # Keep the command term's frequency fixed for each complete episode.  The
    # command term is intentionally fixed-gait, so this update is deterministic
    # and does not resample in the middle of a rollout.
    gait_term = base.command_manager.get_term("gait")
    results = []
    heatmap = []
    short_names = ["FL", "FR", "BL", "BR"]

    for frequency in args.frequencies:
        gait_term._command[:, 3] = float(frequency)
        # Isaac Lab keeps simulation state in inference tensors after the
        # wrapper's initial reset, so explicit episode resets must use the
        # same inference context as the rollout loop.
        with torch.inference_mode():
            obs, _ = env.reset()
        if obs.shape != (1, 323):
            raise RuntimeError(f"Unexpected observation shape: {tuple(obs.shape)}")

        # Keep the transient samples as well.  A low-frequency command can
        # expose a checkpoint that falls before the requested warmup window;
        # that is a useful evaluation result and must not abort the sweep.
        desired_trace = []
        actual_trace = []
        force_trace = []
        horizontal_speed_trace = []
        vertical_speed_trace = []
        done_step = None
        with torch.inference_mode():
            for step in range(args.steps):
                actions = policy(obs)
                obs, _, done, _ = env.step(actions)
                force = torch.linalg.vector_norm(sensor.data.net_forces_w[0, foot_ids], dim=-1)
                velocity = robot.data.body_lin_vel_w[0, foot_ids]
                desired_trace.append(gait.desired_contact(base)[0].cpu().numpy())
                actual_trace.append((force > args.contact_threshold).cpu().numpy().astype(np.float32))
                force_trace.append(force.cpu().numpy())
                horizontal_speed_trace.append(torch.linalg.vector_norm(velocity[:, :2], dim=-1).cpu().numpy())
                vertical_speed_trace.append(velocity[:, 2].cpu().numpy())
                if bool(done[0]):
                    done_step = step
                    break

        def _trace_array(trace, width):
            array = np.asarray(trace, dtype=np.float32)
            if array.size == 0:
                return np.empty((0, width), dtype=np.float32)
            return array.reshape((-1, width))

        all_desired = _trace_array(desired_trace, len(foot_names))
        all_actual = _trace_array(actual_trace, len(foot_names))
        all_force = _trace_array(force_trace, len(foot_names))
        all_horizontal_speed = _trace_array(horizontal_speed_trace, len(foot_names))
        all_vertical_speed = _trace_array(vertical_speed_trace, len(foot_names))

        # Apply the requested warmup whenever the episode survived long
        # enough.  Otherwise use all available samples and report that the
        # warmup could not be applied, preserving an early-fall result.
        warmup_start = args.warmup if all_desired.shape[0] > args.warmup else 0
        warmup_applied = all_desired.shape[0] > args.warmup
        desired = all_desired[warmup_start:]
        actual = all_actual[warmup_start:]
        force = all_force[warmup_start:]
        horizontal_speed = all_horizontal_speed[warmup_start:]
        vertical_speed = all_vertical_speed[warmup_start:]

        per_foot = {}
        alignment = []
        numeric_alignment = []
        for i, name in enumerate(foot_names):
            if desired.shape[0] == 0:
                alignment.append(None)
                per_foot[name] = {
                    "alignment_accuracy": None,
                    "stance_contact_rate": None,
                    "swing_no_contact_rate": None,
                    "mean_slip_speed_contact_mps": None,
                    "p95_force_N": None,
                    "mean_touchdown_downward_speed_mps": None,
                }
                continue
            stance = desired[:, i] > 0.5
            swing = ~stance
            contact = actual[:, i] > 0.5
            alignment_i = float(np.mean(contact == stance))
            alignment.append(alignment_i)
            numeric_alignment.append(alignment_i)
            per_foot[name] = {
                "alignment_accuracy": alignment_i,
                "stance_contact_rate": float(np.mean(contact[stance])) if np.any(stance) else None,
                "swing_no_contact_rate": float(np.mean(~contact[swing])) if np.any(swing) else None,
                "mean_slip_speed_contact_mps": float(np.mean(horizontal_speed[contact, i])) if np.any(contact) else None,
                "p95_force_N": float(np.percentile(force[:, i], 95)),
                "mean_touchdown_downward_speed_mps": float(
                    np.mean(np.maximum(-vertical_speed[:, i], 0.0)[contact])
                ) if np.any(contact) else None,
            }

        results.append({
            "frequency_Hz": float(frequency),
            "steps_requested": args.steps,
            "steps_collected_total": int(all_desired.shape[0]),
            "steps_collected_after_warmup": int(len(desired)),
            "warmup_steps": args.warmup,
            "warmup_steps_used": int(warmup_start),
            "warmup_applied": warmup_applied,
            "step_dt_s": float(base.step_dt),
            "survival_time_s": float(all_desired.shape[0] * base.step_dt),
            "terminated_at_step": done_step,
            "survived_full_rollout": done_step is None,
            "overall_alignment_accuracy": (
                float(np.mean(numeric_alignment)) if numeric_alignment else None
            ),
            "per_foot": per_foot,
        })
        heatmap.append([float(value) if value is not None else np.nan for value in alignment])

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": int(checkpoint.stem.split("_")[-1]) if checkpoint.stem.split("_")[-1].isdigit() else None,
        "command": {"lin_vel_x": args.speed, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "gait_command": {"theta1": 0.5, "theta2": 0.0, "theta3": 0.0, "duty_factor": 0.5},
        "foot_order": foot_names,
        "step_dt_s": float(base.step_dt),
        "contact_threshold_N": args.contact_threshold,
        "results": results,
    }
    (output / "gait_frequency_sweep.json").write_text(json.dumps(summary, indent=2))

    frequencies = np.asarray([item["frequency_Hz"] for item in results], dtype=np.float32)
    alignment = np.asarray(heatmap, dtype=np.float32)
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    alignment = alignment[order]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.15, 1.0]})
    mean_alignment = np.asarray(
        [np.nanmean(row) if np.isfinite(row).any() else np.nan for row in alignment],
        dtype=np.float32,
    )
    ax0.plot(frequencies, mean_alignment, "o-", color="black", label="mean")
    for i, short in enumerate(short_names):
        ax0.plot(frequencies, alignment[:, i], "o--", alpha=0.75, label=short)
    ax0.set_xlabel("Trot frequency [Hz]")
    ax0.set_ylabel("Desired/actual contact alignment")
    ax0.set_ylim(0.0, 1.0)
    ax0.grid(alpha=0.25)
    ax0.legend(ncol=3, fontsize=8)
    if len(frequencies) == 1:
        half_width = 0.5
        frequency_edges = np.asarray(
            [frequencies[0] - half_width, frequencies[0] + half_width], dtype=np.float32
        )
    else:
        midpoints = 0.5 * (frequencies[:-1] + frequencies[1:])
        frequency_edges = np.concatenate(
            ([frequencies[0] - (midpoints[0] - frequencies[0])], midpoints,
             [frequencies[-1] + (frequencies[-1] - midpoints[-1])])
        )
    image = ax1.pcolormesh(
        np.arange(alignment.shape[1] + 1, dtype=np.float32) - 0.5,
        frequency_edges,
        alignment,
        shading="flat",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax1.set_xticks(range(4), short_names)
    ax1.set_ylim(float(frequency_edges[0]), float(frequency_edges[-1]))
    ax1.set_xlabel("Foot")
    ax1.set_ylabel("Trot frequency [Hz]")
    fig.colorbar(image, ax=ax1, label="Alignment")
    fig.suptitle("Wolf trot schedule tracking versus episode-level frequency")
    fig.tight_layout()
    fig.savefig(output / "gait_frequency_sweep.png", dpi=170)
    plt.close(fig)

    print("WOLF_GAIT_FREQUENCY_SWEEP_PASS " + json.dumps(summary), flush=True)
    env.close()
except Exception:
    import traceback

    traceback.print_exc()
    raise
else:
    app.close()

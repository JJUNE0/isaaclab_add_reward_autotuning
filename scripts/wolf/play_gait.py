"""Render a flat-trot Wolf checkpoint and record gait/contact diagnostics.

The optional ``--stand-steps`` prefix sends a zero base-velocity command before
switching to the fixed forward speed.  The gait command itself remains active
throughout the rollout, so the prefix tests in-place trot behavior rather than
switching to an all-feet-down standing mode.
"""

import argparse
import json
import re
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument(
    "--task",
    default="Isaac-Velocity-Flat-Wolf-v2-GaitTrot-ppo-Play",
    help="Play task whose policy and observation contract match the checkpoint.",
)
parser.add_argument("--speed", type=float, default=1.0, help="Fixed forward velocity command [m/s]")
parser.add_argument(
    "--frequency",
    type=float,
    default=None,
    help="Fix the episode-level trot frequency [Hz] when the task exposes gait.frequency_range.",
)
parser.add_argument("--steps", type=int, default=1000, help="Number of control steps to record")
parser.add_argument(
    "--stand-steps",
    type=int,
    default=0,
    help="Initial control steps with zero base velocity before the forward command",
)
parser.add_argument(
    "--contact-threshold",
    type=float,
    default=5.0,
    help="Foot-force norm threshold used to classify contact [N]",
)
parser.add_argument("--output", default="outputs/flat_gait_play")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or not 0 <= args.stand_steps < args.steps:
    parser.error("steps must be positive and stand-steps must satisfy 0 <= stand-steps < steps")
if args.contact_threshold <= 0.0:
    parser.error("contact-threshold must be positive")
args.enable_cameras = True
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

    task = args.task
    checkpoint = Path(args.checkpoint).resolve()
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = 42
    # Sample the same exact command used by the rollout.  For a transition
    # rollout the first observation must already contain the zero command.
    initial_speed = 0.0 if args.stand_steps > 0 else args.speed
    cfg.commands.base_velocity.ranges.lin_vel_x = (initial_speed, initial_speed)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    if args.frequency is not None:
        if args.frequency <= 0.0:
            raise ValueError("frequency must be positive")
        cfg.commands.gait.frequency = args.frequency
        cfg.commands.gait.frequency_range = (args.frequency, args.frequency)
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = (2.8, 3.5, 2.0)
    cfg.viewer.lookat = (0.6, 0.0, 0.0)
    cfg.viewer.resolution = (960, 720)
    cfg.scene.light.spawn.intensity = 400.0
    cfg.sim.render_interval = cfg.decimation

    iteration_match = re.search(r"model_(\d+)", checkpoint.stem)
    checkpoint_iteration = int(iteration_match.group(1)) if iteration_match else None

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = gym.make(task, cfg=cfg, render_mode="rgb_array")
    recorded = gym.wrappers.RecordVideo(
        raw,
        video_folder=str(output),
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        name_prefix=f"wolf-flat-gait-{checkpoint_iteration or 'checkpoint'}",
        disable_logger=True,
    )
    agent_cfg = load_cfg_from_registry(task, "co_rl_cfg_entry_point")
    env = CoRlVecEnvWrapper(recorded, agent_cfg)
    base = env.unwrapped
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)

    velocity_term = base.command_manager.get_term("base_velocity")
    if not hasattr(velocity_term, "vel_command_b"):
        raise RuntimeError("The selected task does not expose a writable velocity command buffer")

    def _set_velocity_command(speed: float):
        # This task has rel_standing_envs=0, but explicitly clear the flag so a
        # playback script remains deterministic if a related task changes it.
        with torch.no_grad():
            velocity_term.vel_command_b[0, :] = torch.tensor(
                [speed, 0.0, 0.0], device=base.device, dtype=velocity_term.vel_command_b.dtype
            )
            if hasattr(velocity_term, "is_standing_env"):
                velocity_term.is_standing_env[0] = False

    _set_velocity_command(initial_speed)
    obs, _ = env.get_observations()
    assert obs.shape == (1, 323), obs.shape
    episodes = []
    robot = env.unwrapped.scene["robot"]
    sensor = env.unwrapped.scene["contact_forces"]
    foot_ids, foot_names = sensor.find_bodies(FEET, preserve_order=True)
    if foot_names != FEET:
        raise RuntimeError(f"Unexpected foot order: {foot_names}; expected {FEET}")
    short_names = ["FL", "FR", "BL", "BR"]
    base_ang_vel_trace = []
    base_lin_vel_x_trace = []
    command_trace = []
    desired_contact_trace = []
    actual_contact_trace = []
    contact_force_trace = []
    current_speed = initial_speed
    for step in range(args.steps):
        scheduled_speed = 0.0 if step < args.stand_steps else args.speed
        if scheduled_speed != current_speed:
            _set_velocity_command(scheduled_speed)
            current_speed = scheduled_speed
        with torch.inference_mode():
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
            foot_force = torch.linalg.vector_norm(
                sensor.data.net_forces_w[0, foot_ids], dim=-1
            )
            desired_contact = gait.desired_contact(base)[0]
            base_lin_vel_x_trace.append(float(robot.data.root_lin_vel_b[0, 0].item()))
            base_ang_vel_trace.append(robot.data.root_ang_vel_b[0, :2].cpu().numpy())
            command_trace.append(float(scheduled_speed))
            desired_contact_trace.append(desired_contact.cpu().numpy())
            contact_force_trace.append(foot_force.cpu().numpy())
            actual_contact_trace.append((foot_force > args.contact_threshold).cpu().numpy().astype(np.float32))
        assert torch.isfinite(obs).all() and torch.isfinite(actions).all()
        if bool(done.any()):
            episodes.append({"step": step, "log": {k: float(v) for k, v in info.get("log", {}).items()}})
        if step % 100 == 0:
            print(f"PLAY_STEP {step}/{args.steps}", flush=True)

    base_ang_vel = np.asarray(base_ang_vel_trace, dtype=np.float32).reshape((-1, 2))
    base_lin_vel_x = np.asarray(base_lin_vel_x_trace, dtype=np.float32)
    command = np.asarray(command_trace, dtype=np.float32)
    desired_contact = np.asarray(desired_contact_trace, dtype=np.float32).reshape((-1, len(foot_names)))
    actual_contact = np.asarray(actual_contact_trace, dtype=np.float32).reshape((-1, len(foot_names)))
    contact_force = np.asarray(contact_force_trace, dtype=np.float32).reshape((-1, len(foot_names)))

    def _distribution(values):
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "mean_abs": float(np.mean(np.abs(values))),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p95_abs": float(np.percentile(np.abs(values), 95)),
        }

    def _contact_rate(values):
        if values.shape[0] == 0:
            return None
        return {name: float(np.mean(values[:, i])) for i, name in enumerate(short_names)}

    def _all_four_fraction(values):
        if values.shape[0] == 0:
            return None
        return float(np.mean(np.all(values > 0.5, axis=1)))

    stand_slice = slice(0, args.stand_steps)
    forward_slice = slice(args.stand_steps, args.steps)
    stand_contact = actual_contact[stand_slice]
    forward_contact = actual_contact[forward_slice]
    stand_desired = desired_contact[stand_slice]
    forward_desired = desired_contact[forward_slice]
    transition_time_s = float(args.stand_steps * base.step_dt)

    summary = {
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "terrain": "flat",
        "command": {"lin_vel_x": args.speed, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "command_schedule": {
            "stand_steps": args.stand_steps,
            "stand_time_s": transition_time_s,
            "stand_lin_vel_x": 0.0,
            "forward_steps": args.steps - args.stand_steps,
            "forward_time_s": float((args.steps - args.stand_steps) * base.step_dt),
            "forward_lin_vel_x": args.speed,
        },
        "gait_frequency_Hz": args.frequency,
        "steps": args.steps,
        "step_dt_s": float(base.step_dt),
        "observation_dim": obs.shape[1],
        "foot_order": foot_names,
        "contact_threshold_N": args.contact_threshold,
        "base_ang_vel_frame": "root/body frame; x=roll rate, y=pitch rate",
        "base_ang_vel_units": "rad/s",
        "base_ang_vel_distribution": {
            "x_body_radps": _distribution(base_ang_vel[:, 0]),
            "y_body_radps": _distribution(base_ang_vel[:, 1]),
            "xy_norm_radps": _distribution(np.linalg.norm(base_ang_vel, axis=1)),
        },
        "base_lin_vel_x_distribution_mps": _distribution(base_lin_vel_x),
        "contact_rate_actual": {
            "stand": _contact_rate(stand_contact),
            "forward": _contact_rate(forward_contact),
        },
        "contact_rate_desired_schedule": {
            "stand": _contact_rate(stand_desired),
            "forward": _contact_rate(forward_desired),
        },
        "all_four_contact_fraction_actual": {
            "stand": _all_four_fraction(stand_contact),
            "forward": _all_four_fraction(forward_contact),
        },
        "all_four_contact_fraction_desired_schedule": {
            "stand": _all_four_fraction(stand_desired),
            "forward": _all_four_fraction(forward_desired),
        },
        "diagnostic_files": {
            "transition_trace": "transition_trace.npz",
            "contact_command_plot": "contact_command_trace.png",
        },
        "episodes": episodes,
        "policy_updates": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        output / "base_ang_vel_trace.npz",
        time_s=np.arange(len(base_ang_vel), dtype=np.float32) * float(base.step_dt),
        base_ang_vel_xy_radps=base_ang_vel,
    )
    time_s = np.arange(len(base_ang_vel), dtype=np.float32) * float(base.step_dt)
    np.savez_compressed(
        output / "transition_trace.npz",
        time_s=time_s,
        command_lin_vel_x_mps=command,
        base_lin_vel_x_mps=base_lin_vel_x,
        base_ang_vel_xy_radps=base_ang_vel,
        desired_contact=desired_contact,
        actual_contact=actual_contact,
        contact_force_N=contact_force,
        foot_order=np.asarray(foot_names),
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(time_s, base_ang_vel[:, 0], color="tab:blue")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("ωx roll rate [rad/s]")
    axes[1].plot(time_s, base_ang_vel[:, 1], color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("ωy pitch rate [rad/s]")
    axes[1].set_xlabel("Time [s]")
    for axis in axes:
        if args.stand_steps > 0:
            axis.axvline(transition_time_s, color="tab:red", linestyle="--", linewidth=1.2,
                         label="stand → forward")
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"Wolf base angular velocity ({args.frequency:g} Hz)"
        if args.frequency is not None else "Wolf base angular velocity"
    )
    fig.tight_layout()
    fig.savefig(output / "base_ang_vel_trace.png", dpi=170)
    plt.close(fig)

    # Contact and command diagnostics make the zero-speed behavior explicit:
    # actual contact is compared with the active trot schedule, not inferred
    # from the velocity command alone.
    fig_contact, axes_contact = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes_contact[0].plot(time_s, command, color="tab:blue", label="commanded vx")
    axes_contact[0].plot(time_s, base_lin_vel_x, color="tab:orange", alpha=0.85, label="actual vx")
    axes_contact[0].set_ylabel("vx [m/s]")
    axes_contact[0].legend(loc="upper right", fontsize=8)
    contact_extent = [0.0, float(time_s[-1] + base.step_dt), -0.5, len(short_names) - 0.5]
    axes_contact[1].imshow(actual_contact.T, origin="lower", aspect="auto", interpolation="nearest",
                           extent=contact_extent, vmin=0.0, vmax=1.0, cmap="Greens")
    axes_contact[1].set_yticks(range(len(short_names)), short_names)
    axes_contact[1].set_ylabel("actual contact")
    axes_contact[2].imshow(desired_contact.T, origin="lower", aspect="auto", interpolation="nearest",
                           extent=contact_extent, vmin=0.0, vmax=1.0, cmap="Blues")
    axes_contact[2].set_yticks(range(len(short_names)), short_names)
    axes_contact[2].set_ylabel("desired stance")
    for index, name in enumerate(short_names):
        axes_contact[3].plot(time_s, contact_force[:, index], label=name)
    axes_contact[3].axhline(args.contact_threshold, color="black", linewidth=0.8, alpha=0.5,
                            label="contact threshold")
    axes_contact[3].set_ylabel("foot force [N]")
    axes_contact[3].set_xlabel("Time [s]")
    axes_contact[3].legend(loc="upper right", fontsize=8, ncol=3)
    for axis in axes_contact:
        if args.stand_steps > 0:
            axis.axvline(transition_time_s, color="tab:red", linestyle="--", linewidth=1.2)
        axis.grid(alpha=0.2)
    fig_contact.suptitle("Wolf trot command transition and foot contacts")
    fig_contact.tight_layout()
    fig_contact.savefig(output / "contact_command_trace.png", dpi=170)
    plt.close(fig_contact)
    env.close()
    print("WOLF_GAIT_PLAY_COMPLETE " + str(output), flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    raise
else:
    app.close()

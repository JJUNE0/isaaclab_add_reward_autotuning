"""Render a flat-trot Wolf checkpoint and record body angular velocity."""

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
parser.add_argument("--output", default="outputs/flat_gait_play")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1:
    parser.error("steps must be positive")
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
    from scripts.co_rl.core.runners import OnPolicyRunner
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper

    task = args.task
    checkpoint = Path(args.checkpoint).resolve()
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = 42
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.speed, args.speed)
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

    obs, _ = env.get_observations()
    assert obs.shape == (1, 323), obs.shape
    episodes = []
    robot = env.unwrapped.scene["robot"]
    base_ang_vel_trace = []
    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
            base_ang_vel_trace.append(robot.data.root_ang_vel_b[0, :2].cpu().numpy())
        assert torch.isfinite(obs).all() and torch.isfinite(actions).all()
        if bool(done.any()):
            episodes.append({"step": step, "log": {k: float(v) for k, v in info.get("log", {}).items()}})
        if step % 100 == 0:
            print(f"PLAY_STEP {step}/{args.steps}", flush=True)

    base_ang_vel = np.asarray(base_ang_vel_trace, dtype=np.float32).reshape((-1, 2))

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

    summary = {
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "terrain": "flat",
        "command": {"lin_vel_x": args.speed, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "gait_frequency_Hz": args.frequency,
        "steps": args.steps,
        "observation_dim": obs.shape[1],
        "base_ang_vel_frame": "root/body frame; x=roll rate, y=pitch rate",
        "base_ang_vel_units": "rad/s",
        "base_ang_vel_distribution": {
            "x_body_radps": _distribution(base_ang_vel[:, 0]),
            "y_body_radps": _distribution(base_ang_vel[:, 1]),
            "xy_norm_radps": _distribution(np.linalg.norm(base_ang_vel, axis=1)),
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
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(time_s, base_ang_vel[:, 0], color="tab:blue")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("ωx roll rate [rad/s]")
    axes[1].plot(time_s, base_ang_vel[:, 1], color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("ωy pitch rate [rad/s]")
    axes[1].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"Wolf base angular velocity ({args.frequency:g} Hz)"
        if args.frequency is not None else "Wolf base angular velocity"
    )
    fig.tight_layout()
    fig.savefig(output / "base_ang_vel_trace.png", dpi=170)
    plt.close(fig)
    env.close()
    print("WOLF_GAIT_PLAY_COMPLETE " + str(output), flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    raise
else:
    app.close()

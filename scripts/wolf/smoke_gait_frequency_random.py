"""Check episode-level trot-frequency sampling for the Wolf gait task."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Velocity-Flat-Wolf-v2-GaitTrotPaperFreqRand-ppo",
)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2 or args.steps < 1:
    parser.error("num_envs must be at least 2 and steps must be positive")
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    import lab.wolf.tasks
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    agent_cfg = load_cfg_from_registry(args.task, "co_rl_cfg_entry_point")
    env = CoRlVecEnvWrapper(gym.make(args.task, cfg=cfg), agent_cfg)
    raw = env.unwrapped
    gait_term = raw.command_manager.get_term("gait")
    low, high = (float(value) for value in gait_term.cfg.frequency_range)
    if (low, high) != (1.0, 3.0):
        raise RuntimeError(f"Unexpected frequency range: {(low, high)}")

    with torch.inference_mode():
        obs, extras = env.reset()
    if obs.shape != (args.num_envs, 323):
        raise RuntimeError(f"Unexpected observation shape: {tuple(obs.shape)}")
    if extras["observations"]["critic"].shape != (args.num_envs, 323):
        raise RuntimeError("Unexpected critic observation shape")

    episode_frequency = gait_term.command[:, 3].clone()
    if not bool(((episode_frequency >= low) & (episode_frequency <= high)).all()):
        raise RuntimeError("Sampled frequency is outside the configured range")
    if float(torch.std(episode_frequency)) <= 0.0:
        raise RuntimeError("Frequency samples are identical across all environments")

    with torch.inference_mode():
        for _ in range(args.steps):
            actions = torch.zeros((args.num_envs, env.num_actions), device=env.device)
            obs, _, done, _ = env.step(actions)
            current_frequency = gait_term.command[:, 3]
            active = ~done.to(torch.bool)
            if bool(active.any()) and not torch.allclose(
                current_frequency[active], episode_frequency[active]
            ):
                raise RuntimeError("Frequency changed during an active episode")
            episode_frequency = torch.where(done.to(torch.bool), current_frequency, episode_frequency)

    with torch.inference_mode():
        env.reset()
    reset_frequency = gait_term.command[:, 3].clone()
    if not bool(((reset_frequency >= low) & (reset_frequency <= high)).all()):
        raise RuntimeError("Reset frequency is outside the configured range")

    summary = {
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "frequency_range_Hz": [low, high],
        "initial_min_Hz": float(episode_frequency.min()),
        "initial_max_Hz": float(episode_frequency.max()),
        "reset_min_Hz": float(reset_frequency.min()),
        "reset_max_Hz": float(reset_frequency.max()),
        "actor_dim": obs.shape[1],
        "critic_dim": extras["observations"]["critic"].shape[1],
    }
    print("WOLF_GAIT_FREQUENCY_RANDOM_SMOKE_PASS " + json.dumps(summary), flush=True)
    env.close()
except Exception:
    import traceback

    traceback.print_exc()
    raise
else:
    app.close()

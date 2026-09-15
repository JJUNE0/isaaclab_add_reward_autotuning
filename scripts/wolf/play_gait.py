"""Render a fixed-speed flat-trot Wolf gait checkpoint."""

import argparse
import json
import re
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--speed", type=float, default=1.0, help="Fixed forward velocity command [m/s]")
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
    import torch

    import lab.wolf.tasks
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from scripts.co_rl.core.runners import OnPolicyRunner
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper

    task = "Isaac-Velocity-Flat-Wolf-v2-GaitTrot-ppo-Play"
    checkpoint = Path(args.checkpoint).resolve()
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = 42
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.speed, args.speed)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
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
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.get_observations()
    assert obs.shape == (1, 323), obs.shape
    episodes = []
    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
        assert torch.isfinite(obs).all() and torch.isfinite(actions).all()
        if bool(done.any()):
            episodes.append({"step": step, "log": {k: float(v) for k, v in info.get("log", {}).items()}})
        if step % 100 == 0:
            print(f"PLAY_STEP {step}/{args.steps}", flush=True)

    summary = {
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": checkpoint_iteration,
        "terrain": "flat",
        "command": {"lin_vel_x": args.speed, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "steps": args.steps,
        "observation_dim": obs.shape[1],
        "episodes": episodes,
        "policy_updates": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    env.close()
    print("WOLF_GAIT_PLAY_COMPLETE " + str(output), flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    raise
else:
    app.close()

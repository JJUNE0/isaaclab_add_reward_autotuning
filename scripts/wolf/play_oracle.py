"""Render a fixed-level Wolf PPO rollout with checkpoint normalization."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--flat", action="store_true", help="Render actual flat mesh terrain with oracle level zero")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--terrain_level", type=int, default=4)
parser.add_argument("--speed", type=float, default=None, help="Fixed forward command; otherwise restore saved training range")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", default="outputs/play_oracle")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or not 0 <= args.terrain_level < 8:
    parser.error("steps must be positive and terrain_level must be 0..7")
args.enable_cameras = True
app = AppLauncher(args).app
try:
    import json
    import gymnasium as gym
    import torch
    import lab.wolf.tasks
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper
    from scripts.co_rl.core.runners import OnPolicyRunner
    task = "Isaac-Velocity-Stairs-Wolf-v2-Oracle-ppo-Play"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    import yaml
    saved = yaml.load((Path(args.checkpoint).resolve().parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    saved_range = tuple(float(x) for x in saved["commands"]["base_velocity"]["ranges"]["lin_vel_x"])
    cfg.commands.base_velocity.ranges.lin_vel_x = saved_range if args.speed is None else (args.speed, args.speed)
    cfg.scene.light.spawn.intensity = 400.0  # Playback lighting only.
    if args.flat:
        from isaaclab.terrains import MeshPlaneTerrainCfg
        cfg.scene.terrain.terrain_generator.sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}
        args.terrain_level = 0
    cfg.seed = 42
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = (2.8, 3.5, 2.0)
    cfg.viewer.lookat = (0.6, 0.0, 0.0)
    cfg.viewer.resolution = (960, 720)
    cfg.sim.render_interval = cfg.decimation
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = gym.make(task, cfg=cfg, render_mode="rgb_array")
    terrain = raw.unwrapped.scene.terrain
    terrain.terrain_levels[:] = args.terrain_level
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    recorded = gym.wrappers.RecordVideo(raw, video_folder=str(output), step_trigger=lambda step: step == 0,
        video_length=args.steps, name_prefix=f"wolf-level{args.terrain_level}", disable_logger=True)
    agent = load_cfg_from_registry(task, "co_rl_cfg_entry_point")
    env = CoRlVecEnvWrapper(recorded, agent)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
    runner.load(str(Path(args.checkpoint).resolve()), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
    obs, extras = env.get_observations()
    episodes = []
    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
        assert torch.isfinite(obs).all() and torch.isfinite(actions).all()
        if done.any():
            episodes.append({"step": step, "log": {k: float(v) for k, v in info.get("log", {}).items()}})
        if step % 100 == 0:
            print(f"PLAY_STEP {step}/{args.steps}", flush=True)
    summary = {"terrain": "flat" if args.flat else "stairs", "checkpoint": str(Path(args.checkpoint).resolve()), "terrain_level": args.terrain_level,
               "command_range": cfg.commands.base_velocity.ranges.lin_vel_x, "steps": args.steps, "episodes": episodes, "policy_updates": 0}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    env.close()
    print("WOLF_PLAY_COMPLETE " + str(output), flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    raise
else:
    app.close()

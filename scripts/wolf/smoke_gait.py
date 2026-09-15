"""Verify fixed-trot command, phase clock, reward routing and PPO dimensions."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("num_envs and steps must be positive")
app = AppLauncher(args).app

try:
    import json
    import gymnasium as gym
    import torch

    import lab.wolf.tasks
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from scripts.co_rl.core.runners import OnPolicyRunner
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper
    from lab.wolf.tasks.manager_based.locomotion.velocity.mdp import gait

    task = "Isaac-Velocity-Flat-Wolf-v2-GaitTrot-ppo"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(task, "co_rl_cfg_entry_point")
    env = CoRlVecEnvWrapper(gym.make(task, cfg=cfg), agent_cfg)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    obs, extras = env.reset()
    assert env.num_actions == 14
    assert obs.shape == (args.num_envs, 323), obs.shape
    assert extras["observations"]["critic"].shape == (args.num_envs, 323)
    assert torch.equal(obs, extras["observations"]["critic"])

    raw = env.unwrapped
    expected = torch.tensor([0.5, 0.0, 0.0, 0.5], device=raw.device).repeat(args.num_envs, 1)
    phases = gait.foot_phase(raw)
    assert torch.allclose(phases, expected, atol=1.0e-6), phases
    desired = gait.desired_contact(raw)
    assert torch.equal(desired, torch.tensor([0.0, 1.0, 1.0, 0.0], device=raw.device).repeat(args.num_envs, 1))
    command = raw.command_manager.get_command("gait")
    assert torch.allclose(command[:, 3], torch.full((args.num_envs,), 2.0, device=raw.device))

    for _ in range(args.steps):
        with torch.inference_mode():
            actions = runner.alg.actor_critic.act_inference(obs)
        obs, rewards, dones, extras = env.step(actions)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(obs).all() and torch.isfinite(rewards).all()
        assert torch.equal(obs, extras["observations"]["critic"])
        assert torch.isfinite(gait.swing_force_penalty(raw)).all()
        assert torch.isfinite(gait.stance_velocity_penalty(raw)).all()

    obs, extras = env.reset()
    assert torch.isfinite(obs).all() and torch.equal(obs, extras["observations"]["critic"])
    print("WOLF_GAIT_SMOKE_PASS " + json.dumps({
        "task": task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "actor_dim": obs.shape[1],
        "critic_dim": extras["observations"]["critic"].shape[1],
        "actions": env.num_actions,
        "gait_command": [0.5, 0.0, 0.0, 2.0],
        "initial_desired_contact": [0, 1, 1, 0],
        "policy_updates": 0,
    }), flush=True)
    env.close()
except Exception:
    import traceback

    traceback.print_exc()
    raise
else:
    app.close()

"""Smoke-test the paper-style Wolf gait task and its one-term ablations."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Wolf-v2-GaitTrotPaper-ppo",
    help="Paper task or one of its NoSlip/NoClearance/NoImpact/NoMaxForce variants.",
)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("num_envs and steps must be positive")
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    import lab.wolf.tasks
    from isaaclab.envs import mdp
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from scripts.co_rl.core.runners import OnPolicyRunner
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper
    from lab.wolf.tasks.manager_based.locomotion.velocity.mdp import gait

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "co_rl_cfg_entry_point")
    env = CoRlVecEnvWrapper(gym.make(args.task, cfg=cfg), agent_cfg)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    obs, extras = env.reset()
    assert env.num_actions == 14
    assert obs.shape == (args.num_envs, 323), obs.shape
    assert extras["observations"]["critic"].shape == (args.num_envs, 323)
    assert torch.equal(obs, extras["observations"]["critic"])

    raw = env.unwrapped
    reward_terms = {
        "base_ang_vel_xy": mdp.ang_vel_xy_l2,
        "foot_slip": gait.foot_slip_penalty,
        "clearance": gait.foot_clearance_penalty,
        "impact_velocity": gait.foot_impact_velocity_penalty,
        "max_contact_force": gait.maximum_contact_force_penalty,
    }
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = runner.alg.actor_critic.act_inference(obs)
        obs, rewards, dones, extras = env.step(actions)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(obs).all() and torch.isfinite(rewards).all()
        assert torch.equal(obs, extras["observations"]["critic"])
        for name, term in reward_terms.items():
            value = term(raw)
            assert value.shape == (args.num_envs,), (name, value.shape)
            assert torch.isfinite(value).all(), name

    obs, extras = env.reset()
    assert torch.isfinite(obs).all() and torch.equal(obs, extras["observations"]["critic"])
    print(
        "WOLF_GAIT_PAPER_SMOKE_PASS "
        + json.dumps(
            {
                "task": args.task,
                "num_envs": args.num_envs,
                "steps": args.steps,
                "actor_dim": obs.shape[1],
                "critic_dim": extras["observations"]["critic"].shape[1],
                "actions": env.num_actions,
                "paper_terms": list(reward_terms),
                "policy_updates": 0,
            }
        ),
        flush=True,
    )
    env.close()
except Exception:
    import traceback

    traceback.print_exc()
    raise
else:
    app.close()

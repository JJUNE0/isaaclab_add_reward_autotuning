"""Verify oracle routing, PPO construction and reset/step; never optimize a policy."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--max_height", action="store_true", help="Use exactly 0.40 m risers for geometry smoke only")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.num_envs < 1:
    parser.error("steps and num_envs must be positive")
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import json
    import torch
    import lab.wolf.tasks
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper
    from scripts.co_rl.core.runners import OnPolicyRunner
    task = "Isaac-Velocity-Stairs-Wolf-v2-Oracle-ppo"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    if args.max_height:
        cfg.scene.terrain.terrain_generator.num_rows = 1
        cfg.scene.terrain.terrain_generator.num_cols = 1
        cfg.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.curriculum.terrain_levels = None
    agent = load_cfg_from_registry(task, "co_rl_cfg_entry_point")
    env = CoRlVecEnvWrapper(gym.make(task, cfg=cfg), agent)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
    obs, extras = env.reset()
    assert env.num_actions == 14
    sensor = env.unwrapped.scene["contact_forces"]
    term = env.unwrapped.reward_manager.get_term_cfg("nonfoot_contact")
    ids = term.params["sensor_cfg"].body_ids
    selected = {sensor.body_names[i] for i in ids}
    from lab.wolf.tasks.manager_based.locomotion.velocity.mdp.oracle import FEET
    assert selected == set(sensor.body_names) - set(FEET), selected
    assert len(selected) == 15
    print("NONFOOT_CONTACT_SELECTION_PASS " + json.dumps(sorted(selected)), flush=True)
    assert obs.shape == (args.num_envs, 311), obs.shape
    assert torch.equal(obs, extras["observations"]["critic"]), "Actor does not receive the same oracle information"
    initial_parameters = [p.detach().clone() for p in runner.alg.actor_critic.parameters()]
    resets = 0
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = runner.alg.actor_critic.act_inference(obs)
        assert torch.isfinite(actions).all()
        obs, rewards, dones, extras = env.step(actions)
        assert torch.isfinite(obs).all() and torch.isfinite(rewards).all()
        assert torch.equal(obs, extras["observations"]["critic"])
        resets += int(dones.sum())
    assert all(torch.equal(before, after) for before, after in zip(initial_parameters, runner.alg.actor_critic.parameters()))
    # Also exercise explicit reset after rollout.
    obs, extras = env.reset()
    assert torch.isfinite(obs).all() and torch.equal(obs, extras["observations"]["critic"])
    print("WOLF_ORACLE_PPO_SMOKE_PASS " + json.dumps({
        "num_envs": args.num_envs, "steps": args.steps, "actor_dim": obs.shape[1],
        "critic_dim": extras["observations"]["critic"].shape[1], "actions": env.num_actions,
        "exact_40cm": args.max_height, "episode_resets": resets, "policy_updates": 0,
    }), flush=True)
    env.close()
except Exception:
    import traceback
    traceback.print_exc()
    raise
else:
    app.close()

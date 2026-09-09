"""Zero-command drift check run natively in Isaac Lab (ground truth), for
direct comparison against eval_sim2sim_zero_command.py's MuJoCo result on the
same checkpoint -- isolates "sim2sim transfer gap" from "cfg-runtime adapter
bug" by giving both a result computed with the exact same obs/action cfg.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Flat-Tita-Play-v3-moo-ppo")
parser.add_argument("--num_policy_stacks", type=int, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_s", type=float, default=1.0)
parser.add_argument("--duration_s", type=float, default=30.0)
parser.add_argument("--out_dir", default="eval/results/zero_command")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat

from scripts.co_rl.core.runners import MOO_OnPolicyRunner
from scripts.co_rl.core.wrapper.vecenv_wrapper_v2 import CoRlVecEnvWrapper


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.events.push_robot = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapper_cfg = type(
        "WrapperCfg",
        (),
        {
            "num_policy_stacks": args_cli.num_policy_stacks,
            "num_critic_stacks": 0,
            "use_constraint_rl": False,
        },
    )()
    env = CoRlVecEnvWrapper(env, wrapper_cfg)

    import yaml

    agent_cfg = yaml.safe_load((Path(args_cli.checkpoint).parent / "params" / "agent.yaml").read_text())
    agent_cfg["store_training_data"] = False
    runner = MOO_OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    command_term = env.unwrapped.command_manager.get_term("base_velocity")

    obs, extras = env.get_observations()
    command_term.vel_command_b[:] = 0.0

    policy_dt = env.unwrapped.step_dt
    settle_steps = max(1, round(args_cli.settle_s / policy_dt))
    measure_steps = max(1, round(args_cli.duration_s / policy_dt))

    with torch.inference_mode():
        for _ in range(settle_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            command_term.vel_command_b[:] = 0.0

        start_pos = env.unwrapped.scene["robot"].data.root_link_pos_w[:, :3].clone()
        rows = []
        for step in range(measure_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            command_term.vel_command_b[:] = 0.0

            pos = env.unwrapped.scene["robot"].data.root_link_pos_w[:, :3]
            quat = env.unwrapped.scene["robot"].data.root_link_quat_w
            roll, pitch, _ = euler_xyz_from_quat(quat)
            disp = (pos - start_pos)[0].cpu().numpy()
            rows.append(
                {
                    "time_s": (step + 1) * policy_dt,
                    "base_z": float(pos[0, 2].item()),
                    "drift_xy_m": float((disp[0] ** 2 + disp[1] ** 2) ** 0.5),
                    "tilt": float((roll[0].item() ** 2 + pitch[0].item() ** 2) ** 0.5),
                }
            )

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args_cli.checkpoint).parent.name}_{Path(args_cli.checkpoint).stem}_isaaclab_zero_command_{args_cli.duration_s:g}s"
    with (out_dir / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": args_cli.checkpoint,
        "domain": "ISAAC_LAB_NATIVE",
        "final_drift_xy_m": rows[-1]["drift_xy_m"],
        "max_drift_xy_m": max(r["drift_xy_m"] for r in rows),
        "min_base_z_m": min(r["base_z"] for r in rows),
        "max_tilt": max(r["tilt"] for r in rows),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print("[RESULT] " + json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

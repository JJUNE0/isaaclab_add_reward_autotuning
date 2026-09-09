#!/usr/bin/env python3
"""Log base height + joint angles for the first few seconds after reset, in
real Isaac Lab (not MuJoCo sim2sim), to see exactly what the policy settles
into during the command's initial_phase_time window -- when the raw height
command is masked to 0.0 (see UniformVelocityWithZCommand.command /
_init_phase_safe_height in feature_functions_common.py). MuJoCo sim2sim
tooling can't answer this because it doesn't replicate that masking; it
always sets an explicit height command with no init-phase concept.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-Tita-GTBaseVel-TargetHeight-v3-moo-ppo")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration_s", type=float, default=7.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import lab.flamingo.tasks  # noqa: F401
from scripts.co_rl.core.runners import MOO_OnPolicyRunner
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper

PHYSICAL_JOINT_NAMES = [
    "joint_left_leg_1", "joint_left_leg_2", "joint_left_leg_3", "joint_left_leg_4",
    "joint_right_leg_1", "joint_right_leg_2", "joint_right_leg_3", "joint_right_leg_4",
]


def _resolve(entry_point):
    if not isinstance(entry_point, str):
        return entry_point() if callable(entry_point) else entry_point
    module_name, attr = entry_point.split(":")
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if callable(obj) else obj


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()

    spec = gym.spec(args_cli.task)
    env_cfg = _resolve(spec.kwargs["env_cfg_entry_point"])
    agent_cfg = _resolve(spec.kwargs["co_rl_cfg_entry_point"])
    env_cfg.seed = 42
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    # Deterministic reset (default pose, no noise/push) so any twisting we
    # see is purely from the policy's own response, not reset randomization.
    env_cfg.observations.stack_policy.enable_corruption = False
    env_cfg.observations.none_stack_policy.enable_corruption = False
    env_cfg.events.physics_material = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_robot_joints.params["position_noise_range"] = (0.0, 0.0)
    env_cfg.events.reset_robot_joints.params["interpolation_range"] = (0.0, 0.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        },
    }

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = CoRlVecEnvWrapper(env, agent_cfg)
    runner = MOO_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    joint_ids = [robot.find_joints(name)[0][0] for name in PHYSICAL_JOINT_NAMES]
    step_dt = float(env.unwrapped.step_dt)
    n_steps = max(1, round(args_cli.duration_s / step_dt))
    initial_phase_time = env.unwrapped.command_manager.get_term("base_velocity").cfg.initial_phase_time

    obs, _ = env.get_observations()
    print(f"[INFO] step_dt={step_dt:.4f}s, initial_phase_time={initial_phase_time:.2f}s, "
          f"logging {n_steps} steps ({args_cli.duration_s:.1f}s)")
    print(f"{'t(s)':>6} {'phase':>8} {'raw_z_cmd':>10} {'base_z':>8} "
          f"{'hip_L':>7} {'shoulder_L':>10} {'leg_L':>7} {'hip_R':>7} {'shoulder_R':>10} {'leg_R':>7}")

    for step in range(n_steps):
        t = (step + 1) * step_dt
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        base_z = float(robot.data.root_link_pos_w[0, 2])
        q = robot.data.joint_pos[0, joint_ids]
        raw_z_cmd = float(env.unwrapped.command_manager.get_command("base_velocity")[0, 3])
        phase = "init" if t <= initial_phase_time else "post"
        if step % 10 == 0 or (t > initial_phase_time and (t - step_dt) <= initial_phase_time):
            print(
                f"{t:6.2f} {phase:>8} {raw_z_cmd:10.4f} {base_z:8.4f} "
                f"{float(q[0]):7.4f} {float(q[1]):10.4f} {float(q[2]):7.4f} "
                f"{float(q[4]):7.4f} {float(q[5]):10.4f} {float(q[6]):7.4f}"
            )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

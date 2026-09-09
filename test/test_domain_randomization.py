# SPDX-License-Identifier: BSD-3-Clause
"""Test script to verify DomainManager parameter randomization for TITA environment."""

import os
import sys

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import lab.flamingo.tasks  # noqa: F401
from lab.flamingo.tasks.moo_based.locomotion.velocity.tita_env.flat_env.stand_drive.flat_env_stand_drive_cfg import TitaFlatEnvCfg
from lab.flamingo.tasks.moo_based.locomotion.velocity.tita_env.agents.co_rl_cfg import TitaMOOPPORunnerCfg_Flat_Stand_Drive
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper


def main():
    task_id = "Isaac-Velocity-Flat-Tita-v3-moo-ppo"
    env_cfg = TitaFlatEnvCfg()
    env_cfg.scene.num_envs = 4
    agent_cfg = TitaMOOPPORunnerCfg_Flat_Stand_Drive()

    print("=== DOMAIN MANAGER PARAMETER RANDOMIZATION VERIFICATION ===")
    raw_env = gym.make(task_id, cfg=env_cfg)
    env = CoRlVecEnvWrapper(raw_env, agent_cfg)

    # 1. Print registered domain manager active terms
    if hasattr(raw_env, "domain_manager"):
        print("DomainManager initialized successfully!")
        info = raw_env.domain_manager.get_curriculum_info()
        print("Curriculum Info:", info)
    elif hasattr(raw_env.unwrapped, "domain_manager"):
        print("DomainManager initialized on unwrapped env!")
        info = raw_env.unwrapped.domain_manager.get_curriculum_info()
        print("Curriculum Info:", info)

    # 2. Reset env and inspect parameters across 4 envs
    obs, extras = env.reset()

    robot = raw_env.unwrapped.scene["robot"]
    print("\n--- Physical Parameters Across 4 Environments After Reset ---")

    # Joint Positions
    print("Joint Positions (rad) shape:", robot.data.joint_pos.shape)
    for i in range(4):
        print(f"Env {i} Joint Pos (first 4):", robot.data.joint_pos[i, :4].cpu().numpy().round(4))

    # Joint Gains (Stiffness & Damping)
    if hasattr(robot.data, "actuator_stiffness"):
        print("\nActuator Kp Stiffness shape:", robot.data.actuator_stiffness.shape)
        for i in range(4):
            print(f"Env {i} Kp (first 4):", robot.data.actuator_stiffness[i, :4].cpu().numpy().round(2))

    if hasattr(robot.data, "actuator_damping"):
        print("\nActuator Kd Damping shape:", robot.data.actuator_damping.shape)
        for i in range(4):
            print(f"Env {i} Kd (first 4):", robot.data.actuator_damping[i, :4].cpu().numpy().round(2))

    # Privileged Physical Observation Group in Policy (priv_physical: 20D)
    if "observations" in extras and "critic" in extras["observations"]:
        critic_obs = extras["observations"]["critic"]
        print("\nCritic Observation Tensor Shape:", critic_obs.shape)
        print("Privileged Physical Group in Critic Obs (body mass, CoM, Kp, Kd):")
        for i in range(4):
            # Last 20 features correspond to priv_physical (mass: 1, com: 3, kp: 8, kd: 8)
            priv_phys = critic_obs[i, -20:]
            print(f"Env {i} Mass offset:", priv_phys[0].item(), "| CoM:", priv_phys[1:4].cpu().numpy().round(3))

    print("\n=== DOMAIN PARAMETER VERIFICATION COMPLETE ===")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

# SPDX-License-Identifier: BSD-3-Clause
"""Discriminator Dimension & Context Verification Script for TITA and Flamingo."""

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
from scripts.co_rl.core.runners import MOO_OnPolicyRunner


def main():
    task_id = "Isaac-Velocity-Flat-Tita-v3-moo-ppo"
    env_cfg = TitaFlatEnvCfg()
    env_cfg.scene.num_envs = 4
    agent_cfg = TitaMOOPPORunnerCfg_Flat_Stand_Drive()

    raw_env = gym.make(task_id, cfg=env_cfg)
    env = CoRlVecEnvWrapper(raw_env, agent_cfg)
    runner = MOO_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.logger_type = None

    obs_tensor, extras = env.get_observations()
    single_obs_dim = runner.alg.single_obs_dim  # 28
    disc_context = obs_tensor[:, :single_obs_dim]
    reward_manager = raw_env.unwrapped.moo_reward_manager
    delta = reward_manager.get_latest_delta(recompute=True, detach=True)

    print("=== TITA RUNTIME DISCRIMINATOR TENSOR VERIFICATION ===")
    print("policy_obs.shape:", obs_tensor.shape)
    print("disc_context.shape:", disc_context.shape)
    print("delta.shape:", delta.shape)
    print("discriminator first layer in_features:", runner.alg.discriminator.net[0].in_features)

    # Invariants assertion
    assert delta.shape[-1] == 14, f"Expected delta dim 14, got {delta.shape[-1]}"
    assert (
        disc_context.shape[-1] + delta.shape[-1] == runner.alg.discriminator.net[0].in_features
    ), f"Mismatch: {disc_context.shape[-1]} + {delta.shape[-1]} != {runner.alg.discriminator.net[0].in_features}"

    # Verification of reward compute context vs update context
    compute_context = raw_env.unwrapped.observation_manager.compute_group("stack_policy")
    print("MOORewardManager.compute() context group ('stack_policy') shape:", compute_context.shape)
    assert torch.equal(compute_context, disc_context), "Reward compute context and update context layout mismatch!"

    print("=== ALL RUNTIME INVARIANTS PASSED SUCCESSFULLY ===")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

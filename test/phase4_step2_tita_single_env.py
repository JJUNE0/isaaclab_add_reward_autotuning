# SPDX-License-Identifier: BSD-3-Clause
"""Phase 4 - Step 2: TITA 1 env reset + 20 steps check."""

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
    env_cfg.scene.num_envs = 1
    agent_cfg = TitaMOOPPORunnerCfg_Flat_Stand_Drive()

    print("=== PHASE 4 - STEP 2: TITA 1 ENV RESET + 20 STEPS ===")
    raw_env = gym.make(task_id, cfg=env_cfg)
    env = CoRlVecEnvWrapper(raw_env, agent_cfg)
    runner = MOO_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="/tmp/test_phase4_step2", device=env.device)

    obs, extras = env.reset()
    print("Initial Reset Successful. Obs shape:", obs.shape)

    for i in range(20):
        actions = torch.zeros((1, 8), device=env.device)
        obs, rewards, dones, infos = env.step(actions)
        assert not torch.isnan(obs).any().item(), f"NaN in obs at step {i}"
        assert not torch.isnan(rewards).any().item(), f"NaN in rewards at step {i}"
        assert "delta" in infos and not torch.isnan(infos["delta"]).any().item(), f"NaN in delta at step {i}"
        print(f"Step {i+1}/20: Reward={rewards.mean().item():.4f}, Delta_Dim={infos['delta'].shape[1]}")

    print("=== STEP 2 PASSED: TITA SINGLE ENV 20 STEPS COMPLETE ===")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

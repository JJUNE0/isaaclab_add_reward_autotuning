# SPDX-License-Identifier: BSD-3-Clause
"""Phase 4 - Step 1: Flamingo 16 envs x 2 iterations regression check."""

import os
import sys

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import lab.flamingo.tasks  # noqa: F401
from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_env.flat_env.stand_drive.flat_env_stand_drive_cfg import FlamingoFlatEnvCfg
from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_env.agents.co_rl_cfg import FlamingoMOOPPORunnerCfg_Flat_Stand_Drive
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper
from scripts.co_rl.core.runners import MOO_OnPolicyRunner


def main():
    task_id = "Isaac-Velocity-Flat-Flamingo-v3-moo-ppo"
    env_cfg = FlamingoFlatEnvCfg()
    env_cfg.scene.num_envs = 16
    agent_cfg = FlamingoMOOPPORunnerCfg_Flat_Stand_Drive()
    agent_cfg.max_iterations = 2
    agent_cfg.save_interval = 1

    log_dir = "/tmp/test_phase4_step1"
    os.makedirs(log_dir, exist_ok=True)

    print("=== PHASE 4 - STEP 1: FLAMINGO 16 ENV x 2 ITERATIONS ===")
    raw_env = gym.make(task_id, cfg=env_cfg)
    env = CoRlVecEnvWrapper(raw_env, agent_cfg)
    runner = MOO_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)
    runner.logger_type = None

    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    print("=== STEP 1 PASSED: FLAMINGO REGRESSION COMPLETE ===")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

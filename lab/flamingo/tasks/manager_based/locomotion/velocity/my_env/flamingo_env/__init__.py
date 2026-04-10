# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    flat_env,
    rough_env,
    diffusion_env,
)

##
# Register Gym environments.
##


#########################################CoRL###################################################
################################################################################################
gym.register(
    id="Flamingo-Flat-Stand-Drive-Base-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Flamingo-Flat-Stand-Drive-Base-Velocity-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Stand_Drive,
    },
)

gym.register(
    id="Flamingo-Flat-Stand-Walk-Base-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_walk_cfg.FlamingoFlatEnvWalkCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Stand_Walk,
    },
)

gym.register(
    id="Flamingo-Flat-Stand-Walk-Base-Velocity-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_walk_cfg.FlamingoFlatEnvWalkCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Stand_Walk,
    },
)

gym.register(
    id="Flamingo-Flat-Rise-Again",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_rise_again_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Rise_Again,
    },
)

gym.register(
    id="Flamingo-Flat-Rise-Again-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_rise_again_cfg.FlamingoFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoFlatPPORunnerCfg_Rise_Again,
    },
)


gym.register(
    id="Flamingo-Diffusion-Policy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": diffusion_env.FlamingoDiffusionPolicyEnvCfg_PLAY,
    },
)

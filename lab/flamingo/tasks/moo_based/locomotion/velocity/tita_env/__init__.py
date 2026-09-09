# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Tita-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.TitaFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.TitaMOOPPORunnerCfg_Flat_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tita-GTBaseVel-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.TitaFlatEnvCfg_GTBaseVel,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.TitaMOOPPORunnerCfg_Flat_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tita-GTBaseVel-TargetHeight-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.TitaFlatEnvCfg_GTBaseVel_TargetHeight,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.TitaMOOPPORunnerCfg_Flat_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tita-Estimator-TargetHeight-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.TitaFlatEnvCfg_Estimator_TargetHeight,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.TitaMOOPPORunnerCfg_Flat_Stand_Drive_Estimator,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tita-RMATeacher-TargetHeight-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.TitaFlatEnvCfg_RMA_TargetHeight,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.TitaMOOPPORunnerCfg_Flat_Stand_Drive_RMATeacher,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tita-RMATeacher808-TargetHeight-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.TitaFlatEnvCfg_GTBaseVel_TargetHeight,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.TitaMOOPPORunnerCfg_Flat_Stand_Drive_RMATeacher808,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tita-RMAStudent-TargetHeight-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.TitaFlatEnvCfg_RMA_TargetHeight,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.TitaMOOPPORunnerCfg_Flat_Stand_Drive_RMAStudent,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tita-RMAStudentMultiHead808-TargetHeight-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.TitaFlatEnvCfg_GTBaseVel_TargetHeight,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.TitaMOOPPORunnerCfg_Flat_Stand_Drive_RMAStudentMultiHead808,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tita-Play-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.TitaFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.TitaMOOPPORunnerCfg_Flat_Stand_Drive,
    },
)

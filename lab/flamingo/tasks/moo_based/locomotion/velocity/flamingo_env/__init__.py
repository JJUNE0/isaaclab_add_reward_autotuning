# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    flat_env,
    rough_env,
)

##
# Register Gym environments.
##


#########################################CoRL###################################################
################################################################################################
gym.register(
    id="Isaac-Velocity-Flat-Flamingo-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoMOOPPORunnerCfg_Flat_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Play-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_stand_drive_cfg.FlamingoFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoMOOPPORunnerCfg_Flat_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Recovery-Flamingo-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_recovery_cfg.FlamingoFlatEnvRecoveryCfg ,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoMOOPPORunnerCfg_Recovery,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Recovery-Flamingo-Play-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.flat_env_recovery_cfg.FlamingoFlatEnvRecoveryCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoMOOPPORunnerCfg_Recovery,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoMOOPPORunnerCfg_Rough_Stand_Drive,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Play-v3-moo-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoMOOPPORunnerCfg_Rough_Stand_Drive,
    },
)


gym.register(
    id="Isaac-Velocity-Rough-Flamingo-v3-moo-Teacher-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.teacher_student_cfg.FlamingoRMATeacherRunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Play-v3-moo-Teacher-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.teacher_student_cfg.FlamingoRMATeacherRunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-v3-moo-Student-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg,
        "co_rl_cfg_entry_point": agents.teacher_student_cfg.FlamingoRMAStudentRunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Flamingo-Play-v3-moo-Student-ppo",
    entry_point="lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env.rough_env_stand_drive_cfg.FlamingoRoughEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.teacher_student_cfg.FlamingoRMAStudentRunnerCfg,
    },
)

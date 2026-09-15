import gymnasium as gym

for suffix, config in [("", "WolfStairEnvCfg"), ("-Play", "WolfStairEnvCfg_PLAY")]:
    gym.register(
        id=f"Isaac-Velocity-Stairs-Wolf-v2-Oracle-ppo{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rough_env.stair:{config}",
            "co_rl_cfg_entry_point": f"{__name__}.agents.co_rl_cfg:WolfStairPPORunnerCfg",
        },
    )

for suffix, config in [("", "WolfGaitFlatEnvCfg"), ("-Play", "WolfGaitFlatEnvCfg_PLAY")]:
    gym.register(
        id=f"Isaac-Velocity-Flat-Wolf-v2-GaitTrot-ppo{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env.gait:{config}",
            "co_rl_cfg_entry_point": f"{__name__}.agents.gait_ppo_cfg:WolfGaitPPORunnerCfg",
        },
    )

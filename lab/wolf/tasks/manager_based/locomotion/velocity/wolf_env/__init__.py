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


# Paper-style contact-quality comparison.  The full task enables all four
# additional terms; each ablation disables exactly one term while retaining the
# same scene, commands, observations and PPO settings.
_PAPER_GAIT_TASKS = {
    "GaitTrotPaper": ("WolfGaitFlatPaperEnvCfg", "WolfGaitFlatPaperEnvCfg_PLAY", "WolfGaitPaperPPORunnerCfg"),
    "GaitTrotPaperFreqRand": (
        "WolfGaitFlatPaperFreqRandEnvCfg",
        "WolfGaitFlatPaperFreqRandEnvCfg_PLAY",
        "WolfGaitPaperFreqRandPPORunnerCfg",
    ),
    "GaitTrotPaperNoSlip": (
        "WolfGaitFlatPaperNoSlipEnvCfg",
        "WolfGaitFlatPaperNoSlipEnvCfg_PLAY",
        "WolfGaitPaperNoSlipPPORunnerCfg",
    ),
    "GaitTrotPaperNoClearance": (
        "WolfGaitFlatPaperNoClearanceEnvCfg",
        "WolfGaitFlatPaperNoClearanceEnvCfg_PLAY",
        "WolfGaitPaperNoClearancePPORunnerCfg",
    ),
    "GaitTrotPaperNoImpact": (
        "WolfGaitFlatPaperNoImpactEnvCfg",
        "WolfGaitFlatPaperNoImpactEnvCfg_PLAY",
        "WolfGaitPaperNoImpactPPORunnerCfg",
    ),
    "GaitTrotPaperNoMaxForce": (
        "WolfGaitFlatPaperNoMaxForceEnvCfg",
        "WolfGaitFlatPaperNoMaxForceEnvCfg_PLAY",
        "WolfGaitPaperNoMaxForcePPORunnerCfg",
    ),
}

for _task_name, (_env_cfg, _env_cfg_play, _runner_cfg) in _PAPER_GAIT_TASKS.items():
    for _suffix, _config in [("", _env_cfg), ("-Play", _env_cfg_play)]:
        gym.register(
            id=f"Isaac-Velocity-Flat-Wolf-v2-{_task_name}-ppo{_suffix}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.flat_env.gait_paper:{_config}",
                "co_rl_cfg_entry_point": f"{__name__}.agents.gait_paper_ppo_cfg:{_runner_cfg}",
            },
        )

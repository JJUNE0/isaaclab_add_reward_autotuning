"""PPO runner names for the paper-style gait baseline and ablations."""

from isaaclab.utils import configclass

from .co_rl_cfg import WolfStairPPORunnerCfg


@configclass
class _WolfGaitPaperPPORunnerCfg(WolfStairPPORunnerCfg):
    max_iterations = 1500
    experiment_description = (
        "Flat Wolf trot with V0 contact shaping plus the four paper-style "
        "foot slip, clearance, impact velocity and maximum-force terms"
    )


@configclass
class WolfGaitPaperPPORunnerCfg(_WolfGaitPaperPPORunnerCfg):
    experiment_name = "Wolf_v2_Flat_Gait_Trot_Paper_Full_PPO"


@configclass
class WolfGaitPaperNoSlipPPORunnerCfg(_WolfGaitPaperPPORunnerCfg):
    experiment_name = "Wolf_v2_Flat_Gait_Trot_Paper_NoSlip_PPO"
    experiment_description = "Paper full baseline without foot-slip shaping"


@configclass
class WolfGaitPaperNoClearancePPORunnerCfg(_WolfGaitPaperPPORunnerCfg):
    experiment_name = "Wolf_v2_Flat_Gait_Trot_Paper_NoClearance_PPO"
    experiment_description = "Paper full baseline without swing-foot clearance shaping"


@configclass
class WolfGaitPaperNoImpactPPORunnerCfg(_WolfGaitPaperPPORunnerCfg):
    experiment_name = "Wolf_v2_Flat_Gait_Trot_Paper_NoImpact_PPO"
    experiment_description = "Paper full baseline without touchdown impact shaping"


@configclass
class WolfGaitPaperNoMaxForcePPORunnerCfg(_WolfGaitPaperPPORunnerCfg):
    experiment_name = "Wolf_v2_Flat_Gait_Trot_Paper_NoMaxForce_PPO"
    experiment_description = "Paper full baseline without maximum contact-force shaping"


__all__ = [
    "WolfGaitPaperPPORunnerCfg",
    "WolfGaitPaperNoSlipPPORunnerCfg",
    "WolfGaitPaperNoClearancePPORunnerCfg",
    "WolfGaitPaperNoImpactPPORunnerCfg",
    "WolfGaitPaperNoMaxForcePPORunnerCfg",
]

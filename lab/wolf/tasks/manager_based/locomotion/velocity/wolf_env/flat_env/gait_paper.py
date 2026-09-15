"""Walk These Ways-style contact shaping and ablations for Wolf on flat ground.

The V0 task in :mod:`gait` remains unchanged.  This module adds the four
contact-quality terms needed for the paper-style comparison, then exposes one
task per one-term ablation so runs can share exactly the same PPO and scene
configuration.
"""

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as Reward
from isaaclab.utils import configclass

from ...mdp import gait
from ..velocity_env_cfg import (
    GaitFrequencyRandomizedCommandsCfg,
    WolfGaitFlatEnvCfg as WolfGaitFlatBaseCfg,
)
from .gait import GaitRewardsCfg


# The terms return bounded costs.  Negative reward weights therefore keep their
# contribution comparable to the two V0 gait costs while the physical units of
# each threshold remain explicit in the config below.
PAPER_FOOT_SLIP_WEIGHT = -0.05
PAPER_CLEARANCE_WEIGHT = -0.05
PAPER_IMPACT_WEIGHT = -0.05
PAPER_MAX_FORCE_WEIGHT = -0.05
BASE_ANG_VEL_XY_WEIGHT = -0.1


@configclass
class GaitPaperRewardsCfg(GaitRewardsCfg):
    """V0 rewards plus contact-quality costs and base roll/pitch damping."""

    base_ang_vel_xy = Reward(
        func=mdp.ang_vel_xy_l2,
        weight=BASE_ANG_VEL_XY_WEIGHT,
    )

    gait_foot_slip = Reward(
        func=gait.foot_slip_penalty,
        weight=PAPER_FOOT_SLIP_WEIGHT,
        params={"velocity_scale": 0.5, "force_threshold": 5.0},
    )
    gait_foot_clearance = Reward(
        func=gait.foot_clearance_penalty,
        weight=PAPER_CLEARANCE_WEIGHT,
        params={
            "target_height": 0.10,
            "height_offset": 0.02,
            "height_scale": 0.04,
            "reference_height": 0.0,
        },
    )
    gait_impact_velocity = Reward(
        func=gait.foot_impact_velocity_penalty,
        weight=PAPER_IMPACT_WEIGHT,
        params={"velocity_scale": 0.75, "force_threshold": 5.0},
    )
    gait_max_contact_force = Reward(
        func=gait.maximum_contact_force_penalty,
        weight=PAPER_MAX_FORCE_WEIGHT,
        params={"max_contact_force": 250.0, "force_scale": 150.0},
    )


@configclass
class GaitPaperNoSlipRewardsCfg(GaitPaperRewardsCfg):
    """Paper baseline with only foot-slip shaping removed."""

    gait_foot_slip = Reward(
        func=gait.foot_slip_penalty,
        weight=0.0,
        params={"velocity_scale": 0.5, "force_threshold": 5.0},
    )


@configclass
class GaitPaperNoClearanceRewardsCfg(GaitPaperRewardsCfg):
    """Paper baseline with only swing-foot clearance shaping removed."""

    gait_foot_clearance = Reward(
        func=gait.foot_clearance_penalty,
        weight=0.0,
        params={
            "target_height": 0.10,
            "height_offset": 0.02,
            "height_scale": 0.04,
            "reference_height": 0.0,
        },
    )


@configclass
class GaitPaperNoImpactRewardsCfg(GaitPaperRewardsCfg):
    """Paper baseline with only touchdown impact shaping removed."""

    gait_impact_velocity = Reward(
        func=gait.foot_impact_velocity_penalty,
        weight=0.0,
        params={"velocity_scale": 0.75, "force_threshold": 5.0},
    )


@configclass
class GaitPaperNoMaxForceRewardsCfg(GaitPaperRewardsCfg):
    """Paper baseline with only maximum-force shaping removed."""

    gait_max_contact_force = Reward(
        func=gait.maximum_contact_force_penalty,
        weight=0.0,
        params={"max_contact_force": 250.0, "force_scale": 150.0},
    )


@configclass
class WolfGaitFlatPaperEnvCfg(WolfGaitFlatBaseCfg):
    """Flat Wolf trot with all four additional contact-quality terms."""

    rewards: GaitPaperRewardsCfg = GaitPaperRewardsCfg()


@configclass
class WolfGaitFlatPaperEnvCfg_PLAY(WolfGaitFlatPaperEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class WolfGaitFlatPaperFreqRandEnvCfg(WolfGaitFlatBaseCfg):
    """Full paper-style flat trot with a 1--3 Hz episode-level command."""

    commands: GaitFrequencyRandomizedCommandsCfg = GaitFrequencyRandomizedCommandsCfg()
    rewards: GaitPaperRewardsCfg = GaitPaperRewardsCfg()


@configclass
class WolfGaitFlatPaperFreqRandEnvCfg_PLAY(WolfGaitFlatPaperFreqRandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class WolfGaitFlatPaperNoSlipEnvCfg(WolfGaitFlatBaseCfg):
    rewards: GaitPaperNoSlipRewardsCfg = GaitPaperNoSlipRewardsCfg()


@configclass
class WolfGaitFlatPaperNoSlipEnvCfg_PLAY(WolfGaitFlatPaperNoSlipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class WolfGaitFlatPaperNoClearanceEnvCfg(WolfGaitFlatBaseCfg):
    rewards: GaitPaperNoClearanceRewardsCfg = GaitPaperNoClearanceRewardsCfg()


@configclass
class WolfGaitFlatPaperNoClearanceEnvCfg_PLAY(WolfGaitFlatPaperNoClearanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class WolfGaitFlatPaperNoImpactEnvCfg(WolfGaitFlatBaseCfg):
    rewards: GaitPaperNoImpactRewardsCfg = GaitPaperNoImpactRewardsCfg()


@configclass
class WolfGaitFlatPaperNoImpactEnvCfg_PLAY(WolfGaitFlatPaperNoImpactEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class WolfGaitFlatPaperNoMaxForceEnvCfg(WolfGaitFlatBaseCfg):
    rewards: GaitPaperNoMaxForceRewardsCfg = GaitPaperNoMaxForceRewardsCfg()


@configclass
class WolfGaitFlatPaperNoMaxForceEnvCfg_PLAY(WolfGaitFlatPaperNoMaxForceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = 0


__all__ = [
    "WolfGaitFlatPaperEnvCfg",
    "WolfGaitFlatPaperEnvCfg_PLAY",
    "WolfGaitFlatPaperFreqRandEnvCfg",
    "WolfGaitFlatPaperFreqRandEnvCfg_PLAY",
    "WolfGaitFlatPaperNoSlipEnvCfg",
    "WolfGaitFlatPaperNoSlipEnvCfg_PLAY",
    "WolfGaitFlatPaperNoClearanceEnvCfg",
    "WolfGaitFlatPaperNoClearanceEnvCfg_PLAY",
    "WolfGaitFlatPaperNoImpactEnvCfg",
    "WolfGaitFlatPaperNoImpactEnvCfg_PLAY",
    "WolfGaitFlatPaperNoMaxForceEnvCfg",
    "WolfGaitFlatPaperNoMaxForceEnvCfg_PLAY",
]

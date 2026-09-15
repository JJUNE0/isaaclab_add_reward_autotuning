"""Flat-terrain fixed-trot Wolf gait task (V0)."""

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as Reward, SceneEntityCfg
from isaaclab.utils import configclass

from ...mdp import gait
from ..velocity_env_cfg import WolfGaitFlatEnvCfg as WolfGaitFlatBaseCfg


@configclass
class GaitRewardsCfg:
    """Flat gait costs; kept local so flat task does not depend on rough task code."""

    track_velocity = Reward(func=mdp.track_lin_vel_xy_exp, weight=2.0,
                            params={"command_name": "base_velocity", "std": 0.5})
    track_yaw = Reward(func=mdp.track_ang_vel_z_exp, weight=1.0,
                       params={"command_name": "base_velocity", "std": 0.5})
    termination = Reward(func=mdp.is_terminated, weight=-10.0)
    torque = Reward(func=mdp.joint_torques_l2, weight=-1.0e-5)
    acceleration = Reward(func=mdp.joint_acc_l2, weight=-1.0e-7)
    smoothness = Reward(func=mdp.action_rate_l2, weight=-0.02)
    joint_limits = Reward(func=mdp.joint_pos_limits, weight=-1.0)
    nonfoot_contact = Reward(func=mdp.undesired_contacts, weight=-1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
            "base_link", "Hip_.*", "Thigh_.*", "Shank_.*", "Ankle_.*"
        ]), "threshold": 1.0})

    gait_swing_force = Reward(
        func=gait.swing_force_penalty,
        weight=-0.20,
        params={"force_scale": 150.0},
    )
    gait_stance_velocity = Reward(
        func=gait.stance_velocity_penalty,
        weight=-0.20,
        params={"velocity_scale": 0.5},
    )


@configclass
class WolfGaitFlatEnvCfg(WolfGaitFlatBaseCfg):
    rewards: GaitRewardsCfg = GaitRewardsCfg()


@configclass
class WolfGaitFlatEnvCfg_PLAY(WolfGaitFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = 0

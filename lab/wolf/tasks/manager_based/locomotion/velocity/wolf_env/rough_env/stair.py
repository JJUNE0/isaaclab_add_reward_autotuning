"""Standard PPO oracle stair task. No MOO or learned terrain estimator."""
from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as Reward, CurriculumTermCfg, TerminationTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from ..velocity_env_cfg import WolfOracleEnvCfg
from ...mdp import oracle

@configclass
class RewardsCfg:
    track_velocity = Reward(func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5})
    track_yaw = Reward(func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5})
    termination = Reward(func=mdp.is_terminated, weight=-10.0)
    torque = Reward(func=mdp.joint_torques_l2, weight=-1.0e-5)
    acceleration = Reward(func=mdp.joint_acc_l2, weight=-1.0e-7)
    smoothness = Reward(func=mdp.action_rate_l2, weight=-0.02)
    joint_limits = Reward(func=mdp.joint_pos_limits, weight=-1.0)
    # Count every non-foot link above 1 N, including shanks/ankles. Do not
    # terminate on these added contacts; base/overturn termination stays separate.
    nonfoot_contact = Reward(func=mdp.undesired_contacts, weight=-1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
            "base_link", "Hip_.*", "Thigh_.*", "Shank_.*", "Ankle_.*"
        ]), "threshold": 1.0})

@configclass
class CurriculumCfg:
    terrain_levels = CurriculumTermCfg(func=oracle.terrain_curriculum)

@configclass
class WolfStairEnvCfg(WolfOracleEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        super().__post_init__()
        self.terminations.out_of_patch = TerminationTermCfg(func=oracle.out_of_patch, time_out=True)

@configclass
class WolfStairEnvCfg_PLAY(WolfStairEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

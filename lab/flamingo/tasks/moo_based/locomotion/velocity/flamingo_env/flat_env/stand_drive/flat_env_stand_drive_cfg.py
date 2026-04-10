# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
import lab.flamingo.tasks.moo_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    CurriculumCfg,
)


from lab.flamingo.isaaclab.isaaclab.managers import MOORewardTermCfg
from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_env.flat_env.stand_drive.feature_functions import(
    error_track_lin_vel_xy,
    error_track_ang_vel_z,
    error_base_height,
    error_flat_euler_rp,
    error_joint_deviation,
    error_joint_deviation_huber,
    action_rate_huber,
    error_track_pos_integral,
    action_rate_l2,
    joint_acc_l2,
    ActionRatePenalty,
    TorqueRatePenalty
)

from lab.flamingo.assets.flamingo.flamingo_rev03_2_0 import FLAMINGO_CFG  # isort: skip


@configclass
class FlamingoCurriculumCfg(CurriculumCfg):

    modify_base_velocity_range = CurrTerm(
        func=mdp.modify_base_velocity_range,
        params={
            "term_name": "base_velocity",
            "mod_range": {"lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-3.14, 3.14)},
            "num_steps": 25000,
        },
    )
@configclass
class FlamingoMOORewardsCfg:
    
    alpha = 1.0
    # 1) 선속도(xy) 트래킹 (오차 벡터 / std 정규화 내장)
    error_track_lin_vel_xy = MOORewardTermCfg(
        func= error_track_lin_vel_xy,
        params = {
            "command_name": "base_velocity",
            "scale": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        }
    )

    error_base_height = MOORewardTermCfg(
        func=error_base_height,
        params= {
            "target_height" : 0.615 , 
            "scale" : 1.0
        }
    )

    # 2) 각속도(yaw) 트래킹
    error_track_ang_vel_z = MOORewardTermCfg(
        func= error_track_ang_vel_z,
        params= {
            "command_name": "base_velocity",
            "scale": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        }
    )

    error_flat_euler_rp = MOORewardTermCfg(
        func = error_flat_euler_rp,
        params= {
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "scale"  : 1.0
        }
    )
    
    error_hip_deviatoin = MOORewardTermCfg(
        func = error_joint_deviation_huber,
        params = {"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]), "delta" : 0.01, "scale" : 1.0} 
    )
    
    error_shoulder_deviatoin = MOORewardTermCfg(
        func = error_joint_deviation_huber,
        params = {"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"]), "delta" : 0.01, "scale" : 1.0} 
    )
    
    error_leg_deviatoin = MOORewardTermCfg(
        func = error_joint_deviation_huber,
        params = {"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"]), "delta" : 0.01, "scale" : 1.0} 
    )
    
    action_smoothness_error = MOORewardTermCfg(func = ActionRatePenalty, params= {"delta" : 0.0, "scale" : 1.0})
    
    torque_rate_error = MOORewardTermCfg(
        func=TorqueRatePenalty,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "max_torque": 60.0,
            "scale": 1.0,
            "delta": 0.1},
    )

    
@configclass
class FlamingoFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    moo_rewards: FlamingoMOORewardsCfg = FlamingoMOORewardsCfg()
    # curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

       #! ************** scene & observations setup - 0 *********** !#
        self.scene.height_scanner = None
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None
        #! ********************************************************* !#

        
        self.observations.priv_extrio.is_discrete_terrain = None        
        self.commands.integral_position = None
        self.observations.priv_extrio.position_commands = None

        #! ********************************************************* !#

        self.domain.init_difficulty_level = 0.0

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        }


        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.0, 0.0),
            },
        }

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]
        self.terminations.terrain_out_of_bounds = None



@configclass
class FlamingoFlatEnvCfg_PLAY(FlamingoFlatEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.debug_vis = True
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.domain.init_difficulty_level = 1.0
        
        # observations
        #! ****************** Observations setup ******************* !#
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.2, 0.2)
        self.events.push_robot.interval_range_s = (5.5, 6.5)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0)},
        }

        # add base mass should be called here
        self.domain.randomize_mass.params["asset_cfg"].body_names = ["base_link"]
        self.domain.randomize_mass.params["mass_distribution_params"] = (-0.75, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]

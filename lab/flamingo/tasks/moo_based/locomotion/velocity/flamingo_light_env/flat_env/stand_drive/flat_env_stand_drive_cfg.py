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
from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_light_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.isaaclab.isaaclab.managers import MOORewardTermCfg
from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_light_env.flat_env.stand_drive.feature_functions import(
    error_track_lin_vel_xy,
    error_track_ang_vel_z,
    error_base_height,
    error_flat_euler_rp,
    error_joint_deviation,
    error_joint_deviation_huber,
    error_ang_vel_x,
    error_ang_vel_y,
    error_lin_vel_z,
    ActionRatePenalty
)

from lab.flamingo.assets.flamingo.flamingo_light_v1 import FLAMINGO_LIGHT_CFG  # isort: skip


@configclass
class FlamingoCurriculumCfg(CurriculumCfg):

    modify_base_velocity_range = CurrTerm(
        func=mdp.modify_base_velocity_range,
        params={
            "term_name": "base_velocity",
            "mod_range": {"lin_vel_x": (-0.5, 0.5), "ang_vel_z": (-1.5, 1.5)},
            "num_steps": 10000,
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

    error_track_ang_vel_z = MOORewardTermCfg(
        func= error_track_ang_vel_z,
        params= {
            "command_name": "base_velocity",
            "scale": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        }
    )

    error_base_height = MOORewardTermCfg(
        func=error_base_height,
        params= {
            "target_height" : 0.3, 
            "scale" : 10.0
        }
    )

    error_flat_euler_rp = MOORewardTermCfg(
        func = error_flat_euler_rp,
        params= {
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "scale"  : 1.0
        }
    )
    # error_ang_vel_x = MOORewardTermCfg(
    #     func= error_ang_vel_x,
    #     params= {
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "kernel" : "linear",
    #         "delta" : 0.0,
    #         "scale": 1.0,
    #     }
    # )
    
    # error_ang_vel_y = MOORewardTermCfg(
    #     func= error_ang_vel_y,
    #     params= {
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "kernel" : "linear",
    #         "delta" : 0.0,
    #         "scale": 1.0,
    #     }
    # )
    
    # error_lin_vel_z = MOORewardTermCfg(
    #     func= error_lin_vel_z,
    #     params= {
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "kernel" : "linear",
    #         "delta" : 0.0,
    #         "scale": 1.0,
    #     }
    # )

    # error_shoulder_deviation = MOORewardTermCfg(
    #     func = error_joint_deviation_huber,
    #     params = {"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"]), "delta" : 0.0, "scale" : 10.0} 
    # )
    
    action_smoothness_error = MOORewardTermCfg(func = ActionRatePenalty, params= {"delta" : 0.0, "scale" : 1.0})
    


@configclass
class FlamingoFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()
    moo_rewards: FlamingoMOORewardsCfg = FlamingoMOORewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # environment
        self.episode_length_s = 20.0
        # scene
        self.scene.robot = FLAMINGO_LIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None
        
        #! ****************** Observations setup - 0 *************** !#

        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        }

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
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

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            # "base_link",
            # "left_leg_link",
            # "right_leg_link",
        ]

@configclass
class FlamingoFlatEnvCfg_PLAY(FlamingoFlatEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.debug_vis = True
        # scene
        self.scene.robot = FLAMINGO_LIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # observations
        #! ****************** Observations setup - 0 *************** !#
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.2, 0.2)
        self.events.push_robot.interval_range_s = (5.5, 6.5)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        }
        # self.events.robot_wheel_stiffness_and_damping.params["stiffness_distribution_params"] = (1.0, 1.0)
        # self.events.robot_wheel_stiffness_and_damping.params["damping_distribution_params"] = (1.0, 1.0)
        # self.events.robot_joint_stiffness_and_damping.params["stiffness_distribution_params"] = (1.0, 1.0)
        # self.events.robot_joint_stiffness_and_damping.params["damping_distribution_params"] = (1.0, 1.0)
        # add base mass should be called here

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (1.5708, 1.5708)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            # "left_leg_link",
            # "right_leg_link",
        ]

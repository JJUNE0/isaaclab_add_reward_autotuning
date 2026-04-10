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
from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_4w4l_env.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.assets.flamingo.flamingo_4w4l_rev02_0_0 import FLAMINGO4W4L_CFG  # isort: skip

from lab.flamingo.isaaclab.isaaclab.managers import MOORewardTermCfg
from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_4w4l_env.flat_env.stand_drive.feature_functions import(
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
class Flamingo4w4lFlatEnvCfg(LocomotionVelocityRoughEnvCfg):

    moo_rewards: FlamingoMOORewardsCfg = FlamingoMOORewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO4W4L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None
        
        self.observations.none_stack_policy.height_scan = None

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
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
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ]

@configclass
class Flamingo4w4lFlatEnvCfg_PLAY(Flamingo4w4lFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 20.0
        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = True

        # scene
        self.scene.robot = FLAMINGO4W4L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None
        
        self.observations.none_stack_policy.height_scan = None
        
        #! ****************** Observations setup ******************* !#
        # disable randomization for play
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]

        # randomize actuator gains
        self.events.randomize_joint_actuator_gains = None

        self.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)
        self.events.push_robot = None
        
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ]

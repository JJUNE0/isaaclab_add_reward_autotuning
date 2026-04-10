
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
    LocomotionVelocityRoughEnvCfg,
    CurriculumCfg,
    CommandsCfg
)

from lab.flamingo.isaaclab.isaaclab.managers import MOORewardTermCfg
from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_env.rough_env.stand_drive.feature_functions import(
    error_track_lin_vel_xy,
    error_track_ang_vel_z,
    error_base_height,
    error_flat_euler_rp,
    joint_action_rate_huber,
    error_joint_deviation_huber,
    error_track_pos_integral,
    error_stuck_dance,
    error_same_foot_x_position,
    error_foot_clearance,
    error_ang_vel_x,
    error_ang_vel_y,
    error_lin_vel_z,
    applied_torque_limits,
    joint_soft_pos_limits,
    velocity_limits,
    error_both_feet_air_time,
    error_projected_gravity,
    ActionRatePenalty,
    TorqueRatePenalty
)


from lab.flamingo.assets.flamingo.flamingo_rev03_2_3 import FLAMINGO_CFG  # isort: skip


@configclass
class FlamingoCurriculumCfg(CurriculumCfg):
    modify_base_velocity_range = CurrTerm(
        func=mdp.modify_base_velocity_range,
        params={
            "term_name": "base_velocity",
            "mod_range": {"lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-3.0, 3.0)},
            "num_steps": 40000,
        },
        
    )
    
    
@configclass
class FlamingoMOORewardsCfg:
    
    alpha = 32.0
    
    # error_both_feet_air_time = MOORewardTermCfg(
    #     func=error_both_feet_air_time,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wheel_link", "right_wheel_link"]),
    #         "scale": 10.0
    #     }
    # )

    error_stuck_dance = MOORewardTermCfg(
        func=error_stuck_dance, # 위에서 만든 함수
        params={
            "command_name": "base_velocity",
            "stuck_threshold": 0.01, # 속도가 5cm/s 미만이면 갇힌 것
            "scale": 1.0
        }
    )
    
    # Target term
    error_track_lin_vel_xy = MOORewardTermCfg(
        func= error_track_lin_vel_xy,
        params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "command_name": "base_velocity",
            "kernel" : "linear",
            "scale": 1.0,
        }
    )

    error_track_pos_integral = MOORewardTermCfg(
        func = error_track_pos_integral,
        params ={ 
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "command_name": "integral_position", # 새로 만든 커맨드 이름
            "kernel": "linear",
            "delta" : 0.2,
            "scale": 10.0,
        }
    )

    error_track_ang_vel_z = MOORewardTermCfg(
        func= error_track_ang_vel_z,
        params= {
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "command_name": "base_velocity",
            "kernel" : "linear",
            "scale": 1.0,
        }
    )
    
    # error_same_foot_x_position = MOORewardTermCfg(
    #     func=error_same_foot_x_position,
    #     params= {
    #         "asset_cfg" : SceneEntityCfg("robot", body_names=["left_wheel_link","right_wheel_link"]),
    #         "scale" : 1.0
    #     }
    # )
    
    error_left_foot_clearance = MOORewardTermCfg(
        func=error_foot_clearance,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wheel_link"), 
            "base_sensor_cfg": SceneEntityCfg("height_scanner"), 
            "command_name": "base_velocity",
            "wheel_height": 0.1, 
            "scale": 1.0
        }
    )
    
    error_right_foot_clearance = MOORewardTermCfg(
        func=error_foot_clearance,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wheel_link"), 
            "base_sensor_cfg": SceneEntityCfg("height_scanner"), 
            "command_name": "base_velocity",
            "wheel_height": 0.1,
            "scale": 1.0
        }
    )
    
    # error_projected_gravity = MOORewardTermCfg(
    #     func = error_projected_gravity,
    #     params= {
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "scale"  : 1.0
    #     }
    # )
    
    # error_flat_euler_rp = MOORewardTermCfg(
    #     func = error_flat_euler_rp,
    #     params= {
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         #"sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "command_name": "base_velocity",
    #         "threshold" : 0.03,
    #         "scale"  : 1.0
    #     }
    # )
    
    error_ang_vel_x = MOORewardTermCfg(
        func= error_ang_vel_x,
        params= {
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "kernel" : "linear",
            "delta" : 1.0,
            "scale": 1.0,
        }
    )
    
    
    error_ang_vel_y = MOORewardTermCfg(
        func= error_ang_vel_y,
        params= {
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "kernel" : "linear",
            "delta" : 1.0,
            "scale": 1.0,
        }
    )
    
    error_lin_vel_z = MOORewardTermCfg(
        func= error_lin_vel_z,
        params= {
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "kernel" : "linear",
            "delta" : 0.1,
            "scale": 1.0,
        }
    )
    
    # Constraint term
    error_hip_deviatoin_huber = MOORewardTermCfg(
        func = error_joint_deviation_huber,
        params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]), 
            #"sensor_cfg": SceneEntityCfg("height_scanner"),
            "kernel" : "linear",
            "delta" : 0.0,
            "threshold" : 0.01, 
            "scale" : 1.0} 
    )
    
    error_shoulder_deviatoin_huber = MOORewardTermCfg(
        func = error_joint_deviation_huber,
        params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"]), 
            #"sensor_cfg": SceneEntityCfg("height_scanner"),
            "kernel" : "linear",
            "delta" : 0.0,
            "threshold" : 0.01, 
            "scale" : 1.0} 
    )

    error_leg_deviatoin_huber = MOORewardTermCfg(
        func = error_joint_deviation_huber,
        params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"]), 
            #"sensor_cfg": SceneEntityCfg("height_scanner"),
            "kernel" : "linear",
            "delta" : 0.0,
            "threshold" : 0.01, 
            "scale" : 1.0} 
    )
    action_smoothness_error = MOORewardTermCfg(func = ActionRatePenalty, params= {"delta" : 0.0, "scale" : 0.1})
    
    # velocity_limits = MOORewardTermCfg(
    #     func = velocity_limits, 
    #     params= {
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint",".*_shoulder_joint",".*_leg_joint"]),
    #         "thresh_hold" : 10.0, 
    #         "scale" : 1.0})
    
    # joint_soft_pos_limits = MOORewardTermCfg(
    #     func = joint_soft_pos_limits,
    #     params= {
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint",".*_shoulder_joint",".*_leg_joint"]),
    #         "soft_ratio" : 0.2, 
    #         "scale" : 1.0})

    # TODO
    # 토크 리밋 낮추고 apllied_torque_limits 없애기 랑 비교 해보기.
    # applied_torque_limits = MOORewardTermCfg(
    #     func = applied_torque_limits,
    #     params = {
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
    #         "scale" : 0.1}
    # )
    
    # torque_rate_error = MOORewardTermCfg(
    #     func=TorqueRatePenalty,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
    #         "max_torque": 60.0,
    #         "scale": 1.0,
    #         "delta": 0.1},
    # )
    


@configclass
class FlamingoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):

    moo_rewards: FlamingoMOORewardsCfg = FlamingoMOORewardsCfg()
    # curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

       #! ************** scene & observations setup - 0 *********** !#
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None

        # If train studeny, please set init_difficulty_level to 1.0
        # self.domain.init_difficulty_level = 0.0
        # self.domain.curriculum_threshold = 0.15
        #! ****************** Observations setup ****************** !#
        self.observations.none_stack_policy.event_commands = None
        
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        }
        # add base mass should be called here
        # self.domain.randomize_mass.params["asset_cfg"].body_names = ["base_link"]
        # self.domain.randomize_mass.params["mass_distribution_params"] = (-0.75, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
             "base_link",
            #".*_hip_link",
            #".*_shoulder_link",
            #".*_leg_link",
        ]


@configclass
class FlamingoRoughEnvCfg_PLAY(FlamingoRoughEnvCfg):
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
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # self.domain.init_difficulty_level = 0.0
        #! ****************** Observations setup ******************* !#
        # disable randomization for play
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # # add base mass should be called here
        # self.domain.randomize_mass.params["asset_cfg"].body_names = ["base_link"]
        # self.domain.randomize_mass.params["mass_distribution_params"] = (-3.0, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        self.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.0, 1.0), "z": (-1.0, 0.5)},
        }
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            #  ".*_hip_link",
            # ".*_shoulder_link",
            # ".*_leg_link",
        ]

        self.commands.base_velocity.resampling_time_range = (2.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) #(-math.pi / 3.0, math.pi / 3.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
 
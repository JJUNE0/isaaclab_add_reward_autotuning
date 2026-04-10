# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
import lab.flamingo.tasks.manager_based.locomotion.velocity.my_env.humanoid_env.flat_env.stand_walk.humanoid_walk_reward as mdp_drive

from lab.flamingo.tasks.manager_based.locomotion.velocity.my_env.humanoid_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.assets.flamingo.humanoid_rev2_1_0 import HUMANOID_CFG  # isort: skip


@configclass
class HumanoidRewardsCfg():
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.25,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.4,
    #     },
    # )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    is_alive = RewTerm(func=mdp.is_alive, weight = 1.0)

    lin_vel_z_l2 = None
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_yaw_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )

    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,  # default: -0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_range_l2,
        weight=0.5, # 상체가 주저앉지 않도록 강한 페널티 부여
        params={"min_height" : 0.9,
                "max_height" : 0.95,
                "in_range_reward" : 1.0}, # 로봇의 기본 높이에 맞춰 조절 (예: 0.88m)
    )
    
    # symmetrical_movement = RewTerm(
    #     func=mdp_drive.symmetrical_leg_movement_l2,
    #     weight=-0.1, # 너무 강하면 다양한 움직임을 방해할 수 있으므로 약하게 시작
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             # 좌우 순서를 맞춰서 정확하게 기입해야 합니다.
    #             joint_names=[
    #                 "left_hip_yaw_joint", "right_hip_yaw_joint",
    #                 "left_hip_roll_joint", "right_hip_roll_joint",
    #                 "left_hip_pitch_joint", "right_hip_pitch_joint",
    #                 "left_knee_joint", "right_knee_joint",
    #                 "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    #                 "left_ankle_roll_joint", "right_ankle_roll_joint",
    #             ],
    #         )
    #     },
    # )
    
    periodic_gait = RewTerm(
        func=mdp_drive.periodic_gait_reward,
        weight=0.5,  # 이 보상을 잘 따르도록 비교적 높은 가중치로 시작
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "gait_frequency": 1.0, # 초당 1.2 스텝의 보행 리듬 목표 (조절 가능)
            "phase_offset": 0.0,
            "command_name": "base_velocity"
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", ".*_shoulder_.*", ".*_elbow_.*"])})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", [".*_hip_.*", ".*_knee_joint", ".*_shoulder_.*", ".*_elbow_.*"])})  # default: -2.5e-7
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)  # default: -0.01


@configclass
class HumanoidFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    rewards: HumanoidRewardsCfg = HumanoidRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.observations.none_stack_critic.height_scan = None
        self.observations.none_stack_policy.height_scan = None
        
        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        }
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 3.0)

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
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis_link",
        ]


@configclass
class HumanoidFlatEnvCfg_PLAY(HumanoidFlatEnvCfg):
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
        self.scene.robot = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        #! ****************** Observations setup ******************* !#
        # disable randomization for play
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.0)

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

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis_link",
        ]

        self.commands.base_velocity.resampling_time_range = (2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
 
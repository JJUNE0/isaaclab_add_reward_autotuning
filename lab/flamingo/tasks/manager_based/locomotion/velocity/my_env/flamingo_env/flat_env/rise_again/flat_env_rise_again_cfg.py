# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
import lab.flamingo.tasks.manager_based.locomotion.velocity.my_env.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.velocity.my_env.flamingo_env.flat_env.velocity_env_cfg import (
    LocomotionVelocityFlatEnvCfg,
    CurriculumCfg,
)

from lab.flamingo.assets.flamingo.flamingo_rev01_5_2 import FLAMINGO_CFG  # isort: skip


@configclass
class FlamingoCurriculumCfg(CurriculumCfg):

    # modify_base_velocity_range = CurrTerm(
    #     func=mdp.modify_base_velocity_range,
    #     params={
    #         "term_name": "base_velocity",
    #         "mod_range": {"lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-3.14, 3.14)},
    #         "num_steps": 25000,
    #     },
    # )
    
    modify_terminataton_condition = CurrTerm(
        func=mdp.modify_terminataton_condition,
        params={
            "term_name": "base_contact",
            "link_name": [],
            "num_steps": 5000,
        })
    
    modify_reset_base_range = CurrTerm(
        func=mdp.modify_reset_base_range,
        params={
            "term_name": "reset_base",
            "mod_range": {          
                "pose_range": {"roll": (-math.pi, math.pi), "pitch": (-math.pi, math.pi)},
                "velocity_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-1.0, 1.0),
                    "roll": (-3.0, 3.0),
                    "pitch": (-3.0, 3.0),
                    }
            },
            "num_steps": 5000,
        }
    )


@configclass
class FlamingoRewardsCfg():
    # -- task
    
    # stop_stand_l2 = RewTerm(
    #     func=mdp.stop_stand_l2, weight=-0.002, params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint")}
    # )
    track_lin_vel_xy_exp_l2 = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp_l2, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_lin_vel_xy_exp_l1 = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp_l1, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )


    alive_reward = RewTerm(func=mdp.is_alive, weight = 1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-0.01)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.01)

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation_shoulder = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    )
    
    joint_deviation_leg = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"])},
    )
    

    dof_pos_limits_hip = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    dof_pos_limits_leg = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link",".*_leg_link" ,".*_shoulder_link", ".*_hip_link"]),
            "threshold": 1.0,
        },
    )
    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1,  # default: -0.1
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    # shoulder_align_l1 = RewTerm(
    #     func=mdp.joint_align_l1,
    #     weight=-0.1,  # default: -0.5
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    # )
    # leg_align_l1 = RewTerm(
    #     func=mdp.joint_align_l1,
    #     weight=-0.1,  # default: -0.5
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint")},
    # )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    rise_again = RewTerm(func=mdp.base_flat_exp_v2, weight=3.0, params={"temperature": 16, "asset_cfg": SceneEntityCfg("robot", body_names="base_link")})
    # base_height = RewTerm(
    #     func=mdp.base_height_adaptive_l2,
    #     weight=-25.0,
    #     params={
    #         "target_height": 0.36288,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         # "sensor_cfg": SceneEntityCfg("base_height_scanner"),
    #     },
    # )

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # default: -2.5e-7
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # default: -0.01



@configclass
class FlamingoFlatEnvCfg(LocomotionVelocityFlatEnvCfg):

    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()
    curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.episode_length_s = 5.0
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

       #! ************** scene & observations setup - 0 *********** !#
        # self.scene.height_scanner = None
        # self.scene.base_height_scanner = None
        # self.scene.left_wheel_height_scanner = None
        # self.scene.right_wheel_height_scanner = None
        # self.scene.left_mask_sensor = None
        # self.scene.right_mask_sensor = None

        self.observations.none_stack_critic.height_scan = None
        self.observations.none_stack_critic.base_height_scan = None
        self.observations.none_stack_critic.left_wheel_height_scan = None
        self.observations.none_stack_critic.right_wheel_height_scan = None
        self.observations.none_stack_critic.lift_mask = None
        #! ********************************************************* !#

        #! ****************** Observations setup ****************** !#
        self.observations.none_stack_policy.base_pos_z.params["sensor_cfg"] = None
        self.observations.none_stack_critic.base_pos_z.params["sensor_cfg"] = None

        self.observations.none_stack_policy.height_scan = None
        self.observations.none_stack_policy.base_lin_vel = None
        self.observations.none_stack_policy.base_pos_z = None
        self.observations.none_stack_policy.current_reward = None
        self.observations.none_stack_policy.is_contact = None
        self.observations.none_stack_policy.lift_mask = None

        self.observations.none_stack_policy.roll_pitch_commands = None
        self.observations.none_stack_policy.event_commands = None
        self.observations.none_stack_critic.roll_pitch_commands = None
        self.observations.none_stack_critic.event_commands = None
        
        self.observations.stack_critic.gait_phase = None
        self.observations.stack_critic.gait_command = None
        
        self.observations.none_stack_policy.gait_phase = None
        self.observations.none_stack_policy.gait_command = None
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.1, 0.1)
        # self.events.push_robot = True
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        }
        # add base mass should be called here
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 3.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14), "roll": (0,0), "pitch": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (0, 0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]



@configclass
class FlamingoFlatEnvCfg_PLAY(FlamingoFlatEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 5.0
        self.sim.render_interval = self.decimation
        self.debug_vis = True
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # observations
        #! ****************** Observations setup ******************* !#
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False
        #! ********************************************************* !#

        # reset_robot_joint_zero should be called here
        self.events.reset_robot_joints.params["position_range"] = (-0.2, 0.2)
        # Domain randomization events
        self.events.push_robot.interval_range_s = (2.5, 2.5)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0), "roll" : (-10.0,10.0), "pitch" : (-10.0,10.0), "yaw" : (0.0,0.0)}
        }
        # self.events.push_robot = None
        # self.events.randomize_joint_actuator_gains = None
        # self.events.randomize_leg_joint_actuator_gains = None
        # self.events.randomize_wheel_actuator_gains = None
        # self.events.randomize_com_positions = None
        # self.events.add_base_mass= None
  
        # # add base mass should be called here
        # self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        # self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)
        
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14), "roll": (-math.pi,math.pi), "pitch": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-3.0, 3.0),
                "pitch": (-3.0, 3.0),
                "yaw": (-0.0, 0.0),
            },
        }

        # commands
        self.commands.base_velocity.resampling_time_range = (4,6)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
        

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            # "base_link",
            # ".*_hip_link",
            # ".*_shoulder_link",
            # ".*_leg_link",
        ]

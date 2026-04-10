# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

import lab.flamingo.tasks.manager_based.locomotion.velocity.my_env.mdp as mdp

from lab.flamingo.tasks.manager_based.locomotion.velocity.sensors import LiftMaskCfg

##
# Pre-defined configs
##
from lab.flamingo.tasks.manager_based.locomotion.velocity.terrain_config.stair_config import ROUGH_TERRAINS_CFG
from lab.flamingo.assets.flamingo.flamingo_rev01_5_2 import FLAMINGO_CFG  # isort: skip
##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment = 'yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.07, size=[0.8, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment = 'yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.025, 0.025]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    left_wheel_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wheel_static_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment = 'yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.025, 0.025]), # (resolution=0.05, size=[0.025, 0.025])
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    right_wheel_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wheel_static_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment = 'yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.025, 0.025]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    left_mask_sensor = LiftMaskCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wheel_static_link",
        history_length=10,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment = 'yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.07, size=[0.35, 0.29]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        gradient_threshold = 0.03,
    )
    right_mask_sensor = LiftMaskCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wheel_static_link",
        history_length=10,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment = 'yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.07, size=[0.35, 0.29]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        gradient_threshold = 0.03,
        last_zero_num = 1,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75), intensity=4000.0
        ),  # Warmer color with higher intensity
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.53, 0.81, 0.98), intensity=1500.0
        ),  # Sky blue color with increased intensity
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    
    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # Fixed resampling time of 5 seconds
        debug_vis=False,  # No debug visualization needed
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(0.8, 1.6), # (1.5, 2.5),  # Gait frequency range [Hz]
            offsets=(0.5, 0.5),  # Phase offset range [0-1]
            durations=(0.5, 0.5),  # Contact duration range [0-1]
        ),
    )
    
    base_velocity = mdp.UniformVelocityWithZCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 8.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityWithZCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-2.0, 2.0), pos_z=(0.1931942, 0.3531942)
        ),
        initial_phase_time=2.0,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    hip_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_joint", "right_hip_joint", 
                     ],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    shoudler_leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_joint", "right_shoulder_joint", 
                     "left_leg_joint", "right_leg_joint"
                     ],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=20.0,
        use_default_offset=False,
        preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class StackPolicyCfg(ObsGroup):
        """Observations for Stack policy group."""
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.05, n_max=0.05),  # default: -0.04
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint", ".*_leg_joint"])
            },
        )
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.15)  # default: -1.5
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_link, noise=Unoise(n_min=-0.15, n_max=0.15), scale=0.25)  # default: -0.15
        # base_euler = ObsTerm(func=mdp.base_euler_angle_link, noise=Unoise(n_min=-0.125, n_max=0.125))  # default: -0.125
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # default: -0.05
        actions = ObsTerm(func=mdp.last_action)
    
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class NoneStackPolicyCfg(ObsGroup):
        """Observations for None-Stack policy group."""
        velocity_commands = ObsTerm(func=mdp.generated_scaled_commands, params={"command_name": "base_velocity", "scale": (2.0, 0.0, 0.25)})
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    stack_policy: StackPolicyCfg = StackPolicyCfg()
    none_stack_policy: NoneStackPolicyCfg = NoneStackPolicyCfg()
    stack_critic = None
    none_stack_critic = None

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_joint_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*shoulder_joint"]),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_leg_joint_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*leg_joint"),
            "stiffness_distribution_params": (0.8, 1.3),
            "damping_distribution_params": (0.8, 1.3),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_wheel_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*wheel_joint"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_com_positions,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_distribution_params": (-0.05, -0.02),
            "operation": "add",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

##
# Environment configuration
##

@configclass
class DiffusionPolicyLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards = None # It will be defined in the task
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Terrain curriculum
        self.curriculum.terrain_levels = None

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self. scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.base_height_scanner is not None:
            self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.left_wheel_height_scanner is not None:
            self.scene.left_wheel_height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.right_wheel_height_scanner is not None:    
            self.scene.right_wheel_height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.left_mask_sensor is not None:
            self.scene.left_mask_sensor.update_period = self.decimation * self.sim.dt
        if self.scene.right_mask_sensor is not None:
            self.scene.right_mask_sensor.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False



@configclass
class FlamingoDiffusionPolicyEnvCfg(DiffusionPolicyLocomotionEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # scene
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

       #! ************** scene & observations setup - 0 *********** !#
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



@configclass
class FlamingoDiffusionPolicyEnvCfg_PLAY(FlamingoDiffusionPolicyEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation * 2
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
        # self.events.push_robot = None       
        # self.events.push_robot.interval_range_s = (3.0, 5.0)
        # self.events.push_robot.params = {
        #     "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0), "roll" : (-10.0,10.0), "pitch" : (-10.0,10.0), "yaw" : (0.0,0.0)},
        # }
        # self.events.randomize_joint_actuator_gains = None
        # self.events.randomize_leg_joint_actuator_gains = None
        # self.events.randomize_wheel_actuator_gains = None
        # self.events.randomize_com_positions = None
        # self.events.add_base_mass= None
        # self.events.push_robot.interval_range_s = (5.5, 6.5)
        # self.events.push_robot.params = {
        #     "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0)},
        # }

        # # add base mass should be called here
        # self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        # self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.0)

        # physics material should be called here
        self.events.physics_material.params["asset_cfg"].body_names = [".*_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14), "roll": (-math.pi,math.pi), "pitch": (-math.pi, math.pi)},
        #     "velocity_range": {
        #         "x": (0.0, 0.0),
        #         "y": (0.0, 0.0),
        #         "z": (0.0, 0.0),
        #         "roll": (-3.0, 3.0),
        #         "pitch": (-3.0, 3.0),
        #         "yaw": (-0.0, 0.0),
        #     },
        # }
        

        # commands
        self.commands.base_velocity.resampling_time_range = (2,3)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)
        self.commands.base_velocity.initial_phase_time = 0.0

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            # "base_link"
            # ".*_hip_link",
            # ".*_shoulder_link",
            # ".*_leg_link",
        ]
        self.terminations.terrain_out_of_bounds = None
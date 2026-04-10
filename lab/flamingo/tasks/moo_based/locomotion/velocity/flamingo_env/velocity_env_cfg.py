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

import lab.flamingo.tasks.moo_based.locomotion.velocity.mdp as mdp

from lab.flamingo.tasks.moo_based.locomotion.velocity.sensors import LiftMaskCfg

##
# Pre-defined configs
##
from lab.flamingo.tasks.moo_based.locomotion.velocity.terrain_config.stair_config import ROUGH_TERRAINS_CFG

##
# Scene definition
##=


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
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.3, 0.3]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.025, 0.025]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    left_wheel_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wheel_static_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.025, 0.025]), # (resolution=0.05, size=[0.025, 0.025])
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    right_wheel_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wheel_static_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.025, 0.025]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    left_mask_sensor = LiftMaskCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wheel_static_link",
        history_length=10,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.07, size=[0.35, 0.29]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
        gradient_threshold = 0.03,
    )
    right_mask_sensor = LiftMaskCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wheel_static_link",
        history_length=10,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.07, size=[0.35, 0.29]),
        debug_vis=True,
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

    base_velocity = mdp.DiscreteAngularVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 5.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.DiscreteAngularVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-2.0, 2.0), pos_z=(0.1931942, 0.3531942)
        ),
        initial_phase_time=2.0,
    )
    
    integral_position = mdp.IntegralPositionCommandCfg(
            asset_name="robot",
            velocity_command_name="base_velocity",
            max_acceleration = 2.0,
            turn_threshold=0.1,
            pos_weight=0.8,
            resampling_time_range=(1.0e9, 1.0e9),
            feet_cfg=SceneEntityCfg(
                name="robot", 
                body_names=["left_wheel_link", "right_wheel_link"]
            ),
            debug_vis=True
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
    # TODO 40
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=40.0,
        use_default_offset=False,
        preserve_order=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class StackPolicyCfg(ObsGroup):
        
        # 관절 위치
        hip_shoulder_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_leg_gear,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"]), "gear_ratio": -1.5},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        hip_shoulder_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint"])},            
            scale=0.15,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_leg_gear, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"]), "gear_ratio": -1.5},            
            scale=0.15,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        wheel_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},            
            scale=0.15,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_link, noise=Unoise(n_min=-0.15, n_max=0.15), scale=0.25)
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True # Student는 노이즈 필수
            self.concatenate_terms = True

    @configclass
    class NoneStackPolicyCfg(ObsGroup):
        """[Current] 현재 시점 정보 (Blind)"""
        velocity_commands = ObsTerm(
            func=mdp.generated_scaled_commands, 
            params={"command_name": "base_velocity", "scale": (2.0, 0.0, 0.25)}
        )
        event_commands = ObsTerm(func=mdp.generated_partial_commands, params={"command_name": "event"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ExtrioceptionCfg(ObsGroup):
        position_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "integral_position"})
        
        # # 2. 지형 스캔 (Scan) -> "계단" 감지용
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), 'offset': 0.0},
            clip=(-1.0, 1.0),
        )
        
        # is_discrete_terrain = ObsTerm(
        #     func=mdp.is_discrete_terrain,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner"), 'threshold': 0.01},
        #     # noise=... (Teacher는 깨끗한 걸 보는 게 좋음, 선택 사항)
        # )
        
        is_contact = ObsTerm(
            func=mdp.is_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel_link"]), "threshold": 1.0},
        )
        
        
        def __post_init__(self):
            self.enable_corruption = False # Teacher는 정답을 봐야 함
            self.concatenate_terms = True
    
    @configclass
    class PhysicalCfg(ObsGroup):
      
        body_mass = ObsTerm(func=mdp.body_mass, params= {"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])}, scale = 0.1)
        
        body_com = ObsTerm(func= mdp.body_com, params= {"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])}, scale =10)
        
        #joint_friction = ObsTerm(func = mdp.joint_friction, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])}, scale=10)
        
        actuator_kp = ObsTerm(func = mdp.actuator_kp, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])}, scale=0.01)       
        
        actuator_kd = ObsTerm(func = mdp.actuator_kd, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])}, scale=1.0)
        
        def __post_init__(self):
            self.enable_corruption = False # Teacher는 정답을 봐야 함
            self.concatenate_terms = True
    
    @configclass
    class ProprioceptionCfg(ObsGroup): # (이름 변경: NoneStackCritic -> Privileged)
        """[Privileged] 시뮬레이터에서만 알 수 있는 정보"""

        base_lin_vel_x = ObsTerm(func=mdp.base_lin_vel_x_link, scale=2.0)
        base_lin_vel_y = ObsTerm(func=mdp.base_lin_vel_y_link)
        base_lin_vel_z = ObsTerm(func=mdp.base_lin_vel_z_link, scale=0.25)
                
        # hip_shoulder_joint_pos_clean = ObsTerm(
        #     func=mdp.joint_pos,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint"])},
        #     # noise 없음
        # )
        # leg_joint_pos_clean = ObsTerm(
        #     func=mdp.joint_pos_leg_gear,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"]), "gear_ratio": -1.5},
        # )
        # hip_shoulder_joint_vel_clean = ObsTerm(
        #     func=mdp.joint_vel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_shoulder_joint"])},
        #     scale=0.15,
        # )
        # joint_vel_clean = ObsTerm(
        #     func=mdp.joint_vel_leg_gear,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"]), "gear_ratio": -1.5},
        #     scale=0.15,
        # )
        # wheel_joint_vel_clean = ObsTerm(
        #     func=mdp.joint_vel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])},
        #     scale=0.15,
        # )

        # # --- IMU도 clean 버전으로 하나 복사해도 됨 ---
        # base_ang_vel_clean = ObsTerm(
        #     func=mdp.base_ang_vel_link,
        #     scale=0.25,
        # )
        # base_projected_gravity_clean = ObsTerm(
        #     func=mdp.projected_gravity,
        # )
        
        # torque = ObsTerm(
        #     func=mdp.joint_torques,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"])},
        #     scale=0.015,

        # )

        
        def __post_init__(self):
            self.enable_corruption = False # Teacher는 정답을 봐야 함
            self.concatenate_terms = True

    # =========================================================================
    # 3. 그룹 등록
    # =========================================================================
    
    # Student용 (Blind)
    stack_policy: StackPolicyCfg = StackPolicyCfg()
    none_stack_policy: NoneStackPolicyCfg = NoneStackPolicyCfg()
    
    priv_extrio = ExtrioceptionCfg()
    priv_prio = ProprioceptionCfg()
    priv_physical = PhysicalCfg()


@configclass
class DomainManagerCfg:
    
    # D 확률이 이보다 높으면 난이도 증가
    init_difficulty_level: float = 0.0
    initial_level: float = 0.05  # Step size for the first increment
    
    tau: float = 0.8
    kappa: float = 0.1
    confidence_level: float = 0.95
    
    w_fm: float = 0.8
    w_gail: float = 0.2
    
    target_threshold_start: float = 0.5  
    target_threshold_end: float = 0.85
    
    min_gate_samples : int = 5
    gate_window : int = 20
    
    
    randomize_mass: EventTerm = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset", 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "operation": "add",
            "mass_distribution_params": (-3.0, 3.0), 
        }
    )

    randomize_com: EventTerm = EventTerm(
        func=mdp.randomize_com_positions,
        mode="reset",
            params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        "com_distribution_params": (-0.1, 0.1), # x,y,z 모두 +/- 5cm 변동
        "operation": "abs",
        "distribution": "uniform",
    },
    )

    randomize_gains: EventTerm = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "operation": "scale",
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "distribution": "log_uniform"
        }
    )

    randomize_joints: EventTerm = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "operation": "add",
            "friction_distribution_params": (0.0, 0.05), 
            "armature_distribution_params": (0.0, 0.01), 
            "distribution": "uniform"
        }
    )


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     # startup
#     physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "static_friction_range": (0.8, 1.0),
#             "dynamic_friction_range": (0.6, 0.8),
#             "restitution_range": (0.0, 0.0),
#             "num_buckets": 64,
#         },
#     )

#     randomize_joint_actuator_gains = EventTerm(
#         func=mdp.randomize_actuator_gains,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*shoulder_joint"]),
#             "stiffness_distribution_params": (0.5, 1.5),
#             "damping_distribution_params": (0.5, 1.5),
#             "operation": "scale",
#             "distribution": "log_uniform",
#         },
#     )

#     randomize_leg_joint_actuator_gains = EventTerm(
#         func=mdp.randomize_actuator_gains,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=".*leg_joint"),
#             "stiffness_distribution_params": (0.5, 1.5),
#             "damping_distribution_params": (0.5, 1.5),
#             "operation": "scale",
#             "distribution": "log_uniform",
#         },
#     )

#     randomize_wheel_actuator_gains = EventTerm(
#         func=mdp.randomize_actuator_gains,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=".*wheel_joint"),
#             "stiffness_distribution_params": (0.5, 1.5),
#             "damping_distribution_params": (0.5, 1.5),
#             "operation": "scale",
#             "distribution": "log_uniform",
#         },

#     )

#     randomize_com_positions = EventTerm(
#         func=mdp.randomize_com_positions,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
#             "com_distribution_params": (-0.1, -0.1),
#             "operation": "add",
#         },
#     )

#     add_base_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
#             "mass_distribution_params": (-3.0, 3.0),
#             "operation": "add",
#         },
#     )

#     reset_base = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
#             "velocity_range": {
#                 "x": (-0.5, 0.5),
#                 "y": (-0.5, 0.5),
#                 "z": (-0.5, 0.5),
#                 "roll": (-0.5, 0.5),
#                 "pitch": (-0.5, 0.5),
#                 "yaw": (-0.5, 0.5),
#             },
#         },
#     )

#     reset_robot_joints = EventTerm(
#         func=mdp.reset_joints_by_offset,
#         mode="reset",
#         params={
#             "position_range": (-0.0, 0.0),
#             "velocity_range": (0.0, 0.0),
#         },
#     )

#     # reset_robot_joints = EventTerm(
#     #     func=mdp.reset_joints_by_scale,
#     #     mode="reset",
#     #     params={
#     #         "position_range": (0.5, 1.5),
#     #         "velocity_range": (0.0, 0.0),
#     #     },
#     # )

#     # interval
#     push_robot = EventTerm(
#         func=mdp.push_by_setting_velocity,
#         mode="interval",
#         interval_range_s=(10.0, 15.0),
#         params={
#             "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
#         },
#     )

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
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 1.0},
        time_out=True,
    )
    position_command_out_of_bounds = DoneTerm(
        func=mdp.position_command_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "integral_position", 'threshold': 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
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
    
    domain = DomainManagerCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        # # Terrain curriculum
        # self.curriculum.terrain_levels = None

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
class LocomotionVelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
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
    
    domain = DomainManagerCfg()

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
        self.terminations.position_command_out_of_bounds = None

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

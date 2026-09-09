# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TITA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/cocel/Desktop/Research/DDT_Lab/tita_gluon_urdf/urdf/tita.urdf",
        activate_contact_sensors=True,
        fix_base=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.385),
        joint_pos={
            "joint_left_leg_1": 0.0,
            "joint_left_leg_2": 0.8,
            "joint_left_leg_3": -1.5,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_1": 0.0,
            "joint_right_leg_2": 0.8,
            "joint_right_leg_3": -1.5,
            "joint_right_leg_4": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.8,
    actuators={
        "joints_hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_leg_1"],
            effort_limit=55.0,
            velocity_limit=20.0,
            min_delay=0,
            max_delay=4,
            stiffness={".*_leg_1": 70.0},
            damping={".*_leg_1": 0.7},
            friction={".*_leg_1": 0.0},
            armature={".*_leg_1": 0.01},
        ),
        "joints_shoulder": DelayedPDActuatorCfg(
            joint_names_expr=[".*_leg_2"],
            effort_limit=55.0,
            velocity_limit=20.0,
            min_delay=0,
            max_delay=4,
            stiffness={".*_leg_2": 70.0},
            damping={".*_leg_2": 0.7},
            friction={".*_leg_2": 0.0},
            armature={".*_leg_2": 0.01},
        ),
        "joints_leg": DelayedPDActuatorCfg(
            joint_names_expr=[".*_leg_3"],
            effort_limit=55.0,
            velocity_limit=20.0,
            min_delay=0,
            max_delay=4,
            stiffness={".*_leg_3": 70.0},
            damping={".*_leg_3": 0.7},
            friction={".*_leg_3": 0.0},
            armature={".*_leg_3": 0.01},
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*_leg_4"],
            effort_limit=36.0,
            velocity_limit=50.0,
            min_delay=0,
            max_delay=4,
            stiffness={".*_leg_4": 0.0},
            damping={".*_leg_4": 0.55},
            friction={".*_leg_4": 0.0},
            armature={".*_leg_4": 0.01},
        ),
    },
)

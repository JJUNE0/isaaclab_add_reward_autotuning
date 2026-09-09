# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import lab.flamingo.tasks.moo_based.locomotion.velocity.mdp as mdp
from lab.flamingo.isaaclab.isaaclab.managers import MOORewardTermCfg
from lab.flamingo.assets.tita.tita import TITA_CFG

from lab.flamingo.tasks.moo_based.locomotion.velocity.tita_env.velocity_env_cfg import TitaVelocityFlatEnvCfg
from lab.flamingo.tasks.moo_based.locomotion.velocity.tita_env.flat_env.stand_drive.feature_functions import (
    error_track_lin_vel_xy,
    error_track_ang_vel_z,
    error_base_height,
    error_flat_euler_rp,
    error_joint_deviation_huber,
    error_shoulder_deviation_height_aware,
    error_leg_deviation_height_aware,
    error_lin_vel_z,
    error_ang_vel_y,
    error_ang_vel_x,
    ActionRatePenalty,
    TorqueRatePenalty,
    PoseIntegralTrackingError,
)


@configclass
class TitaMOORewardsCfg:
    alpha = 1.0

    # 1) linear velocity (xy) tracking error: 2D
    error_track_lin_vel_xy = MOORewardTermCfg(
        func=error_track_lin_vel_xy,
        params={
            "command_name": "base_velocity",
            "scale": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # 2) base height error: 1D
    error_base_height = MOORewardTermCfg(
        func=error_base_height,
        params={"target_height": 0.385, "scale": 1.0},
    )

    # 3) angular velocity (z) tracking error: 1D
    error_track_ang_vel_z = MOORewardTermCfg(
        func=error_track_ang_vel_z,
        params={
            "command_name": "base_velocity",
            "scale": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # 4) base roll/pitch error: 2D
    error_flat_euler_rp = MOORewardTermCfg(
        func=error_flat_euler_rp,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "scale": 1.0,
        },
    )

    # 5) hip joint deviation Huber: 2D
    error_hip_deviation = MOORewardTermCfg(
        func=error_joint_deviation_huber,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_1"]),
            "delta": 0.0,
            "scale": 1.0,
        },
    )

    # 6) shoulder joint deviation Huber (delta=0.1 rad ~= 5.7deg): 2D
    error_shoulder_deviation = MOORewardTermCfg(
        func=error_joint_deviation_huber,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_2"]),
            "delta": 0.1,
            "scale": 1.0,
        },
    )

    # 7) leg joint deviation Huber (delta=0.1 rad ~= 5.7deg): 2D
    error_leg_deviation = MOORewardTermCfg(
        func=error_joint_deviation_huber,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_3"]),
            "delta": 0.1,
            "scale": 1.0,
        },
    )

    # 8) action velocity + acceleration penalty: 1D
    action_smoothness_error = MOORewardTermCfg(
        func=ActionRatePenalty,
        params={"delta": 0.0, "scale": 1.0},
    )

    # 9) normalized torque-rate penalty: 1D
    torque_rate_error = MOORewardTermCfg(
        func=TorqueRatePenalty,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_.*"]),
            "max_torque": 60.0,
            "scale": 1.0,
            "delta": 0.1,
        },
    )

    # 10) integrated base_velocity pose vs. actual base pose error (xy + yaw): 3D
    error_pose_integral = MOORewardTermCfg(
        func=PoseIntegralTrackingError,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "delta": 0.0,
            "scale": 1.0,
        },
    )

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


@configclass
class TitaFlatEnvCfg(TitaVelocityFlatEnvCfg):
    moo_rewards: TitaMOORewardsCfg = TitaMOORewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Scene
        self.scene.robot = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # Clean unused scanners/sensors
        self.scene.height_scanner = None
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None

        self.observations.priv_extrio.is_discrete_terrain = None
        self.commands.integral_position = None
        self.observations.priv_extrio.position_commands = None
        self.observations.priv_extrio.height_scan = None
        self.observations.none_stack_policy.event_commands = None

        self.domain.init_difficulty_level = 0.0

        # Events & Randomization (Parity with Flamingo flat)
        self.events.push_robot.interval_range_s = (13.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        }

        self.events.physics_material.params["asset_cfg"].body_names = [".*_leg_.*", "base_link"]
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
                "yaw": (0.0, 0.0),
            },
        }

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_leg_1",
            ".*_leg_2",
            ".*_leg_3",
        ]
        self.terminations.terrain_out_of_bounds = None


@configclass
class TitaFlatEnvCfg_GTBaseVel(TitaFlatEnvCfg):
    """Oracle test: adds ground-truth base_lin_vel (x/y/z) and base_height to
    the actor's own obs (stack_policy), on top of the flamingo-style
    obs/action rollback.

    Purpose: check whether giving the actor perfect base-velocity/height
    knowledge actually helps before investing in a learned estimator (MLP +
    auxiliary supervised loss, trained to predict both quantities jointly).
    If this oracle doesn't help, a noisy estimator won't either. Not
    deployable as-is (no real robot has ground-truth base velocity/height)
    -- purely a training-time upper-bound probe.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.stack_policy.base_lin_vel_x = ObsTerm(func=mdp.base_lin_vel_x_link, scale=2.0)
        self.observations.stack_policy.base_lin_vel_y = ObsTerm(func=mdp.base_lin_vel_y_link)
        self.observations.stack_policy.base_lin_vel_z = ObsTerm(func=mdp.base_lin_vel_z_link, scale=0.25)
        self.observations.stack_policy.base_height = ObsTerm(func=mdp.base_pos_z_rel_link, scale=1.0)


@configclass
class TitaFlatEnvCfg_GTBaseVel_TargetHeight(TitaFlatEnvCfg_GTBaseVel):
    """GT base_lin_vel oracle + a randomized commanded base height instead of
    the fixed 0.385 m target -- error_base_height now tracks
    commands.base_velocity's pos_z command directly. The height-tracking
    term's scale is raised since its raw magnitude (~0.1m typical) is small
    next to the other error terms' natural scale.

    Shoulder/leg deviation targets are height-aware (FK-derived, see
    error_shoulder/leg_deviation_height_aware): fixed default_joint_pos
    deviation targets don't make sense once the commanded height varies --
    the policy needs the legs to actually reconfigure per height, not just
    hold the 0.385m-default pose. Both this and error_base_height now
    correctly fall back to the natural 0.385m target during the command's
    initial_phase_time window instead of chasing literal height=0 (see
    _init_phase_safe_height) -- ablating shoulder/leg deviation entirely
    didn't fix the earlier weirdness on its own, so it's back in with that
    fix applied."""

    def __post_init__(self):
        super().__post_init__()
        # Start narrow (centered on the natural standing height, 0.35 +/- 0.03) and widen to
        # the full (0.25, 0.45) range over iterations 1000->5000 (num_steps_per_env=24, so
        # start_step=1000*24=24000, end_step=5000*24=120000) -- lets the policy master basic
        # standing/init-phase behavior at an easy height before the range widens, rather than
        # demanding the full range from step 0.
        self.commands.base_velocity.ranges.pos_z = (0.32, 0.38)
        self.curriculum.widen_height_range = CurrTerm(
            func=mdp.linearly_widen_command_range,
            params={
                "term_name": "base_velocity",
                "key": "pos_z",
                "start_range": (0.32, 0.38),
                "end_range": (0.25, 0.45),
                "start_step": 24000,
                "end_step": 120000,
            },
        )
        # scale dropped from 5.0: error_base_height now operates on height normalized to
        # roughly [-1, 1] (see feature_functions_common.normalize_height), which already
        # amplifies the raw ~0.1m-scale error by ~11x (1 / _HEIGHT_NORM_HALF_RANGE).
        self.moo_rewards.error_base_height.params = {"command_name": "base_velocity", "scale": 1.0}
        self.moo_rewards.error_shoulder_deviation.func = error_shoulder_deviation_height_aware
        self.moo_rewards.error_shoulder_deviation.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_2"]),
            "command_name": "base_velocity",
            "delta": 0.1,
            "scale": 1.0,
        }
        self.moo_rewards.error_leg_deviation.func = error_leg_deviation_height_aware
        self.moo_rewards.error_leg_deviation.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_3"]),
            "command_name": "base_velocity",
            "delta": 0.1,
            "scale": 1.0,
        }


@configclass
class TitaFlatEnvCfg_RMA_TargetHeight(TitaFlatEnvCfg_GTBaseVel_TargetHeight):
    """RMA observation contract with no simulator-only actor inputs.

    The actor keeps only robot-observable proprioception plus the current
    command. Ground-truth base velocity and measured base height are exposed
    exclusively through ``priv_prio`` for the Teacher encoder and critic.
    """

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.observations.priv_prio.base_height = ObsTerm(func=mdp.base_pos_z_rel_link, scale=1.0)
        self.observations.stack_policy.base_height = None
        self.observations.stack_policy.base_lin_vel_x = None
        self.observations.stack_policy.base_lin_vel_y = None
        self.observations.stack_policy.base_lin_vel_z = None


@configclass
class TitaFlatEnvCfg_Estimator_TargetHeight(TitaFlatEnvCfg_GTBaseVel_TargetHeight):
    """Base-velocity-and-height-estimator variant (v3): identical to
    TitaFlatEnvCfg_GTBaseVel_TargetHeight (hip-only deviation, height reward
    scale=5.0, randomized target height 0.23-0.42m) except the actor no
    longer sees ground-truth base_lin_vel OR base_height directly -- those 4
    obs terms are removed from stack_policy. Ground truth is still available
    (via the "priv_prio" observation group -- see TitaVelocityFlatEnvCfg's
    ObservationsCfg.ProprioceptionCfg, extended here with base_height as its
    4th term -- which CoRlVecEnvWrapper automatically appends to critic_obs)
    as the supervised target for ActorCriticWithEstimator's estimator MLP
    (estimator_output_dim=4), trained jointly via MOOPPO's
    estimator_loss_coef-weighted auxiliary MSE loss. Use with
    TitaMOOPPORunnerCfg_Flat_Stand_Drive_Estimator."""

    def __post_init__(self):
        super().__post_init__()
        # priv_prio's declaration order is [base_lin_vel_x, base_lin_vel_y, base_lin_vel_z];
        # base_height is appended here as its 4th term. MOOPPO's estimator loss slices the
        # LAST estimator_output_dim columns of critic_obs as the supervised target ("priv_prio"
        # sorts after "priv_extrio"/"priv_physical", so it's the last priv_ group, and within
        # it base_height -- added last -- is the last column), so this ordering must be kept
        # in sync with TitaMOOPPORunnerCfg_Flat_Stand_Drive_Estimator's estimator_output_dim=4.
        self.observations.priv_prio.base_height = ObsTerm(func=mdp.base_pos_z_rel_link, scale=1.0)
        self.observations.stack_policy.base_height = None
        self.observations.stack_policy.base_lin_vel_x = None
        self.observations.stack_policy.base_lin_vel_y = None
        self.observations.stack_policy.base_lin_vel_z = None


@configclass
class TitaFlatEnvCfg_PLAY(TitaFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.debug_vis = True
        self.scene.robot = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.domain.init_difficulty_level = 0.0

        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False

        self.events.push_robot.interval_range_s = (5.5, 6.5)
        self.events.push_robot.params = {
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        }
        self.events.add_base_inertia = None
        self.events.add_base_com = None

        self.domain.randomize_mass.params["asset_cfg"].body_names = ["base_link"]
        self.domain.randomize_mass.params["mass_distribution_params"] = (-0.75, 3.0)

        self.events.physics_material.params["asset_cfg"].body_names = [".*_leg_.*", "base_link"]
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.8)

        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)
        self.commands.base_velocity.ranges.pos_z = (0.0, 0.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_leg_1",
            ".*_leg_2",
            ".*_leg_3",
        ]

# SPDX-License-Identifier: BSD-3-Clause
# POSTECH Flamingo Lab, 2025
# Common feature functions shared across all Flamingo environments.
# Environment-specific overrides live in each env's feature_functions.py.

from __future__ import annotations
import torch
import torch.nn.functional as F
import math
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ------------------------------------------------------------------
# Kernel utility
# ------------------------------------------------------------------

def apply_kernel(error: torch.Tensor, kernel: str = "linear", scale: float = 1.0, temperature: float = 4.0) -> torch.Tensor:
    if kernel == "linear":
        return error * scale
    elif kernel == "tanh":
        return torch.tanh(error * temperature) * scale
    elif kernel == "exp":
        return (1.0 - torch.exp(-torch.abs(error) * temperature)) * torch.sign(error) * scale
    else:
        return error * scale


# ------------------------------------------------------------------
# Velocity tracking (flat / default versions)
# ------------------------------------------------------------------

def error_track_lin_vel_xy(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    vel_xy = asset.data.root_link_lin_vel_b[:, :2]
    return (vel_xy - cmd_xy) * scale


def error_track_ang_vel_z(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_z = env.command_manager.get_command(command_name)[:, 2:3]
    vel_z = asset.data.root_link_ang_vel_b[:, 2:3]
    return (vel_z - cmd_z) * scale


def error_lin_vel_z(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    delta: float = 0.5,
    kernel: str = "tanh",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_z = asset.data.root_link_lin_vel_b[:, 2:3]
    if delta == 0.0:
        error = vel_z
    else:
        error = F.huber_loss(vel_z, torch.zeros_like(vel_z), reduction='none', delta=delta)
    return apply_kernel(error, kernel, scale)


def error_ang_vel_y(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    delta: float = 0.5,
    kernel: str = "tanh",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_y = asset.data.root_link_ang_vel_b[:, 1]
    if delta == 0.0:
        error = ang_vel_y
    else:
        error = F.huber_loss(ang_vel_y, torch.zeros_like(ang_vel_y), reduction='none', delta=delta)
    return apply_kernel(error, kernel, scale)


def error_ang_vel_x(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    delta: float = 0.5,
    kernel: str = "tanh",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_x = asset.data.root_link_ang_vel_b[:, 0]
    if delta == 0.0:
        error = ang_vel_x
    else:
        error = F.huber_loss(ang_vel_x, torch.zeros_like(ang_vel_x), reduction='none', delta=delta)
    return apply_kernel(error, kernel, scale)


# ------------------------------------------------------------------
# Position tracking
# ------------------------------------------------------------------

def error_track_pos_integral(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "integral_position",
    kernel: str = "tanh",
    scale: float = 1.0,
) -> torch.Tensor:
    pos_error_xy = env.command_manager.get_command(command_name)
    return apply_kernel(pos_error_xy, kernel, scale)


# ------------------------------------------------------------------
# Base pose (flat / default versions)
# ------------------------------------------------------------------

_NATURAL_STANDING_HEIGHT = 0.35

# Affine map of the trained pos_z command range ([0.25, 0.45] m -- see
# CommandsCfg.base_velocity.ranges.pos_z) onto [-1, 1]. Kept inside the FK fit's clamp range
# (_FK_HEIGHT_CLAMP = (0.22, 0.48)) -- going below 0.22 previously let error_base_height demand
# a lower actual height than the (frozen-at-0.22) FK-based shoulder/leg deviation target could
# represent, and the policy exploited the gap with a non-canonical splayed-leg pose to hit the
# height target without matching the frozen joint pose. Raw meter values are a very low-
# magnitude signal (a 0.20m span) for both the policy's input and the height-tracking reward;
# this rescaling is applied only at the point of use (observation encoding, reward error) -- the
# command's own internal representation (`vel_command_b`) stays in real meters throughout, so FK
# fits, dead-reckoning integration, and the MuJoCo sim2sim bridge are unaffected.
_HEIGHT_NORM_CENTER = 0.35        # (0.25 + 0.45) / 2
_HEIGHT_NORM_HALF_RANGE = 0.10    # (0.45 - 0.25) / 2


def normalize_height(height_m: torch.Tensor) -> torch.Tensor:
    """Map a real height (meters) in the trained pos_z range to roughly [-1, 1]."""
    return (height_m - _HEIGHT_NORM_CENTER) / _HEIGHT_NORM_HALF_RANGE


def _init_phase_safe_height(height_cmd: torch.Tensor, fallback: float = _NATURAL_STANDING_HEIGHT) -> torch.Tensor:
    """UniformVelocityWithZCommand (mdp/commands/velocity_command.py) masks
    ALL command channels -- including height -- to exactly 0.0 during its
    initial_phase_time window after every reset (5s for TITA). 0.0 is not a
    physically meaningful height target (never sampled otherwise -- the real
    range is 0.25-0.45), so substitute the natural default standing height
    during that window instead of feeding literal 0.0 into height-conditioned
    reward targets (which would otherwise demand an extreme crouch)."""
    return torch.where(height_cmd.abs() < 1e-6, torch.full_like(height_cmd, fallback), height_cmd)


def error_base_height(
    env: ManagerBasedRLEnv,
    target_height: float | None = None,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]

    if command_name is not None:
        tgt = _init_phase_safe_height(env.command_manager.get_command(command_name)[:, 3])
    elif target_height is not None:
        tgt = torch.full_like(base_z, float(target_height))
    else:
        tgt = base_z.detach()

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)
        tgt = tgt + ground_z

    return (normalize_height(base_z) - normalize_height(tgt)) * scale


def error_flat_euler_rp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi
    return torch.stack((roll, pitch), dim=1) * scale


# ------------------------------------------------------------------
# Joint deviation (default / flat versions)
# ------------------------------------------------------------------

def error_joint_deviation_huber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    delta: float = 0.5,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if delta != 0.0:
        error_vector = F.huber_loss(
            angle_error, torch.zeros_like(angle_error), reduction='none', delta=delta
        )
    else:
        error_vector = angle_error
    return error_vector * scale


# Shoulder-angle target as a function of commanded base height, fit against a
# kinematic (hip=0, foot directly under hip) IK solve over the trained
# command range (~0.25-0.45m) -- see docs/exp/ for the derivation. Under that
# same foot-under-hip constraint the leg angle is always exactly -2x the
# shoulder angle, so leg target is derived from shoulder target rather than
# fit independently.
_FK_SHOULDER_COEFFS = (-25.62687550843755, 21.257766050024646, -8.709666591280762, 2.4161909585596337)
_FK_HEIGHT_CLAMP = (0.22, 0.48)


def _fk_shoulder_target_from_height(height: torch.Tensor) -> torch.Tensor:
    h = torch.clamp(height, *_FK_HEIGHT_CLAMP)
    c3, c2, c1, c0 = _FK_SHOULDER_COEFFS
    return c3 * h**3 + c2 * h**2 + c1 * h + c0


def error_shoulder_deviation_height_aware(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    delta: float = 0.5,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    height_cmd = _init_phase_safe_height(env.command_manager.get_command(command_name)[:, 3])
    shoulder_target = _fk_shoulder_target_from_height(height_cmd).unsqueeze(-1)
    angle_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - shoulder_target
    if delta != 0.0:
        error_vector = F.huber_loss(
            angle_error, torch.zeros_like(angle_error), reduction='none', delta=delta
        )
    else:
        error_vector = angle_error
    return error_vector * scale


def error_leg_deviation_height_aware(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    delta: float = 0.5,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    height_cmd = _init_phase_safe_height(env.command_manager.get_command(command_name)[:, 3])
    leg_target = (-2.0 * _fk_shoulder_target_from_height(height_cmd)).unsqueeze(-1)
    angle_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - leg_target
    if delta != 0.0:
        error_vector = F.huber_loss(
            angle_error, torch.zeros_like(angle_error), reduction='none', delta=delta
        )
    else:
        error_vector = angle_error
    return error_vector * scale


def error_joint_symmetry_huber(
    env: ManagerBasedRLEnv,
    left_asset_cfg: SceneEntityCfg,
    right_asset_cfg: SceneEntityCfg,
    delta: float = 0.5,
    scale: float = 1.0,
) -> torch.Tensor:
    """Left/right joint-position mismatch, paired 1:1 by list order (each
    ``left_asset_cfg``/``right_asset_cfg`` joint list must be the same length
    and already correspond joint-for-joint, e.g. ``[".*_leg_2"]`` resolved
    separately against left- and right-prefixed joint names)."""
    asset: Articulation = env.scene[left_asset_cfg.name]
    left_pos = asset.data.joint_pos[:, left_asset_cfg.joint_ids]
    right_pos = asset.data.joint_pos[:, right_asset_cfg.joint_ids]
    angle_error = left_pos - right_pos
    if delta != 0.0:
        error_vector = F.huber_loss(
            angle_error, torch.zeros_like(angle_error), reduction='none', delta=delta
        )
    else:
        error_vector = angle_error
    return error_vector * scale


def error_joint_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) * scale


def error_joint_align(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = -1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
    if cmd_threshold != -1.0:
        mis_aligned = torch.where(
            cmd <= cmd_threshold,
            torch.abs(
                asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
            ),
            torch.tensor(0.0),
        )
    else:
        mis_aligned = torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids[0]] - asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
        )
    return mis_aligned


# ------------------------------------------------------------------
# Action / joint smoothness
# ------------------------------------------------------------------

def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.action_manager.action - env.action_manager.prev_action


def action_rate_huber(
    env: ManagerBasedRLEnv,
    delta: float = 1.0,
    scale: float = 1.0,
) -> torch.Tensor:
    error = env.action_manager.action - env.action_manager.prev_action
    if delta != 0.0:
        error_vector = F.huber_loss(
            error, torch.zeros_like(error), reduction='none', delta=delta
        )
    else:
        error_vector = error
    return error_vector * scale


def action_rate_l2(env: ManagerBasedRLEnv, scale: float = 0.1) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    ) * scale


def dof_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.joint_vel - asset.data.prev_joint_vel


def joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


# ------------------------------------------------------------------
# Velocity / torque limits
# ------------------------------------------------------------------

def velocity_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    thresh_hold: float = 10,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = torch.relu(abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) - thresh_hold)
    return torch.sum(out_of_limits, dim=1) * scale


# ------------------------------------------------------------------
# Stateful penalty classes
# ------------------------------------------------------------------

class ActionRatePenalty(ManagerTermBase):
    """Penalizes action rate (velocity) and action acceleration (smoothness)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev_prev_action = None

    def reset(self, env_ids=None):
        if self._prev_prev_action is not None:
            if env_ids is None:
                self._prev_prev_action.zero_()
            else:
                self._prev_prev_action[env_ids] = 0.0

    def __call__(
        self,
        env,
        delta: float = 0.0,
        scale: float = 1.0,
    ) -> torch.Tensor:
        current_action = env.action_manager.action
        prev_action = env.action_manager.prev_action

        if self._prev_prev_action is None:
            self._prev_prev_action = prev_action.clone()

        reset_buf = getattr(env, "reset_buf", None)
        if reset_buf is None and hasattr(env, "reset_terminated"):
            reset_buf = env.reset_terminated | env.reset_time_outs

        if reset_buf is not None and len(reset_buf) > 0:
            reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self._prev_prev_action[reset_env_ids] = prev_action[reset_env_ids]

        diff_velocity = current_action - prev_action
        diff_acceleration = diff_velocity - (prev_action - self._prev_prev_action)

        if delta > 0.0:
            loss_vel = F.huber_loss(diff_velocity, torch.zeros_like(diff_velocity), reduction='none', delta=delta)
            loss_acc = F.huber_loss(diff_acceleration, torch.zeros_like(diff_acceleration), reduction='none', delta=delta)
        else:
            loss_vel = torch.square(diff_velocity)
            loss_acc = torch.square(diff_acceleration)

        penalty = torch.sum(loss_vel + loss_acc, dim=1)
        self._prev_prev_action.copy_(prev_action)
        return penalty * scale


class TorqueRatePenalty(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.max_torque = cfg.params.get("max_torque", 60.0)
        self.delta = cfg.params.get("delta", 0.0)
        self.scale = cfg.params.get("scale", 1.0)

        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.joint_ids = self.asset_cfg.joint_ids
        if self.joint_ids is None:
            self.joint_ids = slice(None)
        self._prev_torque = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: object,
        max_torque: float = 60.0,
        delta: float = 0.0,
        scale: float = 1.0,
    ) -> torch.Tensor:
        current_torque = self.asset.data.applied_torque[:, self.joint_ids]

        if self._prev_torque is None:
            self._prev_torque = current_torque.clone()

        reset_buf = getattr(env, "reset_buf", None)
        if reset_buf is None and hasattr(env, "reset_terminated"):
            reset_buf = env.reset_terminated | env.reset_time_outs

        if reset_buf is not None and len(reset_buf) > 0:
            reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_env_ids.numel() > 0:
                self._prev_torque[reset_env_ids] = current_torque[reset_env_ids]

        diff_norm = (current_torque - self._prev_torque) / (self.max_torque + 1e-6)

        if self.delta > 0.0:
            loss = F.huber_loss(diff_norm, torch.zeros_like(diff_norm), reduction="none", delta=self.delta)
        else:
            loss = diff_norm * diff_norm

        penalty = torch.clamp(torch.sum(loss, dim=1), 0.0, 1.0)
        self._prev_torque = current_torque.clone()
        return -self.scale * penalty


class PoseIntegralTrackingError(ManagerTermBase):
    """Dead-reckons a base pose by integrating the commanded base velocity, and returns its error
    against the robot's actual base pose (xy position + yaw).

    Only accumulated and reported while the commanded base velocity is (near) zero: the dead-reckoning
    baseline is kept pinned to the actual pose whenever the command is non-zero, so this term measures
    stand-still drift during zero-command windows instead of open-loop drift over the whole episode.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.command_name: str = cfg.params.get("command_name", "base_velocity")
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

        self._integrated_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self._integrated_yaw = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def _snap_to_actual(self, env_ids):
        self._integrated_xy[env_ids] = self.asset.data.root_link_pos_w[env_ids, :2]
        _, _, yaw = euler_xyz_from_quat(self.asset.data.root_link_quat_w[env_ids])
        self._integrated_yaw[env_ids] = yaw
        self._initialized[env_ids] = True

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._snap_to_actual(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "base_velocity",
        cmd_threshold: float = 0.05,
        delta: float = 0.0,
        scale: float = 1.0,
    ) -> torch.Tensor:
        uninitialized = ~self._initialized
        if uninitialized.any():
            self._snap_to_actual(uninitialized)

        cmd = env.command_manager.get_command(command_name)
        lin_vel_x, lin_vel_y, ang_vel_z = cmd[:, 0], cmd[:, 1], cmd[:, 2]
        dt = env.step_dt

        actual_xy = self.asset.data.root_link_pos_w[:, :2]
        _, _, actual_yaw = euler_xyz_from_quat(self.asset.data.root_link_quat_w)

        is_zero_cmd = torch.linalg.norm(cmd[:, :3], dim=1) < cmd_threshold
        moving = ~is_zero_cmd

        # while commanded to move, keep the dead-reckoning baseline pinned to the actual pose so no
        # drift is accumulated; it only starts integrating once the command drops to (near) zero.
        if moving.any():
            self._integrated_xy[moving] = actual_xy[moving]
            self._integrated_yaw[moving] = actual_yaw[moving]

        if is_zero_cmd.any():
            cos_yaw = torch.cos(self._integrated_yaw[is_zero_cmd])
            sin_yaw = torch.sin(self._integrated_yaw[is_zero_cmd])
            self._integrated_xy[is_zero_cmd, 0] += (
                lin_vel_x[is_zero_cmd] * cos_yaw - lin_vel_y[is_zero_cmd] * sin_yaw
            ) * dt
            self._integrated_xy[is_zero_cmd, 1] += (
                lin_vel_x[is_zero_cmd] * sin_yaw + lin_vel_y[is_zero_cmd] * cos_yaw
            ) * dt
            new_yaw = self._integrated_yaw[is_zero_cmd] + ang_vel_z[is_zero_cmd] * dt
            self._integrated_yaw[is_zero_cmd] = (new_yaw + math.pi) % (2 * math.pi) - math.pi

        pos_error = self._integrated_xy - actual_xy
        yaw_error = self._integrated_yaw - actual_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
        scaled_yaw_error = yaw_error * 3
        error = torch.cat([pos_error, scaled_yaw_error.unsqueeze(-1)], dim=1)

        # only report drift for envs currently under a (near) zero command.
        error = error * is_zero_cmd.unsqueeze(-1)

        if delta > 0.0:
            error = F.huber_loss(error, torch.zeros_like(error), reduction="none", delta=delta)

        return 1 - torch.exp(-4*abs(error)) # * scale

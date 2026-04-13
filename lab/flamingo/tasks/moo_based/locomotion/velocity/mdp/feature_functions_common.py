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
        tgt = env.command_manager.get_command(command_name)[:, 3:4]
    elif target_height is not None:
        tgt = torch.full_like(base_z, float(target_height))
    else:
        tgt = base_z.detach()

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)
        tgt = tgt + ground_z

    return (base_z - tgt) * scale


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
    angle_error = asset.data.joint_pos[:, asset_cfg.joint_ids]
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
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
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

        if len(env.reset_buf) > 0:
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
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

        reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
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

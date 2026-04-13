# SPDX-License-Identifier: BSD-3-Clause
# POSTECH Flamingo Lab, 2025
# Rough-terrain overrides and additions to common feature functions (4w4l variant).

from __future__ import annotations
import torch
import torch.nn.functional as F
import math
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import RayCaster, ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, quat_apply_inverse

from ....mdp.feature_functions_common import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ------------------------------------------------------------------
# Rough-terrain overrides (extended signatures)
# ------------------------------------------------------------------

def error_track_lin_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    kernel: str = "linear",
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    vel_xy = asset.data.root_link_lin_vel_b[:, :2]
    return apply_kernel(vel_xy - cmd_xy, kernel, scale)


def error_track_ang_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    kernel: str = "linear",
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_z = env.command_manager.get_command(command_name)[:, 2:3]
    ang_vel_z = asset.data.root_link_ang_vel_b[:, 2:3]
    return apply_kernel(ang_vel_z - cmd_z, kernel, scale)


def error_track_pos_integral(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "integral_position",
    kernel: str = "tanh",
    delta: float = 0.1,
    scale: float = 1.0,
) -> torch.Tensor:
    pos_error_xy = env.command_manager.get_command(command_name)
    if delta == 0:
        error = pos_error_xy
    else:
        error = F.huber_loss(pos_error_xy, torch.zeros_like(pos_error_xy), reduction='none', delta=delta)
    return apply_kernel(error, kernel, scale)


def error_base_height(
    env: ManagerBasedRLEnv,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    kernel: str = "tanh",
    temperature: float = 8.0,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2:3]
    tgt = torch.full_like(base_z, float(target_height))
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z, _ = torch.max(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)
        tgt = tgt + ground_z
    return apply_kernel(base_z - tgt, kernel, scale, temperature)


def error_joint_deviation_huber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    delta: float = 0.5,
    scale: float = 1.0,
    threshold: float = 0.03,
    kernel: str = "tanh",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle_error = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if delta == 0:
        error = angle_error
    else:
        error = F.huber_loss(angle_error, torch.zeros_like(angle_error), reduction='none', delta=delta)
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z_values = sensor.data.ray_hits_w[..., 2]
        finite_mask = torch.isfinite(ground_z_values)
        safe_for_max = torch.where(finite_mask, ground_z_values, -torch.inf)
        safe_for_min = torch.where(finite_mask, ground_z_values, torch.inf)
        max_ground_z, _ = torch.max(safe_for_max, dim=1, keepdim=True)
        min_ground_z, _ = torch.min(safe_for_min, dim=1, keepdim=True)
        is_flat_mask = ((max_ground_z - min_ground_z) <= threshold).float()
        error *= is_flat_mask
    return apply_kernel(error, kernel, scale)


def error_flat_euler_rp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    threshold: float = 0.03,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi
    error = torch.stack((roll, pitch), dim=1)
    if sensor_cfg is not None and command_name is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z_values = sensor.data.ray_hits_w[..., 2]
        finite_mask = torch.isfinite(ground_z_values)
        safe_for_max = torch.where(finite_mask, ground_z_values, -torch.inf)
        safe_for_min = torch.where(finite_mask, ground_z_values, torch.inf)
        max_ground_z, _ = torch.max(safe_for_max, dim=1, keepdim=True)
        min_ground_z, _ = torch.min(safe_for_min, dim=1, keepdim=True)
        is_flat_mask = ((max_ground_z - min_ground_z) <= threshold).float()
        cmd = env.command_manager.get_command(command_name)
        is_stopped_mask = (torch.norm(cmd[:, :3], dim=1, keepdim=True) <= 0.1)
        mask = torch.where(is_stopped_mask, torch.tensor(1.0, device=env.device), is_flat_mask)
        error *= mask
    return error * scale


# ------------------------------------------------------------------
# Rough-terrain only functions
# ------------------------------------------------------------------

def error_ang_vel_xy(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    kernel: str = "tanh",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.root_link_ang_vel_b[:, :2]
    return apply_kernel(ang_vel_xy, kernel, scale)


def error_stuck_dance(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    stuck_threshold: float = 0.1,
    cmd_threshold: float = 0.1,
    scale: float = 1.0,
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1)
    asset: RigidObject = env.scene["robot"]
    vel_norm = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    is_stuck = (cmd_norm > cmd_threshold) & (vel_norm < stuck_threshold)
    curr = env.action_manager.action
    prev = env.action_manager.prev_action
    action_diff = torch.norm(curr - prev, dim=1)
    target_movement = 1.0
    error = torch.where(
        is_stuck,
        torch.clamp(target_movement - action_diff, min=0.0),
        torch.tensor(0.0, device=env.device),
    )
    return error * scale


def error_both_feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = -1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    num_contacts = torch.sum(in_contact.int(), dim=1)
    is_flight = (num_contacts == 0)
    max_flight_duration = torch.max(air_time, dim=1)[0]
    penalty = torch.where(is_flight, max_flight_duration, 0.0)
    return penalty * scale


def error_body_com(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 1.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    local_com = asset.root_physx_view.get_coms()[:, body_ids, :3].to(env.device)
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    com_rel_w = local_com - root_pos_w
    com_error = quat_rotate_inverse(root_quat_w, com_rel_w)
    return com_error * scale


def illegal_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.sum(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)


def joint_action_rate_huber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    command_name: str | None = None,
    threshold: float = 0.03,
    delta: float = 1.0,
    scale: float = 1.0,
    kernel: str = "tanh",
) -> torch.Tensor:
    curr = env.action_manager.action
    prev = env.action_manager.prev_action
    if delta == 0.0:
        error = curr - prev
    else:
        error = F.huber_loss(curr - prev, torch.zeros_like(curr - prev), reduction='none', delta=delta)
    if sensor_cfg is not None and command_name is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z_values = sensor.data.ray_hits_w[..., 2]
        finite_mask = torch.isfinite(ground_z_values)
        safe_for_max = torch.where(finite_mask, ground_z_values, -torch.inf)
        safe_for_min = torch.where(finite_mask, ground_z_values, torch.inf)
        max_ground_z, _ = torch.max(safe_for_max, dim=1, keepdim=True)
        min_ground_z, _ = torch.min(safe_for_min, dim=1, keepdim=True)
        is_flat_mask = ((max_ground_z - min_ground_z) <= threshold).float()
        cmd = env.command_manager.get_command(command_name)
        is_stopped_mask = (torch.norm(cmd[:, :3], dim=1, keepdim=True) <= 0.1)
        mask = torch.where(is_stopped_mask, torch.tensor(1.0, device=env.device), is_flat_mask)
        error *= mask
    return apply_kernel(error, kernel, scale)


def error_projected_gravity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * scale


def link_x_vel_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    wheel_vel_b = torch.mean(
        quat_rotate_inverse(
            asset.data.body_link_quat_w[:, asset_cfg.body_ids],
            asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids],
        ),
        dim=1,
    )
    root_vel_b = asset.data.root_lin_vel_b
    return wheel_vel_b[:, 0:1] - root_vel_b[:, 0:1]


def error_same_foot_x_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:, i, :] = quat_apply_inverse(base_quat, foot_base[:, i, :])
    dx = foot_base[:, 0, 0] - foot_base[:, 1, 0]
    return dx * scale


def applied_torque_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1) * scale


def joint_soft_pos_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    upper_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    out_of_lower = torch.relu((lower_limits * soft_ratio) - joint_pos)
    out_of_upper = torch.relu(joint_pos - (upper_limits * soft_ratio))
    return torch.sum(out_of_lower + out_of_upper, dim=1) * scale


def error_foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    base_sensor_cfg: SceneEntityCfg,
    command_name: str | None = None,
    wheel_height: float = 0.1,
    kernel: str = "tanh",
    temperature: float = 8.0,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    current_foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    sensor: RayCaster = env.scene[base_sensor_cfg.name]
    ground_z_values = sensor.data.ray_hits_w[..., 2]
    safe_ground_z = torch.nan_to_num(ground_z_values, posinf=-1.0e6, neginf=-1.0e6, nan=-1.0e6)
    max_ground_z, _ = torch.max(safe_ground_z, dim=1, keepdim=True)
    wheel_height_t = torch.full_like(current_foot_z, float(wheel_height))
    target_foot_z = max_ground_z + wheel_height_t
    error = torch.clamp(target_foot_z - current_foot_z, min=0.0)
    if command_name is not None:
        cmd = env.command_manager.get_command(command_name)
        cmd_mask = (torch.norm(cmd[:, :3], dim=1, keepdim=True) > 0.1)
        error *= cmd_mask
    return apply_kernel(error, kernel, scale, temperature)

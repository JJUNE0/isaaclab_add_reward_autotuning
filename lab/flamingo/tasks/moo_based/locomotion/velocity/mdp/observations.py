# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import torch
import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from typing import Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def base_lin_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b

def base_lin_vel_x_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 0].unsqueeze(-1)

def base_lin_vel_y_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 1].unsqueeze(-1)

def base_lin_vel_z_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 1].unsqueeze(-1)

def base_ang_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # print("ang vel: ", asset.data.root_link_ang_vel_b[0, 2])
    # print("lin vel z: ", asset.data.root_link_lin_vel_b[0, 2])
    return asset.data.root_link_ang_vel_b
        
def base_pos_z_rel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2]
    else:
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)
    
def base_pos_z_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1) - sensor.data.ray_hits_w[..., 2]
    else:
        return asset.data.root_link_pos_w[:, 2].unsqueeze(-1)


def current_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current reward value. Returns zeros if the reward manager is not initialized."""
    if not hasattr(env, "reward_manager") or env.reward_manager is None:
        # Assuming the shape should be (num_envs,) based on the environment
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

    try:
        return env.reward_buf.unsqueeze(-1)
    except AttributeError:
        # Fallback to zeros if the reward_manager is initialized but compute isn't ready
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)


def joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def is_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact.float()


def contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """각 발(또는 body_ids)에 대한 최근 히스토리 구간 동안의 최대 접촉 힘 [N]."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # (num_envs, history, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # 선택한 body들만 가져와서 norm -> (num_envs, history, num_selected_bodies)
    forces = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)

    # history 차원(최근 몇 스텝)에서 max -> (num_envs, num_selected_bodies)
    max_forces, _ = torch.max(forces, dim=1)

    return max_forces


def lift_mask_by_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    
    """
    Generate a lift mask for the robot's legs based on row-wise height scan gradients from separate left and right sensors.

    Args:
        env (ManagerBasedRLEnv): Simulation environment.
        sensor_cfg_left (SceneEntityCfg): Configuration for the left raycast sensor.
        sensor_cfg_right (SceneEntityCfg): Configuration for the right raycast sensor.
        command_name (str): Command name to check movement intention.
        gradient_threshold (float): Threshold for row-wise height gradient to detect steps.

    Returns:
        torch.Tensor: Lift mask for left and right legs. Shape: [num_envs, 2].
    """
    #* Step 1: Extract ray hit positions (Z coordinates) from left and right sensors
    left_lift_mask_sensor = env.scene.sensors[sensor_cfg_left.name]
    right_lift_mask_sensor = env.scene.sensors[sensor_cfg_right.name]

    left_mask= left_lift_mask_sensor.data.mask 
    right_mask = right_lift_mask_sensor.data.mask  
    
    lift_mask = torch.stack([left_mask, right_mask], dim=1) 

    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)  # Shape: [num_envs]
    lift_mask *= (command_norm > 0.1).unsqueeze(-1).float()  # Apply movement condition

    return lift_mask

def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def base_euler_angle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_com_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    return rpy

def base_euler_angle_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation in the environment frame as Euler angles (roll, pitch, yaw)."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)

    # Map angles from [0, 2*pi] to [-pi, pi]
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    pitch = (pitch + math.pi) % (2 * math.pi) - math.pi
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    rpy = torch.stack((roll, pitch, yaw), dim=-1)
    return rpy


def joint_pos_rel_sin(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as sine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_value_sin = torch.sin(current_value)
    return current_value_sin


def joint_pos_rel_cos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions as cosine values.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    current_value = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_value_cos = torch.cos(current_value)
    return current_value_cos


def height_scan_raw(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.ray_hits_w[..., 2]


def generated_partial_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)[:, 0].unsqueeze(-1)


def generated_scaled_commands(env: ManagerBasedRLEnv, command_name: str, scale: tuple) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    scaled_command = env.command_manager.get_command(command_name).clone()
    scaled_command[:, :3] *= torch.tensor(scale, device=env.device)
    return scaled_command

def generated_scaled_event_commands(env: ManagerBasedRLEnv, command_name: str, scale: tuple) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    scaled_command = env.command_manager.get_command(command_name).clone()
    scaled_command[:, :2] *= torch.tensor(scale, device=env.device)
    return scaled_command

def joint_pos_leg_gear(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint positions for the configured (leg) joints, scaled by `gear_ratio`."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return pos * gear_ratio

def joint_vel_leg_gear(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return joint velocities for the configured (leg) joints, scaled by `gear_ratio`."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return vel * gear_ratio


def body_mass(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """로봇의 링크 질량을 반환합니다 (Randomize된 값 포함)."""
    # asset: Articulation
    asset = env.scene[asset_cfg.name]
    
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    masses = asset.root_physx_view.get_masses().clone()[:, body_ids]
    
#    print("masses: ", masses)
    
    return masses.to(env.device)

def body_com(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """로봇의 CoM 위치를 'Base 프레임 기준(Local Offset)'으로 반환합니다."""
    asset = env.scene[asset_cfg.name]
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    
    local_com = asset.root_physx_view.get_coms()[:, body_ids, :3].to(env.device)
    
#    print("local_com: ", local_com)
    # 5. 차원 펴주기
    return local_com.view(env.num_envs, -1)

def joint_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(Effective) DOF friction coefficients from PhysX view."""
    asset = env.scene[asset_cfg.name]

    if getattr(asset_cfg, "joint_names", None):
        dof_ids = asset.find_joints(asset_cfg.joint_names)[0]
    else:
        dof_ids = slice(None)

    fric_all = asset.root_physx_view.get_dof_friction_coefficients().to(env.device)  # (N, num_dofs)
    
#    print("fric_all: ", fric_all)
    return fric_all[:, dof_ids].clone()

def actuator_kp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    # articulation dof/joint 전체 길이 (버전에 따라 num_joints/num_dofs 중 하나)
    n = getattr(asset, "num_joints", None) or getattr(asset, "num_dofs")
    kp_full = torch.zeros((env.num_envs, n), device=env.device, dtype=torch.float32)

    # 각 actuator가 담당하는 joint index에 현재 stiffness를 채움
    for act in asset.actuators.values():
        kp = act.stiffness
        if kp.ndim == 1:
            kp = kp.unsqueeze(0).repeat(env.num_envs, 1)
        kp_full[:, act.joint_indices] = kp.to(dtype=torch.float32)

#    print("kp_full: ", kp_full)
    if getattr(asset_cfg, "joint_names", None):
        joint_ids = asset.find_joints(asset_cfg.joint_names)[0]
        return kp_full[:, joint_ids]
    
    
    return kp_full

def actuator_kd(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    n = getattr(asset, "num_joints", None) or getattr(asset, "num_dofs")
    kd_full = torch.zeros((env.num_envs, n), device=env.device, dtype=torch.float32)

    for act in asset.actuators.values():
        kd = act.damping
        if kd.ndim == 1:
            kd = kd.unsqueeze(0).repeat(env.num_envs, 1)
        kd_full[:, act.joint_indices] = kd.to(dtype=torch.float32)

#    print("kd_full: ", kd_full)
    if getattr(asset_cfg, "joint_names", None):
        joint_ids = asset.find_joints(asset_cfg.joint_names)[0]
        return kd_full[:, joint_ids]
    
    
    return kd_full

def is_discrete_terrain(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """Determines if the terrain is discrete based on height variation from a ray caster sensor."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_z = sensor.data.ray_hits_w[..., 2]
    max_z, _ = torch.max(hit_z, dim=1, keepdim=True)
    min_z, _ = torch.min(hit_z, dim=1, keepdim=True)
    height_range = max_z - min_z  # Shape: [num_envs, 1]
    is_discrete = (height_range > threshold).float()
    return is_discrete


def physics_material(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene.terrain.static_friction * torch.ones(env.num_envs, 1, device=env.device)
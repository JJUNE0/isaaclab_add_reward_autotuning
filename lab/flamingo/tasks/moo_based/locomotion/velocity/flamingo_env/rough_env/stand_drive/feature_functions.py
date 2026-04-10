# SPDX-License-Identifier: BSD-3-Clause
# POSTECH Flamingo Lab, 2025
# ADD (non-motion imitation) feature extractors rewritten to match user's reward style.

from __future__ import annotations
import torch
import torch.nn.functional as F
import math
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import RayCaster
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_kernel(error: torch.Tensor, kernel: str = "linear", scale: float = 1.0, temperature : float = 4.0) -> torch.Tensor:
    if kernel == "linear":
        return error * scale
    elif kernel == "tanh":
        return torch.tanh(error * temperature) * scale
    elif kernel == "exp":
        # 00님의 아이디어: 1 - e^(-|x|)
        return (1.0 - torch.exp(-torch.abs(error) * temperature)) * torch.sign(error) * scale
    else:
        return error * scale


# ---------------------------------------------------------
# Find Global Optimal 
# ---------------------------------------------------------
def error_stuck_dance(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    stuck_threshold: float = 0.1, # 이 속도보다 느리면 갇힌 것으로 간주
    cmd_threshold: float = 0.1,   # 이 명령보다 클 때만 작동
    scale: float = 1.0
) -> torch.Tensor:
    """
    [논문의 Dancing 모방]
    로봇이 이동 명령을 받았는데도(cmd > 0.1) 실제 속도가 느리다면(stuck),
    가만히 있는 것(Local Optima)을 벌하고, 오히려 '움직임(Action Rate)'을 장려합니다.
    
    - 반환값: 오차 벡터 (목표는 '움직임'이므로, 가만히 있을수록 오차가 커짐)
    """
    # 1. 커맨드 확인
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1)
    
    # 2. 실제 속도 확인
    asset: RigidObject = env.scene["robot"]
    vel_norm = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    
    # 3. Stuck 조건: 명령은 있는데 못 가고 있음
    is_stuck = (cmd_norm > cmd_threshold) & (vel_norm < stuck_threshold)
    
    # 4. 액션의 변화량 (움직임)
    curr = env.action_manager.action
    prev = env.action_manager.prev_action
    action_diff = torch.norm(curr - prev, dim=1) # (N,)
    
    # 5. 오차 계산
    # Stuck 상태일 때: 움직임(action_diff)이 클수록 오차가 0에 가까워짐 (보상 효과)
    # Stuck 상태가 아닐 때: 오차 0
    
    # 목표 움직임 크기 (예: 1.0만큼은 움직여라)
    target_movement = 1.0 
    
    # Stuck일 때만 활성화. 가만히 있으면(action_diff=0) 오차가 1.0이 됨.
    # 막 움직이면(action_diff=1.0) 오차가 0이 됨.
    error = torch.where(
        is_stuck,
        torch.clamp(target_movement - action_diff, min=0.0),
        torch.tensor(0.0, device=env.device)
    )
    
    return error * scale



def error_both_feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = -1.0  # 페널티이므로 통상 음수 값을 사용하거나, 
                         # 함수 외부에서 차감할 경우 양수로 설정
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

# ---------------------------------------------------------
# 1) Linear velocity tracking (XY) → feature (N, 2)
#    (사용자 예시: track_lin_vel_xy_link_exp_l2 와 동일한 데이터 경로/인자)
# ---------------------------------------------------------
def error_track_lin_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    kernel:str = "linear",
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]            # (N,2)
    vel_xy = asset.data.root_link_lin_vel_b[:, :2]                           # (N,2)
    return apply_kernel(vel_xy - cmd_xy ,kernel, scale)                      # (N,2)


def error_track_pos_integral(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "integral_position", # 새로 만든 커맨드 이름
    kernel: str = "tanh",
    delta : float = 0.1,
    scale: float = 1.0,
) -> torch.Tensor:
    
    pos_error_xy = env.command_manager.get_command(command_name) 
    
    if delta == 0 :
        error = pos_error_xy
    else:    
        error = F.huber_loss(
            pos_error_xy, 
            torch.zeros_like(pos_error_xy), 
            reduction='none', 
            delta=delta
        )
    
    return apply_kernel(error, kernel, scale)

def error_lin_vel_z(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    delta: float = 0.5,
    kernel : str = "tanh",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_z = asset.data.root_link_lin_vel_b[:, 2:3]                           # (N,2)
    if delta == 0.0:
        error = vel_z
    else :
        error = F.huber_loss(
            vel_z, 
            torch.zeros_like(vel_z),
            reduction='none', 
            delta=delta
        )
    return apply_kernel(error, kernel, scale)                                # (N,2)


def error_track_ang_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    kernel :str = "linear",
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_z = env.command_manager.get_command(command_name)[:, 2:3]            # (N,1)
    ang_vel_z = asset.data.root_link_ang_vel_b[:, 2:3]                           # (N,1)
    error = cmd_z - ang_vel_z
    # square_error = torch.square(error, dim=1)
    return apply_kernel(ang_vel_z - cmd_z ,kernel, scale)                                   # (N,1)

def error_ang_vel_xy(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    kernel: str = "tanh",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_z = asset.data.root_link_ang_vel_b[:, :2]                           # (N,1)
    return apply_kernel(ang_vel_z, kernel, scale)                                   # (N,1)

def error_ang_vel_y(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    delta: float = 0.5,
    kernel: str = "tanh",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_y = asset.data.root_link_ang_vel_b[:, 1]                           # (N,1)
    if delta == 0.0:
        error = ang_vel_y
    else :
        error = F.huber_loss(
            ang_vel_y, 
            torch.zeros_like(ang_vel_y),
            reduction='none', 
            delta=delta
        )
    
    
    return apply_kernel(error, kernel, scale)                                   # (N,1)

def error_ang_vel_x(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    delta: float = 0.5,
    kernel: str = "tanh",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_x = asset.data.root_link_ang_vel_b[:, 0]                           # (N,1)
    if delta == 0.0:
        error = ang_vel_x
    else :
        error = F.huber_loss(
            ang_vel_x, 
            torch.zeros_like(ang_vel_x),
            reduction='none', 
            delta=delta
        )
    
    
    return apply_kernel(error, kernel, scale)  



def error_base_height(
    env: ManagerBasedRLEnv,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    kernel : str = "tanh",
    temperature : float = 8.0,
    scale: float = 1.0
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2:3]                               # (N,1)
    tgt = torch.full_like(base_z, float(target_height))

    if sensor_cfg is not None:
        # 지면 기반 상대 높이 보정 (사용자 track_pos_z 계열과 동일한 접근)
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z, _ = torch.max(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)  # (N,1)
        #print(f"ray :  {sensor.data.ray_hits_w[..., 2]}")
        # print(f"gronud z : {ground_z}")
        # print(f"tgt_z : {tgt}")
        # print(f"base_z : {base_z}")
        tgt = tgt + ground_z
        
    return apply_kernel(base_z - tgt, kernel ,scale, temperature)

def error_foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,  # "다리" 링크 (예: 'left_wheel_link')
    base_sensor_cfg: SceneEntityCfg, # "몸통"의 'height_scanner'
    command_name : str | None = None,
    wheel_height: float = 0.1,    # 최소 10cm 여유
    kernel:str = "tanh",
    temperature :float = 8.0,
    scale: float = 1.0
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    current_foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] # (N, num_feet)
    
    sensor: RayCaster = env.scene[base_sensor_cfg.name]
    ground_z_values = sensor.data.ray_hits_w[..., 2] # (N, num_rays)

    safe_ground_z = torch.nan_to_num(
        ground_z_values, 
        posinf=-1.0e6, neginf=-1.0e6, nan=-1.0e6
    )
    
    max_ground_z,_ = torch.max(safe_ground_z, dim=1, keepdim=True)
    
    wheel_height = torch.full_like(current_foot_z, float(wheel_height))
    
    target_foot_z = max_ground_z + wheel_height

    error = torch.clamp(target_foot_z - current_foot_z, min=0.0) * scale 
    
    if command_name is not None:
        cmd = env.command_manager.get_command(command_name)
        cmd_mask = (torch.norm(cmd[:, :3], dim=1, keepdim=True) > 0.1)
        error *= cmd_mask
    
    return apply_kernel(error, kernel, scale, temperature)


def error_joint_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, scale: float = 1.0) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) * scale


def error_body_com(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    scale: float = 1.0
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    local_com = asset.root_physx_view.get_coms()[:, body_ids, :3].to(env.device)
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    com_rel_w = local_com - root_pos_w
    
    com_error = quat_rotate_inverse(root_quat_w, com_rel_w)
    
    return com_error * scale

def error_joint_deviation_huber(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    delta: float = 0.5,
    scale: float = 1.0,
    threshold: float = 0.03,
    kernel: str = "tanh"
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    angle_error = asset.data.joint_pos[:, asset_cfg.joint_ids] # (N, num_joints)

    if delta == 0 :
        error = angle_error
    else:    
        error = F.huber_loss(
            angle_error, 
            torch.zeros_like(angle_error), 
            reduction='none', 
            delta=delta
        )

    if sensor_cfg is not None: 
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z_values = sensor.data.ray_hits_w[..., 2] # (N, num_ray)
        finite_mask = torch.isfinite(ground_z_values) # (N, num_rays)
        safe_for_max = torch.where(finite_mask, ground_z_values, -torch.inf)
        safe_for_min = torch.where(finite_mask, ground_z_values, torch.inf)
        max_ground_z, _ = torch.max(safe_for_max, dim=1, keepdim=True) # (N, 1)
        min_ground_z, _ = torch.min(safe_for_min, dim=1, keepdim=True) # (N, 1)

        is_flat_mask = ((max_ground_z - min_ground_z ) <= threshold).float() # (N, 1)
        
        error *= is_flat_mask 
        
    return apply_kernel(error, kernel, scale) 


def error_joint_align(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = -1.0,
) -> torch.Tensor:
    """Penalize joint mis-alignments.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
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


def action_rate(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    # (N, A)
    curr = env.action_manager.action
    prev = env.action_manager.prev_action
    return curr - prev


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)



def illegal_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.sum(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim = 1)



def action_rate_l2(env: ManagerBasedRLEnv, scale : float = 0.1) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1) * scale

def joint_action_rate_huber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    command_name : str | None = None,
    threshold : float = 0.03,
    delta: float = 1.0,         # Huber loss 임계값
    scale: float = 1.0,         # (DiffNormalizer가 있으므로 1.0 권장)
    kernel:str = "tanh"
) -> torch.Tensor:


    # (N, A)
    curr = env.action_manager.action#[:,:6]
    prev = env.action_manager.prev_action#[:,:6]
    
    if delta == 0.0:
        error = curr - prev
    else :
        error = F.huber_loss(
            curr - prev, 
            torch.zeros_like(curr - prev),
            reduction='none', 
            delta=delta
        )
                
    if sensor_cfg is not None and command_name is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z_values = sensor.data.ray_hits_w[..., 2] # (N, num_ray)
        finite_mask = torch.isfinite(ground_z_values) # (N, num_rays)
        safe_for_max = torch.where(finite_mask, ground_z_values, -torch.inf)
        safe_for_min = torch.where(finite_mask, ground_z_values, torch.inf)
        max_ground_z, _ = torch.max(safe_for_max, dim=1, keepdim=True) # (N, 1)
        min_ground_z, _ = torch.min(safe_for_min, dim=1, keepdim=True) # (N, 1)

        is_flat_mask = ((max_ground_z - min_ground_z ) <= threshold).float() # (N, 1)
        cmd = env.command_manager.get_command(command_name)
        is_stopped_mask = (torch.norm(cmd[:, :3], dim=1, keepdim=True) <= 0.1) # (N, 1)
        
        mask = torch.where(
            is_stopped_mask, 
            torch.tensor(1.0, device=env.device),
            is_flat_mask                          
        ) # (N, 1)
        
        error *= mask
        
    return apply_kernel(error, kernel, scale) 


def error_flat_euler_rp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    command_name : str | None = None,
    threshold : float = 0.03,
    scale : float = 1.0
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)  # (N,), (N,), (N,)
    # [-pi, pi]로 정규화 (사용자 코드 동일)
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi
    
    error = torch.stack((roll, pitch), dim=1)
    if sensor_cfg is not None and command_name is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z_values = sensor.data.ray_hits_w[..., 2] # (N, num_ray)
        finite_mask = torch.isfinite(ground_z_values) # (N, num_rays)
        safe_for_max = torch.where(finite_mask, ground_z_values, -torch.inf)
        safe_for_min = torch.where(finite_mask, ground_z_values, torch.inf)
        max_ground_z, _ = torch.max(safe_for_max, dim=1, keepdim=True) # (N, 1)
        min_ground_z, _ = torch.min(safe_for_min, dim=1, keepdim=True) # (N, 1)

        is_flat_mask = ((max_ground_z - min_ground_z ) <= threshold).float() # (N, 1)
        cmd = env.command_manager.get_command(command_name)
        is_stopped_mask = (torch.norm(cmd[:, :3], dim=1, keepdim=True) <= 0.1) # (N, 1)
        
        mask = torch.where(
            is_stopped_mask, 
            torch.tensor(1.0, device=env.device), # 멈췄으면 무조건 1.0
            is_flat_mask                          # 움직이면 지면 상태에 따라 결정
        ) # (N, 1)
        error *= mask
    
    return error * scale                     # (N,2)

def error_projected_gravity(
    env: ManagerBasedRLEnv,  
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale : float = 1.0 ):
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * scale


def link_x_vel_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    # 바디(휠) 속도를 base frame으로 투영 후 평균 → (N,3)
    wheel_vel_b = torch.mean(
        quat_rotate_inverse(asset.data.body_link_quat_w[:, asset_cfg.body_ids],
                            asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]),
        dim=1,
    )
    root_vel_b = asset.data.root_lin_vel_b
    return (wheel_vel_b[:, 0:1] - root_vel_b[:, 0:1])             # (N,1)



def error_same_foot_x_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale : float = 1.0
) -> torch.Tensor:
    """
    Penalize X-axis displacement difference of two feet in base frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos = asset.data.root_link_pos_w
    base_quat = asset.data.root_link_quat_w
    foot_world = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_base = foot_world - base_pos.unsqueeze(1)
    for i in range(len(asset_cfg.body_ids)):
        foot_base[:,i,:] = quat_apply_inverse(base_quat, foot_base[:,i,:])
    dx = foot_base[:,0,0] - foot_base[:,1,0]
    return dx * scale # penalize both feet being too far apart and too close together


def applied_torque_limits(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale : float = 1.0) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1) * scale


def velocity_limits(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    thresh_hold : float = 10, 
    scale : float = 1.0) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.relu(
        abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) - thresh_hold
    )
    return torch.sum(out_of_limits, dim=1) * scale


def joint_soft_pos_limits(
    env: ManagerBasedRLEnv, 
    soft_ratio: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    upper_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]

    out_of_lower = torch.relu((lower_limits * soft_ratio) - joint_pos)    
    out_of_upper = torch.relu(joint_pos - (upper_limits * soft_ratio))
    
    return torch.sum(out_of_lower + out_of_upper, dim=1) * scale

class ActionRatePenalty(ManagerTermBase):
    """
    Action Rate & Acceleration Penalty Reward Term.
    
    Penalizes both:
    1. The rate of change of actions (Velocity): (a_t - a_{t-1})
    2. The acceleration of actions (Smoothness): (a_t - 2*a_{t-1} + a_{t-2})
    
    Encourages smoother control by minimizing higher-order derivatives.
    """

    def __init__(self, cfg, env):
        """Initialize the term."""
        super().__init__(cfg, env)
        
        # a_{t-2} (전전 액션)를 저장할 버퍼
        self._prev_prev_action = None

    def __call__(
        self,
        env,
        delta: float = 0.0,      # Huber Loss Delta (0.0이면 L2 Square)
        scale: float = 1.0,      # 전체 페널티 스케일
    ) -> torch.Tensor:
        """Compute the penalty (Called every step)."""
        
        current_action = env.action_manager.action      # a_t
        prev_action = env.action_manager.prev_action    # a_{t-1}

        if self._prev_prev_action is None:
            self._prev_prev_action = prev_action.clone()

        if len(env.reset_buf) > 0:
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self._prev_prev_action[reset_env_ids] = prev_action[reset_env_ids]

        diff_velocity = current_action - prev_action
        
        # acc_t = v_t - v_{t-1} = a_t - 2*a_{t-1} + a_{t-2}
        diff_acceleration = diff_velocity - (prev_action - self._prev_prev_action)
        
        if delta > 0.0:
            loss_vel = F.huber_loss(diff_velocity, torch.zeros_like(diff_velocity), reduction='none', delta=delta)
            loss_acc = F.huber_loss(diff_acceleration, torch.zeros_like(diff_acceleration), reduction='none', delta=delta)
        else:
            # delta가 0이면 Square(L2) (또는 필요시 abs(L1)로 변경 가능)
            loss_vel = torch.square(diff_velocity)
            loss_acc = torch.square(diff_acceleration)

        total_loss = loss_vel + loss_acc
        penalty = torch.sum(total_loss, dim=1)

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
        asset_cfg: object, # Call signature 맞추기용 (실제론 self.asset 사용)
        max_torque : float = 60.0,
        delta: float = 0.0,
        scale: float = 1.0,
    ) -> torch.Tensor:
        current_torque = self.asset.data.applied_torque[:, self.joint_ids]

        if self._prev_torque is None:
            self._prev_torque = current_torque.clone()

        reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() > 0:
            self._prev_torque[reset_env_ids] = current_torque[reset_env_ids]

        epsilon = 1e-6
        diff_norm = (current_torque - self._prev_torque) / (self.max_torque + epsilon)

        if self.delta > 0.0:
            loss = F.huber_loss(
                diff_norm,
                torch.zeros_like(diff_norm),
                reduction="none",
                delta=self.delta,
            )
        else:
            loss = diff_norm * diff_norm  # L2

        penalty = torch.sum(loss, dim=1)
        penalty = torch.clamp(penalty, 0.0, 1.0)  # optional: avoid crazy outliers

        self._prev_torque = current_torque.clone()

        # 패널티는 음수 reward로 리턴
        return - self.scale * penalty
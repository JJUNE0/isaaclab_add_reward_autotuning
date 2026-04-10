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
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse

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
    return apply_kernel(error, kernel, scale)    


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

def error_track_lin_vel_xy(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]            # (N,2)
    vel_xy = asset.data.root_link_lin_vel_b[:, :2]                           # (N,2)
    # ADD에서는 보상 대신 "오차 벡터"를 반환
    return (vel_xy - cmd_xy) * scale                                # (N,2)


def error_track_ang_vel_z(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_z = env.command_manager.get_command(command_name)[:, 2:3]            # (N,1)
    vel_z = asset.data.root_link_ang_vel_b[:, 2:3]                           # (N,1)
    return (vel_z - cmd_z) * scale                                   # (N,1)

def error_track_pos_integral(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "integral_position", # 새로 만든 커맨드 이름
    kernel: str = "tanh",
    scale: float = 1.0,
) -> torch.Tensor:
    
    pos_error_xy = env.command_manager.get_command(command_name) 
    return apply_kernel(pos_error_xy, kernel, scale)


def error_joint_deviation_huber(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    delta: float = 0.5,         # Huber loss 임계값
    scale: float = 1.0          # (DiffNormalizer를 켠다면 1.0 권장)
) -> torch.Tensor:

    # 1. 원시 오차 벡터 (목표=0)
    asset: Articulation = env.scene[asset_cfg.name]
    angle_error = asset.data.joint_pos[:, asset_cfg.joint_ids] # (N, num_joints)

    # 2. Huber loss (Smooth L1 Loss)를 element-wise로 계산
    #    reduction='none'으로 (N, num_joints) 텐서 반환
    
    if delta != 0.0:
        error_vector = F.huber_loss(
           angle_error, 
           torch.zeros_like(angle_error), # 목표값(target)은 0입니다.
           reduction='none', 
           delta=delta
    )
    else:
        # delta=0인 경우는 절대값 오차와 동일
        error_vector = angle_error      
    
    # 3. (선택적) 스케일링 후 반환
    return error_vector * scale

def error_base_height(
    env: ManagerBasedRLEnv,
    target_height: float | None = None,
    command_name: str | None = None,     # 예: "base_velocity"에 z명령이 들어있다면 사용
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    scale: float = 1.0
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]                               # (N,1)

    if command_name is not None:
        # 커맨드로 목표 높이 전달되는 경우 (사용자 스타일: base_velocity의 [:,3] 등)
        tgt = env.command_manager.get_command(command_name)[:, 3:4]           # (N,1)
    elif target_height is not None:
        tgt = torch.full_like(base_z, float(target_height))
    else:
        # 기본값: 현재 높이를 목표로 (제로에 근접한 피처)
        tgt = base_z.detach()

    if sensor_cfg is not None:
        # 지면 기반 상대 높이 보정 (사용자 track_pos_z 계열과 동일한 접근)
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ground_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)  # (N,1)
        tgt = tgt + ground_z

    return (base_z - tgt) * scale                       # (N,1)
                                                        # (N,1)


def error_joint_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, scale: float = 1.0) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) * scale


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


def action_rate_huber(
    env: ManagerBasedRLEnv,
    delta: float = 1.0,         # Huber loss 임계값
    scale: float = 1.0,         # (DiffNormalizer가 있으므로 1.0 권장)
) -> torch.Tensor:

    curr = env.action_manager.action
    prev = env.action_manager.prev_action
    
    error = curr - prev  # (N, A)
     
    if delta != 0.0:
        error_vector = F.huber_loss(
            error, 
            torch.zeros_like(error), # 목표값(target)은 0입니다.
            reduction='none', 
            delta=delta
        )
    else:
        error_vector = error

    return error_vector * scale


def dof_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    v = asset.data.joint_vel                         # (N, dof)
    v_prev = asset.data.prev_joint_vel               # (N, dof)  *사용자 코드 가정*
    return v - v_prev                                # (N, dof)

def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv, scale : float = 0.1) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1) * scale


def error_flat_euler_rp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale : float = 1.0
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)  # (N,), (N,), (N,)
    # [-pi, pi]로 정규화 (사용자 코드 동일)
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi
    return torch.stack((roll, pitch), dim=1)                     # (N,2)



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
        

        # 6. Loss 계산 (Huber or L2/L1)
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
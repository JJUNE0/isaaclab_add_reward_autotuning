
from __future__ import annotations

import torch
import numpy as np
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from lab.flamingo.tasks.manager_based.locomotion.velocity.sensors import LiftMask
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize arm joint velocities using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get arm joint velocities and compute the penalty
    arm_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(arm_joint_vel), dim=1)

def symmetrical_leg_movement_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize asymmetrical leg joint positions and velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Define left and right leg joint indices from the asset_cfg
    # asset_cfg.joint_names should be ordered: [left_hip_yaw, right_hip_yaw, left_hip_roll, right_hip_roll, ...]
    left_leg_joint_ids = asset_cfg.joint_ids[0::2]
    right_leg_joint_ids = asset_cfg.joint_ids[1::2]

    # Positional asymmetry
    left_pos = asset.data.joint_pos[:, left_leg_joint_ids]
    right_pos = asset.data.joint_pos[:, right_leg_joint_ids]
    pos_asymmetry = torch.sum(torch.square(left_pos - right_pos), dim=1)
    
    # Velocity asymmetry
    left_vel = asset.data.joint_vel[:, left_leg_joint_ids]
    right_vel = asset.data.joint_vel[:, right_leg_joint_ids]
    vel_asymmetry = torch.sum(torch.square(left_vel - right_vel), dim=1)
    
    return pos_asymmetry + 0.1 * vel_asymmetry

def periodic_gait_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_frequency: float,
    phase_offset: float = 0.0,
    command_name : str = "base_velocity",
    cmd_threshold : float = 0.1,
) -> torch.Tensor:
    """
    주기적인 신호(sin)를 기반으로 발의 접촉 상태에 따라 보상을 부여합니다.

    Args:
        env: RL 환경 인스턴스.
        sensor_cfg: 발의 접촉을 감지할 ContactSensor 설정.
        gait_frequency: 보행 주파수 (Hz). 1초에 몇 걸음을 걸을지에 대한 목표.
        phase_offset: 위상 오프셋. 보행 시작점을 조절할 때 사용.

    Returns:
        계산된 보상 값.
    """
    cmd = env.command_manager.get_command(command_name)[:, 0]
    # 1. 센서 및 시간 데이터 가져오기
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 왼쪽 발, 오른쪽 발 body_ids 가져오기
    left_foot_id = contact_sensor.find_bodies(".*left_ankle_roll_link")[0]
    right_foot_id = contact_sensor.find_bodies(".*right_ankle_roll_link")[0]

    # 현재 에피소드 진행 시간
    t = env.episode_length_buf * env.step_dt 

    # 2. 주기적인 시계(Clock) 신호 생성
    # t에 따라 -1과 1 사이를 오가는 sin 신호를 생성합니다.
    clock_signal = torch.sin(2 * math.pi * gait_frequency * t + phase_offset)

    # 3. 발의 공중 상태 확인
    # current_air_time이 0보다 크면 공중에 떠 있는 상태입니다.
    is_left_in_air = contact_sensor.data.current_air_time[:, left_foot_id] > 0
    is_right_in_air = contact_sensor.data.current_air_time[:, right_foot_id] > 0

    # 4. 스윙 구간 정의 및 보상 계산
    # clock_signal > 0.1 이면 '왼발 스윙 구간'으로 간주
    # clock_signal < -0.1 이면 '오른발 스윙 구간'으로 간주
    # (0.1의 작은 임계값은 신호가 0 근처일 때의 모호함을 줄여줍니다)
    left_swing_reward = (clock_signal > 0.1) * is_left_in_air.squeeze(1)
    right_swing_reward = (clock_signal < -0.1) * is_right_in_air.squeeze(1)
    
    reward = (left_swing_reward + right_swing_reward).float()
    # 두 보상을 합산하여 최종 보상 결정 (float 타입으로 변환)
    return torch.where(cmd > cmd_threshold, reward, 0)
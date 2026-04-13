# SPDX-License-Identifier: BSD-3-Clause
# POSTECH Flamingo Lab, 2025
# Recovery-specific overrides of common feature functions.
# Event-conditioned variants replace the default implementations.

from __future__ import annotations
import torch
import math
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

from ....mdp.feature_functions_common import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import euler_xyz_from_quat


# ------------------------------------------------------------------
# Recovery-specific overrides (event-conditioned)
# ------------------------------------------------------------------

def error_joint_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "event",
    init_angle: list[float] | float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    event_command = env.command_manager.get_command(command_name)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_angle = asset.data.joint_pos[:, asset_cfg.joint_ids]

    condition = (event_command[:, 0] == 1).unsqueeze(1)

    if isinstance(init_angle, (float, int)):
        target_val = torch.full_like(current_angle, init_angle)
    else:
        target_val = torch.tensor(init_angle, device=env.device).expand_as(current_angle)

    target_angle = torch.where(condition, target_val, torch.zeros_like(current_angle))
    return (target_angle - current_angle) * scale


def error_base_height(
    env: ManagerBasedRLEnv,
    target_height: float | None = None,
    command_name: str = "event",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]
    tgt = torch.full_like(base_z, float(target_height))
    event_command = env.command_manager.get_command(command_name)
    return (base_z - tgt) * scale * (1 - event_command[:, 0])


def error_flat_euler_rp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "event",
    scale: float = 1.0,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    event_command = env.command_manager.get_command(command_name)
    r, p, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
    roll = (r + math.pi) % (2 * math.pi) - math.pi
    pitch = (p + math.pi) % (2 * math.pi) - math.pi
    return torch.stack((roll, pitch), dim=1) * scale * (1 - event_command[:, 0])

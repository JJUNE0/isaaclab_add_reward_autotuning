# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def modify_base_velocity_range(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, mod_range: dict, num_steps: int
):
    """
    Modifies the range of a command term (e.g., base_velocity) in the environment after a specific number of steps.

    Args:
        env: The environment instance.
        term_name: The name of the command term to modify (e.g., "base_velocity").
        end_range: The target range for the term (e.g., {"lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-1.5, 1.5)}).
        activation_step: The step count after which the range modification is applied.
    """
    # Check if the curriculum step exceeds the activation step
    if env.common_step_counter >= num_steps:
        # Get the term object
        command_term = env.command_manager.get_term(term_name)

        # Update the ranges directly
        for key, target_range in mod_range.items():
            if hasattr(command_term.cfg.ranges, key):
                setattr(command_term.cfg.ranges, key, target_range)


def modify_terminataton_condition(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, link_name: list, num_steps: int
):
    """
    Modifies the termination condition of a command term (e.g., base_contact) in the environment after a specific number of steps.

    Args:
        env: The environment instance.
        term_name: The name of the command term to modify (e.g., "base_contact").
        link_name: The name of the link to be monitored for contact.
        num_steps: The step count after which the range modification is applied.
    """
    
    # Check if the curriculum step exceeds the activation step
    if env.common_step_counter >= num_steps:
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        term_cfg.params["sensor_cfg"].body_names = link_name
        term_cfg.params["sensor_cfg"].body_ids = link_name

def modify_reset_base_range(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, mod_range: dict, num_steps: int
):
    """
    Modifies the range of a command term (e.g., base_velocity) in the environment after a specific number of steps.

    Args:
        env: The environment instance.
        term_name: The name of the command term to modify (e.g., "base_velocity").
        end_range: The target range for the term (e.g., {"lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-1.5, 1.5)}).
        activation_step: The step count after which the range modification is applied.
    """
    # Check if the curriculum step exceeds the activation step
    if env.common_step_counter >= num_steps:
        term_cfg = env.event_manager.get_term_cfg(term_name)

        # mod_range의 각 항목에 대해 반복합니다.
        # 예: param_group_to_update = "pose_range"
        #     new_values = {"roll": (-0.5, 0.5), "pitch": (-0.5, 0.5)}
        for param_group_to_update, new_values in mod_range.items():
            
            # 1. 수정하려는 파라미터 그룹(예: "pose_range")이 실제 설정에 존재하는지 확인합니다.
            if param_group_to_update in term_cfg.params:
                
                # 2. 업데이트할 실제 딕셔너리를 가져옵니다.
                target_dict = term_cfg.params[param_group_to_update]

                # 3. 가져온 대상과 새로운 값이 모두 딕셔너리인지 확인하여 안정성을 높입니다.
                if isinstance(target_dict, dict) and isinstance(new_values, dict):
                    
                    # 4. dict.update() 메소드를 사용하여 새로운 값들로 기존 딕셔너리를 업데이트합니다.
                    target_dict.update(new_values)
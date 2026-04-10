from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
from isaaclab.utils import configclass
import numpy as np
import math


from isaaclab.envs.mdp import UniformVelocityCommand, NormalVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg , NormalVelocityCommandCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityWithZCommand(UniformVelocityCommand):
    r"""Command generator that generates a velocity command in SE(2) with an additional position command in Z from uniform distribution.

    The command comprises of a linear velocity in x and y direction, an angular velocity around
    the z-axis, and a position command in z. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    def __init__(self, cfg: UniformVelocityWithZCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator with an additional position command in Z.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # Call the base class constructor
        super().__init__(cfg, env)

        # Update the command buffer to include position z
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.zero_command_b = torch.zeros_like(self.vel_command_b)
        self.time_elapsed = torch.zeros(self.num_envs, device=self.device)  # Time tracker for each environment
        self.initial_choice = None
        self.track_z_flag = (cfg.ranges.pos_z[0] != 0.0 or cfg.ranges.pos_z[1] != 0.0)

    def __str__(self) -> str:
        """Return a string representation of the command generator with position z."""
        msg = super().__str__()
        msg += f"\n\tPosition z range: {self.cfg.ranges.pos_z}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity and position z command in the base frame. Shape is (num_envs, 4)."""
        is_initial_phase = (self.time_elapsed <= self.cfg.initial_phase_time)
        mask = is_initial_phase.unsqueeze(-1)
        vel_command_b = torch.where(mask, self.zero_command_b, self.vel_command_b)
        return vel_command_b

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        # Convert to tensor for easier splitting
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = env_ids_t.numel()

        perm = torch.randperm(n, device=self.device)
        shuffled_env_ids = env_ids_t[perm]

        n_forward = n // 2          
        n_back = n // 10        
                
        forward_env_ids   = shuffled_env_ids[n_forward:]
        backward_env_ids = shuffled_env_ids[:n_back]
        
        # -------- 2) "샘플링 그룹"은 기존 코드 그대로 --------
        if env_ids_t.numel() > 0:
            r = torch.empty(env_ids_t.numel(), device=self.device)
            # Linear velocity - x direction
            self.vel_command_b[env_ids_t, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
            # Linear velocity - y direction
            self.vel_command_b[env_ids_t, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
            # Angular velocity - z direction
            self.vel_command_b[env_ids_t, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

            # Position command - z direction
            if self.track_z_flag:
                # 기존 코드 그대로: gcd 사용
                self.vel_command_b[env_ids_t, 3] = self.gcd(env_ids_t, 5)
            else:
                self.vel_command_b[env_ids_t, 3] = 0.0

                # -------- 3) "고정 값 그룹"은 원하는 값으로 세팅 --------
            if forward_env_ids.numel() > 0:
                # 👉 여기서 원하는 값을 넣으면 됨
                self.vel_command_b[forward_env_ids, 0] = 0.75
                # 그리고 vy, yaw는 전체 고정
                self.vel_command_b[forward_env_ids, 1] = 0.0
                self.vel_command_b[forward_env_ids, 2] = 0.0
                
                if self.track_z_flag:
                    self.vel_command_b[forward_env_ids, 3] = self.gcd(forward_env_ids, 2)
                else:
                    self.vel_command_b[forward_env_ids, 3] = 0.0
            
            # TODO : change back to rotate
            if backward_env_ids.numel() > 0:
                r_back = torch.empty(backward_env_ids.numel(), device=self.device)            
                self.vel_command_b[backward_env_ids, 0] = 0.0
                self.vel_command_b[backward_env_ids, 1] = 0.0
                self.vel_command_b[backward_env_ids, 2] = r_back.uniform_(-2.5, 2.5)
            
            # -------- 4) heading / standing 플래그는 기존처럼 전체 env_ids에 대해 처리 --------
            r = torch.empty(n, device=self.device)
            # Heading target
            if self.cfg.heading_command:
                self.heading_target[env_ids_t] = r.uniform_(*self.cfg.ranges.heading)
                self.is_heading_env[env_ids_t] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            # Standing envs
            self.is_standing_env[env_ids_t] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
            # Standing environment IDs
            
            standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
            if len(standing_env_ids) > 0:
                if self.track_z_flag:
                    # e.g., choose standing z target
                    self.standing_choice = self.gcd(standing_env_ids, 2)
                else:
                    # If pos_z range is zero, initialize with zeros
                    self.standing_choice = torch.zeros_like(
                        standing_env_ids, dtype=torch.float32, device=self.device
                    )
    
    def _update_command(self):
        self.time_elapsed += self._env.step_dt

        reset_env_ids = self._env.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.time_elapsed[reset_env_ids] = 0.0

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :3] = 0.0
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, 3] = self.standing_choice

    def gcd(self, env_ids: Sequence[int], num_categories: int):
        """Generate a categorical distribution for the given number of categories.

        Args:
            env_ids: The environment IDs for which to generate the distribution.
            num_categories: The number of categories to generate.

        Returns:
            The sampled categories.
        """
        if len(env_ids) == 0:
            return torch.tensor([], device=self.device)
    
        probabilities = torch.ones(num_categories, device=self.device) / num_categories  # Uniform probabilities
        categories = torch.linspace(
            self.cfg.ranges.pos_z[0], self.cfg.ranges.pos_z[1], num_categories, device=self.device
        )
        return categories[torch.multinomial(probabilities, len(env_ids), replacement=True)]

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_link_pos_w.clone()
        base_pos_w[:, 2] += 1.0
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_link_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

@configclass
class UniformVelocityWithZCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform velocity with z command generator."""

    class_type: type = UniformVelocityWithZCommand

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges for the velocity and position commands."""

        pos_z: tuple[float, float] = MISSING  # min max [m]

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity and position commands."""

    initial_phase_time: float = 2.0
    """Time for which the initial phase lasts."""
    
    
    
class DiscreteAngularVelocityCommand(UniformVelocityWithZCommand):
    def __init__(self, cfg: DiscreteAngularVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.PI_6 = math.pi / 6.0
        self.PI_3 = math.pi / 3.0

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        ang_vel_raw = self.command[env_ids, 2]
        abs_val = torch.abs(ang_vel_raw)
        sign_val = torch.sign(ang_vel_raw)

        quantized_mag = torch.zeros_like(abs_val)

        mask_mid = (abs_val >= self.PI_6) & (abs_val < self.PI_3)
        quantized_mag[mask_mid] = self.PI_6

        mask_high = abs_val >= self.PI_3
        quantized_mag[mask_high] = self.PI_3

        self.command[env_ids, 2] = quantized_mag * sign_val

@configclass
class DiscreteAngularVelocityCommandCfg(UniformVelocityWithZCommandCfg):
    class_type: type = DiscreteAngularVelocityCommand
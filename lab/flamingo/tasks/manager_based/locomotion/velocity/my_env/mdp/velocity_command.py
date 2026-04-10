from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
from isaaclab.utils import configclass
import numpy as np


from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg


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
        return self.vel_command_b

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
        # Sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # Linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # Linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # Angular velocity - z direction
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # Position command - z direction
        if self.track_z_flag:
            self.vel_command_b[env_ids, 3] = self.gcd(env_ids, 5)
        else:
            self.vel_command_b[env_ids, 3] = 0.0
        # Heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        
        # Standing environment IDs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            # Initialize standing_choice with appropriate values
            if self.track_z_flag:
                self.standing_choice = self.gcd(standing_env_ids, 2)
            else:
                # If pos_z range is zero, initialize with zeros
                self.standing_choice = torch.zeros_like(standing_env_ids, dtype=torch.float32, device=self.device)

    def _update_command(self):
        self.time_elapsed += self._env.step_dt

        initial_phase_env_ids = (self.time_elapsed <= self.cfg.initial_phase_time).nonzero(as_tuple=False).flatten()

        if len(initial_phase_env_ids) > 0:
            self.vel_command_b[initial_phase_env_ids, :3] = 0.0
            if self.track_z_flag:
                self.vel_command_b[initial_phase_env_ids, 3] = self.gcd(initial_phase_env_ids, 5)
            else:
                self.vel_command_b[initial_phase_env_ids, 3] = 0.0

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


class NormalVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: NormalVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: NormalVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)
        # create buffers for zero commands envs
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _resample_command(self, env_ids):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.normal_(mean=self.cfg.ranges.mean_vel[0], std=self.cfg.ranges.std_vel[0])
        self.vel_command_b[env_ids, 0] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.normal_(mean=self.cfg.ranges.mean_vel[1], std=self.cfg.ranges.std_vel[1])
        self.vel_command_b[env_ids, 1] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- angular velocity - yaw direction
        self.vel_command_b[env_ids, 2] = r.normal_(mean=self.cfg.ranges.mean_vel[2], std=self.cfg.ranges.std_vel[2])
        self.vel_command_b[env_ids, 2] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        # update element wise zero velocity command
        # TODO what is zero prob ?
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Sets velocity command to zero for standing envs."""
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()  # TODO check if conversion is needed
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Enforce zero velocity for individual elements
        # TODO: check if conversion is needed
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0



@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = NormalVelocityCommand
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution (in m/s).

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """

        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution (in m/s).

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """

        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""
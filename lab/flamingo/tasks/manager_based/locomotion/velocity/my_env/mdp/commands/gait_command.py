"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformGaitCommandCfg




class GaitCommand(CommandTerm):
    """Command generator that generates gait frequency, phase offset and contact duration."""

    cfg: UniformGaitCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformGaitCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command
        # command format: [frequency, phase offset, contact duration]
        self.gait_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.time_elapsed = torch.zeros(self.num_envs, device=self.device) 
        # create metrics dictionary for logging
        self.metrics = {}

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The gait command. Shape is (num_envs, 3)."""
        return self.gait_command

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def _resample_command(self, env_ids):
        """Resample the gait command for specified environments."""
        # sample gait parameters

        
        num_resamples = len(env_ids)
        device = self.device

        # -- frequency
        freq_range = self.cfg.ranges.frequencies
        self.gait_command[env_ids, 0] = torch.rand(num_resamples, device=device) * (freq_range[1] - freq_range[0]) + freq_range[0]

        # -- phase offset
        offset_range = self.cfg.ranges.offsets
        self.gait_command[env_ids, 1] = torch.rand(num_resamples, device=device) * (offset_range[1] - offset_range[0]) + offset_range[0]

        # -- contact duration
        duration_range = self.cfg.ranges.durations
        self.gait_command[env_ids, 2] = torch.rand(num_resamples, device=device) * (duration_range[1] - duration_range[0]) + duration_range[0]

          # set phase offset to 0 during initial phase

    def _update_command(self):
        self.time_elapsed +=  self._env.step_dt
        initial_phase_env_ids = (self.time_elapsed <= self.cfg.initial_phase_time).nonzero(as_tuple=False).flatten()
        self.gait_command[initial_phase_env_ids, :] = 0.0
        
        reset_env_ids = self._env.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.time_elapsed[reset_env_ids] = 0.0
            self.gait_command[reset_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        In this implementation, we don't provide any debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        In this implementation, we don't provide any debug visualization.
        """
        pass
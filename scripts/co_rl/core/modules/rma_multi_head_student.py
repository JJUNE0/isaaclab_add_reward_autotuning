"""Multi-head RMA student for the successful 2026-08-08 oracle teacher.

The frozen teacher actor consumes two stacked 32-D frames. The last four
values in each frame are simulator-only base linear velocity (xyz) and base
height. This student removes those values before building its history and
predicts them, together with the teacher latent, from deployable observations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .teacher_student import RMAStudent


class MultiHeadAdaptationModule(nn.Module):
    """Shared temporal backbone with latent, velocity, and height heads."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        history_len: int,
        num_actor_frames: int,
        hidden_dims: tuple[int, int, int] = (128, 128, 128),
        kernel_sizes: tuple[int, int, int] = (5, 3, 3),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.history_len = history_len
        self.num_actor_frames = num_actor_frames

        self.input_norm = nn.LayerNorm(input_dim)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims[0], kernel_size=kernel_sizes[0], dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=kernel_sizes[1], dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=kernel_sizes[2], dilation=4),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, history_len)
            try:
                conv_out = self.conv_layers(dummy)
            except RuntimeError as exc:
                raise ValueError(
                    f"history_len={history_len} is too short for the adaptation TCN"
                ) from exc
            flatten_dim = conv_out.flatten(1).shape[1]

        self.shared_mlp = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )
        self.latent_head = nn.Linear(128, latent_dim)
        self.velocity_head = nn.Linear(128, num_actor_frames * 3)
        self.height_head = nn.Linear(128, num_actor_frames)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if history.dim() == 2:
            expected = self.history_len * self.input_dim
            if history.shape[1] != expected:
                raise ValueError(
                    f"flattened history has width {history.shape[1]}, expected {expected}"
                )
            history = history.view(history.shape[0], self.history_len, self.input_dim)
        if history.dim() != 3 or history.shape[1:] != (self.history_len, self.input_dim):
            raise ValueError(
                "history must have shape "
                f"(batch, {self.history_len}, {self.input_dim}), got {tuple(history.shape)}"
            )

        features = self.input_norm(history).permute(0, 2, 1)
        features = self.conv_layers(features).flatten(1)
        features = self.shared_mlp(features)

        z_hat = self.latent_head(features)
        velocity_hat = self.velocity_head(features).view(-1, self.num_actor_frames, 3)
        height_hat = self.height_head(features).view(-1, self.num_actor_frames, 1)
        return z_hat, velocity_hat, height_hat


class RMAStudentMultiHead(RMAStudent):
    """Distill an oracle teacher while explicitly estimating its GT actor terms.

    The simulator supplies the teacher's full policy observation for supervised
    targets. ``update_history`` strips every GT state slot before storing data,
    and ``forward`` reconstructs those slots exclusively from multi-head
    predictions. The same method accepts an already stripped deployable
    observation at inference time.
    """

    is_multi_head_student = True

    def __init__(
        self,
        teacher_policy,
        num_obs: int,
        num_envs: int,
        device: str = "cpu",
        num_single_obs: int = 32,
        num_stacks: int = 2,
        latent_dim: int = 16,
        history_len: int = 50,
        observable_single_obs_dim: int = 28,
        state_dim: int = 4,
        z_loss_coef: float = 1.0,
        velocity_loss_coef: float = 1.0,
        height_loss_coef: float = 10.0,
        action_loss_coef: float = 1.0,
        **kwargs,
    ):
        if state_dim != 4:
            raise ValueError("RMAStudentMultiHead requires state_dim=4 (velocity xyz + height)")
        if num_single_obs - observable_single_obs_dim != state_dim:
            raise ValueError(
                "teacher frame layout mismatch: expected the final four frame terms to be "
                "base velocity xyz and base height"
            )

        super().__init__(
            teacher_policy=teacher_policy,
            num_obs=num_obs,
            num_envs=num_envs,
            device=device,
            num_single_obs=num_single_obs,
            num_stacks=num_stacks,
            latent_dim=latent_dim,
            history_len=history_len,
            **kwargs,
        )

        self.teacher_num_obs = num_obs
        self.teacher_single_obs_dim = num_single_obs
        self.num_actor_frames = num_stacks
        self.observable_single_obs_dim = observable_single_obs_dim
        self.state_dim = state_dim
        self.nonstack_obs_dim = num_obs - num_single_obs * num_stacks
        if self.nonstack_obs_dim <= 0:
            raise ValueError("teacher observation must include a non-stacked command block")

        self.deployable_num_obs = observable_single_obs_dim * num_stacks + self.nonstack_obs_dim
        self.num_obs = self.deployable_num_obs
        self.single_obs_dim = observable_single_obs_dim
        self.exterio_dim = self.nonstack_obs_dim
        self.frame_dim = observable_single_obs_dim + self.nonstack_obs_dim

        # Replace the base student's GT-contaminated buffers with deployable-only buffers.
        self.history_buffer = torch.zeros(
            num_envs, history_len, self.frame_dim, device=device
        )
        self.current_obs = torch.zeros(num_envs, self.deployable_num_obs, device=device)
        self.adaptation_module = MultiHeadAdaptationModule(
            input_dim=self.frame_dim,
            latent_dim=latent_dim,
            history_len=history_len,
            num_actor_frames=num_stacks,
        )

        self.z_loss_coef = z_loss_coef
        self.velocity_loss_coef = velocity_loss_coef
        self.height_loss_coef = height_loss_coef
        self.action_loss_coef = action_loss_coef

        print(
            "Multi-head RMA Student initialized: "
            f"teacher_obs={self.teacher_num_obs}, deployable_obs={self.deployable_num_obs}, "
            f"history_frame={self.frame_dim}, actor_frames={self.num_actor_frames}"
        )

    def _strip_teacher_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deployable observation, accepting full or stripped input."""
        if obs.shape[-1] == self.deployable_num_obs:
            return obs
        if obs.shape[-1] != self.teacher_num_obs:
            raise ValueError(
                f"observation width must be {self.teacher_num_obs} (teacher) or "
                f"{self.deployable_num_obs} (deployable), got {obs.shape[-1]}"
            )

        stacked_width = self.teacher_single_obs_dim * self.num_actor_frames
        frames = obs[:, :stacked_width].view(
            -1, self.num_actor_frames, self.teacher_single_obs_dim
        )
        observable_frames = frames[:, :, : self.observable_single_obs_dim].flatten(1)
        nonstack_obs = obs[:, stacked_width:]
        return torch.cat((observable_frames, nonstack_obs), dim=-1)

    def update_history(self, obs: torch.Tensor, dones: torch.Tensor | None = None):
        deployable_obs = self._strip_teacher_state(obs)
        self.current_obs.copy_(deployable_obs)

        latest_observable = deployable_obs[:, : self.observable_single_obs_dim]
        latest_command = deployable_obs[:, -self.nonstack_obs_dim :]
        latest_frame = torch.cat((latest_observable, latest_command), dim=-1)

        self.history_buffer[:, :-1] = self.history_buffer[:, 1:].clone()
        self.history_buffer[:, -1] = latest_frame

        if dones is not None:
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                self.history_buffer[reset_ids] = latest_frame[reset_ids].unsqueeze(1).expand(
                    -1, self.history_len, -1
                )

    def _reconstruct_teacher_obs(
        self,
        deployable_obs: torch.Tensor,
        velocity_hat: torch.Tensor,
        height_hat: torch.Tensor,
    ) -> torch.Tensor:
        observable_width = self.observable_single_obs_dim * self.num_actor_frames
        observable_frames = deployable_obs[:, :observable_width].view(
            -1, self.num_actor_frames, self.observable_single_obs_dim
        )
        estimated_state = torch.cat((velocity_hat, height_hat), dim=-1)
        teacher_frames = torch.cat((observable_frames, estimated_state), dim=-1).flatten(1)
        nonstack_obs = deployable_obs[:, observable_width:]
        teacher_obs = torch.cat((teacher_frames, nonstack_obs), dim=-1)
        if teacher_obs.shape[-1] != self.teacher_num_obs:
            raise RuntimeError(
                f"reconstructed teacher obs has width {teacher_obs.shape[-1]}, "
                f"expected {self.teacher_num_obs}"
            )
        return teacher_obs

    def forward(self, obs=None):
        z_hat, velocity_hat, height_hat = self.adaptation_module(self.history_buffer)
        reconstructed_obs = self._reconstruct_teacher_obs(
            self.current_obs, velocity_hat, height_hat
        )
        action_mean = self.actor(torch.cat((reconstructed_obs, z_hat), dim=-1))
        return action_mean, z_hat

    def act_inference(self, obs=None):
        with torch.no_grad():
            action_mean, _ = self.forward()
            return action_mean

    def compute_distillation_loss(
        self, history_batch: torch.Tensor, critic_obs_batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute multi-task supervision against the frozen 808 teacher."""
        teacher_obs = critic_obs_batch[:, : self.teacher_num_obs]
        encoder_input_dim = self.teacher_encoder.net[0].in_features
        privileged_info = critic_obs_batch[:, -encoder_input_dim:]

        z_hat, velocity_hat, height_hat = self.adaptation_module(history_batch)

        stacked_width = self.teacher_single_obs_dim * self.num_actor_frames
        teacher_frames = teacher_obs[:, :stacked_width].view(
            -1, self.num_actor_frames, self.teacher_single_obs_dim
        )
        velocity_target = teacher_frames[:, :, -self.state_dim : -1]
        height_target = teacher_frames[:, :, -1:]

        deployable_obs = self._strip_teacher_state(teacher_obs)
        reconstructed_obs = self._reconstruct_teacher_obs(
            deployable_obs, velocity_hat, height_hat
        )
        student_action = self.actor(torch.cat((reconstructed_obs, z_hat), dim=-1))

        with torch.no_grad():
            z_target = self.teacher_encoder(privileged_info)
            teacher_action = self.actor(torch.cat((teacher_obs, z_target), dim=-1))

        z_loss = F.mse_loss(z_hat, z_target)
        velocity_loss = F.mse_loss(velocity_hat, velocity_target)
        height_loss = F.mse_loss(height_hat, height_target)
        action_loss = F.mse_loss(student_action, teacher_action)
        total_loss = (
            self.z_loss_coef * z_loss
            + self.velocity_loss_coef * velocity_loss
            + self.height_loss_coef * height_loss
            + self.action_loss_coef * action_loss
        )

        infos = {
            "z_loss": z_loss.item(),
            "velocity_loss": velocity_loss.item(),
            "height_loss": height_loss.item(),
            "action_loss": action_loss.item(),
            "velocity_mae": torch.mean(torch.abs(velocity_hat - velocity_target)).item(),
            "height_mae": torch.mean(torch.abs(height_hat - height_target)).item(),
            "action_mae": torch.mean(torch.abs(student_action - teacher_action)).item(),
            "teacher_z": z_target.detach().cpu(),
            "student_z": z_hat.detach().cpu(),
        }
        return total_loss, infos

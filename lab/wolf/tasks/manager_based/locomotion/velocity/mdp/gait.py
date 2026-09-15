"""Phase-based gait commands, observations and rewards for Wolf gait V0.

The gait term intentionally has no hidden mutable phase state.  Isaac Lab computes
rewards before observations in a manager step, so deriving phase from the local
episode step keeps both terms side-effect free and makes resets deterministic.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .oracle import FEET


class FixedGaitCommand(CommandTerm):
    """A fixed gait command (phase offsets and stepping frequency)."""

    def __init__(self, cfg: FixedGaitCommandCfg, env):
        super().__init__(cfg, env)
        self._command = torch.tensor(
            [cfg.theta1, cfg.theta2, cfg.theta3, cfg.frequency],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        return None

    def _resample_command(self, env_ids: Sequence[int]):
        # V0 is deliberately fixed.  Keeping a command term still gives the phase
        # generator the same interface needed for later frequency/gait randomization.
        return None

    def _update_command(self):
        return None


@configclass
class FixedGaitCommandCfg(CommandTermCfg):
    """Configuration for the fixed V0 gait command."""

    class_type: type = FixedGaitCommand
    theta1: float = 0.5
    theta2: float = 0.0
    theta3: float = 0.0
    frequency: float = 2.0
    duty_factor: float = 0.5


def command(env) -> torch.Tensor:
    """Return ``[theta1, theta2, theta3, frequency]`` for all environments."""

    return env.command_manager.get_command("gait")


def phase(env) -> torch.Tensor:
    """Return the global gait phase in ``[0, 1)`` for each environment.

    ``episode_length_buf`` is incremented before reward computation and remains
    zero for the reset observation, so this is the endpoint phase convention:
    the post-step reward and observation describe the same phase.
    """

    gait_command = command(env)
    frequency = gait_command[:, 3]
    return torch.remainder(env.episode_length_buf.to(torch.float32) * env.step_dt * frequency, 1.0)


def foot_phase(env) -> torch.Tensor:
    """Return phases in the canonical ``[FL, FR, BL, BR]`` order."""

    gait_command = command(env)
    global_phase = phase(env)
    theta1, theta2, theta3 = gait_command[:, 0], gait_command[:, 1], gait_command[:, 2]
    offsets = torch.stack(
        [theta1 + theta2 + theta3, theta2, theta3, theta1], dim=-1
    )
    return torch.remainder(global_phase[:, None] + offsets, 1.0)


def desired_contact(env, duty_factor: float | None = None) -> torch.Tensor:
    """Return hard desired stance contacts (one means stance)."""

    if duty_factor is None:
        duty_factor = env.command_manager.get_term("gait").cfg.duty_factor
    if not 0.0 < duty_factor < 1.0:
        raise ValueError(f"duty_factor must be in (0, 1), got {duty_factor}")
    return (foot_phase(env) < duty_factor).to(torch.float32)


def clock(env) -> torch.Tensor:
    """Return sine and cosine gait clocks, eight values in total."""

    foot_phases = foot_phase(env)
    angles = 2.0 * torch.pi * foot_phases
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def _foot_sensor(env):
    sensor = env.scene["contact_forces"]
    ids, names = sensor.find_bodies(FEET, preserve_order=True)
    if names != FEET:
        raise RuntimeError(f"Unexpected foot order: {names}; expected {FEET}")
    return sensor, ids


def _foot_contact_force_norm(env) -> torch.Tensor:
    sensor, ids = _foot_sensor(env)
    return torch.linalg.vector_norm(sensor.data.net_forces_w[:, ids], dim=-1)


def _foot_velocity_world(env) -> torch.Tensor:
    robot = env.scene["robot"]
    ids, names = robot.find_bodies(FEET, preserve_order=True)
    if names != FEET:
        raise RuntimeError(f"Unexpected foot order: {names}; expected {FEET}")
    # Use world link-origin velocity.  The existing oracle foot_state is base
    # relative and is intentionally not used for stance contact velocity.
    return robot.data.body_lin_vel_w[:, ids]


def swing_force_penalty(env, force_scale: float = 150.0) -> torch.Tensor:
    """Bounded cost for contact force on feet scheduled to be in swing."""

    force = _foot_contact_force_norm(env)
    swing = 1.0 - desired_contact(env)
    normalized = force / max(float(force_scale), 1.0e-6)
    return torch.mean(swing * (1.0 - torch.exp(-torch.square(normalized))), dim=-1)


def stance_velocity_penalty(env, velocity_scale: float = 0.5) -> torch.Tensor:
    """Bounded cost for world foot velocity on feet scheduled to be in stance."""

    velocity = torch.linalg.vector_norm(_foot_velocity_world(env), dim=-1)
    stance = desired_contact(env)
    normalized = velocity / max(float(velocity_scale), 1.0e-6)
    return torch.mean(stance * (1.0 - torch.exp(-torch.square(normalized))), dim=-1)

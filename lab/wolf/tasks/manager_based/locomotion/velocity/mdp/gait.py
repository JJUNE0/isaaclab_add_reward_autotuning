"""Phase-based gait commands, observations and rewards for Wolf gait tasks.

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


def _foot_contact_history(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return current and previous foot force magnitudes in canonical order.

    The contact sensor is configured with a three-frame history.  Keeping the
    transition test in the sensor history makes the reward term stateless and
    avoids a Python-side buffer that would have to be reset for a subset of
    environments.  The fallback is useful for unit tests or a scene that was
    created without history enabled.
    """

    sensor, ids = _foot_sensor(env)
    history = getattr(sensor.data, "net_forces_w_history", None)
    if history is None or history.shape[1] < 2:
        current = torch.linalg.vector_norm(sensor.data.net_forces_w[:, ids], dim=-1)
        return current, current
    force_history = torch.linalg.vector_norm(history[:, :, ids], dim=-1)
    return force_history[:, -1], force_history[:, -2]


def foot_slip_penalty(
    env,
    velocity_scale: float = 0.5,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize horizontal foot motion while a foot is in contact.

    This follows the ``feet_slip`` term from Walk These Ways: current or
    immediately previous contact gates the penalty, and only horizontal foot
    velocity is used.  The exponential form keeps the Wolf term in the same
    bounded ``[0, 1]`` range as the V0 gait costs while retaining the paper's
    velocity-squared shaping.
    """

    current_force, previous_force = _foot_contact_history(env)
    contact = (current_force > force_threshold) | (previous_force > force_threshold)
    foot_velocity_xy = torch.linalg.vector_norm(_foot_velocity_world(env)[..., :2], dim=-1)
    normalized = foot_velocity_xy / max(float(velocity_scale), 1.0e-6)
    cost = 1.0 - torch.exp(-torch.square(normalized))
    return torch.mean(contact.to(cost.dtype) * cost, dim=-1)


def _swing_phase(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized swing progress and the corresponding swing mask.

    A foot is in stance for ``phase < duty_factor``.  During the remaining
    part of the cycle, progress runs from zero at lift-off to one at touchdown.
    """

    term_cfg = env.command_manager.get_term("gait").cfg
    duty_factor = float(term_cfg.duty_factor)
    if not 0.0 < duty_factor < 1.0:
        raise ValueError(f"duty_factor must be in (0, 1), got {duty_factor}")
    phases = foot_phase(env)
    desired = (phases < duty_factor).to(torch.float32)
    progress = torch.clamp((phases - duty_factor) / (1.0 - duty_factor), 0.0, 1.0)
    return progress, 1.0 - desired


def foot_clearance_penalty(
    env,
    target_height: float = 0.08,
    height_offset: float = 0.02,
    height_scale: float = 0.04,
    reference_height: float = 0.0,
) -> torch.Tensor:
    """Penalize swing-foot height error against a triangular clearance target.

    ``reference_height`` is the flat-ground height for the V1 flat task.  A
    terrain-relative reference (per-foot ray query) should replace it before
    enabling this term on stairs; using a fixed world height there would reward
    feet that are below a raised step.
    """

    progress, swing = _swing_phase(env)
    # Walk These Ways uses a triangular phase target with a small foot-radius
    # offset.  ``target_height`` is the peak clearance above the reference.
    triangle = 1.0 - torch.abs(2.0 * progress - 1.0)
    desired_height = float(reference_height) + float(height_offset) + float(target_height) * triangle
    robot = env.scene["robot"]
    ids, names = robot.find_bodies(FEET, preserve_order=True)
    if names != FEET:
        raise RuntimeError(f"Unexpected foot order: {names}; expected {FEET}")
    foot_height = robot.data.body_pos_w[:, ids, 2]
    normalized = (desired_height - foot_height) / max(float(height_scale), 1.0e-6)
    cost = 1.0 - torch.exp(-torch.square(normalized))
    return torch.mean(swing * cost, dim=-1)


def foot_impact_velocity_penalty(
    env,
    velocity_scale: float = 0.75,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize downward velocity on a newly contacting foot.

    Contact transitions come from the force history.  Isaac Lab's articulation
    data exposes the post-step foot velocity, so this is a conservative current
    velocity proxy for the pre-impact value used by the original implementation;
    it still gates the cost to touchdown frames rather than every stance frame.
    """

    current_force, previous_force = _foot_contact_history(env)
    touchdown = (current_force > force_threshold) & (previous_force <= force_threshold)
    downward_speed = torch.relu(-_foot_velocity_world(env)[..., 2])
    normalized = downward_speed / max(float(velocity_scale), 1.0e-6)
    cost = 1.0 - torch.exp(-torch.square(normalized))
    return torch.mean(touchdown.to(cost.dtype) * cost, dim=-1)


def maximum_contact_force_penalty(
    env,
    max_contact_force: float = 250.0,
    force_scale: float = 150.0,
) -> torch.Tensor:
    """Penalize foot force above a configurable maximum.

    The hinge follows Walk These Ways' ``relu(||F|| - F_max)`` term.  It is
    normalized and smoothly bounded here so a single rare impact cannot dwarf
    velocity tracking while the threshold remains easy to interpret in N.
    """

    force = _foot_contact_force_norm(env)
    excess = torch.relu(force - float(max_contact_force))
    normalized = excess / max(float(force_scale), 1.0e-6)
    cost = 1.0 - torch.exp(-normalized)
    return torch.mean(cost, dim=-1)

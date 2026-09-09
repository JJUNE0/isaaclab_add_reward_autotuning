"""cfg-driven obs/action runtime for MuJoCo sim2sim eval, built from a training
run's dumped ``params/env.yaml`` + ``params/agent.yaml`` instead of a hardcoded
observation/action contract.

Isaac Lab's own ``dump_yaml`` (via ``configclass``) writes the fully-resolved
env cfg (after every task's ``__post_init__`` override) to
``<run_dir>/params/env.yaml`` at the start of every training run, and the
runner cfg (incl. ``num_policy_stacks``) to ``<run_dir>/params/agent.yaml``.
Both parse with plain ``yaml.UnsafeLoader`` -- no ``isaaclab``/``pxr`` import
needed -- because Isaac Lab's yaml dumper serializes class references as
``"module.path:ClassName"`` strings rather than live objects.

This module reads that recipe (obs term list/order/params/scale/clip, action
term list/order/joint_names/scale/use_default_offset, actuator gains, default
joint pose) and replays it against live MuJoCo state, so a change to the
*cfg* (new obs term, different scale, different num_policy_stacks, ...)
requires zero changes here -- only a rerun of training (which redumps
``params/env.yaml``). The one thing that does need a code change here is a
genuinely new obs *function* the numpy registry below doesn't know about yet
(``FUNC_IMPLS`` raises a clear error naming the missing function in that
case) -- unavoidable without paying the isaaclab/Isaac Sim import+launch cost
inside a lightweight headless MuJoCo script.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Isaac Lab's native (URDF-import) joint order, needed to correctly resolve
# preserve_order=True wildcard obs terms (e.g. joint_names=".*") the same way
# training did. This is a structural fact about the robot asset, not a reward/
# obs/action cfg choice -- see dump_native_joint_order.py's docstring.
_DEFAULT_NATIVE_JOINT_ORDER_PATH = Path(__file__).resolve().parent / "tita_native_joint_order.json"

# ---------------------------------------------------------------------------
# Recipe loading
# ---------------------------------------------------------------------------

# obs-group metadata keys that are not observation terms.
_OBS_GROUP_META_KEYS = {
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
}


@dataclass
class ObsTermSpec:
    name: str
    func: str
    params: dict
    scale: float | list | None
    clip: tuple | None


@dataclass
class ActionTermSpec:
    name: str
    joint_names: list[str]
    scale: float
    offset: float
    use_default_offset: bool
    is_velocity: bool


@dataclass
class ActuatorSpec:
    joint_regexes: list[str]
    stiffness: float
    damping: float
    effort_limit: float


@dataclass
class RunSpec:
    obs_stack_terms: list[ObsTermSpec]
    obs_nonstack_terms: list[ObsTermSpec]
    action_terms: list[ActionTermSpec]
    actuators: list[ActuatorSpec]
    default_joint_pos: dict[str, float]
    num_policy_stacks: int
    native_joint_order: list[str]
    command_name: str = "base_velocity"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.UnsafeLoader)


def _collect_obs_terms(group: dict) -> list[ObsTermSpec]:
    terms = []
    for name, term in group.items():
        if name in _OBS_GROUP_META_KEYS or term is None:
            continue
        clip = term.get("clip")
        terms.append(
            ObsTermSpec(
                name=name,
                func=term["func"],
                params=term.get("params") or {},
                scale=term.get("scale"),
                clip=tuple(clip) if clip is not None else None,
            )
        )
    return terms


def _collect_action_terms(actions: dict) -> list[ActionTermSpec]:
    terms = []
    for name, term in actions.items():
        if term is None:
            continue
        class_type = term.get("class_type", "")
        is_velocity = "JointVelocityAction" in class_type
        joint_names = term["joint_names"]
        if isinstance(joint_names, str):
            joint_names = [joint_names]
        terms.append(
            ActionTermSpec(
                name=name,
                joint_names=list(joint_names),
                scale=float(term.get("scale", 1.0)),
                offset=float(term.get("offset", 0.0)),
                use_default_offset=bool(term.get("use_default_offset", False)),
                is_velocity=is_velocity,
            )
        )
    return terms


def _collect_actuators(robot: dict) -> list[ActuatorSpec]:
    specs = []
    for actuator in (robot.get("actuators") or {}).values():
        if actuator is None:
            continue
        stiffness = actuator.get("stiffness") or {}
        damping = actuator.get("damping") or {}
        stiffness_val = next(iter(stiffness.values()), 0.0) if isinstance(stiffness, dict) else float(stiffness or 0.0)
        damping_val = next(iter(damping.values()), 0.0) if isinstance(damping, dict) else float(damping or 0.0)
        specs.append(
            ActuatorSpec(
                joint_regexes=list(actuator.get("joint_names_expr") or []),
                stiffness=float(stiffness_val),
                damping=float(damping_val),
                effort_limit=float(actuator.get("effort_limit") or 0.0),
            )
        )
    return specs


def load_run_spec(
    run_dir: str | Path,
    policy_group: str = "stack_policy",
    nonstack_group: str = "none_stack_policy",
    native_joint_order_path: str | Path = _DEFAULT_NATIVE_JOINT_ORDER_PATH,
) -> RunSpec:
    """Parse ``<run_dir>/params/{env,agent}.yaml`` into a :class:`RunSpec`."""
    run_dir = Path(run_dir)
    params_dir = run_dir / "params"
    env_cfg = _load_yaml(params_dir / "env.yaml")
    agent_cfg = _load_yaml(params_dir / "agent.yaml")

    obs = env_cfg["observations"]
    stack_terms = _collect_obs_terms(obs[policy_group]) if obs.get(policy_group) else []
    nonstack_terms = _collect_obs_terms(obs[nonstack_group]) if obs.get(nonstack_group) else []

    action_terms = _collect_action_terms(env_cfg["actions"])
    actuators = _collect_actuators(env_cfg["scene"]["robot"])
    default_joint_pos = dict(env_cfg["scene"]["robot"]["init_state"]["joint_pos"])

    native_joint_order = json.loads(Path(native_joint_order_path).read_text())["joint_names"]

    return RunSpec(
        obs_stack_terms=stack_terms,
        obs_nonstack_terms=nonstack_terms,
        action_terms=action_terms,
        actuators=actuators,
        default_joint_pos=default_joint_pos,
        num_policy_stacks=int(agent_cfg.get("num_policy_stacks", 0) or 0),
        native_joint_order=native_joint_order,
    )


# ---------------------------------------------------------------------------
# Joint-name resolution (binds by NAME, never by a hardcoded index
# permutation). Training resolves preserve_order=True wildcard obs terms
# against Isaac Lab's native (URDF-import) joint order, which is generally
# NOT the same order as the runtime's own data arrays (e.g. a MuJoCo config's
# declared joint list) -- so resolution happens against ``resolution_names``
# (native order, for correct match ORDER) but returns indices into
# ``target_names`` (the runtime's actual data order), keeping the two
# concerns cleanly separated.
# ---------------------------------------------------------------------------


def resolve_joint_indices(
    patterns: list[str] | str,
    resolution_names: list[str],
    target_names: list[str] | None = None,
) -> np.ndarray:
    """Resolve a regex (or list of regexes) against ``resolution_names``,
    Isaac-Lab style: each pattern is matched in turn, in ``resolution_names``
    order, and matches are concatenated in pattern order (mirrors
    ``preserve_order`` resolution against a single asset's native joint
    order). Returns, for each matched joint (in that resolution order), its
    index in ``target_names`` (defaults to ``resolution_names`` itself)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    if target_names is None:
        target_names = resolution_names
    target_index = {name: i for i, name in enumerate(target_names)}
    indices: list[int] = []
    seen: set[str] = set()
    for pattern in patterns:
        regex = re.compile(f"^{pattern}$")
        for name in resolution_names:
            if regex.match(name) and name not in seen:
                seen.add(name)
                indices.append(target_index[name])
    return np.asarray(indices, dtype=np.int64)


# ---------------------------------------------------------------------------
# Numpy reimplementations of the mdp observation functions this project
# actually uses. Keyed by the exact "module.path:func_name" string Isaac
# Lab's dumper writes -- a genuinely new obs function needs a new entry here.
# ---------------------------------------------------------------------------


def _impl_base_ang_vel(ctx, **params):
    return ctx.base_ang_vel.copy()


def _impl_projected_gravity(ctx, **params):
    return ctx.projected_gravity.copy()


def _impl_last_action(ctx, **params):
    return ctx.last_action.copy()


def _impl_generated_scaled_commands(ctx, command_name: str, scale, **params):
    command = ctx.command.copy()
    scale = np.asarray(scale, dtype=np.float32)
    command[: scale.shape[0]] *= scale
    return command


def _impl_generated_commands(ctx, command_name: str, **params):
    return ctx.command.copy()


# Must match lab.flamingo...mdp.feature_functions_common._HEIGHT_NORM_CENTER/_HALF_RANGE.
_HEIGHT_NORM_CENTER = 0.35
_HEIGHT_NORM_HALF_RANGE = 0.10


def _impl_generated_commands_height_normalized(ctx, command_name: str, vel_scale=(1.0, 1.0, 1.0), **params):
    command = ctx.command.copy()
    scale = np.asarray(vel_scale, dtype=np.float32)
    command[: scale.shape[0]] *= scale
    command[3] = (command[3] - _HEIGHT_NORM_CENTER) / _HEIGHT_NORM_HALF_RANGE
    return command


def _impl_joint_pos_rel_without_wheel(ctx, asset_cfg: dict, wheel_asset_cfg: dict, **params):
    # ids: for each joint matched by asset_cfg in NATIVE (training) order,
    # its index into ctx.joint_pos (the runtime's own data order) -- so
    # rel[ids] reorders runtime data into the order training expects.
    ids = resolve_joint_indices(asset_cfg["joint_names"], ctx.native_joint_names, ctx.joint_names)
    wheel_ids = resolve_joint_indices(wheel_asset_cfg["joint_names"], ctx.native_joint_names, ctx.joint_names)
    rel = ctx.joint_pos - ctx.default_joint_pos
    rel = rel.copy()
    rel[wheel_ids] = 0.0
    return rel[ids]


def _impl_joint_vel_rel(ctx, asset_cfg: dict, **params):
    ids = resolve_joint_indices(asset_cfg["joint_names"], ctx.native_joint_names, ctx.joint_names)
    # default joint velocity is always 0 for this robot.
    return ctx.joint_vel[ids]


def _impl_joint_pos(ctx, asset_cfg: dict, **params):
    # absolute joint position (no default-pos subtraction), unlike joint_pos_rel*.
    ids = resolve_joint_indices(asset_cfg["joint_names"], ctx.native_joint_names, ctx.joint_names)
    return ctx.joint_pos[ids]


def _impl_joint_vel(ctx, asset_cfg: dict, **params):
    ids = resolve_joint_indices(asset_cfg["joint_names"], ctx.native_joint_names, ctx.joint_names)
    return ctx.joint_vel[ids]


def _impl_base_lin_vel_x_link(ctx, **params):
    return ctx.base_lin_vel[0:1].copy()


def _impl_base_lin_vel_y_link(ctx, **params):
    return ctx.base_lin_vel[1:2].copy()


def _impl_base_lin_vel_z_link(ctx, **params):
    return ctx.base_lin_vel[2:3].copy()


def _impl_base_pos_z_rel_link(ctx, sensor_cfg=None, **params):
    # sensor_cfg (terrain-relative height via raycaster) isn't modeled by
    # this flat-terrain runtime; only the sensor_cfg=None (absolute world z)
    # case is supported.
    if sensor_cfg is not None:
        raise NotImplementedError(
            "base_pos_z_rel_link with sensor_cfg (terrain-relative height) has no numpy impl."
        )
    return ctx.base_z.copy()


FUNC_IMPLS = {
    "isaaclab.envs.mdp.observations:base_ang_vel": _impl_base_ang_vel,
    "isaaclab.envs.mdp.observations:projected_gravity": _impl_projected_gravity,
    "isaaclab.envs.mdp.observations:last_action": _impl_last_action,
    "isaaclab.envs.mdp.observations:joint_vel_rel": _impl_joint_vel_rel,
    "isaaclab.envs.mdp.observations:joint_pos": _impl_joint_pos,
    "isaaclab.envs.mdp.observations:joint_vel": _impl_joint_vel,
    "lab.flamingo.tasks.moo_based.locomotion.velocity.mdp.observations:generated_scaled_commands": _impl_generated_scaled_commands,
    "lab.flamingo.tasks.moo_based.locomotion.velocity.mdp.observations:generated_commands": _impl_generated_commands,
    "lab.flamingo.tasks.moo_based.locomotion.velocity.mdp.observations:joint_pos_rel_without_wheel": _impl_joint_pos_rel_without_wheel,
    # root_link_* frame is what the MuJoCo runtime's own base_ang_vel input already is.
    "lab.flamingo.tasks.moo_based.locomotion.velocity.mdp.observations:base_ang_vel_link": _impl_base_ang_vel,
    "lab.flamingo.tasks.moo_based.locomotion.velocity.mdp.observations:base_lin_vel_x_link": _impl_base_lin_vel_x_link,
    "lab.flamingo.tasks.moo_based.locomotion.velocity.mdp.observations:base_lin_vel_y_link": _impl_base_lin_vel_y_link,
    "lab.flamingo.tasks.moo_based.locomotion.velocity.mdp.observations:base_lin_vel_z_link": _impl_base_lin_vel_z_link,
    "lab.flamingo.tasks.moo_based.locomotion.velocity.mdp.observations:base_pos_z_rel_link": _impl_base_pos_z_rel_link,
    "lab.flamingo.tasks.moo_based.locomotion.velocity.mdp.observations:generated_commands_height_normalized": _impl_generated_commands_height_normalized,
}


@dataclass
class _ObsContext:
    joint_names: list[str]
    native_joint_names: list[str]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    default_joint_pos: np.ndarray
    base_ang_vel: np.ndarray
    projected_gravity: np.ndarray
    last_action: np.ndarray
    command: np.ndarray
    base_lin_vel: np.ndarray
    base_z: np.ndarray


def _eval_term(term: ObsTermSpec, ctx: _ObsContext) -> np.ndarray:
    impl = FUNC_IMPLS.get(term.func)
    if impl is None:
        raise NotImplementedError(
            f"No numpy implementation registered for obs func '{term.func}' "
            f"(term '{term.name}'). Add one to FUNC_IMPLS in tita_cfg_runtime.py."
        )
    value = np.atleast_1d(np.asarray(impl(ctx, **term.params), dtype=np.float32))
    if term.scale is not None:
        value = value * np.asarray(term.scale, dtype=np.float32)
    if term.clip is not None:
        value = np.clip(value, term.clip[0], term.clip[1])
    return value.astype(np.float32)


def _eval_group(terms: list[ObsTermSpec], ctx: _ObsContext) -> np.ndarray:
    if not terms:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([_eval_term(t, ctx) for t in terms])


# ---------------------------------------------------------------------------
# Observation builder: replicates StateHandler's stacking semantics exactly
# (fill-by-repeat on reset, prepend-new/drop-oldest on step, concat order is
# [newest, ..., oldest] then the non-stacked group).
# ---------------------------------------------------------------------------


class ObsBuilder:
    def __init__(self, spec: RunSpec, joint_names: list[str]):
        self.spec = spec
        self.joint_names = joint_names
        self.default_joint_pos = np.asarray(
            [spec.default_joint_pos[name] for name in joint_names], dtype=np.float32
        )
        self.total_frames = spec.num_policy_stacks + 1
        self._buffer: list[np.ndarray] | None = None

    def _make_context(
        self, joint_pos, joint_vel, base_ang_vel, projected_gravity, last_action, command, base_lin_vel=None, base_z=None
    ) -> _ObsContext:
        return _ObsContext(
            joint_names=self.joint_names,
            native_joint_names=self.spec.native_joint_order,
            joint_pos=np.asarray(joint_pos, dtype=np.float32),
            joint_vel=np.asarray(joint_vel, dtype=np.float32),
            default_joint_pos=self.default_joint_pos,
            base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32),
            projected_gravity=np.asarray(projected_gravity, dtype=np.float32),
            last_action=np.asarray(last_action, dtype=np.float32),
            command=np.asarray(command, dtype=np.float32),
            base_lin_vel=np.zeros(3, dtype=np.float32) if base_lin_vel is None else np.asarray(base_lin_vel, dtype=np.float32),
            base_z=np.zeros(1, dtype=np.float32) if base_z is None else np.atleast_1d(np.asarray(base_z, dtype=np.float32)),
        )

    def reset(self, joint_pos, joint_vel, base_ang_vel, projected_gravity, last_action, command, base_lin_vel=None, base_z=None) -> np.ndarray:
        ctx = self._make_context(joint_pos, joint_vel, base_ang_vel, projected_gravity, last_action, command, base_lin_vel, base_z)
        stack_frame = _eval_group(self.spec.obs_stack_terms, ctx)
        nonstack_frame = _eval_group(self.spec.obs_nonstack_terms, ctx)
        self._buffer = [stack_frame.copy() for _ in range(self.total_frames)]
        return np.concatenate([*self._buffer, nonstack_frame]).astype(np.float32)

    def step(self, joint_pos, joint_vel, base_ang_vel, projected_gravity, last_action, command, base_lin_vel=None, base_z=None) -> np.ndarray:
        if self._buffer is None:
            return self.reset(joint_pos, joint_vel, base_ang_vel, projected_gravity, last_action, command, base_lin_vel, base_z)
        ctx = self._make_context(joint_pos, joint_vel, base_ang_vel, projected_gravity, last_action, command, base_lin_vel, base_z)
        stack_frame = _eval_group(self.spec.obs_stack_terms, ctx)
        nonstack_frame = _eval_group(self.spec.obs_nonstack_terms, ctx)
        self._buffer = [stack_frame.copy()] + self._buffer[:-1]
        return np.concatenate([*self._buffer, nonstack_frame]).astype(np.float32)


# ---------------------------------------------------------------------------
# Action -> torque adapter, cfg-driven (per-term joint/scale/offset resolved
# from the dumped ActionsCfg; PD gains resolved from the dumped ActuatorCfg).
# ---------------------------------------------------------------------------


class ActionTorqueAdapter:
    def __init__(self, spec: RunSpec, joint_names: list[str]):
        self.joint_names = joint_names
        self.default_joint_pos = np.asarray(
            [spec.default_joint_pos[name] for name in joint_names], dtype=np.float32
        )
        num_joints = len(joint_names)

        # Each action term contributes one raw-action slot PER joint it
        # covers (in the term's declared joint_names order), not one slot
        # per term -- a term like "shoudler_leg_joint_pos" with 4 joint
        # names occupies 4 consecutive slots in the ONNX policy's flat
        # action vector. Flatten term -> joints here so the slot list lines
        # up 1:1 with that flat vector regardless of how many joints any
        # single term covers.
        slot_joint_idx: list[int] = []
        slot_scale: list[float] = []
        slot_offset: list[float] = []
        slot_use_default_offset: list[bool] = []
        slot_is_velocity: list[bool] = []
        for term in spec.action_terms:
            ids = resolve_joint_indices(term.joint_names, spec.native_joint_order, joint_names)
            if len(ids) == 0:
                raise ValueError(
                    f"Action term '{term.name}' resolved to 0 joints ({term.joint_names})."
                )
            for joint_idx in ids:
                slot_joint_idx.append(int(joint_idx))
                slot_scale.append(term.scale)
                slot_offset.append(term.offset)
                slot_use_default_offset.append(term.use_default_offset)
                slot_is_velocity.append(term.is_velocity)

        self._term_joint_idx = np.asarray(slot_joint_idx, dtype=np.int64)
        self._term_scale = np.asarray(slot_scale, dtype=np.float32)
        self._term_offset = np.asarray(slot_offset, dtype=np.float32)
        self._term_use_default_offset = np.asarray(slot_use_default_offset, dtype=bool)
        self._term_is_velocity = np.asarray(slot_is_velocity, dtype=bool)

        self.kp = np.zeros(num_joints, dtype=np.float32)
        self.kd = np.zeros(num_joints, dtype=np.float32)
        self.effort_limit = np.full(num_joints, np.inf, dtype=np.float32)
        for actuator in spec.actuators:
            ids = resolve_joint_indices(actuator.joint_regexes, spec.native_joint_order, joint_names)
            self.kp[ids] = actuator.stiffness
            self.kd[ids] = actuator.damping
            self.effort_limit[ids] = actuator.effort_limit

    def torque(self, action_policy_order: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """``action_policy_order`` must be in the exact per-term order the
        run's ActionsCfg declared (i.e. the raw ONNX policy output order)."""
        offset = np.where(
            self._term_use_default_offset,
            self.default_joint_pos[self._term_joint_idx],
            self._term_offset,
        )
        targets = action_policy_order * self._term_scale + offset

        torque = np.zeros_like(q)
        for i, joint_idx in enumerate(self._term_joint_idx):
            kp = self.kp[joint_idx]
            kd = self.kd[joint_idx]
            if self._term_is_velocity[i]:
                t = kd * (targets[i] - qd[joint_idx])
            else:
                t = kp * (targets[i] - q[joint_idx]) - kd * qd[joint_idx]
            limit = self.effort_limit[joint_idx]
            torque[joint_idx] = np.clip(t, -limit, limit)
        return torque

    @property
    def action_dim(self) -> int:
        return len(self._term_joint_idx)

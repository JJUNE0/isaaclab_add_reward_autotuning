"""Simulator ground truth, expressed in the robot frame where appropriate."""
import torch
from isaaclab.utils.math import quat_apply_inverse

FEET = [f"Foot_{leg}_link" for leg in ("front_left", "front_right", "back_left", "back_right")]
JOINTS = [f"{joint}_{leg}_joint" for leg in ("front_left", "front_right", "back_left", "back_right")
          for joint in (("HAA", "HFE", "KFE") if leg.startswith("front") else ("HAA", "HFE", "KFE", "AFE"))]


def foot_state(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_bodies(FEET, preserve_order=True)
    q = robot.data.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
    pos = robot.data.body_pos_w[:, ids] - robot.data.root_pos_w[:, None, :]
    vel = robot.data.body_lin_vel_w[:, ids] - robot.data.root_lin_vel_w[:, None, :]
    return torch.cat([quat_apply_inverse(q, x.reshape(-1, 3)).reshape(env.num_envs, -1) for x in (pos, vel)], dim=-1)


def contact_forces(env):
    sensor = env.scene["contact_forces"]
    ids, _ = sensor.find_bodies(FEET, preserve_order=True)
    q = env.scene["robot"].data.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
    force = sensor.data.net_forces_w[:, ids]
    return quat_apply_inverse(q, force.reshape(-1, 3)).reshape(env.num_envs, -1)


def body_masses(env):
    return env.scene["robot"].root_physx_view.get_masses().to(env.device)


def terrain_level(env):
    return env.scene.terrain.terrain_levels[:, None].float() / max(env.scene.terrain.cfg.terrain_generator.num_rows - 1, 1)


def terrain_curriculum(env, env_ids):
    terrain = env.scene.terrain
    distance = torch.linalg.vector_norm(env.scene["robot"].data.root_pos_w[env_ids, :2] - terrain.env_origins[env_ids, :2], dim=-1)
    advance = distance > terrain.cfg.terrain_generator.size[0] / 2.0
    expected = torch.linalg.vector_norm(env.command_manager.get_command("base_velocity")[env_ids, :2], dim=-1) * env.max_episode_length_s
    regress = (distance < expected * 0.5) & ~advance
    terrain.update_env_origins(env_ids, advance, regress)
    return terrain.terrain_levels.float().mean()


def out_of_patch(env):
    delta = (env.scene["robot"].data.root_pos_w[:, :2] - env.scene.terrain.env_origins[:, :2]).abs()
    return (delta > env.scene.terrain.cfg.terrain_generator.size[0] * 0.5).any(dim=-1)

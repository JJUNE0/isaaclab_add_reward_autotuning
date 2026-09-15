"""Fixed-level/speed evaluation with pre-reset episode accounting; no learning."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--flat", action="store_true", help="Replace all stair tiles with flat mesh terrain; oracle level zero")
parser.add_argument("--telemetry", action="store_true", help="Save per-physics-step foot forces and joint telemetry")
parser.add_argument("--replicas", type=int, default=8)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--output", default="outputs/results_20260915/evaluation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.replicas < 1 or args.episodes < 1:
    parser.error("replicas and episodes must be positive")
app = AppLauncher(args).app
try:
    import json
    import types
    import torch
    import gymnasium as gym
    import lab.wolf.tasks
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper
    from scripts.co_rl.core.runners import OnPolicyRunner
    task = "Isaac-Velocity-Stairs-Wolf-v2-Oracle-ppo-Play"
    n = 16 * args.replicas
    cfg = parse_env_cfg(task, device=args.device, num_envs=n)
    cfg.seed = 42
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
    if args.flat:
        from isaaclab.terrains import MeshPlaneTerrainCfg
        cfg.scene.terrain.terrain_generator.sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}
    raw = gym.make(task, cfg=cfg)
    base = raw.unwrapped
    terrain = base.scene.terrain
    indices = torch.arange(n, device=base.device)
    levels = (indices // args.replicas) % 8
    speeds = (indices // (8 * args.replicas)).float() + 1.0
    if args.flat:
        levels = torch.zeros_like(levels)
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = indices % terrain.cfg.terrain_generator.num_cols
    terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
    command = base.command_manager.get_term("base_velocity")
    original_resample = command._resample_command
    def fixed_resample(self, ids):
        original_resample(ids)
        self.vel_command_b[ids, 0] = speeds[ids]
    command._resample_command = types.MethodType(fixed_resample, command)
    agent = load_cfg_from_registry(task, "co_rl_cfg_entry_point")
    env = CoRlVecEnvWrapper(raw, agent)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
    runner.load(str(Path(args.checkpoint).resolve()), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
    obs, _ = env.reset()
    completed = torch.zeros(n, dtype=torch.long, device=base.device)
    velocity_sum = torch.zeros(n, device=base.device)
    errors = torch.zeros(n, device=base.device)
    frame_count = torch.zeros(n, device=base.device)
    records = []
    telemetry = []
    sampling = {"enabled": False}
    if args.telemetry:
        import numpy as np
        from lab.wolf.tasks.manager_based.locomotion.velocity.mdp.oracle import FEET, JOINTS
        robot = base.scene["robot"]
        sensor = base.scene["contact_forces"]
        foot_ids, foot_names = sensor.find_bodies(FEET, preserve_order=True)
        joint_ids, joint_names = robot.find_joints(JOINTS, preserve_order=True)
        assert foot_names == FEET and joint_names == JOINTS
        original_update = base.scene.update
        def telemetry_update(self, dt):
            original_update(dt)
            if not sampling["enabled"]:
                return
            active = completed < args.episodes
            ids = indices[active]
            if ids.numel() == 0:
                return
            data = robot.data
            values = torch.cat([
                ids[:, None], completed[ids, None], levels[ids, None], speeds[ids, None],
                (base.episode_length_buf[ids] * base.step_dt +
                 ((base._sim_step_counter - 1) % base.cfg.decimation + 1) * dt)[:, None],
                sensor.data.net_forces_w[ids][:, foot_ids].reshape(-1, 12),
                data.joint_pos[ids][:, joint_ids], data.joint_vel[ids][:, joint_ids],
                data.applied_torque[ids][:, joint_ids]], dim=1)
            assert torch.isfinite(values).all()
            telemetry.append(values.detach().cpu().numpy())
        base.scene.update = types.MethodType(telemetry_update, base.scene)
    original_reset = base._reset_idx
    def capture_reset(self, ids):
        robot = self.scene["robot"]
        pos = robot.data.root_pos_w[ids] - terrain.env_origins[ids]
        reasons = {key: self.termination_manager.get_term(key)[ids].detach().cpu().tolist()
                   for key in self.termination_manager.active_terms}
        for i, idx in enumerate(ids.detach().cpu().tolist()):
            if completed[idx] < args.episodes:
                record = {"env": idx, "level": int(levels[idx]), "speed": float(speeds[idx]),
                          "episode": int(completed[idx]), "seconds": float(self.episode_length_buf[idx]) * self.step_dt,
                          "final_relative_xyz": pos[i].detach().cpu().tolist(),
                          "mean_base_vx": float(velocity_sum[idx] / frame_count[idx].clamp_min(1)),
                          "mean_abs_vx_error": float(errors[idx] / frame_count[idx].clamp_min(1)),
                          "terminations": {key: bool(values[i]) for key, values in reasons.items()}}
                record["clean_forward_exit"] = (record["terminations"]["out_of_patch"] and
                    not record["terminations"]["base_contact"] and not record["terminations"]["overturned"] and
                    record["final_relative_xyz"][0] > 5.0)
                records.append(record)
                completed[idx] += 1
        velocity_sum[ids] = 0
        errors[ids] = 0
        frame_count[ids] = 0
        was_sampling = sampling["enabled"]
        sampling["enabled"] = False
        try:
            original_reset(ids)
        finally:
            sampling["enabled"] = was_sampling
    base._reset_idx = types.MethodType(capture_reset, base)
    for step in range(args.episodes * 1000 + 10):
        vx = base.scene["robot"].data.root_lin_vel_b[:, 0]
        velocity_sum += vx
        errors += (vx - speeds).abs()
        frame_count += 1
        with torch.inference_mode():
            actions = policy(obs)
            sampling["enabled"] = True
            obs, reward, done, info = env.step(actions)
            sampling["enabled"] = False
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        if step % 250 == 0:
            print(f"EVAL_STEP {step} completed={int(completed.sum())}/{n * args.episodes}", flush=True)
        if bool((completed >= args.episodes).all()):
            break
    assert bool((completed >= args.episodes).all()), "Incomplete episode quota"
    groups = []
    for speed in (1.0, 2.0):
        for level in ([0] if args.flat else range(8)):
            rows = [r for r in records if r["speed"] == speed and r["level"] == level]
            count = len(rows)
            groups.append({"speed": speed, "level": level, "episodes": count,
                "clean_forward_exits": sum(r["clean_forward_exit"] for r in rows),
                "failures": sum(r["terminations"]["base_contact"] or r["terminations"]["overturned"] for r in rows),
                "timeouts": sum(r["terminations"]["time_out"] for r in rows),
                "mean_seconds": sum(r["seconds"] for r in rows)/count,
                "mean_base_vx": sum(r["mean_base_vx"] for r in rows)/count,
                "mean_abs_vx_error": sum(r["mean_abs_vx_error"] for r in rows)/count})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"terrain": "flat" if args.flat else "stairs", "checkpoint": str(Path(args.checkpoint).resolve()), "seed": cfg.seed,
              "replicas": args.replicas, "episodes_per_replica": args.episodes,
              "groups": groups, "records": records, "policy_updates": 0}
    if args.telemetry:
        samples = np.concatenate(telemetry)
        np.savez_compressed(output / "telemetry.npz", samples=samples)
        metadata = {"columns": ["env", "episode", "level", "command_vx", "episode_seconds"] +
                    [f"force_world_{axis}:{foot}" for foot in foot_names for axis in "xyz"] +
                    [f"{kind}:{joint}" for kind in ("position_rad", "velocity_rad_s", "applied_torque_Nm") for joint in joint_names],
                    "feet": foot_names, "joints": joint_names, "physics_dt": base.physics_dt,
                    "terrain": "flat" if args.flat else "stairs",
                    "expected_episodes": n * args.episodes,
                    "samples": len(samples), "checkpoint": str(Path(args.checkpoint).resolve()),
                    "force_definition": "world-frame net resultant per foot body, sampled every physics step before reset",
                    "torque_definition": "ArticulationData.applied_torque, simulated actuator applied torque"}
        (output / "telemetry_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"TELEMETRY_PASS samples={len(samples)}", flush=True)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2))
    print("EVALUATION_PASS " + json.dumps(groups), flush=True)
    env.close()
except Exception:
    import traceback
    traceback.print_exc()
    raise
else:
    app.close()

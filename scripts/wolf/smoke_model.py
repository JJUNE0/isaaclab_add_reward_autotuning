"""Headless Wolf v2 asset/physics check; does not create a learner."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")
app = AppLauncher(args).app

try:
    import json
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from lab.flamingo.assets.wolf_v2 import wolf_v2_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    # Local cuboid avoids a network dependency on the default ground-plane USD.
    ground = sim_utils.CuboidCfg(size=(20.0, 20.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg())
    ground.func("/World/Ground", ground, translation=(0.0, 0.0, -0.05))
    cfg = wolf_v2_cfg().replace(prim_path="/World/Wolf")
    robot = Articulation(cfg)
    sim.reset()
    expected = set(cfg.actuators)
    assert set(robot.joint_names) == expected, robot.joint_names
    assert robot.num_joints == 14
    feet = [f"Foot_{leg}_link" for leg in ("front_left", "front_right", "back_left", "back_right")]
    assert set(feet).issubset(robot.body_names), robot.body_names
    limits = robot.data.joint_pos_limits[0]
    assert torch.all(robot.data.default_joint_pos[0] >= limits[:, 0])
    assert torch.all(robot.data.default_joint_pos[0] <= limits[:, 1])
    for _ in range(args.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        assert torch.isfinite(robot.data.root_state_w).all()
        assert torch.isfinite(robot.data.joint_pos).all()
        assert torch.isfinite(robot.data.joint_vel).all()
        assert torch.isfinite(robot.data.applied_torque).all()
    print("WOLF_MODEL_SMOKE_PASS " + json.dumps({
        "steps": args.steps, "device": str(robot.device), "joint_names": robot.joint_names,
        "body_count": robot.num_bodies, "feet": feet,
        "final_base_z": robot.data.root_pos_w[0, 2].item(),
        "mass_kg": robot.root_physx_view.get_masses().sum().item(),
        "source": cfg.spawn.asset_path,
    }), flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    raise
else:
    # Release asset callbacks before Kit closes the stage (same order as the env).
    del robot
    sim.clear_all_callbacks()
    sim.clear_instance()
    app.close()

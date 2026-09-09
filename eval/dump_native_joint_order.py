"""One-time dump of Isaac Lab's native (URDF-import) joint order for a robot
asset, needed because ``preserve_order=True`` wildcard obs terms (e.g.
``joint_names=".*"``) resolve against *that* order at training time -- which
does not necessarily match any convenient/declared order (e.g. the MuJoCo
sim2sim config's joint list). Isaac Lab's native order is a structural fact
about the URDF import, not something exposed in the dumped ``params/env.yaml``,
so it has to be queried once via a live (but otherwise idle) Isaac Lab launch
and cached here. Re-run this only if the robot asset/URDF changes; it is
independent of reward/obs/action cfg edits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Velocity-Flat-Tita-Play-v3-moo-ppo")
parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "tita_native_joint_order.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from lab.flamingo.isaaclab.isaaclab.envs import ManagerBasedMOORLEnvCfg  # noqa: F401  (triggers task registration)
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.joint_names)
    Path(args_cli.out).write_text(json.dumps({"task": args_cli.task, "joint_names": joint_names}, indent=2))
    print(f"[RESULT] native joint order for {args_cli.task}: {joint_names}")
    print(f"[RESULT] written to {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

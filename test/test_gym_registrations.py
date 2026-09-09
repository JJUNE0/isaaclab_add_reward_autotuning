# SPDX-License-Identifier: BSD-3-Clause
"""Contract test: Verify Gym environment registrations for Flamingo baseline and TITA."""

import sys
import os

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

import unittest
from isaaclab.app import AppLauncher

# Launch Omniverse AppLauncher in headless mode before importing IsaacLab / Gym modules
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import lab.flamingo.tasks


class TestGymRegistrations(unittest.TestCase):
    def test_flamingo_registration(self):
        """1. Verify existing Flamingo Gym registration is maintained."""
        task_id = "Isaac-Velocity-Flat-Flamingo-v3-moo-ppo"
        self.assertIn(task_id, gym.envs.registry, f"Flamingo task {task_id} not registered.")
        spec = gym.spec(task_id)
        self.assertEqual(spec.entry_point, "lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv")

    def test_tita_registration(self):
        """2. Verify TITA Gym registration."""
        task_id = "Isaac-Velocity-Flat-Tita-v3-moo-ppo"
        self.assertIn(task_id, gym.envs.registry, f"TITA task {task_id} not registered.")
        spec = gym.spec(task_id)
        self.assertEqual(spec.entry_point, "lab.flamingo.isaaclab.isaaclab.envs:ManagerBasedMOORLEnv")

        play_task_id = "Isaac-Velocity-Flat-Tita-Play-v3-moo-ppo"
        self.assertIn(play_task_id, gym.envs.registry, f"TITA play task {play_task_id} not registered.")


if __name__ == "__main__":
    try:
        unittest.main(exit=False)
    finally:
        simulation_app.close()

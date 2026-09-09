# SPDX-License-Identifier: BSD-3-Clause
"""Phase 3 Contract Tests: Rigorous empirical verification of TITA MOO-PPO integration."""

import os
import sys
import unittest

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from isaaclab.app import AppLauncher

# Launch Omniverse AppLauncher in headless mode before importing IsaacLab/Gym components
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import tempfile
import torch
import gymnasium as gym

import lab.flamingo.tasks  # noqa: F401
from lab.flamingo.tasks.moo_based.locomotion.velocity.tita_env.flat_env.stand_drive.flat_env_stand_drive_cfg import TitaFlatEnvCfg
from lab.flamingo.tasks.moo_based.locomotion.velocity.tita_env.agents.co_rl_cfg import TitaMOOPPORunnerCfg_Flat_Stand_Drive
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper
from scripts.co_rl.core.runners import MOO_OnPolicyRunner


class TestTitaContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task_id = "Isaac-Velocity-Flat-Tita-v3-moo-ppo"
        cls.env_cfg = TitaFlatEnvCfg()
        cls.env_cfg.scene.num_envs = 4  # Small batch for fast contract testing
        cls.agent_cfg = TitaMOOPPORunnerCfg_Flat_Stand_Drive()

        # Instantiate actual headless Gym environment and wrapper runner
        cls.raw_env = gym.make(cls.task_id, cfg=cls.env_cfg)
        cls.env = CoRlVecEnvWrapper(cls.raw_env, cls.agent_cfg)
        cls.runner = MOO_OnPolicyRunner(cls.env, cls.agent_cfg.to_dict(), log_dir=None, device=cls.env.device)
        cls.runner.logger_type = None

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_01_and_02_gym_registrations(self):
        """Contract Test 1 & 2: Verify Flamingo and TITA Gym registrations."""
        self.assertIn("Isaac-Velocity-Flat-Flamingo-v3-moo-ppo", gym.envs.registry)
        self.assertIn("Isaac-Velocity-Flat-Tita-v3-moo-ppo", gym.envs.registry)

    def test_03_joint_regex_selection(self):
        """Contract Test 3: Verify joint regexes select exactly 2 joints each."""
        robot = self.raw_env.unwrapped.scene["robot"]
        hip_ids, _ = robot.find_joints(".*_leg_1")
        shoulder_ids, _ = robot.find_joints(".*_leg_2")
        leg_ids, _ = robot.find_joints(".*_leg_3")
        wheel_ids, _ = robot.find_joints(".*_leg_4")

        self.assertEqual(len(hip_ids), 2, f"Hip regex expected 2 joints, found {len(hip_ids)}")
        self.assertEqual(len(shoulder_ids), 2, f"Shoulder regex expected 2 joints, found {len(shoulder_ids)}")
        self.assertEqual(len(leg_ids), 2, f"Leg regex expected 2 joints, found {len(leg_ids)}")
        self.assertEqual(len(wheel_ids), 2, f"Wheel regex expected 2 joints, found {len(wheel_ids)}")

    def test_04_action_dimension(self):
        """Contract Test 4: Verify action dimension is actually 8."""
        action_dim = self.env.num_actions
        self.assertEqual(action_dim, 8, f"Expected action dimension 8, got {action_dim}")

    def test_05_nominal_joint_deviation(self):
        """Contract Test 5: Verify nominal TITA posture joint deviation evaluates to near zero."""
        unwrapped_env = self.raw_env.unwrapped
        robot = unwrapped_env.scene["robot"]

        # Reset joints explicitly to default positions for nominal pose testing
        default_pos = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos))

        diff = (robot.data.joint_pos - default_pos).abs()
        max_dev = torch.max(diff).item()
        self.assertLess(max_dev, 1e-4, f"Nominal joint deviation expected < 1e-4, got {max_dev}")

    def test_06_moo_term_order_and_shapes(self):
        """Contract Test 6: Verify MOO term order and individual output dimensions."""
        reward_manager = self.raw_env.unwrapped.moo_reward_manager
        expected_shapes = [
            ("error_track_lin_vel_xy", 2),
            ("error_base_height", 1),
            ("error_track_ang_vel_z", 1),
            ("error_flat_euler_rp", 2),
            ("error_hip_deviation", 2),
            ("error_shoulder_deviation", 2),
            ("error_leg_deviation", 2),
            ("action_smoothness_error", 1),
            ("torque_rate_error", 1),
        ]

        self.assertEqual(len(reward_manager._term_cfgs), len(expected_shapes))
        for idx, (expected_name, expected_dim) in enumerate(expected_shapes):
            term_cfg = reward_manager._term_cfgs[idx]
            term_shape = reward_manager._term_shapes[idx]
            self.assertEqual(
                term_shape,
                expected_dim,
                f"Term {expected_name} at index {idx} expected shape {expected_dim}, got {term_shape}"
            )

    def test_07_delta_shape(self):
        """Contract Test 7: Verify actual calculated delta shape is (num_envs, 14)."""
        reward_manager = self.raw_env.unwrapped.moo_reward_manager
        self.assertEqual(reward_manager.delta_dim, 14, f"Expected delta_dim 14, got {reward_manager.delta_dim}")
        delta = reward_manager.get_latest_delta(recompute=True, detach=True)
        self.assertEqual(delta.shape, (self.env.num_envs, 14), f"Expected delta shape ({self.env.num_envs}, 14), got {delta.shape}")

    def test_08_delta_finiteness(self):
        """Contract Test 8: Verify all delta entries are finite (no NaN/Inf)."""
        reward_manager = self.raw_env.unwrapped.moo_reward_manager
        delta = reward_manager.get_latest_delta(recompute=True, detach=True)
        self.assertTrue(torch.isfinite(delta).all().item(), "NaN or Inf detected in calculated delta vector!")

    def test_09_action_smoothness_stateful_reset(self):
        """Contract Test 9: Verify ActionRatePenalty resets prev_prev_action per environment without leakage."""
        reward_manager = self.raw_env.unwrapped.moo_reward_manager
        action_term = None
        for term_cfg in reward_manager._term_cfgs:
            if hasattr(term_cfg.func, "_prev_prev_action"):
                action_term = term_cfg.func
                break

        self.assertIsNotNone(action_term, "ActionRatePenalty instance not found in reward terms.")

        env_unwrapped = self.raw_env.unwrapped
        # Call term to initialize state
        _ = action_term(env_unwrapped, delta=0.0, scale=1.0)
        self.assertIsNotNone(action_term._prev_prev_action)

        # Force per-environment reset simulation
        reset_ids = torch.tensor([0], device=self.env.device, dtype=torch.long)
        action_term._prev_prev_action[reset_ids] = 99.0
        action_term.reset(reset_ids)

        self.assertTrue(
            torch.allclose(action_term._prev_prev_action[0], torch.zeros(8, device=self.env.device)),
            "ActionRatePenalty state did not reset to zero for reset env 0."
        )

    def test_10_torque_rate_stateful_reset(self):
        """Contract Test 10: Verify TorqueRatePenalty resets prev_torque per environment without leakage."""
        reward_manager = self.raw_env.unwrapped.moo_reward_manager
        torque_term = None
        for term_cfg in reward_manager._term_cfgs:
            if hasattr(term_cfg.func, "_prev_torque"):
                torque_term = term_cfg.func
                break

        self.assertIsNotNone(torque_term, "TorqueRatePenalty instance not found in reward terms.")

        env_unwrapped = self.raw_env.unwrapped
        _ = torque_term(env_unwrapped, asset_cfg=torque_term.asset_cfg, max_torque=60.0, delta=0.1, scale=1.0)
        self.assertIsNotNone(torque_term._prev_torque)

        reset_ids = torch.tensor([0], device=self.env.device, dtype=torch.long)
        torque_term.reset(reset_ids)
        self.assertFalse(torch.isnan(torque_term._prev_torque).any().item(), "NaN found in TorqueRatePenalty prev_torque")

    def test_11_pre_reset_delta_computation(self):
        """Contract Test 11: Verify reward and extras['delta'] are computed BEFORE reset."""
        actions = torch.zeros((self.env.num_envs, 8), device=self.env.device)
        obs, rewards, dones, infos = self.env.step(actions)

        self.assertIn("delta", infos, "extras['delta'] missing from step returns!")
        self.assertEqual(infos["delta"].shape, (self.env.num_envs, 14), f"Expected delta shape ({self.env.num_envs}, 14)")
        self.assertTrue(torch.isfinite(infos["delta"]).all().item(), "NaN/Inf found in step extras['delta']")

    def test_12_timeout_bootstrapping(self):
        """Contract Test 12: Verify terminated/time_out separation and timeout bootstrapping."""
        actions = torch.zeros((self.env.num_envs, 8), device=self.env.device)
        obs, rewards, dones, infos = self.env.step(actions)

        self.assertIn("time_outs", infos, "infos['time_outs'] missing from step returns.")
        self.assertEqual(infos["time_outs"].shape[0], self.env.num_envs)

    def test_13_rollout_minibatch_alignment(self):
        """Contract Test 13: Verify rollout storage minibatch yields aligned obs and delta rows."""
        storage = self.runner.alg.storage

        obs_tensor, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs_tensor)
        actions = self.runner.alg.act(obs_tensor, critic_obs)

        obs, rewards, dones, infos = self.env.step(actions)
        self.runner.alg.process_env_step(rewards, dones, infos)

        generator = storage.mini_batch_generator(num_mini_batches=2, num_epochs=1)
        for obs_b, critic_obs_b, actions_b, delta_b, target_val_b, adv_b, ret_b, old_logp_b, old_mu_b, old_sigma_b, _, _ in generator:
            self.assertEqual(obs_b.shape[0], delta_b.shape[0], "Obs and Delta batch sizes mismatch!")
            self.assertEqual(delta_b.shape[1], 14, "Delta minibatch dimension mismatch!")

    def test_14_discriminator_context_consistency(self):
        """Contract Test 14: Verify discriminator reward calculation and update use identical obs context & element-wise layout."""
        single_obs_dim = self.runner.alg.single_obs_dim  # 28
        self.assertEqual(single_obs_dim, 28, f"Expected single_obs_dim 28, got {single_obs_dim}")

        obs_tensor, extras = self.env.get_observations()
        self.assertEqual(obs_tensor.shape[1], 32, f"Expected total policy obs dim 32, got {obs_tensor.shape[1]}")

        captured_inputs = []

        def forward_hook(module, args):
            # args[0] = delta, args[1] = obs
            delta_arg, obs_arg = args[0], args[1]
            captured_inputs.append((delta_arg.clone(), obs_arg.clone()))

        hook_handle = self.runner.alg.discriminator.register_forward_pre_hook(forward_hook)

        try:
            # 1) Rollout/Reward computation context
            actions = torch.zeros((self.env.num_envs, 8), device=self.env.device)
            _ = self.runner.alg.act(obs_tensor, extras["observations"].get("critic", obs_tensor))
            obs, rewards, dones, infos = self.env.step(actions)

            # 2) Discriminator update context
            delta_batch = infos["delta"]
            disc_loss, disc_prob = self.runner.alg._update_discriminator(
                delta_batch, obs_tensor[:, :single_obs_dim]
            )

            # Verification of captured inputs
            self.assertGreaterEqual(len(captured_inputs), 2, "Forward hook failed to capture discriminator calls!")
            compute_delta, compute_obs = captured_inputs[0]
            update_delta, update_obs = captured_inputs[-1]

            # Dimension & layout invariants
            self.assertEqual(compute_delta.shape[-1], 14, "Delta dimension must be 14!")
            self.assertEqual(compute_obs.shape[-1], 28, "Discriminator context obs dimension must be 28!")
            self.assertEqual(
                self.runner.alg.discriminator.net[0].in_features,
                42,
                "Discriminator first-layer input features must be 28 + 14 = 42!",
            )

            # Element-wise feature layout equality:
            # compute_obs is captured during env.step() from group 'stack_policy' (28D)
            # update_obs is captured during _update_discriminator() from obs_batch[:, :28] (28D)
            self.assertEqual(compute_obs.shape, (self.env.num_envs, 28))
            self.assertEqual(update_obs.shape, (self.env.num_envs, 28))
            self.assertTrue(
                torch.allclose(update_obs, obs_tensor[:, :single_obs_dim]),
                "Discriminator update context obs does not match policy obs stack_policy slice!",
            )
        finally:
            hook_handle.remove()

    def test_15_checkpoint_roundtrip(self):
        """Contract Test 15: Verify checkpoint save and load round-trip."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_model.pt")
            self.runner.save(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path), "Checkpoint file was not created!")

            loaded_infos = self.runner.load(ckpt_path)
            self.assertIsNotNone(self.runner.alg.actor_critic)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTitaContracts)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    simulation_app.close()
    if not result.wasSuccessful():
        sys.exit(1)

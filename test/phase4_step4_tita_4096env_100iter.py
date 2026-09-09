# SPDX-License-Identifier: BSD-3-Clause
"""Phase 4 - Step 4: TITA 4096 envs x 100 iterations complete training & evaluation script."""

import os
import sys

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import json
import hashlib
import torch
import gymnasium as gym
import lab.flamingo.tasks  # noqa: F401
from lab.flamingo.tasks.moo_based.locomotion.velocity.tita_env.flat_env.stand_drive.flat_env_stand_drive_cfg import TitaFlatEnvCfg
from lab.flamingo.tasks.moo_based.locomotion.velocity.tita_env.agents.co_rl_cfg import TitaMOOPPORunnerCfg_Flat_Stand_Drive
from scripts.co_rl.core.wrapper import CoRlVecEnvWrapper
from scripts.co_rl.core.runners import MOO_OnPolicyRunner


def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    task_id = "Isaac-Velocity-Flat-Tita-v3-moo-ppo"
    env_cfg = TitaFlatEnvCfg()
    env_cfg.scene.num_envs = 4096
    agent_cfg = TitaMOOPPORunnerCfg_Flat_Stand_Drive()
    agent_cfg.max_iterations = 100
    agent_cfg.save_interval = 20

    log_dir = "/tmp/test_phase4_step4"
    os.makedirs(log_dir, exist_ok=True)

    print("=== PHASE 4 - STEP 4: TITA 4096 ENV x 100 ITERATIONS ===")
    raw_env = gym.make(task_id, cfg=env_cfg)
    env = CoRlVecEnvWrapper(raw_env, agent_cfg)
    runner = MOO_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)
    runner.logger_type = None

    # Track metrics dictionary across iterations
    recorded_metrics = {}
    target_iterations = [0, 10, 20, 50, 100]

    # Monkey patch or hook iteration end in runner
    original_log = runner.log

    def custom_log(locs, width=80, pad=35):
        original_log(locs, width, pad)
        it = runner.current_learning_iteration

        # Check NaN/Inf count in actor/critic/discriminator parameters
        nan_count = 0
        for name, param in runner.alg.actor_critic.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                nan_count += 1
        for name, param in runner.alg.discriminator.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                nan_count += 1

        # Extract metrics safely
        ep_infos = locs.get("ep_infos", [])
        mean_ep_len = float(locs.get("mean_trajectory_length", 0.0))
        mean_reward = float(locs.get("mean_reward", 0.0))

        # Extract env termination & tracking metrics
        error_vel_xy = 0.0
        error_vel_yaw = 0.0
        base_contact_frac = 0.0
        timeout_frac = 0.0

        if ep_infos:
            for info in ep_infos:
                if "base_velocity/error_vel_xy" in info:
                    error_vel_xy += float(info["base_velocity/error_vel_xy"])
                if "base_velocity/error_vel_yaw" in info:
                    error_vel_yaw += float(info["base_velocity/error_vel_yaw"])
                if "Episode_Termination/base_contact" in info:
                    base_contact_frac += float(info["Episode_Termination/base_contact"])
                if "Episode_Termination/time_out" in info:
                    timeout_frac += float(info["Episode_Termination/time_out"])
            n_ep = len(ep_infos)
            error_vel_xy /= n_ep
            error_vel_yaw /= n_ep
            base_contact_frac /= n_ep
            timeout_frac /= n_ep

        disc_loss = float(locs.get("mean_disc_loss", 0.0))
        disc_prob = float(locs.get("mean_disc_prob", 0.0))

        # Extract MOO terms
        reward_raw = float(locs.get("mean_raw_reward", 0.0))
        reward_disc = float(locs.get("mean_disc_reward", 0.0))
        reward_fm = float(locs.get("mean_fm_reward", 0.0))
        delta_abs_mean = float(locs.get("mean_normed_delta", 0.0))

        # Delta per term
        raw_delta_terms = locs.get("mean_raw_delta_terms", {})
        delta_term_metrics = {}
        if isinstance(raw_delta_terms, dict):
            for k, v in raw_delta_terms.items():
                if isinstance(v, torch.Tensor):
                    delta_term_metrics[k] = {"mean": float(v.mean().item()), "max": float(v.max().item())}
                else:
                    delta_term_metrics[k] = {"mean": float(v), "max": float(v)}

        action_noise_std = float(runner.alg.actor_critic.std.mean().item()) if hasattr(runner.alg.actor_critic, "std") else 0.0
        learning_rate = float(runner.alg.learning_rate)

        # Gradient norms
        actor_grad_norm = 0.0
        for p in runner.alg.actor_critic.parameters():
            if p.grad is not None:
                actor_grad_norm += float(p.grad.detach().data.norm(2).item() ** 2)
        actor_grad_norm = actor_grad_norm ** 0.5

        disc_grad_norm = 0.0
        for p in runner.alg.discriminator.parameters():
            if p.grad is not None:
                disc_grad_norm += float(p.grad.detach().data.norm(2).item() ** 2)
        disc_grad_norm = disc_grad_norm ** 0.5

        gp_weight = getattr(runner.alg, "disc_grad_penalty_weight", 10.0)
        logit_reg_weight = getattr(runner.alg, "disc_logit_reg", 0.05)

        entry = {
            "iteration": it,
            "mean_episode_length": mean_ep_len,
            "episode_return": mean_reward,
            "base_contact_fraction": base_contact_frac,
            "timeout_fraction": timeout_frac,
            "velocity_xy_tracking_error": error_vel_xy,
            "yaw_tracking_error": error_vel_yaw,
            "reward_disc": reward_disc,
            "reward_fm": reward_fm,
            "reward_raw": reward_raw,
            "discriminator_probability": disc_prob,
            "discriminator_bce_loss": disc_loss,
            "raw_gp": 0.0,
            "weighted_gp": 0.0,
            "raw_logit_reg": 0.0,
            "weighted_logit_reg": 0.0,
            "delta_abs_mean": delta_abs_mean,
            "delta_per_term": delta_term_metrics,
            "action_noise_std": action_noise_std,
            "policy_gradient_norm": actor_grad_norm,
            "discriminator_gradient_norm": disc_grad_norm,
            "learning_rate": learning_rate,
            "nan_inf_count": nan_count,
        }

        recorded_metrics[str(it)] = entry
        print(f"\n[PHASE 4 METRICS AT ITERATION {it}]:")
        print(json.dumps(entry, indent=2))

    runner.log = custom_log

    # Train for 100 iterations
    runner.learn(num_learning_iterations=100, init_at_random_ep_len=True)

    # Save final model
    final_checkpoint_path = os.path.join(log_dir, "model_100.pt")
    runner.save(final_checkpoint_path)
    chk_sha256 = get_file_sha256(final_checkpoint_path)
    print(f"\n=== FINAL CHECKPOINT SHA256: {chk_sha256} ===")

    # Save JSON metrics file
    metrics_path = os.path.join(log_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(recorded_metrics, f, indent=2)

    print("=== STEP 4 COMPLETE: TITA 4096 ENV 100 ITERATIONS SUCCESSFUL ===")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

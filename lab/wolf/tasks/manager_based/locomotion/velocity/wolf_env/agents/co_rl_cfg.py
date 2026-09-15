"""Ordinary PPO; identical oracle information for actor and critic."""
from isaaclab.utils import configclass
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg, CoRlPpoActorCriticCfg, CoRlPpoAlgorithmCfg

@configclass
class WolfStairPPORunnerCfg(CoRlPolicyRunnerCfg):
    num_steps_per_env = 24
    num_policy_stacks = 0
    num_critic_stacks = 0
    max_iterations = 1500
    save_interval = 100
    experiment_name = "Wolf_v2_Stairs_Oracle_PPO"
    experiment_description = "Oracle baseline; 40 cm is an unvalidated target"
    empirical_normalization = True
    policy = CoRlPpoActorCriticCfg(init_noise_std=0.3, actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128], activation="elu")
    algorithm = CoRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )

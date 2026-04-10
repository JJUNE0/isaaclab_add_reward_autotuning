# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlPpoActorCriticCfg,
    CoRlPpoAlgorithmCfg,
    CoRlSrmPpoAlgorithmCfg,
    # CoRlPpoAsymmetricActorCriticCfg
)

from scripts.co_rl.core.wrapper import CoRlMOOPolicyRunnerCfg, ADDCfg, CoRlMooPpoAlgorithmCfg


######################################## [ PPO CONFIG] ########################################

@configclass
class ConcurrentRMAActorCriticCfg(CoRlPpoActorCriticCfg):
    """Configuration for RMA Teacher Policy."""
    class_name: str = "ConcurrentRMA" # eval()에서 이 클래스를 로드함
    latent_dim: int = 16          # 환경 인코더가 압축할 차원
    history_len: int = 50  

    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    
 


@configclass
class FlamingoMOOPPORunnerCfg(CoRlMOOPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "FlamingoStand-v3-mooppo"
    experiment_description = "test"
    empirical_normalization = False
    
    policy = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = CoRlMooPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
    add_cfg = ADDCfg(
        mode='nonmotion',
        disc_coef= 0.5,
        disc_grad_penalty_weight= 0.1,
        disc_logit_reg= 1e-4,
        disc_weight_reg= 1e-5,
        disc_hidden_dims = [256, 256],
        disc_learning_rate = 1e-4,
        enable_reward_norm=False,
        reward_norm_momentum = 0.95
    )

@configclass
class FlamingoMOOPPORunnerCfg_Flat_Stand_Drive(FlamingoMOOPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo_Flat_Stand_Drive-v3-mooppo"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        
        
@configclass
class FlamingoMOOPPORunnerCfg_Recovery(FlamingoMOOPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "Flamingo-v3_Recovery-mooppo"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
         
        
@configclass
class FlamingoMOOPPORunnerCfg_Rough_Stand_Drive(FlamingoMOOPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        # self.policy = CoRlPpoAsymmetricActorCriticCfg(
        #     init_noise_std=1.0,
        #     actor_hidden_dims=[512, 256, 128],
        #     critic_hidden_dims=[512, 256, 128],
        #     tcn_hidden_dims=[128, 128, 64],
        #     activation="elu",
        # )
        self.experiment_name = "Flamingo_Rough_Stand_Drive-v3-mooppo"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]






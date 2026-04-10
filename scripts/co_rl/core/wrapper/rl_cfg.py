# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class CoRlOffPolicyCfg:
    """Configuration for the Off-Policy networks."""

    class_name: str = MISSING
    """The policy class name. Default is SAC."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""


####################################################################################
####################################################################################


@configclass
class CoRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic. ActorCriticRecurrent is for RNNs."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    
@configclass
class CoRlSIEActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "SIEActorCritic"
    """The policy class name. Default is ActorCritic. ActorCriticRecurrent is for RNNs."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

@configclass
class CoRlLipsNetActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "LipsNetActorCritic"
    """The policy class name. Default is ActorCritic. ActorCriticRecurrent is for RNNs."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    
    lambda_t : float = MISSING
     
    lambda_k : float = MISSING
    
    kernel_scale : float = MISSING
    enable_fft_2d : bool = MISSING
    norm_layer_type : str = MISSING


@configclass
class CoRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""


@configclass
class ADDCfg:
    
    mode : str = MISSING
    
    disc_coef : float = MISSING
    
    disc_grad_penalty_weight : float = MISSING
    
    disc_logit_reg : float = MISSING
    
    disc_weight_reg : float = MISSING
    
    disc_hidden_dims : list[int] = MISSING
    
    disc_learning_rate : float = MISSING   
    
    enable_reward_norm : bool = MISSING
    
    reward_norm_momentum : float = MISSING
    

@configclass
class CoRlMooPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "MOOPPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""


@configclass
class CoRlLipsPpoAlgorithmCfg:
    """Configuration for the LipsPPO algorithm."""

    class_name: str = "LipsPPO"
    """The algorithm class name. Default is LipsPPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""



@configclass
class CoRlAcapsPpoAlgorithmCfg:
    """Configuration for the ACAPSPPO algorithm."""

    class_name: str = "ACAPSPPO"
    """The algorithm class name. Default is ACAPSPPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
        
    use_acaps: bool = MISSING
    """Whether to use ACAPS."""

    acaps_lambda_t_coef: float = MISSING
    """The coefficient for the ACAPS temporal loss."""

    acaps_lambda_s_coef: float = MISSING
    """The coefficient for the ACAPS spatial loss."""


@configclass
class CoRlSIEPpoAlgorithmCfg:
    """Configuration for the SIEPPO algorithm."""

    class_name: str = "SIEPPO"
    """The algorithm class name. Default is SIEPPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
        
    use_acaps: bool = MISSING
    """Whether to use ACAPS."""

    acaps_lambda_t_coef: float = MISSING
    """The coefficient for the ACAPS temporal loss."""

    acaps_lambda_s_coef: float = MISSING
    """The coefficient for the ACAPS spatial loss."""



@configclass
class CoRlSrmPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "SRMPPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    srm_net: str = MISSING
    """The SRM network type."""

    srm_input_dim: int = MISSING
    """The input dimension for the SRM model."""

    cmd_dim: int = MISSING
    """The command dimension for the SRM model."""

    srm_hidden_dim: int = MISSING
    """The hidden dimension for the SRM model."""

    srm_output_dim: int = MISSING
    """The output dimension for the SRM model."""

    srm_num_layers: int = MISSING
    """The number of layers for the SRM model."""

    srm_r_loss_coef: float = MISSING
    """The coefficient for the SRM reconstruction loss."""

    srm_rc_loss_coef: float = MISSING
    """The coefficient for the SRM reward consistency loss."""

    use_acaps: bool = MISSING
    """Whether to use ACAPS."""

    acaps_lambda_t_coef: float = MISSING
    """The coefficient for the ACAPS temporal loss."""

    acaps_lambda_s_coef: float = MISSING
    """The coefficient for the ACAPS spatial loss."""


@configclass
class CoRlPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: CoRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: CoRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    experiment_description: str = MISSING
    """The experiment description."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    num_policy_stacks: int = 0
    """The number of frames to stack. Default is 0."""

    num_critic_stacks: int = 0
    """The number of frames to stack. Default is 0."""

    use_constraint_rl: bool = False
    """Whether to use constraints as termination."""
    
    store_training_data : bool = False
########################################################################################################################
########################################################################################################################
########################################################################################################################


@configclass
class CoRlLipcsNetPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: CoRlLipsNetActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: CoRlLipsPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    experiment_description: str = MISSING
    """The experiment description."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    num_policy_stacks: int = 0
    """The number of frames to stack. Default is 0."""

    num_critic_stacks: int = 0
    """The number of frames to stack. Default is 0."""

    use_constraint_rl: bool = False
    """Whether to use constraints as termination."""
    
    #use_lipsnet : bool = False
    
    
    
    

@configclass
class CoRlMOOPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: CoRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: CoRlMooPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
    
    add_cfg : ADDCfg = MISSING

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    experiment_description: str = MISSING
    """The experiment description."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    num_policy_stacks: int = 0
    """The number of frames to stack. Default is 0."""

    num_critic_stacks: int = 0
    """The number of frames to stack. Default is 0."""

    use_constraint_rl: bool = False
    """Whether to use constraints as termination."""
    
    store_training_data : bool = False
    
    use_rma_teacher : bool = False
    
    use_rma_student : bool = False
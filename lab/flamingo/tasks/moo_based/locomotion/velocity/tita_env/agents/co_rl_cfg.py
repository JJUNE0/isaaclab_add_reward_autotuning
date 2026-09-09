# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from scripts.co_rl.core.wrapper import (
    CoRlMOOPolicyRunnerCfg,
    CoRlPpoActorCriticCfg,
    CoRlEstimatorActorCriticCfg,
    CoRlRMATeacherActorCriticCfg,
    CoRlRMAStudentActorCriticCfg,
    CoRlRMAStudentMultiHeadActorCriticCfg,
    CoRlMooPpoAlgorithmCfg,
    ADDCfg,
)


@configclass
class TitaMOOPPORunnerCfg(CoRlMOOPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "TitaStand-v3-mooppo"
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
        mode="nonmotion",
        disc_coef=0.5,
        disc_grad_penalty_weight=0.1,
        disc_logit_reg=1e-4,
        disc_weight_reg=1e-5,
        disc_hidden_dims=[256, 256],
        disc_learning_rate=1e-4,
        enable_reward_norm=False,
        reward_norm_momentum=0.95,
    )


@configclass
class TitaMOOPPORunnerCfg_Flat_Stand_Drive(TitaMOOPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "Tita_Flat_Stand_Drive-v3-mooppo"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class TitaMOOPPORunnerCfg_Flat_Stand_Drive_Estimator(TitaMOOPPORunnerCfg_Flat_Stand_Drive):
    """Base-velocity-and-height-estimator variant (ActorCriticWithEstimator):
    actor sees an MLP-estimated [base_lin_vel_x, base_lin_vel_y, base_lin_vel_z,
    base_height] instead of ground truth, trained jointly via an auxiliary MSE
    loss against the env's priv_prio group (see
    TitaFlatEnvCfg_Estimator_TargetHeight, which appends base_height as
    priv_prio's 4th term to match estimator_output_dim=4)."""

    def __post_init__(self):
        super().__post_init__()
        self.policy = CoRlEstimatorActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            estimator_hidden_dims=[128, 64],
            estimator_output_dim=4,
        )
        self.algorithm.estimator_loss_coef = 1.0


@configclass
class TitaMOOPPORunnerCfg_Flat_Stand_Drive_RMATeacher(TitaMOOPPORunnerCfg_Flat_Stand_Drive):
    """RMA teacher variant (RMATeacher): actor sees [proprio, z], z =
    Encoder(privileged_info), privileged_info = the env's full concatenated
    priv_extrio + priv_physical + priv_prio bundle (system-ID quantities,
    terrain scan, and ground-truth base_lin_vel/base_height) -- maximal use of
    sim-only oracle information, compressed to a latent_dim=16 latent. The
    actor receives only the current robot-observable frame plus command;
    ground-truth base velocity and measured height live exclusively in the
    privileged bundle. Intended as the frozen teacher for a later RMAStudent
    (history-based adaptation module, no privileged info at deployment)."""

    def __post_init__(self):
        super().__post_init__()
        self.use_rma_teacher = True
        self.num_policy_stacks = 0
        self.max_iterations = 10000
        self.experiment_description = (
            "RMA Teacher deployable-observation v5: no GT base velocity/height in policy obs, "
            "current frame only (num_policy_stacks=0), privileged 26D -> latent 16D, actor input 48D"
        )
        self.logger = "wandb"
        self.wandb_project = "tita-moo-rma-teacher"
        self.policy = CoRlRMATeacherActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            latent_dim=16,
        )


@configclass
class TitaMOOPPORunnerCfg_Flat_Stand_Drive_RMATeacher808(
    TitaMOOPPORunnerCfg_Flat_Stand_Drive
):
    """Exact network/stack contract of the successful 2026-08-08 teacher."""

    def __post_init__(self):
        super().__post_init__()
        self.use_rma_teacher = True
        self.use_rma_student = False
        self.num_policy_stacks = 1
        self.max_iterations = 10000
        self.logger = "tensorboard"
        self.experiment_description = (
            "2026-08-08 teacher evaluation contract: GT state in two 32D actor frames, "
            "25D privileged encoder input, 16D latent, 84D actor input"
        )
        self.policy = CoRlRMATeacherActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            latent_dim=16,
        )


@configclass
class TitaMOOPPORunnerCfg_Flat_Stand_Drive_RMAStudent(TitaMOOPPORunnerCfg_Flat_Stand_Drive):
    """RMA student variant (RMAStudent): distills a frozen RMATeacher (actor +
    encoder, both frozen) via a history-based AdaptationModule (dilated TCN)
    that predicts z_hat from a window of past proprio+command frames, without
    privileged info -- for sim2real deployment. Must be trained against the
    exact same env cfg AND num_policy_stacks the referenced teacher was
    trained with (the teacher's frozen actor has a fixed input dimension).
    teacher_checkpoint_path must be set (via cfg edit or a future CLI
    override) before training/playing this variant."""

    def __post_init__(self):
        super().__post_init__()
        self.use_rma_teacher = False
        self.use_rma_student = True
        self.logger = "wandb"
        self.wandb_project = "tita-moo-rma-student"
        # Must match the referenced teacher's training config exactly (frozen
        # actor has a fixed input dim tied to num_policy_stacks).
        self.num_policy_stacks = 0
        self.policy = CoRlRMAStudentActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            latent_dim=16,
            history_len=50,
            # Must be replaced with the new deployable-observation Teacher
            # checkpoint (actor input 48D). The old 84D Teacher is incompatible.
            teacher_checkpoint_path="MISSING",
        )


@configclass
class TitaMOOPPORunnerCfg_Flat_Stand_Drive_RMAStudentMultiHead808(
    TitaMOOPPORunnerCfg_Flat_Stand_Drive
):
    """Multi-head student distilled from the successful 2026-08-08 teacher.

    The environment exposes the teacher's exact 68-D oracle observation for
    targets, while RMAStudentMultiHead strips all velocity/height GT before
    creating its 50-frame deployable history. It predicts latent z plus both
    stacked frames' velocity and height slots for the frozen 84-D actor.
    """

    def __post_init__(self):
        super().__post_init__()
        self.use_rma_teacher = False
        self.use_rma_student = True
        self.num_policy_stacks = 1
        self.max_iterations = 10000
        self.logger = "tensorboard"
        self.wandb_project = "tita-moo-rma-student"
        self.experiment_description = (
            "808 frozen teacher multi-head RMA student: deployable 50-frame history -> "
            "z_hat(16), stacked lin_vel_hat(6), stacked height_hat(2); exact 84D actor reconstruction"
        )
        self.policy = CoRlRMAStudentMultiHeadActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            latent_dim=16,
            history_len=50,
            teacher_checkpoint_path="MISSING",
            observable_single_obs_dim=28,
            state_dim=4,
            z_loss_coef=1.0,
            velocity_loss_coef=1.0,
            height_loss_coef=10.0,
            action_loss_coef=1.0,
        )

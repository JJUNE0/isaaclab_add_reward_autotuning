#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from scripts.co_rl.core.modules import ActorCritic, DiscriminatorGRU, DiscriminatorMLP
from scripts.co_rl.core.storage import MOORolloutStorage

from scripts.co_rl.core.modules.teacher_student import RMAStudent, RMATeacher
from scripts.co_rl.core.modules.utils import DiffNormalizer, EMA

class MOOPPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        env,
        actor_critic,
        moo_manager,
        add_cfg, 
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        
        self.is_student = isinstance(self.actor_critic, RMAStudent)
        if self.is_student:
            print("[MOOPPO] Detected RMAStudent. Switching to Supervised Learning (MSE).")
            self.ppo_optimizer = optim.Adam(
                self.actor_critic.adaptation_module.parameters(), 
                lr=learning_rate
            )
        else:
    
            self.ppo_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        self.transition = MOORolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.cfg = None
        
        # discriminaotr parameters
        self.disc_coef = add_cfg["disc_coef"]
        self.disc_grad_penalty_weight = add_cfg["disc_grad_penalty_weight"]
        self.disc_logit_reg = add_cfg["disc_logit_reg"]
        self.disc_weight_reg = add_cfg["disc_weight_reg"]
        self.disc_learning_rate = add_cfg["disc_learning_rate"]

        # ADD Components
        self.moo_manager = moo_manager
        self.mode = self.moo_manager.mode
        self.delta_dim = self.moo_manager.delta_dim
        self.feature_dim = self.moo_manager.feature_dim
        self.window_size = self.moo_manager.window_size
        self.single_obs_dim = env.num_single_obs
        
        if self.mode == "motion":
            self.discriminator = DiscriminatorGRU(
                in_dim=self.feature_dim, 
                hidden_dims=add_cfg["disc_hidden_dims"]
            )
        else:
            self.discriminator = DiscriminatorMLP(
                delta_dim=self.delta_dim,
                obs_dim = self.single_obs_dim, 
                hidden_dims=add_cfg["disc_hidden_dims"]
            )
        self.discriminator.to(self.device)
        print(f"Discriminator : {self.discriminator}")
        
        self.delta_normalizer = DiffNormalizer(self.delta_dim, device=self.device)
        
        self.enable_reward_norm = add_cfg["enable_reward_norm"]

        self.reward_ema = EMA(momentum=add_cfg["reward_norm_momentum"], device=self.device)

        self.target_delta = torch.zeros((1, self.delta_dim), device=self.device)

        self.moo_manager.set(
            discriminator=self.discriminator,
            delta_normalizer=self.delta_normalizer,
            reward_ema=self.reward_ema,
            enable_reward_norm=self.enable_reward_norm,
            w_pot=add_cfg.get("w_pot", 0.0),
            w_disc=add_cfg.get("w_disc", 0.2),
            w_fm=add_cfg.get("w_fm", 0.8),
        )
        
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.disc_learning_rate)

    def init_storage(self, cfg, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, delta_shape):
        self.update_late_cfg(cfg)
        self.storage = MOORolloutStorage(
            cfg , num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, delta_shape, self.device
        )
    
    def update_late_cfg(self, cfg):
        self.cfg = cfg

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Recurrent Check (기존 코드 안전하게 처리)
        if getattr(self.actor_critic, "is_recurrent", False):
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # ---------------------------------------------------------
        # Case 1: Student Mode (지도 학습)
        # ---------------------------------------------------------
        if self.is_student: # __init__에서 self.is_student = isinstance(...) 해뒀다고 가정
            self.transition.actions, _ = self.actor_critic()
            self.transition.actions = self.transition.actions.detach()
            
            batch_size = obs.shape[0]
            self.transition.values = torch.zeros(batch_size, 1, device=self.device)
            self.transition.actions_log_prob = torch.zeros(batch_size, 1, device=self.device)
            self.transition.action_mean = torch.zeros_like(self.transition.actions)
            self.transition.action_sigma = torch.zeros_like(self.transition.actions)
            
            flat_hist = self.actor_critic.history_buffer.view(batch_size, -1).clone()
            self.transition.observations = flat_hist
            
        # ---------------------------------------------------------
        # Case 2: Teacher Mode (강화 학습 - PPO)
        # ---------------------------------------------------------
        else:
            # 1. Action 계산
            if isinstance(self.actor_critic, RMATeacher):            
                split_idx = self.actor_critic.num_proprio_obs
                privileged_obs = critic_obs[:, split_idx:]
                self.transition.actions = self.actor_critic.act(obs, privileged_info=privileged_obs).detach()
            else:
                self.transition.actions = self.actor_critic.act(obs).detach()
            
            # 2. Value & LogProb 계산 (여기는 Critic이 있으므로 호출 가능)
            self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
            self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
            self.transition.action_mean = self.actor_critic.action_mean.detach()
            self.transition.action_sigma = self.actor_critic.action_std.detach()
            
            # 3. 데이터 저장
            self.transition.observations = obs

        # 공통 저장
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.deltas = infos["delta"].clone()
        
        self.transition.dones = dones
        self.transition.time_outs = infos["time_outs"].int()
        
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _update_discriminator(self, delta_batch: torch.Tensor, obs) -> float:
        expert_delta = self.target_delta.expand(delta_batch.shape[0], -1) 
        self.delta_normalizer.record(delta_batch)
        
        # 정규화
        norm_expert_delta = self.delta_normalizer.normalize(expert_delta)
        norm_policy_delta = self.delta_normalizer.normalize(delta_batch) # 정책 델타로 노멀라이저 업데이트

        # Discriminator 로짓 계산
        expert_logits = self.discriminator(norm_expert_delta, obs)
        policy_logits = self.discriminator(norm_policy_delta.detach(), obs) # Policy 학습에 영향 주지 않도록 detach

        policy_prob = torch.sigmoid(policy_logits).mean().item()
        # Loss 계산 (BCE Loss 예시)
        expert_loss = nn.functional.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        )
        policy_loss = nn.functional.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits)
        )
        
        # Gradient Penalty
        grad_penalty = self.discriminator.compute_gradient_penalty(
            norm_expert_delta, norm_policy_delta, obs, self.disc_grad_penalty_weight
        )
        
        # Logit Regularization
        logit_reg = self.discriminator.compute_logit_regularization(
            expert_logits, policy_logits, self.disc_logit_reg
        )
        
        disc_loss = expert_loss + policy_loss + grad_penalty + logit_reg
        
        # 디스크리미네이터 파라미터 업데이트
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
        self.disc_optimizer.step()
        
        return disc_loss.item(), policy_prob

    def _update_policy(
        self,
        obs_batch,
        critic_obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hid_states_batch,
        masks_batch,
    ) -> dict:
        """ PPO (Policy/Critic)를 1회 업데이트하고 손실(loss)을 반환합니다. """
        
        if isinstance(self.actor_critic, RMATeacher):
            split_idx = self.actor_critic.num_proprio_obs
            real_priv_info = critic_obs_batch[:, split_idx:]
            z_teacher = self.actor_critic.encoder(real_priv_info)
            
            self.actor_critic.act(
                obs_batch, 
                privileged_info=real_priv_info, # [핵심] 전달
                z =z_teacher, 
                masks=masks_batch, 
                hidden_states=hid_states_batch[0]
            )
        else:
            self.actor_critic.act(
                obs_batch, 
                masks=masks_batch, 
                hidden_states=hid_states_batch[0]
            )
        
#        self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(
            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
        )
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy

        # KL
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                for param_group in self.ppo_optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
        # if isinstance(self.actor_critic, RMATeacher):
        #     latent_penalty = 0.0001 * torch.mean(z_teacher ** 2)
        #     loss += latent_penalty
        
        # Gradient step
        self.ppo_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.ppo_optimizer.step()
        
        
        infos = {
                "value_loss" : value_loss.item(),
                "surrogate_loss" : surrogate_loss.item(),
                }
        
        if isinstance(self.actor_critic, RMATeacher):
            infos["z_teacher"] = z_teacher.detach().cpu()
        
        return infos
    
    def _update_student(self, obs_batch, critic_obs_batch) -> dict:
        """
        Student Distillation Update (MSE Loss)
        z_teacher(Target) <-> z_student(Pred)
        """
        z_pred = self.actor_critic.adaptation_module(obs_batch)

        enc_input_dim = self.actor_critic.teacher_encoder.net[0].in_features
        priv_info_batch = critic_obs_batch[:, -enc_input_dim:]

        with torch.no_grad():
            z_target = self.actor_critic.teacher_encoder(priv_info_batch)

        # 3. Loss & Step
        loss = nn.functional.mse_loss(z_pred, z_target)

        self.ppo_optimizer.zero_grad() 
        loss.backward()
        self.ppo_optimizer.step()
        
        with torch.no_grad():
            # (1) L1 Error (직관적인 오차 절댓값)
            l1_error = torch.mean(torch.abs(z_target - z_pred)).item()
            
            # (2) Magnitude (벡터의 크기 비교 - Collpase 확인용)
            t_norm = torch.norm(z_target, dim=1).mean().item()
            s_norm = torch.norm(z_pred, dim=1).mean().item()
            norm_ratio = s_norm / (t_norm + 1e-6) # 1.0에 가까워야 함

            # (3) Direction (코사인 유사도 - 방향성 확인용)
            # -1 ~ 1 사이 값. 1에 가까울수록 방향이 일치함.
            cos_sim = torch.nn.functional.cosine_similarity(z_target, z_pred, dim=1).mean().item()
            
            # (4) Debug Print (가끔씩 눈으로 확인)
            # 학습 초반이나 디버깅 때는 100번에 한 번씩 찍어보세요.
            if torch.rand(1) < 0.1: 
                print(f"\n[DEBUG] Z Stats:")
                print(f"  Target  (First 3): {z_target[0, :].cpu().numpy().round(3)}")
                print(f"  Student (First 3): {z_pred[0, :].cpu().numpy().round(3)}")
                print(f"  Cos Sim: {cos_sim:.4f} | Norm Ratio: {norm_ratio:.4f}")

        # 단순 loss만 리턴하지 말고, 분석 정보를 같이 리턴해서 Logger가 찍게 하세요.
        
        infos = {
            "rma_loss" : loss.item(),
            "teacher_z" : z_target.detach().cpu(), 
            "student_z" : z_pred.detach().cpu()}
        
        return infos
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_disc_loss = 0
        mean_disc_prob = 0
        mean_rma_loss = 0
        
        extras = {}
        
        if getattr(self.actor_critic, "is_recurrent", False):
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            delta_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            
            disc_loss, disc_prob = self._update_discriminator(delta_batch, obs_batch[:, :self.single_obs_dim])
            mean_disc_prob += disc_prob
            mean_disc_loss += disc_loss
        
            if self.is_student:

                student_infos = self._update_student(obs_batch, critic_obs_batch)
                mean_rma_loss += student_infos["rma_loss"] # 로그 호환성을 위해 disc_loss 자리에 mse 저장
                
                
                extras["teacher_z"] = student_infos["teacher_z"]
                extras["student_z"] = student_infos["student_z"]
                
                enc_input_dim = self.actor_critic.teacher_encoder.net[0].in_features
                extras["privileged_info"] = critic_obs_batch[:, -enc_input_dim:].detach().cpu()
                
            else:

                policy_infos = self._update_policy(
                    obs_batch, critic_obs_batch, actions_batch, target_values_batch, 
                    advantages_batch, returns_batch, old_actions_log_prob_batch, 
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch
                )
                
                mean_value_loss += policy_infos["value_loss"]
                mean_surrogate_loss += policy_infos["surrogate_loss"]
                
                if "z_teacher" in policy_infos:

                    extras["teacher_z"] = policy_infos["z_teacher"]
                    
                    if hasattr(self.actor_critic, "num_proprio_obs"):
                         split_idx = self.actor_critic.num_proprio_obs
                         extras["privileged_info"] = critic_obs_batch[:, split_idx:].detach().cpu()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_disc_loss /= num_updates
        mean_disc_prob /= num_updates
        mean_rma_loss /= num_updates
        self.storage.clear()

        # self.delta_normalizer.reset()
        
        return mean_value_loss, mean_surrogate_loss, mean_disc_loss, mean_disc_prob, mean_rma_loss, extras



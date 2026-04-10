#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from scripts import co_rl
from scripts.co_rl.core.utils import *
from scripts.co_rl.core.algorithms import MOOPPO

from scripts.co_rl.core.env import VecEnv
from scripts.co_rl.core.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization, RMAStudent, RMATeacher
from scripts.co_rl.core.utils import store_code_state

from dataclasses import is_dataclass, fields

from scripts.co_rl.core.storage import SequenceDataStorage
# 

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.manifold import TSNE

def log_teacher_z_tsne(extras: dict, step: int, num_samples: int = 1000):
    """
    extras 딕셔너리에서 'teacher_z'와 'privileged_info'를 꺼내 
    t-SNE 분석 후 WandB에 업로드합니다.
    
    Args:
        extras (dict): env.step() 후 리턴받은 정보. 'teacher_z', 'priv_info' 키가 있어야 함.
        step (int): 현재 학습 Step (WandB 로깅용)
        num_samples (int): t-SNE 속도를 위해 샘플링할 개수 (기본 1000개)
    """
    
    # 1. 키 존재 확인
    if "teacher_z" not in extras:
        return # z가 없으면 스킵
    
    # 2. 데이터 추출 및 Sampling (CPU 이동)
    # 전체 배치를 다 돌리면 t-SNE가 너무 느리므로 랜덤 샘플링
    z_tensor = extras["teacher_z"]
    priv_tensor = extras.get("privileged_info", None) # 색칠용 (없으면 단색)
    
    total_samples = z_tensor.shape[0]
    idx = torch.randperm(total_samples)[:min(total_samples, num_samples)]
    
    z_np = z_tensor[idx].detach().cpu().numpy()
    
    # 3. t-SNE 수행
    # perplexity: 보통 30~50 사용. 데이터 구조 파악의 핵심 파라미터.
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        z_embedded = tsne.fit_transform(z_np)
    except Exception as e:
        print(f"[Warning] t-SNE visualization failed: {e}") 
        return

    # 4. 시각화 (Matplotlib)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if priv_tensor is not None:
        # [핵심] 물리 정보(Privileged Info)의 첫 번째 차원(예: 마찰력)으로 색칠
        # 그래야 "마찰력끼리 뭉쳤는지" 확인 가능
        priv_np = priv_tensor[idx].detach().cpu().numpy()
        c_val = priv_np[:, 0] 
        scatter = ax.scatter(z_embedded[:, 0], z_embedded[:, 1], c=c_val, cmap='viridis', s=15, alpha=0.7)
        plt.colorbar(scatter, label='Privileged Info [Dim 0]')
    else:
        # 물리 정보가 없으면 그냥 단색으로 분포만 확인
        ax.scatter(z_embedded[:, 0], z_embedded[:, 1], c='blue', s=15, alpha=0.5)

    ax.set_title(f"Teacher Latent Space t-SNE (Step {step})")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.grid(True, alpha=0.3)

    # 5. WandB 업로드
    wandb.log({"Analysis/Teacher_Z_t-SNE": wandb.Image(fig)}, step=step)
    
    # 메모리 누수 방지
    plt.close(fig)

class MOO_OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg

        self.alg_cfg = train_cfg["algorithm"]
        
        self.policy_cfg = train_cfg["policy"]
        
        self.add_cfg = train_cfg["add_cfg"]
        
        assert self.alg_cfg["class_name"] == "MOOPPO"

        self.device = device
        self.env = env
        
        obs, extras = self.env.get_observations()
        
        num_obs = obs.shape[1]
        self.num_single_obs = env.num_single_obs
        self.num_stacks = env.num_policy_stacks + 1
        
        
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        
        obs_dict = extras["observations"]
        
        priv_keys = sorted([k for k in obs_dict.keys() if k.startswith("priv_")])
        
        print(f"[SysID Setup] Sorted Keys: {priv_keys}")
        
        # [수정] 특정 키만 찾는 게 아니라, 전체 차원을 계산
        total_priv_dim = 0
        for key in priv_keys:
            dim = obs_dict[key].shape[1]
            total_priv_dim += dim
            
        if total_priv_dim > 0:
            num_privileged_obs = total_priv_dim
        else:
            # 하위 호환성
            if "privileged" in obs_dict:
                num_privileged_obs = obs_dict["privileged"].shape[1]
                print(f"[MOO Runner] Target: Single 'privileged' Key. Dim: {num_privileged_obs}")
            else:
                num_privileged_obs = 0
                print("[MOO Runner] No Privileged Observations detected.")
        
        self.is_student_mode = self.cfg.get("use_rma_student", False)
        
        actor_critic = self.init_actor_critic(num_obs, num_critic_obs, num_privileged_obs,self.env.num_actions)
        
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        
        self.moo_reward_manager = env.unwrapped.moo_reward_manager
        num_deltas = self.moo_reward_manager.delta_dim

        self.alg: MOOPPO = alg_class(env, actor_critic, self.moo_reward_manager, self.add_cfg, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        
        if self.cfg.get("use_rma_student", False) and self.empirical_normalization:
            print("[MOO Runner] Loading Teacher's Obs Normalizer Statistics...")
            
            teacher_ckpt_path = self.policy_cfg.teacher_checkpoint_path
            # 체크포인트 로드
            ckpt = torch.load(teacher_ckpt_path, map_location=self.device)
            
            if "obs_norm_state_dict" in ckpt:
                self.obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
                
                # [중요] Student 학습 중에는 관측 통계치가 변하면 안 됨 (Teacher 기준 고정)
                self.obs_normalizer.eval() 
                print(f"[MOO Runner] Teacher Obs Norm Loaded & Frozen! (Count: {self.obs_normalizer.count.item()})")
            else:
                print("[WARNING] Teacher Checkpoint has no 'obs_norm_state_dict'! Training might fail.")
        
        
        if self.cfg.get("use_rma_student", False):
            num_storage_obs = actor_critic.frame_dim * actor_critic.history_len
            print(f"[Runner] Student Mode Detected. Storage Obs Dim adjusted: {num_obs} -> {num_storage_obs}")
        else:
            num_storage_obs = num_obs
        
        self.alg.init_storage(
            self.cfg,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_storage_obs],
            [num_critic_obs],
            [self.env.num_actions],
            [num_deltas]
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [co_rl.__file__]
        
        # offlinedata
        if self.cfg["store_training_data"]:
            self.offline_data_log_dir = os.path.join(log_dir, "offline_data")
            self.offline_data = SequenceDataStorage(num_envs=self.env.num_envs, obs_dim= num_obs,action_dim=self.env.num_actions  ,zarr_path=self.offline_data_log_dir ,high_reward_episodes=190, low_reward_episodes=10) #OfflineDataStorage(obs_dim= num_obs,action_dim=self.env.num_actions  ,zarr_path=self.offline_data_log_dir ,high_reward_capacity=450000, low_reward_capacity=50000)
            self.save_iteration = 200


    def init_actor_critic(self, num_obs,  num_critic_obs, num_privileged_obs,num_actions):
        # class_name을 pop하지 않고 읽기만 함 (Config 객체 재사용성을 위해)
        class_name = self.policy_cfg['class_name'] 
        actor_critic_class = eval(class_name)
        
        # ---------------------------------------------------------
        # 1. RMA Teacher 모드
        # ---------------------------------------------------------
        
        if self.cfg.get("use_rma_teacher", False):
            print("[MOO Runner] Initializing RMA Teacher...")
                    
            actor_critic = actor_critic_class(
                num_proprio_obs=num_obs,       # Student 입력(History)을 Proprio로 사용
                num_privileged_obs=num_privileged_obs,   # 계산된 특권 정보 차원
                num_critic_obs=num_critic_obs, # Critic 입력 (Full)
                num_actions=num_actions,
                **self.policy_cfg # latent_dim 등이 여기 포함됨
            ).to(self.device)
        
        # ---------------------------------------------------------
        # 2. RMA Student 모드
        # ---------------------------------------------------------
        elif self.cfg.get("use_rma_student", False):
            print("[MOO Runner] Initializing RMA Student...")
            
            teacher_ckpt_path = self.policy_cfg['teacher_checkpoint_path']
            if teacher_ckpt_path is None or teacher_ckpt_path == "MISSING":
                raise ValueError("Student 학습을 위해 'teacher_checkpoint_path'가 필요합니다!")
            
            teacher_policy = RMATeacher(
                num_proprio_obs=num_obs,
                num_privileged_obs=num_privileged_obs,
                num_critic_obs=num_critic_obs,
                num_actions=num_actions,
                **self.policy_cfg
            ).to(self.device)
            
            # [단계 B] 가중치 로드
            checkpoint = torch.load(teacher_ckpt_path, map_location=self.device)
            teacher_policy.load_state_dict(checkpoint['model_state_dict'])
            teacher_policy.eval()
            print(f"[MOO Runner] Teacher loaded from: {teacher_ckpt_path}")

            # [단계 C] Student 생성
            # Student는 Teacher 객체와 자신의 입력(num_obs)만 알면 됨
            actor_critic = actor_critic_class(
                teacher_policy=teacher_policy,
                num_obs=num_obs, 
                num_envs = self.env.num_envs,
                device=self.device,
                num_single_obs=self.num_single_obs,
                num_stacks=self.num_stacks,
                **self.policy_cfg
            ).to(self.device)

        # ---------------------------------------------------------
        # 3. 일반 PPO
        # ---------------------------------------------------------
        else:  
            print(f"[MOO Runner] Initializing Standard {class_name}...")
            actor_critic = actor_critic_class(
                num_actor_obs=num_obs, # 키워드 인자 이름 확인 필요 (__init__ 정의에 따름)
                num_critic_obs=num_critic_obs,
                num_actions=num_actions,
                **self.policy_cfg
            ).to(self.device)
        
        return actor_critic


    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from scripts.co_rl.core.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from scripts.co_rl.core.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")
            

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                mean_potential_reward = 0.0
                rollout_raw_cnt = 0
                for i in range(self.num_steps_per_env):
                    if self.is_student_mode:
                        current_dones = dones if i > 0 else torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
                        self.alg.actor_critic.update_history(obs, current_dones)
                    
                    actions = self.alg.act(obs, critic_obs)
                    
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # move to the right device
                    
                    mean_potential_reward += self.moo_reward_manager.debug_metrics["mean_potential_reward"]
                    rollout_raw_cnt += 1
                    
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs
                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        if not self.cfg["use_constraint_rl"]:
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                        else:
                            new_ids = (dones == 1.0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                # OfflineDataset
                if self.cfg["store_training_data"]:
                    self.offline_data.add_rollout_data(self.alg.storage, obs)
                    if it % self.save_iteration == 0 and it !=0:
                        self.offline_data.flush_to_zarr()

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if not self.is_student_mode:
                    self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_disc_loss, mean_disc_prob, mean_rma_loss, extras = self.alg.update()

            mean_potential = mean_potential_reward / max(rollout_raw_cnt, 1)

            if hasattr(self.env.unwrapped, "domain_manager") and self.env.unwrapped.domain_manager is not None:
                self.env.unwrapped.domain_manager.update_curriculum(mean_potential)

            if self.log_dir is not None and self.writer is not None and it % 100 == 0:
                if "teacher_z" in extras and "privileged_info" in extras:
                    # t-SNE 유틸 함수 호출
                    log_teacher_z_tsne(extras, step=it)
            
            
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    if "Reward" not in key:
                        self.writer.add_scalar(key, value, locs["it"])
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        if self.cfg.get("use_rma_student", False):
            self.writer.add_scalar("Loss/rma_student_mse_loss", locs["mean_rma_loss"], locs["it"])
        
        if hasattr(self.moo_reward_manager, "debug_metrics"):
            locs["moo_metrics"] = self.moo_reward_manager.debug_metrics
        
        if "moo_metrics" in locs:
            # 1. 메인 MOO 지표들을 wandb/tensorboard에 기록
            self.writer.add_scalar("MOO/Mean_D_Prob", locs['moo_metrics']['mean_discriminator_prob'], locs["it"])
            self.writer.add_scalar("MOO/Mean_Raw_Reward", locs['moo_metrics']['mean_raw_reward'], locs["it"])
            self.writer.add_scalar("MOO/Mean_Disc_Reward", locs['moo_metrics']['mean_disc_reward'], locs["it"])
            self.writer.add_scalar("MOO/Mean_Feature_Matching_Reward", locs['moo_metrics']['mean_feature_matching_reward'], locs["it"])
            self.writer.add_scalar("MOO/Mean_Potential_reward", locs['moo_metrics']['mean_potential_reward'], locs["it"])
            self.writer.add_scalar("MOO/Mean_Normed_Delta_Abs", locs['moo_metrics']['mean_norm_delta_abs'], locs["it"])
            self.writer.add_scalar("MOO/Mean_Potential_U", locs['moo_metrics']['mean_potential_U'], locs["it"])

            # 2. 개별 피처(per_feature) 지표들을 wandb/tensorboard에 기록
            if "per_feature_abs_mean" in locs['moo_metrics']:
                for key, value in locs['moo_metrics']['per_feature_abs_mean'].items():
                    # wandb가 이해할 수 있는 깔끔한 키 생성 (예: "MOO_Features/error_track_lin_vel_xy")
                    scalar_key = f"MOO_Features/{key}"
                    self.writer.add_scalar(scalar_key, value, locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        
        domain_level = 0.0
        
        if hasattr(self.env.unwrapped, "domain_manager") and \
           self.env.unwrapped.domain_manager is not None:
            
            curriculum_info = self.env.unwrapped.domain_manager.get_curriculum_info()
            
            for key, value in curriculum_info.items():
                self.writer.add_scalar(f"Domain/{key}", value, locs["it"])
            
            domain_level = curriculum_info.get("Difficulty_Level", 0.0)
        
        log_string += f"""{'Domain Level:':>{pad}} {domain_level:.4f}\n"""
        
        if self.cfg.get("use_rma_student", False):
            log_string += (f"""{'-' * width}\n""" \
            f"""{'RMA Student MSE Loss:':>{pad}} {locs['mean_rma_loss']:.4f}\n"""
            )
        
        if hasattr(self.moo_reward_manager, "debug_metrics"):
            locs["moo_metrics"] = self.moo_reward_manager.debug_metrics
        
        if "moo_metrics" in locs:
            moo_str = (f"""{'-' * width}\n""" \
              f"""{'MOO/ADD Metrics'.center(width, ' ')}\n""" \
              f"""{f'Mean D(Δ) Prob:':>{pad}} {locs['moo_metrics']['mean_discriminator_prob']:.4f}\n""" \
              f"""{f'Mean Raw Reward:':>{pad}} {locs['moo_metrics']['mean_raw_reward']:.4f}\n""" \
              f"""{f'Mean Disc Reward:':>{pad}} {locs['moo_metrics']['mean_disc_reward']:.4f}\n""" \
              f"""{f'Mean Feature Matching Reward:':>{pad}} {locs['moo_metrics']['mean_feature_matching_reward']:.4f}\n""" \
              f"""{f'Mean Potential Reward:':>{pad}} {locs['moo_metrics']['mean_potential_reward']:.4f}\n""" \
              f"""{f'Mean Normed |Δ|:':>{pad}} {locs['moo_metrics']['mean_norm_delta_abs']:.4f}\n""" \
              f"""{f'Mean Potential U:':>{pad}} {locs['moo_metrics']['mean_potential_U']:.4f}\n""" \
            )


            if "per_feature_abs_mean" in locs['moo_metrics']:
                moo_str += f"""{f'Mean Raw |Δ| per feature:':>{pad}}\n"""
                for key, value in locs['moo_metrics']['per_feature_abs_mean'].items():
                    key_str = f"    {key}:" 
                    moo_str += f"""{f'{key_str}':>{pad}} {value:.4f}\n"""

            log_string += moo_str
        

        
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Domain Level:':>{pad}} {domain_level:.4f}\n""" #
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "ppo_optimizer_state_dict": self.alg.ppo_optimizer.state_dict(),
            "disc_optimizer_state_dict": self.alg.disc_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service``
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        if "model_state_dict" in loaded_dict:
            model_dict = loaded_dict["model_state_dict"]
            filtered_dict = {
                k: v for k, v in model_dict.items() 
                if "history_buffer" not in k and "current_obs" not in k
            }
            
            self.alg.actor_critic.load_state_dict(filtered_dict, strict=False)
        
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)


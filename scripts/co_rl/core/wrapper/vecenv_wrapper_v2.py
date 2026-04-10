# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
from scripts.co_rl.core.env import VecEnv
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from lab.flamingo.isaaclab.isaaclab.envs import ManagerBasedMOORLEnv 
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg
from scripts.co_rl.core.utils.state_handler import StateHandler

import os
from datetime import datetime
import numpy as np

class CoRlVecEnvWrapper(VecEnv):
    def __init__(self, env: ManagerBasedRLEnv, agent_cfg: CoRlPolicyRunnerCfg):
        """
        Args:
            env: The environment to wrap around.
            agent_cfg: Configuration for the agent.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv) and not isinstance(env.unwrapped, ManagerBasedMOORLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv, DirectRLEnv, ManagerBasedMOORLEnv. Environment type:"
                f" {type(env)}"
            )

        # initialize the wrapper
        self.env = env
        self.csv_path = os.path.join("logs", f"torque_vel_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.torque_log = []
        self.vel_log = []

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # Determine the number of policy and critic stacks
        self.num_policy_stacks = agent_cfg.num_policy_stacks
        self.num_critic_stacks = agent_cfg.num_critic_stacks

        
        # Determine if constraint RL is used
        self.use_constraint_rl = agent_cfg.use_constraint_rl
        
        # ---------------------------------------------------------------------
        # [수정] 초기화 및 차원 계산 (예외 처리 강화)
        # ---------------------------------------------------------------------
        if hasattr(self.unwrapped, "observation_manager"):
            group_obs_dim = self.unwrapped.observation_manager.group_obs_dim
            self.num_single_obs = group_obs_dim["stack_policy"][0]
            
            # Policy Keys Check
            if "stack_policy" not in group_obs_dim or "none_stack_policy" not in group_obs_dim:
                raise KeyError('"stack_policy" or "none_stack_policy" missing in observation_manager')

            self.priv_keys = sorted([
                k for k in group_obs_dim.keys() 
                if k.startswith("priv") or k == "privileged"
            ])
            
            self.has_privileged = len(self.priv_keys) > 0
            self.has_standard_critic = "stack_critic" in group_obs_dim
            
            self.priv_obs_dims = {}  # 개별 차원 저장용 (예: {'priv_prio': 31, ...})
            self.total_priv_dim = 0  
    
            if self.has_privileged:
                for key in self.priv_keys:
                    dim = group_obs_dim[key][0]
                    self.priv_obs_dims[key] = dim
                    self.total_priv_dim += dim
                
                print(f"[Wrapper] Privileged Groups Found: {self.priv_obs_dims}")
                print(f"[Wrapper] Total Privileged Dim: {self.total_priv_dim}")
            

        # Determine action dimension
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = self.unwrapped.num_actions

        # -- Policy observations setup (Student)
        if hasattr(self.unwrapped, "observation_manager"):
            stack_policy_dim = group_obs_dim["stack_policy"][0]
            nonstack_policy_dim = group_obs_dim["none_stack_policy"][0]
            self.policy_state_handler = StateHandler(self.num_policy_stacks + 1, stack_policy_dim, nonstack_policy_dim)
            
            self.unwrapped.observation_manager.group_obs_dim["policy"] = (self.policy_state_handler.num_obs,)
            self.num_obs = self.policy_state_handler.num_obs
        else:
            self.num_obs = self.unwrapped.num_observations

        # -- Privileged observations setup (Critic / Teacher)
        if hasattr(self.unwrapped, "observation_manager"):
            if self.has_privileged:
                total_priv_dim = sum(group_obs_dim[k][0] for k in self.priv_keys)
                
                # Teacher Input = Student Obs + Total Privileged Obs
                self.num_privileged_obs = self.num_obs + total_priv_dim
                
                self.unwrapped.observation_manager.group_obs_dim["critic"] = (self.num_privileged_obs,)
                
            elif self.has_standard_critic:
                stack_critic_dim = group_obs_dim["stack_critic"][0]
                nonstack_critic_dim = group_obs_dim["none_stack_critic"][0]
                self.critic_state_handler = StateHandler(self.num_critic_stacks + 1, stack_critic_dim, nonstack_critic_dim)
                self.num_privileged_obs = self.critic_state_handler.num_obs
            else:
                self.num_privileged_obs = self.num_obs

        # reset at the start
        self.env.reset()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    """
    Properties -- Gym.Wrapper
    """
    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        return self.env.unwrapped

    """
    Properties
    """
    def _get_concatenated_privileged_obs(self, obs_dict):
        if not self.has_privileged:
            return None

        tensors = [obs_dict[k] for k in self.priv_keys]
        return torch.cat(tensors, dim=-1)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment with robust handling."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        # 1. Policy Observations (Student/Actor)
        if hasattr(self, "policy_state_handler"):
            if self.policy_state_handler.stack_buffer is None:
                policy_obs = self.policy_state_handler.reset(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
            else:
                policy_obs = self.policy_state_handler.update(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
        else:
            policy_obs = obs_dict["policy"]
        
        obs_dict["policy"] = policy_obs

        # 2. Critic/Teacher Observations (Robust Branching)
        
        # Case A: RMA Mode (Privileged Group Exists)
        if self.has_privileged:
            merged_priv_obs = self._get_concatenated_privileged_obs(obs_dict)
            
            full_state = torch.cat([policy_obs, merged_priv_obs], dim=-1)
            
            obs_dict["critic"] = full_state
            obs_dict["teacher_policy"] = full_state
            
        elif hasattr(self, "critic_state_handler"):
            # Standard Critic
            if self.critic_state_handler.stack_buffer is None:
                critic_obs = self.critic_state_handler.reset(obs_dict.get("stack_critic"), obs_dict.get("none_stack_critic"))
            else:
                critic_obs = self.critic_state_handler.update(obs_dict.get("stack_critic"), obs_dict.get("none_stack_critic"))
            obs_dict["critic"] = critic_obs
        else:
            obs_dict["critic"] = policy_obs

        return policy_obs, {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs_dict, _ = self.env.reset()
        
        # 1. Policy Reset
        if hasattr(self, "policy_state_handler") and self.policy_state_handler is not None:
            policy_obs = self.policy_state_handler.reset(obs_dict["stack_policy"], obs_dict["none_stack_policy"])
        else:
            policy_obs = obs_dict["policy"]
            
        obs_dict["policy"] = policy_obs

        # 2. Critic/Teacher Reset (Robust Branching)
        if self.has_privileged:
            merged_priv_obs = self._get_concatenated_privileged_obs(obs_dict)
            full_state = torch.cat([policy_obs, merged_priv_obs], dim=-1)
            obs_dict["critic"] = full_state
            obs_dict["teacher_policy"] = full_state
            
        elif hasattr(self, "critic_state_handler"):
            critic_obs = self.critic_state_handler.reset(obs_dict.get("stack_critic"), obs_dict.get("none_stack_critic"))
            obs_dict["critic"] = critic_obs
        else:
            obs_dict["critic"] = policy_obs

        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        if not self.use_constraint_rl:
            dones = (terminated | truncated).to(dtype=torch.long)
        else:
            dones = torch.max(terminated, truncated).to(dtype=torch.float32)

        # 1. Policy Update
        if hasattr(self, "policy_state_handler"):
            policy_obs = self.policy_state_handler.update(
                obs_dict["stack_policy"], obs_dict["none_stack_policy"]
            )
            obs_dict["policy"] = policy_obs        
        
        policy_obs = obs_dict["policy"]

        # 2. Critic/Teacher Update (Robust Branching)
        if self.has_privileged:
            merged_priv_obs = self._get_concatenated_privileged_obs(obs_dict)
            
            # 디버깅용
            extras["privileged_obs"] = merged_priv_obs 
            
            full_state = torch.cat([policy_obs, merged_priv_obs], dim=-1)
            obs_dict["critic"] = full_state
            
        elif hasattr(self, "critic_state_handler"):
            critic_obs = self.critic_state_handler.update(obs_dict.get("stack_critic"), obs_dict.get("none_stack_critic"))
            obs_dict["critic"] = critic_obs
        else:
            obs_dict["critic"] = policy_obs

        extras["observations"] = obs_dict # Agent가 여기서 priv_physical을 꺼내 씀
        
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return policy_obs, rew, dones, extras


    def close(self):
        return self.env.close()
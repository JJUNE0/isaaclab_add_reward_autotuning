#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.distributions import Normal , MultivariateNormal, Independent
from copy import deepcopy

def torch_gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))

class Qnet(nn.Module):
    def __init__(
        self, 
        o_dim, 
        a_dim, 
        eqi_o_idx, 
        reg_o_idx,
        eqi_a_idx,
        reg_a_idx, 
        hidden_dims = [256,256,256],
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        # symmetry info
        self.inv_o_idx = eqi_o_idx
        self.reg_o_idx = reg_o_idx
        self.inv_o_dim = len(self.inv_o_idx)
        self.reg_o_dim = o_dim - self.inv_o_dim
        
        self.inv_a_idx = eqi_a_idx
        self.reg_a_idx = reg_a_idx
        self.inv_a_dim = len(self.inv_a_idx)
        self.reg_a_dim = len(self.reg_a_idx)
        
        all_indices = torch.arange(o_dim)  # [0, 1, 2, ..., 158]

# self.inv_o_idx는 예: [3, 7, 12]
        inv_o_idx = torch.tensor(self.inv_o_idx)  # 제외할 인덱스

# boolean 마스크 생성 후 제외
        mask = torch.ones(o_dim, dtype=torch.bool)
        mask[inv_o_idx] = False

# 남은 인덱스
        self.reg_o_idx = all_indices[mask].tolist()

        # default info
        self.o_dim, self.a_dim = o_dim, a_dim

        self.inv_layer1 = nn.Linear(self.inv_o_dim , hidden_dims[0])  # + self.inv_a_dim
        self.reg_layer1 = nn.Linear(self.reg_o_dim , hidden_dims[0])  # + self.reg_a_dim
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.output_layer = nn.Linear(hidden_dims[2], 1)
    
    def forward(self, o):
        # observation 을 invariant 와 regular 로 분리
        inv_o = o[:, self.inv_o_idx]
        reg_o = o[:, self.reg_o_idx]
        # action 을 invariant 와 regular 로 분리
        # inv_a = a[:, self.inv_a_idx]
        # reg_a = a[:, self.reg_a_idx]
        
        # invariant observation 과 action concat 해 invariant feature 추출
        #inv_inputs = torch.cat([inv_o, inv_a], dim=1)
        inv_feature = self.inv_layer1(inv_o)
        
        # regular observation 의 feature 추출
        #reg_inputs = torch.cat([reg_o, reg_a], dim=1)
        reg_feature = self.reg_layer1(reg_o)
        
        # |invariant| + regular feature 특징 추출
        # invarainant 의 부호를 왜 절대값으로 할까? symmetry 면 + 는 -, - 는 + 로 해야되는 게 아닌가.
        feature = torch.abs(inv_feature) + reg_feature
        
        # bias 더해주고 3개의 layer 통과
        layer = self.layer2(feature)
        layer = F.gelu(layer)
        layer = self.layer3(layer)
        layer = F.gelu(layer)
        
        V = self.output_layer(layer)

        return V
    
    

class Policynet(nn.Module):
    def __init__(
        self, 
        o_dim,
        a_dim,   
        eqi_o_idx, 
        reg_o_idx,
        hidden_dims = [256,256,256],
#        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        self.LOG_SIG_MIN = -20.0
        self.LOG_SIG_MAX = 2.0
        # symmetry info
        self.eqi_o_idx = eqi_o_idx
        self.reg_o_idx = reg_o_idx
        self.eqi_o_dim = len(self.eqi_o_idx)
        self.reg_o_dim = len(self.reg_o_idx)

        # default info
        self.o_dim = o_dim
        self.inv_layer1 = nn.Linear(self.eqi_o_dim , hidden_dims[0])
        self.reg_layer1 = nn.Linear(self.reg_o_dim , hidden_dims[0])
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.std_layer = nn.Linear(hidden_dims[2], a_dim)
        self.mu_layer = nn.Linear(hidden_dims[2], a_dim)
        
    
    def forward(self, o, eval = False):
        # observation 을 invariant 와 regular 로 분리
        eqi_o = o[:, self.eqi_o_idx]
        reg_o = o[:, self.reg_o_idx]

        eqi_feature = self.inv_layer1(eqi_o)
        source = torch.sum(eqi_feature, dim=1, keepdim=True)
        multiplier = torch.where(source < 0.0, -1.0 * torch.ones_like(source), torch.ones_like(source))
        
        reg_feature = self.reg_layer1(reg_o)
        # |invariant| + regular feature 특징 추출
        # invarainant 의 부호를 왜 절대값으로 할까? symmetry 면 + 는 -, - 는 + 로 해야되는 게 아닌가.
        feature = torch.abs(eqi_feature) + reg_feature
        
        # bias 더해주고 3개의 layer 통과
        layer = self.layer2(feature)
        layer = F.gelu(layer)
        layer = self.layer3(layer)
        layer = F.gelu(layer)
        
        mu = self.mu_layer(layer)
        mu *= multiplier
        log_std = self.std_layer(layer)
        std = torch.exp(torch.clamp(log_std, self.LOG_SIG_MIN, self.LOG_SIG_MAX))
        #dist = MultivariateNormal(mu, std)
        dist = Independent(Normal(mu, std), 1)
        
        
        if eval :
            samples = mu
        else :
            samples = dist.sample()
        # samples : batch, num_actions
        
        # [batch, num_actions]
        actions = torch.tanh(samples)

        log_probs = dist.log_prob(samples).view(-1, 1)

        log_probs -= torch.sum(torch.log(1 - actions ** 2 + 1e-10), dim=1, keepdim=True)


        return actions, log_probs


class SIEActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        eqi_o_idx,
        reg_o_idx,
        eqi_a_idx,
        reg_a_idx,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
        # Actor 부분: Policynet을 사용
        # self.actor = Policynet(
        #     o_dim=num_actor_obs,
        #     a_dim=num_actions,
        #     eqi_o_idx=eqi_o_idx,
        #     reg_o_idx=reg_o_idx,
        #     hidden_dims=actor_hidden_dims,
        # )
        
        # Critic 부분: Qnet을 사용
        self.critic = Qnet(
            o_dim=mlp_input_dim_c,
            a_dim=num_actions,
            eqi_o_idx=eqi_o_idx,
            reg_o_idx=reg_o_idx,
            eqi_a_idx=eqi_a_idx,  # 이 부분은 Qnet에서 액션을 받아야 하므로 입력을 수정해야 할 수 있습니다.
            reg_a_idx=reg_a_idx,  # 위와 동일하게 수정 필요
            hidden_dims=critic_hidden_dims,
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(1.0 * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

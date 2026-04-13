import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from .actor_critic import ActorCritic, get_activation

# ----------------------------------------------------------------------
# 1. Helper Modules (Encoder, Adaptation)
# ----------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], activation="elu"):
        super().__init__()
        self.activation = get_activation(activation)
        
        layers = []
        prev_dim = input_dim
        
        # 1. 히든 레이어: Spectral Norm으로 안정성 확보
        for h_dim in hidden_dims:
            layers.append(spectral_norm(nn.Linear(prev_dim, h_dim)))
            layers.append(self.activation)
            prev_dim = h_dim
            
        # 2. 출력 레이어: Spectral Norm + Linear (No Tanh, No LayerNorm)
        # 물리적 크기 정보 보존 + 립시츠 제약으로 안정성 확보
        layers.append(spectral_norm(nn.Linear(prev_dim, output_dim)))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------------------------
# 1. Refined Adaptation Module (Dilated TCN + Norms)
# ----------------------------------------------------------------------

class AdaptationModule(nn.Module):
    """
    [Student Eye] History -> Latent z AND System ID
    Improvements:
      - Input Normalization (LayerNorm)
      - Dilated Convolutions (Exponential Receptive Field)
      - Feature Normalization before Heads
    """
    def __init__(self, input_dim, output_dim, history_len, hidden_dims=[128, 128, 128], kernel_sizes=[5, 3, 3]):
        super().__init__()
        self.input_dim = input_dim
        self.history_len = history_len
        
        # [Improvement 1] Input Normalization
        # 시계열 데이터(B, Dim, T)로 변환 전, Feature 차원(Dim)에 대해 정규화 수행
        self.input_norm = nn.LayerNorm(input_dim)
        
        # [Improvement 2] Dilated Convolution Layers
        # Dilation을 1 -> 2 -> 4로 증가시켜 Receptive Field를 대폭 확장
        layers = []
        
        # Layer 1: Standard Conv (Capture immediate dynamics)
        layers.append(nn.Conv1d(input_dim, hidden_dims[0], kernel_size=kernel_sizes[0], stride=1, padding=0, dilation=1))
        layers.append(nn.ReLU())
        
        # Layer 2: Dilated Conv (dilation=2)
        layers.append(nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=kernel_sizes[1], stride=1, padding=0, dilation=2))
        layers.append(nn.ReLU())
        
        # Layer 3: Dilated Conv (dilation=4)
        layers.append(nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=kernel_sizes[2], stride=1, padding=0, dilation=4))
        layers.append(nn.ReLU())
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Flatten Dimension Calculation
        with torch.no_grad():
            # (Batch, Input_dim, History_len)
            dummy = torch.zeros(1, input_dim, history_len)
            out = self.conv_layers(dummy)
            self.flatten_dim = out.view(1, -1).shape[1]
            print(f"Adaptation Module - History: {history_len}, Flatten Dim: {self.flatten_dim}")

        # Shared Backbone
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            # [Improvement 3] Feature Normalization
            # Teacher의 Encoder도 끝단에 LayerNorm이 있으므로, Student도 분포를 맞춰줍니다.
            nn.LayerNorm(128) 
        )

        # Heads
        self.latent_head = nn.Linear(128, output_dim)
    

    def forward(self, x):
        # x shape: (Batch, History, Dim) or (Batch, History * Dim)
        
        # 1. Reshape if flattened
        if x.dim() == 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, self.history_len, self.input_dim)
            
        # 2. Input Normalization
        # LayerNorm은 마지막 차원(Dim)에 대해 작동하므로 (Batch, History, Dim) 상태에서 적용
        x = self.input_norm(x)
        
        # 3. Permute for Conv1d: (Batch, Dim, History)
        x = x.permute(0, 2, 1)
        
        # 4. Convolution (Feature Extraction)
        feat = self.conv_layers(x)
        feat = feat.view(feat.size(0), -1)
        
        # 5. Shared Features
        shared_feat = self.shared_mlp(feat)
        
        # 6. Heads
        z_hat = self.latent_head(shared_feat)
        
        return z_hat

# ----------------------------------------------------------------------
# 2. Main Agents (Teacher, Student)
# ----------------------------------------------------------------------

class RMATeacher(ActorCritic):
    def __init__(self, num_proprio_obs, num_privileged_obs, num_critic_obs, num_actions, latent_dim=16, **kwargs):
        super().__init__(
            num_actor_obs=num_proprio_obs + latent_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            **kwargs
        )
        self.encoder = Encoder(num_privileged_obs, latent_dim)
        print(f"Encoder : {self.encoder}")
        self.num_actions = num_actions
        self.num_proprio_obs = num_proprio_obs
        
    def act(self, obs, privileged_info=None, z=None,**kwargs):
        if z is None:
            z = self.encoder(privileged_info)
        policy_input = torch.cat([obs, z], dim=-1)
        self.update_distribution(policy_input)
        return self.distribution.sample()
    
    def act_inference(self, obs, privileged_info=None):
        z = self.encoder(privileged_info)
        policy_input = torch.cat([obs, z], dim=-1)
        return self.actor(policy_input)

class RMAStudent(ActorCritic):
    def __init__(self, teacher_policy, num_obs, num_envs, device="cpu", 
                 num_single_obs=28, num_stacks=3, latent_dim=16, history_len=50, 
                 **kwargs):
        super().__init__(
            num_actor_obs=num_obs + latent_dim, 
            num_critic_obs=1,                   
            num_actions=teacher_policy.num_actions,
            **kwargs
        )
        
        # Freeze Teacher Components
        self.actor = teacher_policy.actor
        self.teacher_encoder = teacher_policy.encoder
        
        for p in self.actor.parameters(): p.requires_grad = False
        for p in self.teacher_encoder.parameters(): p.requires_grad = False
        
        if hasattr(teacher_policy, "std"):
            self.std.data.copy_(teacher_policy.std.data)
            self.std.requires_grad = False
        else:
            self.std = nn.Parameter(torch.zeros(teacher_policy.num_actions, device=device))
            self.std.requires_grad = False

        # Dimensions
        self.num_obs = num_obs
        self.history_len = history_len
        self.num_envs = num_envs
        self.single_obs_dim = num_single_obs
        self.exterio_dim = num_obs - num_single_obs * num_stacks
        self.frame_dim = self.single_obs_dim + self.exterio_dim

        # Buffers
        self.register_buffer("history_buffer", torch.zeros(num_envs, history_len, self.frame_dim, device=device))
        self.register_buffer("current_obs", torch.zeros(num_envs, num_obs, device=device)) 

        # [Check] Initialization Logic for Adaptation Module
        self.adaptation_module = AdaptationModule(
            input_dim=self.frame_dim, 
            output_dim=latent_dim, 
            history_len=history_len,
            # Dilation을 고려한 커널 사이즈 조정 권장
            hidden_dims=[128, 128, 128],
            kernel_sizes=[5, 3, 3] 
        )

        print(f"Refined Adaptation Module Initialized with Dilated Convs & LayerNorm")
        
    def update_history(self, obs, dones=None):
        """ FIFO History Buffer Update """
        self.current_obs.copy_(obs)
        
        latest_obs = obs[:, :self.single_obs_dim]
        latest_command = obs[:, -self.exterio_dim:] 
        latest_frame = torch.cat([latest_obs, latest_command], dim=-1)
        
        # Shift Buffer
        self.history_buffer[:, :-1, :] = self.history_buffer[:, 1:, :].clone()
        self.history_buffer[:, -1, :] = latest_frame

        if dones is not None:
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                self.history_buffer[reset_ids] = latest_frame[reset_ids].unsqueeze(1).expand(-1, self.history_len, -1).clone()

    def forward(self, obs=None):
        # Adaptation Module Forward
        z_hat = self.adaptation_module(self.history_buffer)
        
        policy_in = torch.cat([self.current_obs, z_hat], dim=-1)
        action_mean = self.actor(policy_in)
        
        return action_mean, z_hat
        
    def update_distribution(self, action_mean):
        """action_mean을 직접 분포의 mean으로 사용 (actor를 다시 실행하지 않음)"""
        self.distribution = torch.distributions.Normal(action_mean, action_mean * 0.0 + self.std)

    def act(self, obs=None, **kwargs):
        """ Rollout Mode """
        action_mean, _ = self.forward()
        self.update_distribution(action_mean)
        return self.distribution.sample()
        
    def act_inference(self, obs=None):
        """ Inference Mode """
        with torch.no_grad():
            action_mean, _,  = self.forward()
            return action_mean
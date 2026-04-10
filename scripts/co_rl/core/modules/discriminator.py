import torch
import torch.nn as nn
import torch.autograd as autograd # Gradient Penalty 계산을 위해 필요
import torch.nn.utils as utils


class DiscriminatorBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, delta: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_gradient_penalty(self, 
                                 expert_delta: torch.Tensor, 
                                 policy_delta: torch.Tensor, 
                                 obs: torch.Tensor,  # [추가] Condition
                                 weight: float) -> torch.Tensor:
        if weight == 0.0:
            return 0.0

        batch_size = expert_delta.shape[0]
        
        alpha = torch.rand(batch_size, 1, device=expert_delta.device)
        
        if expert_delta.dim() > 2:
            alpha = alpha.unsqueeze(-1)

        interpolates = (alpha * expert_delta + (1 - alpha) * policy_delta.detach()).requires_grad_(True)
        
        disc_interpolates = self(interpolates, obs)
        
        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients_flat = gradients.view(batch_size, -1)
        gradient_norm = gradients_flat.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty * weight

    def compute_logit_regularization(self, 
                                     expert_logits: torch.Tensor, 
                                     policy_logits: torch.Tensor, 
                                     weight: float) -> torch.Tensor:
        if weight == 0.0:
            return 0.0
        
        logit_reg = (torch.square(expert_logits).mean() + 
                     torch.square(policy_logits).mean())
        
        return logit_reg * weight


# ----------------------------
# Context-Aware Discriminator MLP
# ----------------------------
class DiscriminatorMLP(DiscriminatorBase):
    def __init__(self, delta_dim: int, obs_dim: int, hidden_dims: list[int] = [256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = delta_dim + obs_dim
        
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            layers.append(utils.spectral_norm(linear)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = h_dim
            
        # Final Layer
        self.final_layer = utils.spectral_norm(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers, self.final_layer)

    def forward(self, delta: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # (Batch, Delta_Dim) + (Batch, Obs_Dim) -> (Batch, Delta+Obs)
        d_input = torch.cat([delta, obs], dim=-1)
        return self.net(d_input)
    
    def extract_features(self, delta: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        h = torch.cat([delta, obs], dim=-1)
        for layer in self.net[:-1]: 
            h = layer(h)
        return h

# ----------------------------
# Base Class (신규 추가)
# ----------------------------
# class DiscriminatorBase(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError

#     def compute_gradient_penalty(self, 
#                                  expert_data: torch.Tensor, 
#                                  policy_data: torch.Tensor, 
#                                  weight: float) -> torch.Tensor:

#         if weight == 0.0:
#             return 0.0

#         batch_size = expert_data.shape[0]
        
#         alpha = torch.rand(batch_size, 1, device=expert_data.device)
        
#         if expert_data.dim() > 2:
#             alpha = alpha.unsqueeze(-1) # (N, 1, 1)

#         interpolates = (alpha * expert_data + (1 - alpha) * policy_data.detach()).requires_grad_(True)
        
#         disc_interpolates = self(interpolates)
        
#         gradients = autograd.grad(
#             outputs=disc_interpolates,
#             inputs=interpolates,
#             grad_outputs=torch.ones_like(disc_interpolates),
#             create_graph=True,  
#             retain_graph=True,
#         )[0]

#         gradients_flat = gradients.view(batch_size, -1)
#         gradient_norm = gradients_flat.norm(2, dim=1)
        
#         gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
#         return gradient_penalty * weight

#     def compute_logit_regularization(self, 
#                                      expert_logits: torch.Tensor, 
#                                      policy_logits: torch.Tensor, 
#                                      weight: float) -> torch.Tensor:
        
#         if weight == 0.0:
#             return 0.0
            
#         # expert와 policy 양쪽의 logit 제곱합에 대한 페널티
#         logit_reg = (torch.square(expert_logits).mean() + 
#                      torch.square(policy_logits).mean())
        
#         return logit_reg * weight

# # ----------------------------
# # Non-Motion (수정)
# # ----------------------------
# class DiscriminatorMLP(DiscriminatorBase):
#     def __init__(self, in_dim: int, hidden_dims: list[int] = [128, 128]):
#         super().__init__()
#         layers = []
#         prev_dim = in_dim
#         for h_dim in hidden_dims:
#             linear = nn.Linear(prev_dim, h_dim)
#             # [수정] LayerNorm 대신 Spectral Normalization 사용
#             # 논문 포인트: "We enforce Lipschitz continuity via Spectral Normalization to stabilize adversarial training under domain shifts."
#             layers.append(utils.spectral_norm(linear)) 
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             prev_dim = h_dim
            
#         # 마지막 레이어에도 SN 적용
#         self.final_layer = utils.spectral_norm(nn.Linear(prev_dim, 1))
#         self.net = nn.Sequential(*layers, self.final_layer)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)
    
#     def extract_features(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         마지막 확률값(0~1)이 아니라, 그 직전의 풍부한 특징 벡터를 반환
#         """
#         h = x
#         # 마지막 레이어 하나 전까지만 통과시킴
#         for layer in self.net[:-1]: 
#             h = layer(h)
#         return h  # (Batch, Hidden_Dim) 벡터 반환
# ----------------------------
# Imitation Motion (수정)
# ----------------------------
class DiscriminatorGRU(DiscriminatorBase): # <- nn.Module 대신 DiscriminatorBase 상속
    def __init__(self, in_dim: int, hidden_dims: list[int] = [128], num_layers: int = 1):
        super().__init__() # <- 부모 클래스(DiscriminatorBase) 초기화
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dims[0], num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dims[0], 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (N, T, D)
        out, _ = self.gru(x_seq)            # (N, T, H)
        h_last = out[:, -1]                 # (N, H)
        # self.head가 (H, 1)을 출력하므로 (N, 1)이 됨.
        # .unsqueeze(-1)는 불필요하며 (N, 1, 1)을 만들어 BCE Loss와 충돌할 수 있습니다.
        return self.head(h_last)            # (N, 1) 반환

# import torch
# import torch.nn as nn
# import torch.autograd as autograd # Gradient Penalty 계산을 위해 필요
# import torch.nn.utils as utils

# # ----------------------------
# # Base Class (신규 추가)
# # ----------------------------
# class DiscriminatorBase(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """ 
#         서브클래스(MLP, GRU)에서 이 메서드를 반드시 구현해야 합니다. 
#         """
#         raise NotImplementedError

#     def compute_gradient_penalty(self, 
#                                  expert_data: torch.Tensor, 
#                                  policy_data: torch.Tensor, 
#                                  weight: float) -> torch.Tensor:
#         """ WGAN-GP에서 사용하는 Gradient Penalty를 계산합니다. """
        
#         # 가중치가 0이면 계산 스킵
#         if weight == 0.0:
#             return 0.0

#         batch_size = expert_data.shape[0]
        
#         # 1. expert와 policy 샘플 사이를 보간(interpolate)
#         alpha = torch.rand(batch_size, 1, device=expert_data.device)
        
#         # 데이터가 시퀀스(N, T, D)인 경우 (GRU)를 대비해 차원 확장
#         if expert_data.dim() > 2:
#             alpha = alpha.unsqueeze(-1) # (N, 1, 1)

#         interpolates = (alpha * expert_data + (1 - alpha) * policy_data.detach()).requires_grad_(True)
        
#         # 2. 보간된 샘플을 디스크리미네이터에 통과
#         #    (self(interpolates)는 서브클래스의 forward를 호출)
#         disc_interpolates = self(interpolates)
        
#         # 3. 보간된 입력(interpolates)에 대한 출력(disc_interpolates)의 그래디언트 계산
#         gradients = autograd.grad(
#             outputs=disc_interpolates,
#             inputs=interpolates,
#             grad_outputs=torch.ones_like(disc_interpolates),
#             create_graph=True,  # disc_loss.backward() 시 이 그래프를 또 미분해야 함
#             retain_graph=True,
#         )[0] # .grad()는 튜플을 반환하므로 [0] 선택

#         # 4. 그래디언트의 L2-norm 계산
#         #    (N, T, D) 또는 (N, D) -> (N, -1)로 flatten
#         gradients_flat = gradients.view(batch_size, -1)
#         gradient_norm = gradients_flat.norm(2, dim=1)
        
#         # 5. (norm - 1)^2 형태의 페널티 계산
#         gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
#         return gradient_penalty * weight

#     def compute_logit_regularization(self, 
#                                      expert_logits: torch.Tensor, 
#                                      policy_logits: torch.Tensor, 
#                                      weight: float) -> torch.Tensor:
#         """ 
#         디스크리미네이터의 출력이 과도하게 커지는 것을 방지하기 위해
#         logit 값 자체에 L2 페널티를 부여합니다.
#         """
        
#         if weight == 0.0:
#             return 0.0
            
#         # expert와 policy 양쪽의 logit 제곱합에 대한 페널티
#         logit_reg = (torch.square(expert_logits).mean() + 
#                      torch.square(policy_logits).mean())
        
#         return logit_reg * weight

# # ----------------------------
# # Non-Motion (수정)
# # ----------------------------
# class DiscriminatorMLP(DiscriminatorBase):
#     def __init__(self, in_dim: int, hidden_dims: list[int] = [128, 128]):
#         super().__init__()
#         layers = []
#         prev_dim = in_dim
#         for h_dim in hidden_dims:
#             linear = nn.Linear(prev_dim, h_dim)
#             # [수정] LayerNorm 대신 Spectral Normalization 사용
#             # 논문 포인트: "We enforce Lipschitz continuity via Spectral Normalization to stabilize adversarial training under domain shifts."
#             layers.append(utils.spectral_norm(linear)) 
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             prev_dim = h_dim
            
#         # 마지막 레이어에도 SN 적용
#         self.final_layer = utils.spectral_norm(nn.Linear(prev_dim, 1))
#         self.net = nn.Sequential(*layers, self.final_layer)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)
    
#     def extract_features(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         마지막 확률값(0~1)이 아니라, 그 직전의 풍부한 특징 벡터를 반환
#         """
#         h = x
#         # 마지막 레이어 하나 전까지만 통과시킴
#         for layer in self.net[:-1]: 
#             h = layer(h)
#         return h  # (Batch, Hidden_Dim) 벡터 반환
# # ----------------------------
# # Imitation Motion (수정)
# # ----------------------------
# class DiscriminatorGRU(DiscriminatorBase): # <- nn.Module 대신 DiscriminatorBase 상속
#     def __init__(self, in_dim: int, hidden_dims: list[int] = [128], num_layers: int = 1):
#         super().__init__() # <- 부모 클래스(DiscriminatorBase) 초기화
#         self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dims[0], num_layers=num_layers, batch_first=True)
#         self.head = nn.Linear(hidden_dims[0], 1)

#     def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
#         # x_seq: (N, T, D)
#         out, _ = self.gru(x_seq)            # (N, T, H)
#         h_last = out[:, -1]                 # (N, H)
#         # self.head가 (H, 1)을 출력하므로 (N, 1)이 됨.
#         # .unsqueeze(-1)는 불필요하며 (N, 1, 1)을 만들어 BCE Loss와 충돌할 수 있습니다.
#         return self.head(h_last)            # (N, 1) 반환
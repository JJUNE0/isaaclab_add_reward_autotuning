import torch
import torch.optim as optim

class LagrangianTuner:
    def __init__(self, target_value, learning_rate=3e-4, device='cpu'):
        self.target_value = target_value
        self.device = device
        
        self.log_lagrange_multiplier = torch.zeros(1, requires_grad=True, device=device)
        
        self.optimizer = optim.Adam([self.log_lagrange_multiplier], lr=learning_rate)

    def update(self, current_value):

        multiplier = self.log_lagrange_multiplier.exp()
        
        # 수식 의미: (현재값 - 목표값) 차이만큼 multiplier를 조절해라
        loss = - (multiplier * (current_value - self.target_value).detach()).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 현재 조절된 승수 값을 반환 (이 값을 메인 알고리즘에서 loss에 곱해서 씀)
        return multiplier.item()
    


# [SAC] Automatic Entropy Tuning
class EntropyTuner(LagrangianTuner):
    def update(self, log_prob):
        """
        log_prob: 현재 상태에서 Action을 뽑았을 때의 로그 확률 (Batch Size, 1)
        """

        alpha = self.log_lagrange_multiplier.exp()

        # 2. Loss 계산 (SAC 논문 공식)
        # 공식 유도: alpha * (entropy - target_entropy) 
        # entropy = -log_prob 이므로 -> -alpha * log_prob - alpha * target.
        entropy_loss = - (alpha * (log_prob + self.target_value).detach()).mean()

        self.optimizer.zero_grad()
        entropy_loss.backward()
        self.optimizer.step()

        return alpha.item()
    

# [GAN] Constraint Tuning (예: Reconstruction Error 제한)
class GANTuner(LagrangianTuner):
    def update(self, current_constraint_val):
        """
        current_constraint_val: 제약 조건 위반 값 (예: L2 Loss, Gradient Penalty 등)
        """
        lam = self.log_lagrange_multiplier.exp()

        # 목표: current_val <= target_value (이하로 유지해라)
        # current가 target보다 크면 -> lam을 키워서 Penalty를 강화
        # current가 target보다 작으면 -> lam을 줄여서 자유도 부여
        constraint_loss = - (lam * (current_constraint_val - self.target_value).detach()).mean()

        self.optimizer.zero_grad()
        constraint_loss.backward()
        self.optimizer.step()

        return lam.item()
    
# [PPO] Adaptive KL Penalty Tuning
class KLTuner(LagrangianTuner):
    def update(self, kl_divergence):
        """
        kl_divergence: old_policy와 new_policy 사이의 KL 값
        """
        beta = self.log_lagrange_multiplier.exp()

        # 목표: KL Div가 target(0.01)보다 커지면 -> Beta를 키워서 Loss를 엄청 줌 (학습 억제)
        # 목표: KL Div가 target보다 작으면 -> Beta를 줄여서 학습 가속
        kl_loss = - (beta * (kl_divergence - self.target_value).detach()).mean()

        self.optimizer.zero_grad()
        kl_loss.backward()
        self.optimizer.step()
        
        return beta.item()
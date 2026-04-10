# scripts/co_rl/core/modules/utils.py

import torch
import torch.nn as nn
from typing import Optional

class DiffNormalizer(nn.Module):
    """
    Simple online normalizer for difference vectors (Δ).
    Records absolute mean magnitude over samples and normalizes new data.
    
    [수정] nn.Module을 상속받아 state_dict 저장/로드를 지원합니다.
    """

    def __init__(self, dim: int, device: torch.device, eps: float = 0.1):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # [핵심] register_buffer로 등록하면 state_dict에 자동으로 저장/로드됩니다.
        # 또한 model.to(device) 호출 시 자동으로 따라갑니다.
        self.register_buffer("mean_abs", torch.zeros(dim, device=device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.float32, device=device))

    @torch.no_grad()
    def record(self, delta: torch.Tensor):
        """입력된 Δ 샘플들의 절댓값 평균을 누적"""
        batch_mean = delta.abs().mean(dim=0)
        
        if self.count == 0:
            self.mean_abs.copy_(batch_mean)
        else:
            # EMA 방식 또는 누적 평균 방식
            new_mean = (self.mean_abs * self.count + batch_mean) / (self.count + 1)
            self.mean_abs.copy_(new_mean)
            
        self.count += 1

    @torch.no_grad()
    def normalize(self, delta: torch.Tensor) -> torch.Tensor:
        """Δ를 러닝 평균 |Δ|로 정규화"""
        denom = torch.clamp(self.mean_abs, min=self.eps)
        return delta / denom

    @torch.no_grad()
    def reset(self):
        """정규화기 초기화"""
        self.mean_abs.zero_()
        self.count.zero_()


# ----------------------------
# 보상 정규화 EMA
# ----------------------------
class EMA:
    def __init__(self, momentum: float = 0.99, eps: float = 1e-8, device: Optional[torch.device] = None):
        self.m = None
        self.v = None
        self.mom = momentum
        self.eps = eps
        self.device = device

    def update(self, x: torch.Tensor):
        mean = x.mean()
        var = x.var(unbiased=False) + self.eps
        if self.m is None:
            self.m = mean.detach()
            self.v = var.detach()
        else:
            self.m = self.mom * self.m + (1 - self.mom) * mean.detach()
            self.v = self.mom * self.v + (1 - self.mom) * var.detach()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.m is None:
            return x
        return (x - self.m) / torch.sqrt(self.v + self.eps)
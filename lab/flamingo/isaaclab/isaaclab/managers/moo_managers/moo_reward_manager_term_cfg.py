# moo_reward_manager_term_cfg.py
from __future__ import annotations
from typing import Any, Optional, Callable
from dataclasses import MISSING, field
from isaaclab.utils import configclass
from isaaclab.managers import ManagerTermBaseCfg

@configclass
class MOORewardTermCfg(ManagerTermBaseCfg):
    """
    개별 MOO term: '특징 → 스칼라 오차' 계산만 담당.
    학습 하이퍼파라미터는 env/manager 전역으로 이동.
    """
    # 필수: 스칼라 보상(오차) 계산 함수 (env, obs, **params) -> torch.Tensor [N]
    func: Callable = MISSING

    # 필수: 특징/목표 생성기 등 함수 포인터와 상수들이 담긴 컨테이너
    # 예: {"disc_obs_policy_func": fn_policy, "disc_obs_demo_func": fn_demo, "std": 0.5, ...}
    params: dict[str, Any] = field(default_factory=dict)
    
    alpha : float = 1.0
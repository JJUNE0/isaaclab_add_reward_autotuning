from __future__ import annotations

import torch
import numpy as np
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def moo_feature_error(env, **params):
    f_pol = params["disc_obs_policy_func"]     # <- 피처/오차 벡터 함수
    err   = f_pol(env, **params)               # (N, D)  e.g., (vel_xy - cmd_xy)/std
    loss  = (err * err).sum(dim=-1)            # (N,)   L2
    reward = -loss
    return reward

def moo_feature_error_huber(env, delta=0.1, **params):
    f_pol = params["disc_obs_policy_func"]
    err   = f_pol(env, **params)                 # (N, D)
    abs_e = err.abs()
    quad  = 0.5 * (abs_e**2)
    lin   = delta * (abs_e - 0.5 * delta)
    huber = torch.where(abs_e <= delta, quad, lin).sum(dim=-1)  # (N,)
    return -huber

# --- 0) 공통 유틸: deadzone(허용대역) 기반 범위 손실 ---
def range_huber_loss(x, lo, hi, delta=0.1):
    # lo <= x <= hi 이면 0, 넘어가면 허버 손실
    under = (lo - x).clamp_min(0.0)
    over  = (x - hi).clamp_min(0.0)
    z = under + over  # 바깥으로 벗어난 양 (>=0)
    quad = 0.5 * (z ** 2)
    lin  = delta * (z - 0.5 * delta)
    huber = torch.where(z <= delta, quad, lin)
    return huber  # x와 동일 shape

# --- 1) 관측기반 리듀서: 허용대역 범위 패널티 ---
def moo_obs_range_loss(env, **params):
    """
    disc_obs_policy_func(env, **params)  -> obs (N, D)
    target_fn(env, obs) -> (lo, hi)  각각 (N, D) 또는 브로드캐스트 가능
    반환: -loss (N,)  (보상은 음의 손실)
    """
    f_obs  = params["disc_obs_policy_func"]
    target_fn = params["target_fn"]   # 필수
    delta = params.get("delta", 0.1)

    obs = f_obs(env, **params)              # (N, D)
    lo, hi = target_fn(env, obs, **params)  # (N, D) or scalar
    loss = range_huber_loss(obs, lo, hi, delta=delta).sum(dim=-1)  # (N,)
    return -loss

# --- 2) 관측→오차 변환 후 L2/허버 ---
def moo_obs_to_err_l2(env, **params):
    """
    obs -> err = (obs - target) / std,   reward = -||err||^2
    """
    f_obs = params["disc_obs_policy_func"]
    target_fn = params["target_fn"]
    std = params.get("std", 1.0)

    obs = f_obs(env, **params)             # (N, D)
    tgt = target_fn(env, obs, **params)    # (N, D) or (N,1) or scalar
    err = (obs - tgt) / (std + 1e-8)
    return -(err * err).sum(dim=-1)
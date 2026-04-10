# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.reward_manager import RewardManager


from .moo_reward_manager_term_cfg import MOORewardTermCfg          # 각 텀: 오차(feature) 반환용 콜러블 등록
from scripts.co_rl.core.modules.utils import DiffNormalizer, EMA          # record(x), normalize(x) 인터페이스 가정


try:
    from prettytable import PrettyTable
    _HAS_PRETTY = True
except Exception:
    _HAS_PRETTY = False


class MOORewardManager(RewardManager):

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        # RewardManager 초기화(버퍼 등 생성). _prepare_terms()를 우리가 override하므로
        # super().__init__() 안에서 호출될 때도 우리의 로직이 적용됨.
        super().__init__(cfg, env)

        # 매니저 전역 설정 로드f
        manager_cfg = cfg if isinstance(cfg, dict) else cfg.__dict__

        self.mode = manager_cfg.get("mode", "nonmotion")
        self.window_size = int(manager_cfg.get("window_size", 4))

        self.feature_dim = int(sum(self._term_shapes)) if len(self._term_shapes) > 0 else 1
        self.delta_dim = self.feature_dim if self.mode == "nonmotion" else (self.feature_dim * self.window_size)

        # Δ 버퍼
        self._delta_buf = torch.zeros(self.num_envs, self.delta_dim, device=self.device)
        
        self.alpha = 32
        self.sigma = 1
        
        self._discriminator: Optional[nn.Module] = None
        self._delta_norm : Optional[DiffNormalizer] = None
        self._reward_ema : Optional[EMA] = None


        if self.mode == "motion":
            self._delta_hist_buf = torch.zeros((self.num_envs, self.window_size, self.feature_dim),
                                               device=self.device)
        
        self._term_names = ["total_moo_reward"]
        self._step_reward = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        
    def set(self, 
            discriminator: nn.Module,
            delta_normalizer : DiffNormalizer,
            reward_ema : EMA,
            enable_reward_norm 
            ):
        
        self._discriminator = discriminator
        self._delta_norm = delta_normalizer
        self._reward_ema = reward_ema
        self._enable_reward_norm = enable_reward_norm
        
        print(f"[MOORewardManager] Learning components (D, Norm, EMA) successfully set by Agent.")
    
    def _prepare_terms(self):
        
        self._term_names: list[str] = []
        self._term_cfgs: list[MOORewardTermCfg] = []
        self._term_shapes: list[int] = []

        cfg_items = self.cfg.items() if isinstance(self.cfg, dict) else self.cfg.__dict__.items()

        dummy_dims: Dict[str, int] = {}
        with torch.no_grad():
            for term_name, term_cfg in cfg_items:
                if not isinstance(term_cfg, MOORewardTermCfg):
                    continue

                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
                try:
                    val = term_cfg.func(self._env, **term_cfg.params)
                    # (N,) → dim=1 간주, (N, d) → d 추출
                    if val.dim() <= 1:
                        d = 1
                    else:
                        d = int(val.shape[-1])
                except Exception:
                    d = 1
                dummy_dims[term_name] = d

        for term_name, term_cfg in cfg_items:
            if term_cfg is None or not isinstance(term_cfg, MOORewardTermCfg):
                continue
            #self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            self._term_shapes.append(dummy_dims.get(term_name, 1))

        if len(self._term_shapes) == 0:
            raise RuntimeError(
                "No MOO terms registered. Make sure your cfg uses MOORewardTermCfg entries, "
                "not RewardTermCfg, and that term.func returns (N,) or (N, d)."
            )

        try:
            self._feature_names_for_debug = self._term_names.copy()
            self._feature_shapes_for_debug = self._term_shapes.copy()
        except Exception:
            # 방어 코드
            self._feature_names_for_debug = ["error_copying_names"]
            self._feature_shapes_for_debug = [sum(self._term_shapes)]


        self._class_term_cfgs = []

    def get_latest_delta(
        self,
        normalized: bool = True,
        as_sequence: bool = False,
        recompute: bool = False,
        detach: bool = True,
    ) -> torch.Tensor:

        assert hasattr(self, "_delta_buf"), "Delta buffer is not initialized yet."
        assert isinstance(self.delta_dim, int) and self.delta_dim > 0, "Invalid delta_dim."
        assert isinstance(self.feature_dim, int) and self.feature_dim > 0, "Invalid feature_dim."

        if recompute:
            self._compute_delta_vector()

        if as_sequence:
            if self.mode == "motion":
                delta_seq = self._delta_buf.view(self.num_envs, self.window_size, self.feature_dim)
            else:
                delta_seq = self._delta_buf.view(self.num_envs, 1, self.feature_dim)

            return delta_seq.detach().clone() if detach else delta_seq
        else:
            delta_vec = self._delta_buf

            return delta_vec.detach().clone() if detach else delta_vec


    def get_latest_delta_seq(
        self,
        normalized: bool = True,
        recompute: bool = False,
        detach: bool = True,
    ) -> torch.Tensor:
        """항상 (N, T, D_feat) 시퀀스로 반환. nonmotion은 T=1."""
        return self.get_latest_delta(
            normalized=normalized, as_sequence=True, recompute=recompute, detach=detach
        )

  
    def _compute_delta_vector(self):

        temp = []
        for term_cfg in self._term_cfgs:
            err = term_cfg.func(self._env, **term_cfg.params)  # (N, d_i) or (N,)
            if err.dim() == 1:
                err = err.unsqueeze(-1)  # (N,1)
            temp.append(err)

        features = torch.cat(temp, dim=-1)  # (N, D_feat)

        self._last_raw_features = features
        
        if self.mode == "nonmotion":
            self._delta_buf = features  # (N, D_feat)
        else:
            # 에피소드 리셋된 env의 히스토리 누수 방지
            reset_mask = getattr(self._env, "reset_buf", None)
            if reset_mask is not None:
                self._delta_hist_buf[reset_mask.bool()] = 0.0

            self._delta_hist_buf = torch.roll(self._delta_hist_buf, shifts=-1, dims=1)
            self._delta_hist_buf[:, -1, :] = features
            self._delta_buf = self._delta_hist_buf.view(self.num_envs, -1)  # (N, T*D_feat)

    # ---------------------------
    # 보상 계산(ADD Eq.10)
    # ---------------------------
    def compute(self, dt: float, obs: torch.Tensor | None) -> torch.Tensor:
        """
        [OVERRIDE] RewardManager API
        Hybrid Reward:
        - Discriminator Input: norm_delta (Statistically Normalized)
        - Potential Input: _delta_buf (Physically Scaled via Config)
        """
        if obs is None:
            pass
        
        if self._discriminator is None:
            raise RuntimeError("MOORewardManager: Discriminator not set.")

        self._compute_delta_vector()  
        
        norm_delta = self._delta_norm.normalize(self._delta_buf)

        self._discriminator.eval()
        with torch.no_grad():
            zeros = torch.zeros_like(norm_delta)
            
            # TODO Dynamic Aware Discriminator 모드 구현
            if self.mode == "nonmotion":
                logits = self._discriminator(norm_delta, obs)
                feat_policy = self._discriminator.extract_features(norm_delta, obs)
            else:
                seq = norm_delta.view(self.num_envs, self.window_size, self.feature_dim)
                logits = self._discriminator(seq, obs)
                feat_policy = self._discriminator.extract_features(seq, obs)
                zeros = zeros.view(self.num_envs, self.window_size, self.feature_dim)

            feat_expert = self._discriminator.extract_features(zeros, obs).mean(dim=0, keepdim=True)
            prob = torch.sigmoid(logits).view(-1)
            
            # --- [B] Reward Calculation ---
            reward_fm = torch.exp(-torch.norm(feat_policy - feat_expert, dim=1) / self.sigma)
            reward_disc = -torch.log(torch.clamp(1.0 - prob, min=1e-6))
            
            if self.mode == "nonmotion":
                U = 0.5 * (self._delta_buf ** 2).mean(dim=1)
            else:
                seq_raw = self._delta_buf.view(self.num_envs, self.window_size, self.feature_dim)
                U = 0.5 * (seq_raw ** 2).mean(dim=(1, 2))

            reward_pot = torch.exp(-self.alpha * U)
            
            w_pot = 0.0
            w_disc = 0.2
            w_fm = 0.8
        
            reward_raw = w_pot * reward_pot + w_disc * reward_disc + w_fm * reward_fm
        
        if self._enable_reward_norm:
            self._reward_ema.update(reward_raw.detach())
            reward = self._reward_ema.normalize(reward_raw)
        else:
            reward = reward_raw

        self._reward_buf[:] = reward
        # 에피소드 합 기록
        if "total_moo_reward" not in self._episode_sums:
            self._episode_sums["total_moo_reward"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_sums["total_moo_reward"] += reward
        # 스텝 보상 기록
        if self._step_reward.shape[1] != 1:
            self._step_reward = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
            self._term_names = ["total_moo_reward"]
        self._step_reward[:, 0] = reward

        self.debug_metrics = {
            "mean_discriminator_prob": torch.mean(prob).item(),
            "mean_raw_reward": torch.mean(reward_raw).item(),
            "mean_disc_reward": torch.mean(reward_disc).item(),
            "mean_feature_matching_reward": torch.mean(reward_fm).item(),
            "mean_potential_reward": torch.mean(reward_pot).item(),
            "mean_norm_delta_abs": torch.mean(torch.abs(norm_delta)).item(),
            "mean_potential_U": torch.mean(U).item(),
            "mean_raw_feature_abs": torch.mean(torch.abs(self._last_raw_features)).item(),
            "per_feature_abs_mean": {}
        }
        
        if hasattr(self, "_last_raw_features") and hasattr(self, "_feature_shapes_for_debug"):
            try:
                # (N, D_feat) 텐서를 [ (N,d1), (N,d2), ... ] 리스트로 분할
                split_features = torch.split(self._last_raw_features, self._feature_shapes_for_debug, dim=1)
                
                # self._term_names: ['error_track_lin_vel_xy', 'error_track_ang_vel_z', ...]
                for name, feat_tensor in zip(self._feature_names_for_debug, split_features):
                    # 각 텐서의 (N, di) 배치에 대해 |Error|의 평균 계산
                    mean_abs_err = torch.mean(torch.abs(feat_tensor)).item()
                    self.debug_metrics["per_feature_abs_mean"][name] = mean_abs_err
            
            except Exception as e:
                # (예: self._term_shapes의 합이 D_feat와 불일치)
                self.debug_metrics["per_feature_abs_mean"]["_error"] = f"Split failed: {e}"
           

        return self._reward_buf

    # ---------------------------
    # 정보 출력
    # ---------------------------
    def __str__(self) -> str:
        msg = f"<MOORewardManager (ADD)> mode={self.mode}, feature_dim={self.feature_dim}, delta_dim={self.delta_dim}\n"
        try:
            d_str = str(self._discriminator)
        except Exception:
            d_str = self._discriminator.__class__.__name__
        msg += f"Discriminator:\n{d_str}\n"

        if _HAS_PRETTY and len(getattr(self, "_term_names", [])) > 1:
            table = PrettyTable()
            table.title = "Active Objective Terms (Δ features)"
            table.field_names = ["Index", "Name", "Dim"]
            table.align["Name"] = "l"
            table.align["Dim"] = "r"
            for i, (name, dim) in enumerate(zip(self._term_names, self._term_shapes)):
                table.add_row([i, name, dim])
            msg += table.get_string() + "\n"
        else:
            if hasattr(self, "_term_shapes") and len(self._term_shapes) > 0:
                msg += "Terms: " + ", ".join(
                    f"{n}(d={d})" for n, d in zip(self._term_names, self._term_shapes)
                ) + "\n"
        return msg
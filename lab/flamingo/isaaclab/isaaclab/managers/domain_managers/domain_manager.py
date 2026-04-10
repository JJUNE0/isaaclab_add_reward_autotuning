from __future__ import annotations

import math
import inspect
import torch
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from copy import deepcopy

from scipy import stats
from rich import print as rprint

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.event_manager import EventManager, EventTermCfg

if TYPE_CHECKING:
    from .domain_manager_term_cfg import DomainManagerTermCfg


class DomainManager(EventManager):
    """
    A manager that handles domain randomization with an automatic curriculum.
    It expands randomization ranges based on the agent's performance (Gate mechanism).
    """

    cfg: DomainManagerTermCfg

    def __init__(self, cfg: DomainManagerTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 1. Initialize Curriculum State
        # 0.0 (Easy/No Random) -> 1.0 (Hard/Full Random)
        self.difficulty_level = cfg.init_difficulty_level
        
        # 2. Setup Theory Constants
        # S_E: Ideal expert score estimate
        self.expert_score_est = 1.0
        self.threshold_start = cfg.target_threshold_start
        self.threshold_end = cfg.target_threshold_end

        # 3. Capture Initial/Max Parameters for Interpolation
        self.max_params = {}
        self._capture_max_params()
        
        # 4. Calculate Step Size (rho) based on Dimensions
        self.param_dim = self._count_uncertainty_dims(self.max_params)
        self.rho = math.exp(cfg.kappa / max(self.param_dim, 1)) - 1.0

        # 5. Running Statistics for Gate
        self.gate_window = cfg.gate_window
        self._gate_buf = deque(maxlen=self.gate_window)
        
        # Logging stats
        self.last_confidence = 0.0
        self.last_gate_mean = 0.0

        # 6. Apply Initial Curriculum
        self._apply_curriculum()
        
        rprint(f"[DomainManager] Initialized. Level: {self.difficulty_level:.4f}, Dim(d): {self.param_dim}, Rho: {self.rho:.6f}")

    @property
    def target_score(self) -> float:
        """Calculate the target score threshold for the current difficulty level."""
        # Linear interpolation between min and max reward based on difficulty
        target = self.threshold_start + (self.threshold_end - self.threshold_start) * self.difficulty_level
        # Clamp to theoretical max (Expert Score) just in case
        return min(target, self.expert_score_est)

    def update_curriculum(self, mean_potential_reward: float):
        """
        Updates the curriculum level based on the agent's performance.
        Should be called at the end of an episode or regularly.
        """
        # Add new sample
        self._gate_buf.append(float(mean_potential_reward))

        n = len(self._gate_buf)
        min_n = max(int(self.cfg.min_gate_samples), 2)

        # Not enough samples yet
        if n < min_n:
            self.last_confidence = 0.0
            if n > 0:
                self.last_gate_mean = sum(self._gate_buf) / n
            return

        # Calculate Confidence (Student's t-test)
        conf, mu = self._bayes_confidence_mu_gt_delta(self.target_score)
        self.last_confidence = conf
        self.last_gate_mean = mu

        # Check if we should advance
        if conf >= self.cfg.confidence_level and self.difficulty_level < 1.0:
            self._advance_curriculum(mu, conf)
        
    def get_curriculum_info(self) -> dict[str, float]:
        """Return current status for logging."""
        return {
            "Difficulty_Level": self.difficulty_level,
            "Target_Score": self.target_score,
            "Gate_Mean": self.last_gate_mean,
            "Gate_Conf": self.last_confidence,
            "Param_Dim": float(self.param_dim),
        }

    # ----------------------------------------------------------------------
    # Override: Prepare Terms (FIX FOR TYPE ERROR)
    # ----------------------------------------------------------------------
    def _prepare_terms(self):
        """
        Parses the configuration parameters to prepare the event terms.
        Overridden to SKIP non-EventTermCfg attributes (like init_difficulty_level).
        """
        # Initialize internal storage for event manager
        self._mode_term_names = dict()
        self._mode_term_cfgs = dict()
        self._mode_class_term_cfgs = dict()
        
        # Buffers for interval and reset checks
        self._interval_term_time_left = list()
        self._reset_term_last_triggered_step_id = list()
        self._reset_term_last_triggered_once = list()

        # Handle both dict and object config
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        for term_name, term_cfg in cfg_items:
            # --- [CRITICAL FIX] Skip curriculum scalars ---
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, EventTermCfg):
                continue
            # ----------------------------------------------
            
            # Below is the standard EventManager logic
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)

            # Check pre-startup conflicts
            if term_cfg.mode == "prestartup" and self._env.scene.cfg.replicate_physics:
                raise RuntimeError("Scene replication enabled, cannot use prestartup events.")

            # Resolve class-based terms
            if inspect.isclass(term_cfg.func) and term_cfg.mode == "prestartup":
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

            # Register term modes
            if term_cfg.mode not in self._mode_term_names:
                self._mode_term_names[term_cfg.mode] = list()
                self._mode_term_cfgs[term_cfg.mode] = list()
                self._mode_class_term_cfgs[term_cfg.mode] = list()
            
            self._mode_term_names[term_cfg.mode].append(term_name)
            self._mode_term_cfgs[term_cfg.mode].append(term_cfg)

            if inspect.isclass(term_cfg.func):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)

            # Init buffers for interval/reset modes
            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(f"Event term '{term_name}' mode 'interval' requires 'interval_range_s'.")
                if term_cfg.is_global_time:
                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(1) * (upper - lower) + lower
                    self._interval_term_time_left.append(time_left)
                else:
                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(self.num_envs, device=self.device) * (upper - lower) + lower
                    self._interval_term_time_left.append(time_left)
            elif term_cfg.mode == "reset":
                if term_cfg.min_step_count_between_reset < 0:
                    raise ValueError(f"Negative min_step_count for term '{term_name}'.")
                step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
                self._reset_term_last_triggered_step_id.append(step_count)
                no_trigger = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                self._reset_term_last_triggered_once.append(no_trigger)

    # ----------------------------------------------------------------------
    # Internal Logic
    # ----------------------------------------------------------------------

    def _advance_curriculum(self, current_mu: float, confidence: float):
        """Increase difficulty level using trust-region step."""
        old_level = self.difficulty_level
        
        if self.difficulty_level <= 0.0:
            self.difficulty_level = max(1e-6, self.cfg.initial_level)
        else:
            # Trust-region update: level * (1 + rho)
            self.difficulty_level = min(1.0, self.difficulty_level * (1.0 + self.rho))

        rprint(
            f"[DomainManager] 🆙 UPGRADE! "
            f"Lvl: {old_level:.4f} -> {self.difficulty_level:.4f} | "
            f"Mean: {current_mu:.4f} (Target: {self.target_score:.4f}) | "
            f"Conf: {confidence*100:.1f}%"
        )

        self._apply_curriculum()
        self._gate_buf.clear()

    def _bayes_confidence_mu_gt_delta(self, target: float) -> tuple[float, float]:
        """Calculates P(mu > target) using Student's t-distribution."""
        data = list(self._gate_buf)
        n = len(data)
        mu = sum(data) / n
        
        # Calculate sample variance (unbiased)
        var = sum((x - mu) ** 2 for x in data) / (n - 1)
        std_dev = math.sqrt(max(var, 1e-12))
        std_err = std_dev / math.sqrt(n)

        # t-statistic
        t_score = (mu - target) / max(std_err, 1e-12)
        
        # CDF gives P(T < t_score).
        prob = stats.t.cdf(t_score, df=n - 1)
        
        return float(prob), float(mu)

    def _apply_curriculum(self):
        """Interpolates randomization parameters based on current difficulty level."""
        level = self.difficulty_level
        
        for term_name, max_p in self.max_params.items():
            if not hasattr(self.cfg, term_name): 
                continue
            
            current_term_cfg = getattr(self.cfg, term_name)
            # Skip if it's not an event config (double check)
            if not isinstance(current_term_cfg, EventTermCfg):
                continue
                
            operation = max_p.get("operation", "add")
            base_val = 1.0 if operation == "scale" else 0.0
            
            # Interpolate parameters
            new_params = {}
            for param_key, max_val in max_p.items():
                if param_key == "operation": 
                    continue
                
                if isinstance(max_val, (tuple, list)) and len(max_val) == 2 and isinstance(max_val[0], (int, float)):
                    min_v, max_v = max_val
                    cur_min = self._lerp(base_val, min_v, level)
                    cur_max = self._lerp(base_val, max_v, level)
                    new_params[param_key] = (cur_min, cur_max)
                    
                elif isinstance(max_val, dict):
                    new_axis_dict = {}
                    for axis, axis_range in max_val.items():
                        if isinstance(axis_range, (tuple, list)) and len(axis_range) == 2:
                            a_min, a_max = axis_range
                            axis_base = 0.0 
                            cur_a_min = self._lerp(axis_base, a_min, level)
                            cur_a_max = self._lerp(axis_base, a_max, level)
                            new_axis_dict[axis] = (cur_a_min, cur_a_max)
                    new_params[param_key] = new_axis_dict
                
                elif isinstance(max_val, (int, float)):
                    new_params[param_key] = self._lerp(base_val, max_val, level)

            # Update the term configuration
            current_term_cfg.params.update(new_params)

        self._update_internal_term_params()

    def _update_internal_term_params(self):
        """Syncs the python config changes to the active event manager terms."""
        if "reset" not in self._mode_term_cfgs:
            return

        for i, term_cfg in enumerate(self._mode_term_cfgs["reset"]):
            if i < len(self._mode_term_names["reset"]):
                term_name = self._mode_term_names["reset"][i]
                if hasattr(self.cfg, term_name):
                    new_params = getattr(self.cfg, term_name).params
                    term_cfg.params.update(new_params)

    def _capture_max_params(self):
        target_terms = [
            "randomize_mass", 
            "randomize_com", 
            "randomize_gains", 
            "randomize_joints", 
            "randomize_friction", 
            "randomize_base_mass"
        ]
        
        for term_name in target_terms:
            if hasattr(self.cfg, term_name):
                term_cfg = getattr(self.cfg, term_name)
                # Check instance to be safe
                if isinstance(term_cfg, EventTermCfg) and hasattr(term_cfg, "params") and term_cfg.params:
                    self.max_params[term_name] = deepcopy(term_cfg.params)

    def _count_uncertainty_dims(self, max_params: dict) -> int:
        d = 0
        for term in max_params.values():
            if not isinstance(term, dict):
                continue
            for k, v in term.items():
                if k == "operation": continue
                
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    d += 1
                elif isinstance(v, dict):
                    d += len(v)
                elif isinstance(v, (int, float)):
                    d += 1
        return max(d, 1)

    @staticmethod
    def _lerp(start: float, end: float, t: float) -> float:
        return start + (end - start) * t
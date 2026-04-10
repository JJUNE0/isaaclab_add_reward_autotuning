# domain_manager_term_cfg.py

from __future__ import annotations
from dataclasses import field
from isaaclab.utils import configclass
from isaaclab.managers.event_manager import EventTermCfg


@configclass
class DomainManagerTermCfg(EventManagerTermCfg):
    """
    Domain Manager Configuration for Curriculum-based Domain Randomization.
    """
    # ---------------------------------------------------------------------
    # Curriculum Settings (Bayesian Gate)
    # ---------------------------------------------------------------------
    init_difficulty_level: float = 0.0
    initial_level: float = 0.05  # Step size for the first increment

    # Target Reward Range for Curriculum
    target_threshold_start: float = 0.5
    target_threshold_end: float = 0.85  

    # Trust-Region / Confidence Parameters
    tau: float = 0.8        # Not directly used in current code but reserved for threshold scaling
    kappa: float = 0.1      # Determines the step size (rho) based on dimensions
    confidence_level: float = 0.95
    
    # Expert Score Weights (Theory: S_E)
    w_fm: float = 0.8
    w_gail: float = 0.2

    # Gate Statistics
    min_gate_samples: int = 20
    gate_window: int = 100
#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .distillation import Distillation
from .moo_ppo import MOOPPO

from .demo.srmppo import SRMPPO
from .demo.sac import SAC
from .demo.tqc import TQC
from .demo.taco import TACO
from .demo.acapsppo import ACAPSPPO
from .demo.sieppo import SIEPPO



__all__ = ["PPO", "SRMPPO", "SAC", "TQC", "TACO", "ACAPSPPO" , "SIEPPO", "Distillation", "MOOPPO"]

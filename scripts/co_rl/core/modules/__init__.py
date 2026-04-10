#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .discriminator import DiscriminatorMLP, DiscriminatorGRU
from .teacher_student import RMATeacher, RMAStudent


from .demo.replay_memory import ReplayMemory, TACOReplayMemory
from .demo.student_teacher import StudentTeacher
from .demo.student_teacher_recurrent import StudentTeacherRecurrent



__all__ = ["ActorCritic", "ActorCriticRecurrent", "StudentTeacher", "StudentTeacherRecurrent"]

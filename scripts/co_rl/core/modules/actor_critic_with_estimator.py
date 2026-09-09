#  SPDX-License-Identifier: BSD-3-Clause

"""``ActorCritic`` subclass that prepends a learned base-velocity estimator
MLP to the actor's input, trained jointly with PPO via an auxiliary
supervised loss (see ``MOOPPO._update_policy`` in
``scripts/co_rl/core/algorithms/moo_ppo.py``).

The actor's own obs (``num_actor_obs``, e.g. ``stack_policy`` +
``none_stack_policy``) must be pure proprioception -- no ground-truth
``base_lin_vel`` term. The estimator predicts it from that same proprio, and
its (un-detached) output is concatenated with the raw proprio before being
fed to the actor -- so the actor never sees ground truth directly, only the
estimate, making the trained policy deployable without simulator-only
privileged state. The supervised target comes from the env's ``priv_prio``
observation group (ground-truth ``base_lin_vel_x/y/z``, already
automatically appended to ``critic_obs`` by ``CoRlVecEnvWrapper`` for any
``priv_*`` group -- see its ``_get_concatenated_privileged_obs``), sliced out
of ``critic_obs`` by ``MOOPPO`` at update time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import ActorCritic, get_activation


class ActorCriticWithEstimator(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        estimator_hidden_dims=[128, 64],
        estimator_output_dim=3,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        # Base class builds self.actor for an (estimator_output + proprio)-wide
        # input -- num_actor_obs here is the raw proprio width, so pad it.
        super().__init__(
            num_actor_obs=num_actor_obs + estimator_output_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        self.num_proprio_obs = num_actor_obs
        self.estimator_output_dim = estimator_output_dim

        est_activation = get_activation(activation)
        estimator_layers = []
        prev_dim = num_actor_obs
        for hidden_dim in estimator_hidden_dims:
            estimator_layers.append(nn.Linear(prev_dim, hidden_dim))
            estimator_layers.append(est_activation)
            prev_dim = hidden_dim
        estimator_layers.append(nn.Linear(prev_dim, estimator_output_dim))
        self.estimator = nn.Sequential(*estimator_layers)
        print(f"Estimator MLP: {self.estimator}")

        # Populated by every forward pass through the estimator (act/act_inference);
        # MOOPPO's estimator auxiliary loss reads this after re-running act() on a batch.
        self._last_estimator_output: torch.Tensor | None = None

    def _augmented_actor_input(self, proprio: torch.Tensor) -> torch.Tensor:
        est_out = self.estimator(proprio)
        # Not detached -- both the PPO loss (via the actor's dependency on it)
        # and the explicit MSE loss in MOOPPO shape the estimator.
        self._last_estimator_output = est_out
        return torch.cat([est_out, proprio], dim=-1)

    def update_distribution(self, observations):
        policy_input = self._augmented_actor_input(observations)
        mean = self.actor(policy_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        policy_input = self._augmented_actor_input(observations)
        return self.actor(policy_input)

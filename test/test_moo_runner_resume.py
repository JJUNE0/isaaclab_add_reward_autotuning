from types import SimpleNamespace

import torch

from scripts.co_rl.core.runners.moo_on_policy_runner import MOO_OnPolicyRunner


def _make_runner():
    runner = MOO_OnPolicyRunner.__new__(MOO_OnPolicyRunner)
    runner.device = "cpu"
    runner.empirical_normalization = False
    actor = torch.nn.Linear(3, 2)
    discriminator = torch.nn.Linear(2, 1)
    runner.alg = SimpleNamespace(
        actor_critic=actor,
        ppo_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        disc_optimizer=torch.optim.Adam(discriminator.parameters(), lr=2.0e-3),
    )
    runner.current_learning_iteration = 0
    return runner


def _take_optimizer_step(module, optimizer):
    optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in module.parameters()).backward()
    optimizer.step()


def test_load_restores_optimizers_and_resumes_at_next_iteration(tmp_path):
    source = _make_runner()
    _take_optimizer_step(source.alg.actor_critic, source.alg.ppo_optimizer)
    disc_parameter = source.alg.disc_optimizer.param_groups[0]["params"][0]
    source.alg.disc_optimizer.zero_grad()
    disc_parameter.square().sum().backward()
    source.alg.disc_optimizer.step()
    source.current_learning_iteration = 999

    checkpoint = tmp_path / "model_999.pt"
    source.save(checkpoint)

    target = _make_runner()
    target.load(checkpoint)

    assert target.current_learning_iteration == 1000
    assert target.alg.ppo_optimizer.state_dict()["state"]
    assert target.alg.disc_optimizer.state_dict()["state"]

    source_ppo_state = source.alg.ppo_optimizer.state_dict()["state"]
    target_ppo_state = target.alg.ppo_optimizer.state_dict()["state"]
    for source_state, target_state in zip(source_ppo_state.values(), target_ppo_state.values(), strict=True):
        torch.testing.assert_close(source_state["step"], target_state["step"])
        torch.testing.assert_close(source_state["exp_avg"], target_state["exp_avg"])
        torch.testing.assert_close(source_state["exp_avg_sq"], target_state["exp_avg_sq"])

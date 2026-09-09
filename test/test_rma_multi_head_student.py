import torch
import torch.nn as nn

from scripts.co_rl.core.modules.rma_multi_head_student import RMAStudentMultiHead


class _CaptureActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(84, 8, bias=False)
        self.last_input = None

    def forward(self, actor_input):
        self.last_input = actor_input
        return self.projection(actor_input)


class _TeacherEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(25, 16))

    def forward(self, privileged):
        return self.net(privileged)


class _FakeTeacher:
    def __init__(self):
        self.actor = _CaptureActor()
        self.encoder = _TeacherEncoder()
        self.num_actions = 8
        self.std = nn.Parameter(torch.ones(8))


class _FixedHeads(nn.Module):
    def __init__(self, z, velocity, height):
        super().__init__()
        self.register_buffer("z", z)
        self.register_buffer("velocity", velocity)
        self.register_buffer("height", height)

    def forward(self, history):
        batch = history.shape[0]
        return (
            self.z.expand(batch, -1),
            self.velocity.expand(batch, -1, -1),
            self.height.expand(batch, -1, -1),
        )


def _make_student(num_envs=2):
    return RMAStudentMultiHead(
        teacher_policy=_FakeTeacher(),
        num_obs=68,
        num_envs=num_envs,
        num_single_obs=32,
        num_stacks=2,
        latent_dim=16,
        history_len=50,
        observable_single_obs_dim=28,
        state_dim=4,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        activation="elu",
        init_noise_std=1.0,
    )


def test_history_strips_all_teacher_gt_terms():
    student = _make_student()
    full_obs = torch.randn(2, 68)
    full_obs[:, 28:32] = 1000.0
    full_obs[:, 60:64] = -1000.0

    student.update_history(full_obs, dones=torch.ones(2))

    assert student.current_obs.shape == (2, 60)
    assert student.history_buffer.shape == (2, 50, 32)
    assert not torch.any(torch.abs(student.current_obs) == 1000.0)
    assert not torch.any(torch.abs(student.history_buffer) == 1000.0)


def test_forward_reconstructs_exact_808_actor_layout_from_heads():
    student = _make_student(num_envs=1)
    full_obs = torch.arange(68, dtype=torch.float32).unsqueeze(0)
    student.update_history(full_obs, dones=torch.ones(1))

    z = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    velocity = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    height = torch.tensor([[[0.31], [0.32]]])
    student.adaptation_module = _FixedHeads(z, velocity, height)

    action, z_hat = student.forward()
    actor_input = student.actor.last_input

    assert action.shape == (1, 8)
    assert actor_input.shape == (1, 84)
    torch.testing.assert_close(z_hat, z)
    torch.testing.assert_close(actor_input[:, 28:31], velocity[:, 0])
    torch.testing.assert_close(actor_input[:, 31:32], height[:, 0])
    torch.testing.assert_close(actor_input[:, 60:63], velocity[:, 1])
    torch.testing.assert_close(actor_input[:, 63:64], height[:, 1])
    torch.testing.assert_close(actor_input[:, 64:68], full_obs[:, 64:68])
    torch.testing.assert_close(actor_input[:, 68:], z)


def test_distillation_loss_backpropagates_to_all_heads():
    student = _make_student(num_envs=2)
    history = torch.randn(2, 50 * 32)
    critic_obs = torch.randn(2, 68 + 25)

    loss, infos = student.compute_distillation_loss(history, critic_obs)
    loss.backward()

    assert torch.isfinite(loss)
    assert {"z_loss", "velocity_loss", "height_loss", "action_loss"} <= infos.keys()
    for head in (
        student.adaptation_module.latent_head,
        student.adaptation_module.velocity_head,
        student.adaptation_module.height_head,
    ):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()

from types import SimpleNamespace

import torch
import pytest

from fastvideo.train.methods.rl.dmdr import (
    _DMDRTimestepSamplingConfig,
    DMDRMethod,
)


def test_weighted_x0_mse_returns_per_sample_loss():
    prediction = torch.tensor([[[[[2.0]]]], [[[[4.0]]]]])
    target = torch.tensor([[[[[1.0]]]], [[[[1.0]]]]])

    loss = DMDRMethod._weighted_x0_mse(prediction, target)

    assert loss.shape == (2, )
    assert torch.allclose(loss, torch.tensor([1.0, 3.0], dtype=loss.dtype))


def test_optimizer_dict_omits_critic_during_parent_init():
    method = object.__new__(DMDRMethod)
    method._student_optimizer = "student"
    method._student_lr_scheduler = "student_sched"

    assert method._optimizer_dict == {"student": "student"}
    assert method._lr_scheduler_dict == {"student": "student_sched"}


def test_dmd_cfg_uncond_defaults_to_zero_text():
    method = object.__new__(DMDRMethod)
    method.method_config = {}

    assert method._parse_dmd_cfg_uncond() == {
        "text": "zero",
        "on_missing": "ignore",
    }


def test_dmd_cfg_uncond_rejects_drop_text():
    method = object.__new__(DMDRMethod)
    method.method_config = {"cfg_uncond": {"text": "drop"}}

    with pytest.raises(ValueError, match="cfg_uncond.text"):
        method._parse_dmd_cfg_uncond()


def test_dmdr_requires_teacher_and_critic_after_parent_init(monkeypatch):

    def fake_init(self, cfg, role_models):
        self.method_config = {}
        self.training_config = cfg.training
        self.student = role_models["student"]
        self._max_grad_norm = 1.0

    monkeypatch.setattr(
        "fastvideo.train.methods.rl.dmdr.DiffusionNFTMethod.__init__",
        fake_init,
    )

    cfg = SimpleNamespace(training=SimpleNamespace())
    student = SimpleNamespace(_trainable=True)

    with pytest.raises(ValueError, match="requires role 'teacher'"):
        DMDRMethod(cfg=cfg, role_models={"student": student})


def test_fake_score_loss_uses_sampled_timestep_context():

    class FakeStudent:
        num_train_timesteps = 1000

        def shift_and_clamp_timestep(self, timestep):
            return timestep

        def add_noise(self, clean_latents, noise, timestep):
            del timestep
            return clean_latents + noise

    class FakeCritic:

        def __init__(self):
            self.seen_timestep = None
            self.seen_batch_timestep = None

        def predict_noise(self, noisy_x0, timestep, batch, *, conditional, attn_kind):
            del conditional, attn_kind
            self.seen_timestep = timestep
            self.seen_batch_timestep = batch.timesteps
            return torch.zeros_like(noisy_x0)

    method = object.__new__(DMDRMethod)
    method.student = FakeStudent()
    method.critic = FakeCritic()
    method.cuda_generator = torch.Generator().manual_seed(0)
    method._dynamic_step = 0
    method._fake_score_timestep_sampling = _DMDRTimestepSamplingConfig(kind="uniform")
    original_timesteps = torch.tensor([123])
    batch = SimpleNamespace(timesteps=original_timesteps, attn_metadata="attn")

    _, ctx = method._fake_score_flow_matching_loss(
        torch.zeros(2, 1, 1, 1, 1),
        batch,
        iteration=1,
    )

    assert batch.timesteps is original_timesteps
    assert method.critic.seen_batch_timestep is method.critic.seen_timestep
    assert ctx[0] is method.critic.seen_timestep
    assert ctx[1] == "attn"


def test_dmdr_cosine_guidance_scale_decays_to_zero():
    method = object.__new__(DMDRMethod)
    method._dynamic_step = 10
    method._real_score_guidance_scale = 2.0

    assert method._real_guidance_scale(0) == 2.0
    assert method._real_guidance_scale(10) == 0.0
    assert 0.0 < method._real_guidance_scale(5) < 2.0


def test_dmdr_annealed_beta_params_match_reference_schedule():
    method = object.__new__(DMDRMethod)
    method._dynamic_step = 10
    cfg = _DMDRTimestepSamplingConfig(kind="logit_normal", alpha=4.0, beta=1.5)

    assert method._annealed_beta_params(cfg, 0) == (4.0, 1.5)
    assert method._annealed_beta_params(cfg, 10) == (1.0, 1.0)


def test_dmdr_reward_activation_honors_cold_start_steps():
    method = object.__new__(DMDRMethod)
    method._cold_start_steps = 5

    assert method._reward_is_active(5) is False
    assert method._reward_is_active(6) is True

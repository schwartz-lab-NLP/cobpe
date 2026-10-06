import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig


class _CaptureBlock(nn.Module):
    def __init__(self, delta=None):
        super().__init__()
        self.delta = delta
        self.inputs = []

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        self.inputs.append(x.detach().clone())
        if self.delta is not None:
            x = x + self.delta.to(device=x.device, dtype=x.dtype)
        return x


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


def _build_model(*, n_layer=2):
    model = GPT(
        GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=n_layer,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            architecture_preset="speedrun_cobpe",
            window_pattern="L",
            modifier_group_sizes=(3, 4),
        )
    )
    model.init_weights()
    return model


def _modifier_ids():
    return torch.tensor([[[1, 2], [2, 3], [0, 1]]], dtype=torch.long)


def test_speedrun_cobpe_smears_base_before_adding_modifiers():
    model = _build_model()
    capture = _CaptureBlock()
    model.transformer.h = nn.ModuleList([capture, _CaptureBlock()])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
        model.smear_lambda.fill_(1.0)
        model.smear_gate.weight.zero_()

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    modifiers = _modifier_ids()
    model(ids, modifier_ids=modifiers, return_hidden_only=True)

    base = _norm(model.transformer.wte(ids).to(model.transformer.wte.weight.dtype))
    smeared_base = torch.cat([base[:, :1], base[:, 1:] + 0.5 * base[:, :-1]], dim=1)
    expected = _norm(smeared_base + model.cobpe.embed_sum(modifiers).to(smeared_base.dtype))
    assert torch.allclose(capture.inputs[0], expected, atol=1e-5, rtol=1e-5)


def test_speedrun_cobpe_backout_applies_to_base_logits_not_modifier_hidden():
    model = _build_model(n_layer=3)
    delta = torch.linspace(-0.5, 0.5, model.config.n_embd).view(1, 1, -1)
    model.transformer.h = nn.ModuleList([_CaptureBlock(), _CaptureBlock(), _CaptureBlock(delta)])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
        model.backout_lambda.fill_(0.5)

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    modifiers = _modifier_ids()
    logits, modifier_hidden = model(ids, modifier_ids=modifiers, return_hidden=True)

    trunk_input = _norm(
        _norm(model.transformer.wte(ids))
        + model.cobpe.embed_sum(modifiers).to(model.transformer.wte.weight.dtype)
    )
    expected_modifier_hidden = _norm(trunk_input + delta)
    expected_base_hidden = _norm(trunk_input + delta - 0.5 * trunk_input)
    expected_logits = model.lm_head(expected_base_hidden)[..., :model.config.vocab_size].float()
    expected_logits = 15 * torch.tanh(expected_logits / 15)

    assert torch.allclose(modifier_hidden, expected_modifier_hidden, atol=1e-5, rtol=1e-5)
    assert torch.allclose(logits, expected_logits, atol=1e-5, rtol=1e-5)

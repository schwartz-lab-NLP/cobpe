import pytest
import torch
import torch.nn as nn

from cobpe.modeling.modifiers import CoBPEModule


def _module(*, gate_mode="per_group", conditioning_mode="concat_gated_refine"):
    module = CoBPEModule(
        group_sizes=(2, 3),
        n_embd=4,
        padded_vocab_size=7,
        linear_cls=nn.Linear,
        conditioning_mode=conditioning_mode,
        gate_mode=gate_mode,
    )
    module.init_weights(compute_dtype=torch.float32)
    return module


def test_per_group_gates_have_distinct_outputs_and_receive_gradients():
    torch.manual_seed(17)
    module = _module()
    with torch.no_grad():
        module.head.weight.zero_()
        module.base_proj.weight.zero_()
        module.base_proj.weight[:5].fill_(0.25)
        module.gate.weight.copy_(torch.tensor([[0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]]))

    hidden = torch.tensor([[[1.0, 0.2, -0.1, 0.3]]], requires_grad=True)
    token_ids = torch.tensor([[2]])
    base_unembedding = torch.arange(28, dtype=torch.float32).view(7, 4) / 10
    outputs = module.logits(
        hidden,
        token_ids,
        token_embedding_weight=base_unembedding,
        base_unembedding=base_unembedding,
    )

    assert len(outputs) == 2
    assert outputs[0].shape == (1, 1, 2)
    assert outputs[1].shape == (1, 1, 3)
    assert not torch.allclose(outputs[0].mean(), outputs[1].mean())
    sum(output.sum() for output in outputs).backward()
    assert module.gate.weight.grad is not None
    assert torch.all(module.gate.weight.grad.abs().sum(dim=1) > 0)


def test_scalar_gate_remains_available_as_single_shared_gate():
    module = _module(gate_mode="scalar", conditioning_mode="concat_gated")
    assert module.gate.weight.shape == (1, 4)
    assert module.logit_refine is None


@pytest.mark.parametrize("mode", ["mlp", "film", "add_residual", "add_residual_gated"])
def test_removed_modifier_heads_fail_clearly(mode):
    with pytest.raises(ValueError, match="Unsupported modifier_conditioning_mode"):
        _module(conditioning_mode=mode)

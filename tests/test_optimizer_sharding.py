import torch

from nanochat import optim as optim_mod
from nanochat.optim import DistMuonAdamW


class _FakeWork:
    def __init__(self, future=None):
        self.future = object() if future is None else future

    def get_future(self):
        return self.future


def test_dist_adamw_handles_nonshardable_and_unused_parameters(monkeypatch):
    calls = []

    def fake_all_reduce(grad, op, async_op):
        calls.append(("all_reduce", tuple(grad.shape)))
        return _FakeWork()

    def fake_reduce_scatter_tensor(output, input, op, async_op):
        calls.append(("reduce_scatter", tuple(output.shape), tuple(input.shape)))
        return _FakeWork()

    monkeypatch.setattr(optim_mod.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(optim_mod.dist, "reduce_scatter_tensor", fake_reduce_scatter_tensor)

    nonshardable = torch.nn.Parameter(torch.zeros(61, 32))
    shardable = torch.nn.Parameter(torch.zeros(64, 32))
    unused = torch.nn.Parameter(torch.zeros(64, 32))
    for param in (nonshardable, shardable):
        param.grad = torch.ones_like(param)

    optimizer = DistMuonAdamW([
        dict(
            kind="adamw",
            params=[nonshardable, shardable, unused],
            lr=0.1,
            betas=(0.8, 0.96),
            eps=1e-10,
            weight_decay=0.01,
        )
    ])
    info = optimizer._reduce_adamw(optimizer.param_groups[0], world_size=4)

    assert info["param_infos"][nonshardable]["is_small"]
    assert not info["param_infos"][shardable]["is_small"]
    assert unused not in info["param_infos"]
    assert ("all_reduce", (61, 32)) in calls
    assert ("reduce_scatter", (16, 32), (64, 32)) in calls


def test_dist_muon_reduces_missing_grad_as_zero_and_tracks_global_activity(monkeypatch):
    calls = []
    activity_future = object()

    def fake_all_reduce(value, op, async_op):
        calls.append(("activity", value.clone()))
        return _FakeWork(activity_future)

    def fake_reduce_scatter_tensor(output, input, op, async_op):
        calls.append(("grads", input.clone()))
        return _FakeWork()

    monkeypatch.setattr(optim_mod.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(optim_mod.dist, "reduce_scatter_tensor", fake_reduce_scatter_tensor)

    active = torch.nn.Parameter(torch.zeros(2, 2))
    inactive = torch.nn.Parameter(torch.full((2, 2), 7.0))
    active.grad = torch.ones_like(active)
    optimizer = DistMuonAdamW([
        dict(
            kind="muon",
            params=[active, inactive],
            lr=0.1,
            momentum=0.95,
            ns_steps=5,
            beta2=0.9,
            weight_decay=0.01,
        )
    ])

    info = optimizer._reduce_muon(optimizer.param_groups[0], world_size=2)

    assert info["activity_future"] is activity_future
    assert torch.equal(calls[0][1], torch.tensor([1, 0], dtype=torch.int32))
    assert torch.equal(calls[1][1][0], torch.ones(2, 2))
    assert torch.equal(calls[1][1][1], torch.zeros(2, 2))

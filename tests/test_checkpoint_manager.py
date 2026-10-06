import json
from pathlib import Path

import pytest
import torch

from nanochat.checkpoint_manager import (
    ensure_sharded_optimizer_world_size_compatibility,
    find_last_complete_step,
    get_exact_resume_compatibility_mismatches,
    patch_missing_model_config_keys,
    patch_missing_model_keys,
    validate_modifier_checkpoint_compatibility,
    resolve_optimizer_shard_rank,
    validate_checkpoint_tokenizer_fingerprint,
    validate_checkpoint_tokenizer_compatibility,
)
from nanochat.gpt import GPT, GPTConfig, materialize_gpt_config_kwargs
from nanochat.tokenizer import get_tokenizer_fingerprint


def _tiny_config(**overrides):
    return GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        **overrides,
    )


def _tiny_model(**overrides):
    model = GPT(_tiny_config(**overrides))
    model.init_weights()
    return model


def test_gpt_config_materializes_speedrun_defaults():
    config = _tiny_config()

    assert config.architecture_preset == "speedrun"
    assert config.use_smear is True
    assert config.use_backout is True
    assert config.use_residual_scalars is True
    assert config.use_value_embeds is True
    assert config.qk_norm_scale == 1.2
    assert config.c_fc_init_scale == 0.4
    assert config.residual_scalar_init_mode == "speedrun"
    assert config.window_pattern == "SSSL"
    assert config.short_window_divisor == 4


@pytest.mark.parametrize(
    ("preset", "qk_norm_scale", "c_fc_init_scale", "residual_scalar_init_mode", "short_window_divisor"),
    [
        ("speedrun_cobpe", 1.2, 0.4, "speedrun", 4),
    ],
)
def test_gpt_config_materializes_cobpe_speedrun_defaults(
    preset, qk_norm_scale, c_fc_init_scale, residual_scalar_init_mode, short_window_divisor
):
    config = _tiny_config(architecture_preset=preset)

    expected_smear_backout = preset == "speedrun_cobpe"
    assert config.use_smear is expected_smear_backout
    assert config.use_backout is expected_smear_backout
    assert config.use_residual_scalars is True
    assert config.use_value_embeds is True
    assert config.qk_norm_scale == qk_norm_scale
    assert config.c_fc_init_scale == c_fc_init_scale
    assert config.residual_scalar_init_mode == residual_scalar_init_mode
    assert config.window_pattern == "SSSL"
    assert config.short_window_divisor == short_window_divisor
    assert config.cobpe_smear_backout_scope == ("base" if expected_smear_backout else "full")


def test_gpt_config_materializes_vanilla_defaults_and_preserves_overrides():
    config = _tiny_config(architecture_preset="vanilla", window_pattern="SL")

    assert config.architecture_preset == "vanilla"
    assert config.use_smear is False
    assert config.use_backout is False
    assert config.use_residual_scalars is False
    assert config.use_value_embeds is False
    assert config.qk_norm_scale == 1.0
    assert config.c_fc_init_scale == 1.0
    assert config.residual_scalar_init_mode == "neutral"
    assert config.window_pattern == "SL"
    assert config.short_window_divisor == 4


@pytest.mark.parametrize("conditioning_mode", ["mlp", "film", "add_residual"])
def test_removed_modifier_architectures_fail_during_config_validation(conditioning_mode):
    with pytest.raises(ValueError, match="Unsupported modifier_conditioning_mode"):
        _tiny_config(modifier_conditioning_mode=conditioning_mode)


def test_base_conditioned_modifier_gates_are_rejected_in_config():
    with pytest.raises(ValueError, match="Only hidden-state-conditioned gates"):
        _tiny_config(modifier_gate_conditioning="hidden_and_base")


def test_removed_modifier_refine_dim_is_only_accepted_at_legacy_default():
    config_kwargs = {"modifier_refine_dim": 0}
    patch_missing_model_config_keys(config_kwargs)
    assert "modifier_refine_dim" not in config_kwargs
    with pytest.raises(ValueError, match="Unsupported modifier_refine_dim"):
        materialize_gpt_config_kwargs({"modifier_refine_dim": 64})


def test_patch_missing_model_config_keys_materializes_architecture_fields():
    model_config_kwargs = {"n_layer": 4}

    patch_missing_model_config_keys(model_config_kwargs)

    assert model_config_kwargs["architecture_preset"] == "speedrun"
    assert model_config_kwargs["use_smear"] is True
    assert model_config_kwargs["use_backout"] is True
    assert model_config_kwargs["use_residual_scalars"] is True
    assert model_config_kwargs["use_value_embeds"] is True
    assert model_config_kwargs["qk_norm_scale"] == 1.2
    assert model_config_kwargs["c_fc_init_scale"] == 0.4
    assert model_config_kwargs["residual_scalar_init_mode"] == "speedrun"
    assert model_config_kwargs["window_pattern"] == "L"
    assert model_config_kwargs["short_window_divisor"] == 4


def test_legacy_gated_cobpe_config_requires_explicit_gate_architecture():
    with pytest.raises(ValueError, match="modifier_gate_mode is missing"):
        patch_missing_model_config_keys(
            {"modifier_group_sizes": [2, 8], "modifier_conditioning_mode": "concat_gated_refine"}
        )

    # Gate mode does not affect the bias head.
    bias_config = {"modifier_group_sizes": [2, 8], "modifier_conditioning_mode": "base_bias"}
    patch_missing_model_config_keys(bias_config)
    assert bias_config["modifier_gate_mode"] == "per_group"


def test_patch_missing_model_keys_adds_speedrun_defaults():
    model_data = {}
    config = _tiny_config()

    patched_keys = patch_missing_model_keys(model_data, config)

    assert patched_keys == [
        "resid_lambdas",
        "x0_lambdas",
        "smear_gate.weight",
        "smear_lambda",
        "backout_lambda",
    ]
    assert torch.equal(model_data["resid_lambdas"], torch.ones(4))
    assert torch.equal(model_data["x0_lambdas"], torch.zeros(4))
    assert model_data["smear_gate.weight"].shape == (1, 24)
    assert torch.equal(model_data["smear_lambda"], torch.zeros(1))
    assert torch.equal(model_data["backout_lambda"], 0.2 * torch.ones(1))


def test_patch_missing_model_keys_skips_vanilla_only_params():
    model_data = {}
    config = _tiny_config(architecture_preset="vanilla")

    patched_keys = patch_missing_model_keys(model_data, config)

    assert patched_keys == []
    assert model_data == {}


def test_patch_missing_model_keys_honors_prefix():
    model_data = {}
    config = _tiny_config()

    patched_keys = patch_missing_model_keys(model_data, config, prefix="backbone.")

    assert patched_keys == [
        "backbone.resid_lambdas",
        "backbone.x0_lambdas",
        "backbone.smear_gate.weight",
        "backbone.smear_lambda",
        "backbone.backout_lambda",
    ]


def test_vanilla_model_disables_speedrun_parameters_and_optimizer_groups():
    speedrun_model = _tiny_model()
    vanilla_model = _tiny_model(architecture_preset="vanilla")

    assert speedrun_model.smear_gate is not None
    assert speedrun_model.backout_lambda is not None
    assert speedrun_model.resid_lambdas is not None
    assert len(speedrun_model.value_embeds) > 0

    assert vanilla_model.smear_gate is None
    assert vanilla_model.backout_lambda is None
    assert vanilla_model.resid_lambdas is None
    assert vanilla_model.x0_lambdas is None
    assert len(vanilla_model.value_embeds) == 0

    speedrun_optimizer = speedrun_model.setup_optimizer()
    vanilla_optimizer = vanilla_model.setup_optimizer()

    speedrun_adamw_groups = [group for group in speedrun_optimizer.param_groups if group["kind"] == "adamw"]
    vanilla_adamw_groups = [group for group in vanilla_optimizer.param_groups if group["kind"] == "adamw"]

    assert len(speedrun_adamw_groups) == 6
    assert len(vanilla_adamw_groups) == 2


@pytest.mark.parametrize(
    "conditioning_mode",
    ["base_bias", "concat_gated", "concat_gated_refine"],
)
def test_modifier_parameters_use_unembedding_optimizer_bucket(conditioning_mode):
    model = _tiny_model(
        architecture_preset="vanilla",
        modifier_group_sizes=(2, 3),
        modifier_conditioning_mode=conditioning_mode,
    )
    optimizer = model.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.2)

    group_by_label = {group["label"]: group for group in optimizer.param_groups}
    lm_head_group = group_by_label["lm_head"]
    embedding_group = group_by_label["embeddings"]
    lm_head_param_ids = {id(p) for p in lm_head_group["params"]}
    embedding_param_ids = {id(p) for p in embedding_group["params"]}
    modifier_params = {
        name: param
        for name, param in model.named_parameters()
        if name.startswith("cobpe.")
    }

    assert modifier_params
    assert "cobpe.embed.weight" in modifier_params
    if conditioning_mode == "base_bias":
        assert "cobpe.base_bias.weight" in modifier_params

    for name, param in modifier_params.items():
        assert id(param) in lm_head_param_ids, name
        assert id(param) not in embedding_param_ids, name

    dmodel_lr_scale = (model.config.n_embd / 768) ** -0.5
    assert lm_head_group["kind"] == "adamw"
    assert lm_head_group["lr"] == pytest.approx(0.004 * dmodel_lr_scale)
    assert lm_head_group["betas"] == (0.8, 0.96)
    assert lm_head_group["weight_decay"] == pytest.approx(0.01)


@pytest.mark.parametrize("conditioning_mode", ["base_bias", "concat_gated", "concat_gated_refine"])
def test_compositional_model_scaling_parameter_counts_start_up(conditioning_mode):
    model = _tiny_model(
        architecture_preset="vanilla",
        modifier_group_sizes=(2, 3),
        modifier_conditioning_mode=conditioning_mode,
    )

    counts = model.num_scaling_params()

    assert counts["total"] == sum(param.numel() for param in model.parameters())
    assert counts["modifier_embeds"] == model.cobpe.embed.weight.numel()
    assert counts["modifier_heads"] == model.cobpe.head.weight.numel()
    assert counts["modifier_base_biases"] == (
        0 if model.cobpe.base_bias is None else model.cobpe.base_bias.weight.numel()
    )


def test_concat_refine_shapes_initialization_and_forward():
    refined = _tiny_model(
        architecture_preset="vanilla",
        modifier_group_sizes=(2, 3),
        modifier_conditioning_mode="concat_gated_refine",
    )
    assert refined.cobpe.logit_refine.weight.shape == (64, 64)
    assert torch.equal(
        refined.cobpe.logit_refine.weight,
        torch.zeros_like(refined.cobpe.logit_refine.weight),
    )
    ids = torch.randint(0, refined.config.vocab_size, (2, 4))
    modifiers = torch.stack(
        [
            torch.randint(0, 2, (2, 4)),
            torch.randint(0, 3, (2, 4)),
        ],
        dim=-1,
    )
    loss = refined(
        ids,
        ids,
        modifier_ids=modifiers,
        target_modifier_ids=modifiers,
    )
    assert torch.isfinite(loss)


def test_modifier_heads_pad_large_dimensions_for_distributed_adamw():
    model = _tiny_model(
        architecture_preset="vanilla",
        modifier_group_sizes=(31, 30),
        modifier_conditioning_mode="concat_gated_refine",
    )
    assert model.cobpe.padded_total_size == 64
    assert model.cobpe.head.weight.shape[0] == 64
    assert model.cobpe.base_proj.weight.shape[0] == 64
    assert model.cobpe.gate.weight.shape[0] == 2
    assert model.cobpe.logit_refine.weight.shape == (64, 64)


def test_modifier_gate_defaults_to_one_row_per_group():
    config = _tiny_config(modifier_group_sizes=(2, 3))
    assert config.modifier_gate_mode == "per_group"
    assert _tiny_model(modifier_group_sizes=(2, 3)).cobpe.gate.weight.shape == (2, 16)


def test_modifier_checkpoint_gate_shape_must_match_config():
    config = _tiny_config(modifier_group_sizes=(2, 3), modifier_gate_mode="scalar")
    checkpoint = {"cobpe.gate.weight": torch.zeros(2, config.n_embd)}
    with pytest.raises(ValueError, match="gate weights are not reinterpreted"):
        validate_modifier_checkpoint_compatibility(checkpoint, config)


@pytest.mark.parametrize("source", ["sft", "rl"])
def test_checkpoint_convenience_loaders_accept_only_base_source(source):
    from nanochat.checkpoint_manager import load_model, load_optimizer_state

    with pytest.raises(ValueError, match="only 'base' is available"):
        load_model(source, device="cpu", phase="eval")
    with pytest.raises(ValueError, match="only 'base' is available"):
        load_optimizer_state(source, device="cpu", rank=0)


class _TokenizerShape:
    def __init__(self, group_sizes=()):
        self.group_sizes = tuple(group_sizes)

    def has_compositional_mode(self):
        return bool(self.group_sizes)

    def get_modifier_group_sizes(self):
        return list(self.group_sizes)


def test_checkpoint_rejects_plain_tokenizer_for_cobpe_model():
    with pytest.raises(ValueError, match="model requires a CoBPE tokenizer"):
        validate_checkpoint_tokenizer_compatibility(
            _TokenizerShape(), _tiny_config(modifier_group_sizes=(2, 3))
        )


def test_checkpoint_rejects_cobpe_tokenizer_for_plain_model():
    with pytest.raises(ValueError, match="model requires a plain BPE tokenizer"):
        validate_checkpoint_tokenizer_compatibility(
            _TokenizerShape((2, 3)), _tiny_config()
        )


def test_checkpoint_rejects_cobpe_tokenizer_with_different_group_sizes():
    with pytest.raises(ValueError, match="modifier group sizes differ"):
        validate_checkpoint_tokenizer_compatibility(
            _TokenizerShape((2, 4)), _tiny_config(modifier_group_sizes=(2, 3))
        )


def test_checkpoint_tokenizer_fingerprint_rejects_changed_artifact(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_path = tokenizer_dir / "tokenizer.pkl"
    tokenizer_path.write_bytes(b"tokenizer-v1")
    meta = {"tokenizer_fingerprint": get_tokenizer_fingerprint(str(tokenizer_dir))}
    validate_checkpoint_tokenizer_fingerprint(meta, str(tokenizer_dir))

    tokenizer_path.write_bytes(b"tokenizer-v2")
    with pytest.raises(ValueError, match="Tokenizer fingerprint mismatch"):
        validate_checkpoint_tokenizer_fingerprint(meta, str(tokenizer_dir))


def test_vanilla_forward_and_kv_cache_smoke():
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.engine import KVCache

    model = _tiny_model(architecture_preset="vanilla")
    inputs = torch.randint(0, model.config.vocab_size, (2, 4))

    logits = model(inputs)
    assert logits.shape == (2, 4, model.config.vocab_size)

    kv_cache = KVCache(
        batch_size=2,
        num_heads=model.config.n_kv_head,
        seq_len=model.config.sequence_len,
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=model.config.n_layer,
        device=model.get_device(),
        dtype=COMPUTE_DTYPE,
    )
    prefill_logits = model(inputs[:, :3], kv_cache=kv_cache)
    decode_logits = model(inputs[:, 3:], kv_cache=kv_cache)

    assert prefill_logits.shape == (2, 3, model.config.vocab_size)
    assert decode_logits.shape == (2, 1, model.config.vocab_size)
    assert kv_cache.get_pos() == 4


def test_gpt_config_custom_n_inner_changes_mlp_shapes():
    model = _tiny_model(n_inner=24)
    block0 = model.transformer.h[0]
    assert block0.mlp.c_fc.weight.shape == (24, 16)
    assert block0.mlp.c_proj.weight.shape == (16, 24)


def _create_checkpoint_files(
    checkpoint_dir: Path,
    step: int,
    optimizer_ranks: list[int],
    *,
    saved_world_size: int | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / f"model_{step:06d}.pt").write_bytes(b"model")
    meta_data = {}
    if saved_world_size is not None:
        meta_data["training_env"] = {"ddp_world_size": int(saved_world_size)}
    (checkpoint_dir / f"meta_{step:06d}.json").write_text(json.dumps(meta_data), encoding="utf-8")
    for rank in optimizer_ranks:
        (checkpoint_dir / f"optim_{step:06d}_rank{rank}.pt").write_bytes(b"optim")


def _base_exact_meta() -> dict:
    return {
        "num_iterations": 100,
        "lr_schedule": {
            "type": "linear_warmdown",
            "warmup_iters": 4,
            "warmdown_iters": 65,
            "fixed_lr_start_step": 4,
            "fixed_lr_end_step": 35,
            "decay_step_1": None,
            "decay_step_2": None,
            "decay_step_1_lr_frac": None,
            "decay_step_2_lr_frac": None,
        },
        "training_env": {"ddp_world_size": 2},
        "total_batch_size": 4096,
        "device_batch_size": 8,
        "max_seq_len": 256,
        "train_data_signature_hash": "sig-ok",
    }


def _assert_exact_meta_matches(
    meta_data: dict,
    *,
    expected_ddp_world_size: int = 2,
    expected_device_batch_size: int = 8,
    expected_total_batch_size: int = 4096,
    allow_world_size_change: bool = False,
    expected_global_microbatch_tokens: int | None = None,
) -> list[str]:
    return get_exact_resume_compatibility_mismatches(
        meta_data=meta_data,
        expected_num_iterations=100,
        expected_lr_schedule={
            "type": "linear_warmdown",
            "warmup_iters": 4,
            "warmdown_iters": 65,
            "fixed_lr_start_step": 4,
            "fixed_lr_end_step": 35,
            "decay_step_1": None,
            "decay_step_2": None,
            "decay_step_1_lr_frac": None,
            "decay_step_2_lr_frac": None,
        },
        expected_ddp_world_size=expected_ddp_world_size,
        expected_total_batch_size=expected_total_batch_size,
        expected_device_batch_size=expected_device_batch_size,
        expected_max_seq_len=256,
        expected_train_data_signature_hash="sig-ok",
        allow_world_size_change=allow_world_size_change,
        expected_global_microbatch_tokens=expected_global_microbatch_tokens,
    )


def test_find_last_complete_step_skips_partial_newest_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    _create_checkpoint_files(checkpoint_dir, step=9, optimizer_ranks=[0, 1])
    _create_checkpoint_files(checkpoint_dir, step=10, optimizer_ranks=[0])  # partial newest
    (checkpoint_dir / "latest_step.txt").write_text("10\n", encoding="utf-8")
    (checkpoint_dir / "final_step.txt").write_text("10\n", encoding="utf-8")

    resolved = find_last_complete_step(
        str(checkpoint_dir),
        expected_world_size=2,
        require_optimizer=True,
        prefer_markers=True,
    )
    assert resolved == 9


def test_find_last_complete_step_ignores_stale_marker_and_returns_newest_complete(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    _create_checkpoint_files(checkpoint_dir, step=8, optimizer_ranks=[0, 1])
    _create_checkpoint_files(checkpoint_dir, step=9, optimizer_ranks=[0, 1])
    (checkpoint_dir / "latest_step.txt").write_text("8\n", encoding="utf-8")

    resolved = find_last_complete_step(
        str(checkpoint_dir),
        expected_world_size=2,
        require_optimizer=True,
        prefer_markers=True,
    )
    assert resolved == 9


def test_find_last_complete_step_uses_saved_world_size_when_expected_world_is_unknown(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    _create_checkpoint_files(checkpoint_dir, step=9, optimizer_ranks=[0, 1, 2, 3], saved_world_size=4)
    _create_checkpoint_files(checkpoint_dir, step=10, optimizer_ranks=[0, 1], saved_world_size=4)
    (checkpoint_dir / "latest_step.txt").write_text("10\n", encoding="utf-8")

    resolved = find_last_complete_step(
        str(checkpoint_dir),
        expected_world_size=None,
        require_optimizer=True,
        prefer_markers=True,
    )
    assert resolved == 9


def test_exact_resume_compatibility_passes_when_metadata_matches():
    mismatches = _assert_exact_meta_matches(_base_exact_meta())
    assert mismatches == []


def test_exact_resume_compatibility_detects_schedule_mismatch():
    meta_data = _base_exact_meta()
    meta_data["lr_schedule"] = {
        **meta_data["lr_schedule"],
        "warmdown_iters": 64,
    }
    mismatches = _assert_exact_meta_matches(meta_data)
    assert any("lr_schedule mismatch" in msg for msg in mismatches)


def test_exact_resume_compatibility_detects_batch_and_world_size_mismatch():
    meta_data = _base_exact_meta()
    meta_data["training_env"]["ddp_world_size"] = 4
    meta_data["total_batch_size"] = 2048
    mismatches = _assert_exact_meta_matches(meta_data)
    assert any("ddp_world_size mismatch" in msg for msg in mismatches)
    assert any("total_batch_size mismatch" in msg for msg in mismatches)


def test_exact_resume_compatibility_detects_data_signature_mismatch():
    meta_data = _base_exact_meta()
    meta_data["train_data_signature_hash"] = "different"
    mismatches = _assert_exact_meta_matches(meta_data)
    assert any("train_data_signature_hash mismatch" in msg for msg in mismatches)


def test_exact_resume_compatibility_allows_world_size_change_with_matching_microbatch_and_total():
    meta_data = _base_exact_meta()
    meta_data["training_env"]["ddp_world_size"] = 4
    meta_data["device_batch_size"] = 4  # saved global microbatch = 4 * 256 * 4 = 4096
    mismatches = _assert_exact_meta_matches(
        meta_data,
        expected_ddp_world_size=8,
        expected_device_batch_size=2,  # current global microbatch = 2 * 256 * 8 = 4096
        expected_total_batch_size=4096,
        allow_world_size_change=True,
        expected_global_microbatch_tokens=4096,
    )
    assert mismatches == []


def test_exact_resume_compatibility_rejects_world_size_change_on_microbatch_mismatch():
    meta_data = _base_exact_meta()
    meta_data["training_env"]["ddp_world_size"] = 4
    meta_data["device_batch_size"] = 4
    mismatches = _assert_exact_meta_matches(
        meta_data,
        expected_ddp_world_size=8,
        expected_device_batch_size=3,
        expected_total_batch_size=4096,
        allow_world_size_change=True,
        expected_global_microbatch_tokens=6144,
    )
    assert any("global_microbatch_tokens mismatch" in msg for msg in mismatches)


def test_resolve_optimizer_shard_rank_uses_requested_rank_for_same_world():
    meta_data = {"training_env": {"ddp_world_size": 4}}
    resolved = resolve_optimizer_shard_rank(
        meta_data,
        requested_rank=3,
        current_world_size=4,
        allow_world_size_change=True,
    )
    assert resolved == 3


def test_resolve_optimizer_shard_rank_uses_canonical_rank_for_world_size_change():
    meta_data = {"training_env": {"ddp_world_size": 4}}
    resolved = resolve_optimizer_shard_rank(
        meta_data,
        requested_rank=7,
        current_world_size=8,
        allow_world_size_change=True,
        canonical_rank=0,
    )
    assert resolved == 0


def test_ensure_sharded_optimizer_world_size_compatibility_accepts_same_world():
    ensure_sharded_optimizer_world_size_compatibility(
        saved_world_size=4,
        current_world_size=4,
    )


def test_ensure_sharded_optimizer_world_size_compatibility_rejects_world_change():
    with pytest.raises(ValueError, match="cannot be resumed across a DDP topology change"):
        ensure_sharded_optimizer_world_size_compatibility(
            saved_world_size=4,
            current_world_size=8,
        )

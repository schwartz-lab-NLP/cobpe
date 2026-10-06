import argparse

from nanochat.modifier_cli import add_modifier_training_args


def test_modifier_training_cli_aliases_keep_config_destinations():
    parser = add_modifier_training_args(argparse.ArgumentParser())

    default = parser.parse_args([])
    aliases = parser.parse_args([
        "--modifier-head", "concat_gated",
        "--modifier-gates", "scalar",
    ])
    legacy_names = parser.parse_args([
        "--modifier-conditioning-mode", "base_bias",
        "--modifier-gate-mode", "per_group",
    ])

    assert default.modifier_conditioning_mode == "concat_gated_refine"
    assert default.modifier_gate_mode == "per_group"
    assert aliases.modifier_conditioning_mode == "concat_gated"
    assert aliases.modifier_gate_mode == "scalar"
    assert legacy_names.modifier_conditioning_mode == "base_bias"
    assert legacy_names.modifier_gate_mode == "per_group"


def test_modifier_training_cli_describes_retained_heads():
    help_text = " ".join(add_modifier_training_args(argparse.ArgumentParser()).format_help().split())
    assert "lexical base-token bias" in help_text
    assert "gated context and base-token logits" in help_text
    assert "residual refinement" in help_text


def test_public_head_names_map_to_checkpoint_modes():
    parser = add_modifier_training_args(argparse.ArgumentParser())
    for name, mode in {
        "lexical-bias": "base_bias",
        "gated-concat": "concat_gated",
        "gated-refinement": "concat_gated_refine",
    }.items():
        assert parser.parse_args(["--modifier-head", name]).modifier_conditioning_mode == mode

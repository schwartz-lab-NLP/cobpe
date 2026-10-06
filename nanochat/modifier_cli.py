"""Command-line options shared by CoBPE training entry points."""


HEAD_NAMES = {
    "lexical-bias": "base_bias",
    "gated-concat": "concat_gated",
    "gated-refinement": "concat_gated_refine",
}


def _head_mode(value):
    return HEAD_NAMES.get(value, value)


def add_modifier_training_args(parser):
    parser.add_argument(
        "--modifier-head",
        "--modifier-conditioning-mode",
        dest="modifier_conditioning_mode",
        type=_head_mode,
        metavar="{lexical-bias,gated-concat,gated-refinement}",
        default="concat_gated_refine",
        choices=["base_bias", "concat_gated", "concat_gated_refine"],
        help=(
            "lexical-bias: lexical base-token bias; gated-concat: gated context and base-token logits; "
            "gated-refinement (default): gated context and base-token logits with residual refinement"
        ),
    )
    parser.add_argument(
        "--modifier-gates",
        "--modifier-gate-mode",
        dest="modifier_gate_mode",
        type=str,
        default="per_group",
        choices=["scalar", "per_group"],
        help="gate shared across all modifier groups, or one gate per modifier group",
    )
    return parser

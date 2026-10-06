"""Script entrypoints for the nanochat framework backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NanochatEntrypoints:
    """Module names for nanochat-backed training, evaluation, and inference."""

    tokenizer_training: str = "scripts.tok_train"
    metadata_export: str = "scripts.export_compositional_metadata"
    pretraining: str = "scripts.base_train"
    evaluation: str = "scripts.base_eval"
    cli_inference: str = "scripts.generate"


entrypoints = NanochatEntrypoints()

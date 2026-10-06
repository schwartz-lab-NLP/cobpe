"""Nanochat training-framework adapter for compositional tokenization experiments.

The upstream-derived ``nanochat`` package remains the implementation backend.
New framework-facing code should import through this namespace so nanochat is a
peer of Megatron under ``cobpe.integrations``.
"""

from cobpe.integrations.nanochat.entrypoints import NanochatEntrypoints, entrypoints

__all__ = [
    "NanochatEntrypoints",
    "entrypoints",
]

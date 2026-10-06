"""Nanochat runtime helpers."""

from __future__ import annotations

import os

from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_dist_info,
    get_peak_flops,
    is_ddp_initialized,
    print0,
    print_banner,
)


def configure_output_base_dir(output_base_dir: str) -> str | None:
    """Apply a nanochat output directory override and return the resolved path."""

    if not output_base_dir.strip():
        return None
    resolved = os.path.abspath(os.path.expanduser(output_base_dir))
    os.environ["NANOCHAT_BASE_DIR"] = resolved
    return resolved


__all__ = [
    "COMPUTE_DTYPE",
    "COMPUTE_DTYPE_REASON",
    "DummyWandb",
    "autodetect_device_type",
    "compute_cleanup",
    "compute_init",
    "configure_output_base_dir",
    "get_base_dir",
    "get_dist_info",
    "get_peak_flops",
    "is_ddp_initialized",
    "print0",
    "print_banner",
]

"""Training and evaluation of small alternative-architecture memory backbones
(Mamba-2, TTT, DeltaNet, compressive transformer, memory mosaics, STM, ...)
on the benchmark's token sequences."""

from __future__ import annotations

import os

from eccbench import config


def _configure_hf_cache() -> None:
    default_cache = config.MODEL_CACHE
    legacy_cache = os.environ.pop("TRANSFORMERS_CACHE", None)
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(legacy_cache or default_cache)


_configure_hf_cache()

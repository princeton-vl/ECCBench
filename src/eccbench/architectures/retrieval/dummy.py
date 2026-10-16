"""Deterministic random embeddings for testing the pipeline without a GPU.

Probe scores with this backend should match the random baseline.
"""
from __future__ import annotations

import hashlib

import numpy as np

DIM = 256


def _vec(key: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


class Backend:
    def embed_video_files(self, paths, batch=8):
        return np.stack([_vec(str(p)) for p in paths])

    def embed_text(self, text, event=None):
        return _vec(f"text::{text}")

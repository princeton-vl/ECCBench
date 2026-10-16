"""Per-GPU memory budget for local checkpoints."""

from __future__ import annotations

from typing import Optional

import torch

# Approximate weight memory in GB at BF16 plus 25% headroom, keyed by registry alias.
MODEL_SIZES_GB = {
    "minicpm": 10.725,
    "glm45v": 248,
    "qwen3_full": 28.875,
    "qwen3_omni": 90.75,
    "internvl-3-5": 26.5,
    "internvl-3-5-thinking": 26.5,
    "internvl-3-5-38b": 96.75,
    "internvl-3-5-38b-thinking": 96.75,
    "internvl-3-5-30b-a3b": 80.375,
    "internvl-3-5-30b-a3b-thinking": 80.375,
    "minicpm-4-5": 18.48,
    "mimo-vl": 29.875,
    "longvila": 28.875,
    "default": 28.875,
}


def calculate_max_gpu_mem(model_name: str, override: Optional[float] = None) -> float:
    """Per-GPU memory limit in GB: ``1.05 * model size / number of visible GPUs``."""
    if override is not None:
        return override
    model_size_gb = MODEL_SIZES_GB.get(model_name, MODEL_SIZES_GB["default"])
    num_gpus = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    return (model_size_gb / num_gpus) * 1.05

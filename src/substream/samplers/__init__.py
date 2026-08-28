"""Frame sampling utilities for different video-language models."""

import os

# CRITICAL: Set backend BEFORE any qwen_vl_utils/qwen_omni_utils imports.
# The torchcodec backend intermittently raises std::bad_alloc when decoding
# long videos; forcing the torchvision reader avoids it.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")


def get_frame_sampler(model_type: str):
    """Return the frame sampler instance for a registry alias."""
    import importlib

    from substream.registry import REGISTRY

    module_name, class_name = REGISTRY[model_type].sampler.split(".")
    module = importlib.import_module(f"substream.samplers.{module_name}")
    return getattr(module, class_name)()

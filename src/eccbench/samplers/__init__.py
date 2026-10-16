"""Frame samplers, one per model family."""

import os

# Must be set before qwen_vl_utils or qwen_omni_utils is imported: their default
# torchcodec reader can fail with std::bad_alloc on long videos.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")


def get_frame_sampler(model_type: str):
    """Return the frame sampler instance for a registry alias."""
    import importlib

    from eccbench.registry import REGISTRY

    module_name, class_name = REGISTRY[model_type].sampler.split(".")
    module = importlib.import_module(f"eccbench.samplers.{module_name}")
    return getattr(module, class_name)()

"""Pytest configuration for the substream test suite.

Two jobs:

1. Make ``src/`` importable so the tests run against the checkout without
   requiring ``pip install -e .``.
2. On CPU-only environments where torch is not installed, register a minimal
   torch stub so the pure-python modules (which import torch at module level)
   can be exercised. When the real torch is importable (e.g. inside the GPU
   envs driven by ``bin/envctl``) nothing is stubbed.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _real_torch_available() -> bool:
    if "torch" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


if not _real_torch_available():
    fake_torch = ModuleType("torch")

    class _FakeCuda:
        OutOfMemoryError = RuntimeError

        def is_available(self) -> bool:
            return False

        def empty_cache(self) -> None:
            return None

        def reset_peak_memory_stats(self, *args, **kwargs) -> None:
            return None

    fake_torch.cuda = _FakeCuda()
    fake_torch.Tensor = SimpleNamespace  # minimal placeholder
    sys.modules["torch"] = fake_torch

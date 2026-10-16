"""Make ``src/`` importable, and stub torch when it is not installed.

The stub lets the pure-Python modules that import torch at module level run
in a base install. When torch is installed, nothing is stubbed.
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

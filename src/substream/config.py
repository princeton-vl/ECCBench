"""Filesystem locations, each overridable through one environment variable.

All values are plain module-level constants resolved once at import time:

    SUBSTREAM_SCRATCH_ROOT  scratch space (default: ~/.cache/substream)
    SUBSTREAM_MODEL_CACHE   model weights cache, used as HF_HOME
    SUBSTREAM_DATA_ROOT     local datasets, when not loading from the Hub
    SUBSTREAM_RUN_ROOT      where evaluation state files and CSV logs are written
"""
from __future__ import annotations

import os
from pathlib import Path

SCRATCH_ROOT = Path(os.environ.get("SUBSTREAM_SCRATCH_ROOT", Path.home() / ".cache" / "substream"))
MODEL_CACHE = Path(os.environ.get("SUBSTREAM_MODEL_CACHE", SCRATCH_ROOT / "model-cache"))
DATA_ROOT = Path(os.environ.get("SUBSTREAM_DATA_ROOT", SCRATCH_ROOT / "data"))
RUN_ROOT = Path(os.environ.get("SUBSTREAM_RUN_ROOT", "runs"))

# Model weights cache doubles as the Hugging Face cache unless the caller
# already chose one, so every wrapper caches to the same place.
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))

# Point Python's tempfile module at scratch space unless the caller already
# chose a location. Model weights and decoded video frames are far larger than
# a typical /tmp partition.
_TEMP_DIR = SCRATCH_ROOT / "tmp"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)
for _name in ("TMPDIR", "TEMP", "TMP"):
    os.environ.setdefault(_name, str(_TEMP_DIR))

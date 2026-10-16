"""Filesystem locations, read once at import time from these environment variables:

    ECCBENCH_SCRATCH_ROOT  scratch space (default: ~/.cache/eccbench)
    ECCBENCH_MODEL_CACHE   model weights cache, also the default HF_HOME
    ECCBENCH_DATA_ROOT     local dataset files (filtered manifests, RAG inputs)
    ECCBENCH_RUN_ROOT      state files and CSV logs of evaluation runs
"""
from __future__ import annotations

import os
from pathlib import Path

SCRATCH_ROOT = Path(os.environ.get("ECCBENCH_SCRATCH_ROOT", Path.home() / ".cache" / "eccbench"))
MODEL_CACHE = Path(os.environ.get("ECCBENCH_MODEL_CACHE", SCRATCH_ROOT / "model-cache"))
DATA_ROOT = Path(os.environ.get("ECCBENCH_DATA_ROOT", SCRATCH_ROOT / "data"))
RUN_ROOT = Path(os.environ.get("ECCBENCH_RUN_ROOT", "runs"))

# Use the model cache as the Hugging Face cache unless HF_HOME is already set.
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))

# Put temporary files in scratch space unless a temp directory is already set;
# decoded videos can be larger than a typical /tmp.
_TEMP_DIR = SCRATCH_ROOT / "tmp"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)
for _name in ("TMPDIR", "TEMP", "TMP"):
    os.environ.setdefault(_name, str(_TEMP_DIR))

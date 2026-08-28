"""The core package must import from any working directory, torch-free.

Runs a subprocess with cwd=/tmp and PYTHONPATH pointing only at src/, so any
hidden dependence on cwd == repo root (or on the test process's sys.path
mutations) fails loudly.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_IMPORT_CHECK = (
    "import sys; "
    "import substream.config, substream.registry, "
    "substream.data.state, substream.data.manifest; "
    "assert 'torch' not in sys.modules, 'core modules must stay torch-free'"
)


def test_core_modules_import_from_any_cwd(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    # substream.config creates its scratch dir at import time; keep that
    # side effect inside the test sandbox.
    env["SUBSTREAM_SCRATCH_ROOT"] = str(tmp_path / "scratch")

    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

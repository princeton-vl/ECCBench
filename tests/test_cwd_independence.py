"""The core modules import from any working directory and without torch."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_IMPORT_CHECK = (
    "import sys; "
    "import eccbench.config, eccbench.registry, "
    "eccbench.data.state, eccbench.data.manifest; "
    "assert 'torch' not in sys.modules, 'core modules must stay torch-free'"
)


def test_core_modules_import_from_any_cwd(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    # eccbench.config creates its scratch dir at import time.
    env["ECCBENCH_SCRATCH_ROOT"] = str(tmp_path / "scratch")

    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

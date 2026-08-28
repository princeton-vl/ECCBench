"""Fetch a benchmark config from the Hugging Face Hub.

The published dataset (https://huggingface.co/datasets/anonstreammem/substream-recollection)
ships four configs — ``text``, ``synthetic_video``, ``natural_video``, and
``easyhuman`` — each containing a nested ``manifest.json`` that the evaluation
runner ingests directly, plus the referenced media files. ``fetch`` downloads
one config into the local Hub cache and returns the manifest path together
with the directory that the manifest's relative media paths resolve against.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_REPO = "anonstreammem/substream-recollection"
CONFIGS = ("text", "synthetic_video", "natural_video", "easyhuman")


def fetch(config: str, repo_id: str = DEFAULT_REPO, revision: str | None = None) -> tuple[Path, Path]:
    """Download one dataset config and return (manifest_path, asset_root).

    Media paths inside the manifest are relative to the config directory, so
    the returned asset root is ``<snapshot>/<config>``. A subset of
    ``natural_video`` clips is published as provenance only (their rows carry
    ``media_included: false``); the runner skips those when the file is absent.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {CONFIGS}")

    from huggingface_hub import snapshot_download

    patterns = [f"{config}/**"]
    root = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
        )
    )
    manifest = root / config / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} missing after download; the repo layout changed?")
    return manifest, root / config

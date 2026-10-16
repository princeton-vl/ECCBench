"""Download one benchmark config from the Hugging Face Hub.

The dataset (https://huggingface.co/datasets/anonstreammem/substream-recollection)
has four configs: ``text``, ``synthetic_video``, ``natural_video`` and
``easyhuman``. Each config holds ``questions.json`` (one row per question),
a nested ``manifest.json`` that the runner reads, and the media files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from eccbench import config as eccbench_config

DEFAULT_REPO = "anonstreammem/substream-recollection"
CONFIGS = ("text", "synthetic_video", "natural_video", "easyhuman")

logger = logging.getLogger(__name__)


def fetch(
    config: str,
    repo_id: str = DEFAULT_REPO,
    revision: str | None = None,
    media: bool = True,
) -> tuple[Path, Path]:
    """Download one config and return ``(manifest_path, asset_root)``.

    Media paths in the manifest are relative to ``asset_root``. With
    ``media=False`` only the question files are downloaded; get the media of
    selected streams later with :func:`fetch_media`. Rows with
    ``media_included: false`` have no media file; the runner skips them.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {CONFIGS}")

    from huggingface_hub import snapshot_download

    patterns = [f"{config}/**"] if media else [f"{config}/manifest.json", f"{config}/questions.json"]
    root = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
        )
    )
    config_dir = root / config
    manifest = config_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is missing from {repo_id}")
    cache_dir = eccbench_config.DATA_ROOT / "manifests" / repo_id.replace("/", "--") / root.name
    return reconcile_manifest(manifest, config_dir / "questions.json", cache_dir / f"{config}.json"), config_dir


def fetch_media(asset_root: Path, paths: Iterable[str], repo_id: str = DEFAULT_REPO) -> None:
    """Download the media files ``paths`` (relative to ``asset_root``) that are not yet local.

    ``asset_root`` must come from :func:`fetch`; its parent directory name is
    the dataset commit, so the media match the downloaded questions.
    """
    missing = sorted({p for p in paths if p and not (asset_root / p).exists()})
    if not missing:
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=asset_root.parent.name,
        allow_patterns=[f"{asset_root.name}/{p}" for p in missing],
    )


def _row_key(stream: dict, question: dict) -> tuple:
    return (
        stream.get("stream_id"),
        stream.get("length_L"),
        stream.get("entropy_band"),
        stream.get("modality"),
        question.get("question_id"),
        question.get("question_variant"),
    )


def reconcile_manifest(manifest: Path, questions: Path, out: Path) -> Path:
    """Keep only the manifest questions that are also rows of ``questions.json``.

    ``questions.json`` is the list of released rows. If the manifest lists
    extra questions, write a filtered copy to ``out`` and return its path.
    Otherwise return ``manifest`` unchanged.
    """
    if not questions.exists():
        return manifest
    with questions.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keep = {_row_key(row, row) for row in rows}

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    total = 0
    for stream in payload["videos"]:
        total += len(stream["questions"])
        stream["questions"] = [q for q in stream["questions"] if _row_key(stream, q) in keep]
    kept = sum(len(stream["questions"]) for stream in payload["videos"])
    if kept == total:
        return manifest

    payload["videos"] = [stream for stream in payload["videos"] if stream["questions"]]
    logger.warning(
        "%s lists %d questions that are not in %s; using the %d released questions.",
        manifest, total - kept, questions.name, kept,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out

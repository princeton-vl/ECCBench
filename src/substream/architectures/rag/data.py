"""Rows and media for the RAG pipeline, read from the published dataset.

Everything comes from the Hub configs ``natural_video`` and ``synthetic_video``
via :func:`substream.data.hf_loader.fetch` (the same download the evaluation
runner uses): the nested ``manifest.json`` supplies streams and questions, the
config directory supplies the media, and ``questions.json`` supplies the
per-question ``query_class`` used as the text-query target.

Selection is expressed with dataset fields only: ``length_L`` picks the stream
length, ``question_set`` (``designed`` / ``density_topup``) picks the natural
question set, ``density_band`` (``low`` / ``high``) optionally restricts to one
codec-distortion band, and rows with ``media_included == false`` (provenance
only) are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from substream.data import hf_loader

QUESTION_SETS = ("density_topup", "designed", "all")
DENSITY_BANDS = ("all", "low", "high")


def add_dataset_args(ap: argparse.ArgumentParser) -> None:
    """Shared CLI options for locating the published dataset."""
    ap.add_argument("--repo", default=hf_loader.DEFAULT_REPO,
                    help="Hub dataset repo id (default: %(default)s)")
    ap.add_argument("--revision", default=None,
                    help="Hub revision to pin (default: latest)")


def add_natural_filter_args(ap: argparse.ArgumentParser) -> None:
    """Shared CLI options selecting natural-video rows."""
    ap.add_argument("--question-set", choices=QUESTION_SETS, default="density_topup",
                    help="natural_video question_set to use (default: %(default)s)")
    ap.add_argument("--density-band", choices=DENSITY_BANDS, default="all",
                    help="restrict natural streams to one density_band (default: %(default)s)")


def load_config(config: str, repo: str = hf_loader.DEFAULT_REPO,
                revision: str | None = None) -> tuple[dict, Path]:
    """Download one config and return (nested manifest dict, asset root)."""
    manifest, root = hf_loader.fetch(config, repo_id=repo, revision=revision)
    return json.loads(manifest.read_text()), root


def _query_classes(root: Path) -> dict[str, str]:
    """question_id -> query_class from the config's NDJSON ``questions.json``.

    Natural question ids are unique across lengths, so a flat map suffices.
    """
    path = root / "questions.json"
    out = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("query_class"):
                out[r["question_id"]] = r["query_class"]
    return out


def natural_items(length_L: int, question_set: str = "density_topup",
                  density_band: str = "all", repo: str = hf_loader.DEFAULT_REPO,
                  revision: str | None = None) -> list[dict]:
    """Natural streams at one length with their text questions.

    Returns dicts with ``video_key`` (the stream id), ``video`` (absolute media
    path), ``hub_path`` (``<media_config>/<video_path>``), ``density_band``,
    ``event`` (the clip's designed event span in seconds, or None) and
    ``queries``: ``kind="text"`` dicts carrying ``question_id``, ``text``,
    ``event`` (query class) and ``answer``.
    """
    man, root = load_config("natural_video", repo, revision)
    classes = _query_classes(root)
    items = []
    for v in man["videos"]:
        if int(v.get("length_L", -1)) != length_L:
            continue
        if v.get("media_included") is False:
            continue
        if density_band != "all" and v.get("density_band") != density_band:
            continue
        path = root / v["video_path"]
        if not path.exists():
            continue
        qs = []
        for q in v["questions"]:
            if question_set != "all" and q.get("question_set") != question_set:
                continue
            qs.append(dict(kind="text", question_id=q["question_id"], text=q["question"],
                           event=classes.get(q["question_id"], q.get("source_class")),
                           answer=q["answer"], question_set=q.get("question_set")))
        if not qs:
            continue
        a, b = v.get("event_clip_start"), v.get("event_clip_end")
        items.append(dict(
            domain="natural", video_key=v["stream_id"], video=str(path),
            hub_path=f'{v.get("media_config", "natural_video")}/{v["video_path"]}',
            density_band=v.get("density_band"),
            event=(float(a), float(b)) if a is not None and b is not None else None,
            queries=qs))
    return items


def synthetic_items(length_L: int, entropy_bands: tuple[str, ...] | None = None,
                    repo: str = hf_loader.DEFAULT_REPO,
                    revision: str | None = None) -> list[dict]:
    """Synthetic streams at one length with their candidate-clip questions.

    ``video_key`` is ``<bucket dir>_<stream_id>`` (stream ids recur across
    buckets). Each query is a ``kind="video"`` dict with ``question_id``
    (``<video_key>__<question_id>``, unique across lengths and bands), ``clip``
    (absolute path), ``answer``, ``present`` and ``gold`` = (clip_start, clip_end).
    """
    man, root = load_config("synthetic_video", repo, revision)
    items = []
    for v in man["videos"]:
        if int(v.get("length_L", -1)) != length_L:
            continue
        if entropy_bands and v.get("entropy_band") not in entropy_bands:
            continue
        path = root / v["video_path"]
        if not path.exists():
            continue
        key = f'{Path(v["video_path"]).parent.name}_{v["stream_id"]}'
        qs = []
        for q in v["questions"]:
            c = q.get("candidate") or {}
            if not c.get("clip_path"):
                continue
            clip = root / c["clip_path"]
            if not clip.exists():
                continue
            present = bool(c.get("present"))
            qs.append(dict(kind="video", question_id=f'{key}__{q["question_id"]}',
                           clip=str(clip), present=present,
                           answer="yes" if present else "no",
                           gold=[(float(c.get("clip_start") or 0.0),
                                  float(c.get("clip_end") or 0.0))]))
        if qs:
            items.append(dict(domain="synthetic", video_key=key, video=str(path),
                              hub_path=f'synthetic_video/{v["video_path"]}',
                              entropy_band=v.get("entropy_band"), queries=qs))
    return items

"""Tests for the Hugging Face dataset fetch helper (no network)."""

import pytest

pytest.importorskip("huggingface_hub")

from eccbench.data.hf_loader import CONFIGS, DEFAULT_REPO, fetch  # noqa: E402


def test_fetch_rejects_unknown_config():
    with pytest.raises(ValueError, match="unknown config"):
        fetch("not_a_config")


def test_published_config_names():
    assert CONFIGS == ("text", "synthetic_video", "natural_video", "easyhuman")
    assert DEFAULT_REPO == "anonstreammem/substream-recollection"


def _write_config(tmp_path, manifest_questions, released_ids):
    import json

    stream = {"stream_id": "s0", "length_L": 8, "entropy_band": "low", "video_path": "text/low/L8/s0"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"videos": [dict(stream, questions=[
        {"question_id": qid, "question_variant": "sequential"} for qid in manifest_questions
    ])]}))
    questions = tmp_path / "questions.json"
    questions.write_text("\n".join(
        json.dumps(dict(stream, question_id=qid, question_variant="sequential")) for qid in released_ids
    ))
    return manifest, questions


def test_reconcile_keeps_manifest_when_rows_match(tmp_path):
    from eccbench.data.hf_loader import reconcile_manifest

    manifest, questions = _write_config(tmp_path, ["q0", "q1"], ["q0", "q1"])
    assert reconcile_manifest(manifest, questions, tmp_path / "out.json") == manifest


def test_reconcile_drops_questions_missing_from_rows(tmp_path):
    import json

    from eccbench.data.hf_loader import reconcile_manifest

    manifest, questions = _write_config(tmp_path, ["q0", "q1", "q2"], ["q0", "q2"])
    out = reconcile_manifest(manifest, questions, tmp_path / "out.json")
    assert out == tmp_path / "out.json"
    kept = [q["question_id"] for q in json.loads(out.read_text())["videos"][0]["questions"]]
    assert kept == ["q0", "q2"]

"""Tests for the Hugging Face dataset fetch helper (no network)."""

import pytest

pytest.importorskip("huggingface_hub")

from substream.data.hf_loader import CONFIGS, DEFAULT_REPO, fetch  # noqa: E402


def test_fetch_rejects_unknown_config():
    with pytest.raises(ValueError, match="unknown config"):
        fetch("not_a_config")


def test_published_config_names():
    assert CONFIGS == ("text", "synthetic_video", "natural_video", "easyhuman")
    assert DEFAULT_REPO == "anonstreammem/substream-recollection"

"""Tests for the CALIB_ARM payoff suffix and RAG_PREAMBLE gating."""

import pytest

from substream.eval.processors.question_processor import (
    _calibration_payoff_suffix,
    _rag_preamble,
)


def test_payoff_suffix_empty_when_arm_unset(monkeypatch):
    monkeypatch.delenv("CALIB_ARM", raising=False)
    assert _calibration_payoff_suffix("{2}") == ""

    monkeypatch.setenv("CALIB_ARM", "")
    assert _calibration_payoff_suffix("{2}") == ""


def test_payoff_suffix_arm_a(monkeypatch):
    monkeypatch.setenv("CALIB_ARM", "A")
    suffix = _calibration_payoff_suffix("{2}")

    assert "+3 points" in suffix
    assert "-1 point," in suffix  # singular: penalty is one point
    assert "25% confident" in suffix
    assert "{2}" in suffix


def test_payoff_suffix_arm_b(monkeypatch):
    monkeypatch.setenv("CALIB_ARM", "B")
    suffix = _calibration_payoff_suffix("{uncertain}")

    assert "+1 point," in suffix  # singular: reward is one point
    assert "-4 points" in suffix
    assert "80% confident" in suffix
    assert "{uncertain}" in suffix


def test_payoff_suffix_arm_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("CALIB_ARM", " b ")
    assert "-4 points" in _calibration_payoff_suffix("{2}")


def test_payoff_suffix_rejects_unknown_arm(monkeypatch):
    monkeypatch.setenv("CALIB_ARM", "C")
    with pytest.raises(ValueError):
        _calibration_payoff_suffix("{2}")


def test_rag_preamble_gated_on_env(monkeypatch):
    monkeypatch.delenv("RAG_PREAMBLE", raising=False)
    assert _rag_preamble() == ""

    monkeypatch.setenv("RAG_PREAMBLE", "0")
    assert _rag_preamble() == ""

    monkeypatch.setenv("RAG_PREAMBLE", "1")
    preamble = _rag_preamble()
    assert preamble.startswith("NOTE: A retrieval system")
    assert "SEGMENT BREAK" in preamble

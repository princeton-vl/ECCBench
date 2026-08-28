"""Tests for the per-question CSV logging helper."""

import csv

from substream.data.csv_log import QUESTION_LOG_HEADERS, write_question_log_csv


def test_write_question_log_csv(tmp_path):
    rows = [
        {
            "video_id": 7,
            "question_id": "q-1",
            "video_entropy": 0.25,
            "correct_answer": "3",
            "model_answer": "3",
        },
        {
            "video_id": 7,
            "question_id": "q-2",
            "video_entropy": None,
            "correct_answer": "4",
            "model_answer": "2",
        },
    ]

    destination = tmp_path / "logs" / "questions.csv"
    written_path = write_question_log_csv(str(destination), rows)

    assert written_path == destination
    assert destination.exists()

    with destination.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == QUESTION_LOG_HEADERS
        parsed = list(reader)

    assert len(parsed) == 2
    assert parsed[0]["video_id"] == "7"
    assert parsed[0]["question_id"] == "q-1"
    assert parsed[0]["video_entropy"] == "0.25"
    assert parsed[0]["correct_answer"] == "3"
    assert parsed[0]["model_answer"] == "3"

    # None values serialize to empty cells.
    assert parsed[1]["video_entropy"] == ""
    assert parsed[1]["correct_answer"] == "4"
    assert parsed[1]["model_answer"] == "2"


def test_write_question_log_csv_accepts_result_key_aliases(tmp_path):
    """Rows sourced from question results use 'correct'/'predicted' keys."""
    rows = [
        {
            "video_id": 1,
            "question_id": "q-0",
            "correct": "0",
            "predicted": "1",
            "response": "x" * 2000,
        }
    ]

    destination = tmp_path / "questions.csv"
    write_question_log_csv(str(destination), rows)

    with destination.open(newline="") as handle:
        parsed = list(csv.DictReader(handle))

    assert parsed[0]["correct_answer"] == "0"
    assert parsed[0]["model_answer"] == "1"
    # Long responses are kept in full, and additionally truncated to <= 1024.
    assert parsed[0]["response_full"] == "x" * 2000
    assert len(parsed[0]["response_truncated"]) <= 1024
    assert "..." in parsed[0]["response_truncated"]

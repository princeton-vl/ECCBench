"""Round-trip tests for the resumable run state manager."""

from substream.data.state import RunStateManager


def _result(question_id: str, question_time: float) -> dict:
    return {
        "video_index": 0,
        "question_id": question_id,
        "question_order": 1,
        "predicted": "0",
        "correct": "0",
        "is_correct": True,
        "is_dont_know": False,
        "response": "The answer is {0}.",
        "question_time": question_time,
    }


def test_state_round_trip_with_resume(tmp_path):
    state_path = tmp_path / "state.json"
    manager = RunStateManager(str(state_path), resume=False)

    manager.mark_question_complete(0, "q1", result=_result("q1", 1.0), bucket="L64")
    # Recording the identical result twice must append it only once.
    manager.mark_question_complete(0, "q1", result=_result("q1", 1.0), bucket="L64")
    manager.mark_question_complete(0, "q2", result=_result("q2", 2.0), bucket="L64")
    manager.mark_video_complete(0, bucket="L64")

    assert state_path.exists()

    resumed = RunStateManager(str(state_path), resume=True)
    assert resumed.get_completed_questions(0, bucket="L64") == {"q1", "q2"}
    assert resumed.is_video_completed(0, bucket="L64")

    saved = resumed.get_saved_results()
    assert len(saved) == 2
    assert {row["question_id"] for row in saved} == {"q1", "q2"}

    # Bucket is part of the key: the same video index in another bucket is fresh.
    assert resumed.get_completed_questions(0, bucket="L128") == set()
    assert resumed.get_completed_questions(0) == set()
    assert not resumed.is_video_completed(0, bucket="L128")


def test_resume_false_starts_fresh_in_memory(tmp_path):
    state_path = tmp_path / "state.json"
    manager = RunStateManager(str(state_path), resume=False)
    manager.mark_question_complete(0, "q1", result=_result("q1", 1.0), bucket="L64")

    fresh = RunStateManager(str(state_path), resume=False)
    assert fresh.get_completed_questions(0, bucket="L64") == set()
    assert fresh.get_saved_results() == []


def test_disabled_manager_is_noop(tmp_path):
    manager = RunStateManager(None, resume=False)
    manager.mark_question_complete(0, "q1", result=_result("q1", 1.0))
    manager.mark_video_complete(0)
    manager.flush()

    assert manager.get_completed_questions(0) == set()
    assert manager.get_saved_results() == []
    assert not manager.is_video_completed(0)
    assert not any(tmp_path.iterdir())

"""Parse an evaluation manifest: a list of streams, each with its questions.

A question is either native binary (one ``candidate`` and a yes/no
``answer``; every released config uses this) or multiple choice
(``options`` with a ``correct_index``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OptionEntry:
    """One option of a multiple-choice question."""

    source_index: int
    label: str
    clip_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_sequence: Optional[List[str]] = None


@dataclass
class BinaryCandidateEntry:
    """The candidate clip or token sequence of a binary question."""

    sequence: List[str]
    sequences: Dict[str, List[str]]  # e.g., {'S_tokens': [...], 'S_lanes': [...]}
    clip_path: str
    present: bool
    clip_start: Optional[float] = None
    clip_end: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionEntry:
    """One question.

    Binary questions set ``is_native_binary``, ``candidate`` and
    ``binary_answer`` (``"yes"`` or ``"no"``) and leave ``options`` empty.
    Multiple-choice questions set ``options`` and ``correct_answer_index``.
    """

    question_id: str
    prompt: str
    question_time: float
    options: List[OptionEntry]
    correct_answer_index: int
    dont_know_index: int
    clip_start_time: Optional[float]
    clip_end_time: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    question_order: int = 0
    question_mode: Optional[str] = None
    sequence_prefixes: Dict[str, List[str]] = field(default_factory=dict)
    is_native_binary: bool = False
    candidate: Optional[BinaryCandidateEntry] = None
    binary_answer: Optional[str] = None


@dataclass
class VideoEntry:
    """One stream and its questions."""

    video_index: int
    video_path: str
    questions: List[QuestionEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def main_video_path(self) -> str:
        return self.video_path


def load_patternvideos_manifest(
    json_path: str,
    *,
    require_video_assets: bool = True,
    asset_root: Optional[str] = None,
) -> List[VideoEntry]:
    """Parse the manifest at ``json_path``; relative media paths are joined to ``asset_root``."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Manifest must include a non-empty 'videos' list.")

    return [
        _parse_video(
            idx,
            raw_video,
            require_video_assets=require_video_assets,
            asset_root=asset_root,
        )
        for idx, raw_video in enumerate(videos)
    ]


def _parse_video(
    fallback_index: int,
    raw_video: Dict[str, Any],
    *,
    require_video_assets: bool,
    asset_root: Optional[str],
) -> VideoEntry:
    video_path = _resolve_asset_path(raw_video.get("video_path"), asset_root)
    if require_video_assets and not video_path:
        raise ValueError(f"Video entry {fallback_index} is missing 'video_path'.")
    video_path = str(video_path or "")

    raw_index = raw_video.get("video_index")
    try:
        video_index = int(raw_index) if raw_index is not None else fallback_index
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid video_index for entry {fallback_index}: {raw_index}") from exc

    raw_questions = raw_video.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError(f"Video {video_index} must include at least one question.")

    questions = [
        _parse_question(
            video_index,
            position,
            raw_question,
            require_option_clips=require_video_assets,
            asset_root=asset_root,
        )
        for position, raw_question in enumerate(raw_questions)
    ]

    metadata = {
        key: value
        for key, value in raw_video.items()
        if key not in {"video_path", "questions", "video_index"}
    }

    return VideoEntry(video_index=video_index, video_path=video_path, questions=questions, metadata=metadata)


def _is_native_binary_format(raw_question: Dict[str, Any]) -> bool:
    """True for a question with a ``candidate`` and an ``answer`` and no ``options``."""
    has_candidate = "candidate" in raw_question and isinstance(raw_question["candidate"], dict)
    has_answer = "answer" in raw_question
    has_options = "options" in raw_question and isinstance(raw_question.get("options"), list)
    return has_candidate and has_answer and not has_options


def _parse_binary_candidate(
    raw_candidate: Dict[str, Any],
    asset_root: Optional[str],
) -> BinaryCandidateEntry:
    sequence = _normalize_sequence(raw_candidate.get("sequence"))
    sequences_raw = raw_candidate.get("sequences", {})
    sequences = {
        key: _normalize_sequence(value)
        for key, value in sequences_raw.items()
        if isinstance(value, list)
    }
    clip_path = _resolve_asset_path(raw_candidate.get("clip_path"), asset_root)
    present = bool(raw_candidate.get("present", False))
    clip_start = _coerce_optional_float(raw_candidate.get("clip_start"))
    clip_end = _coerce_optional_float(raw_candidate.get("clip_end"))

    reserved_keys = {"sequence", "sequences", "clip_path", "present", "clip_start", "clip_end"}
    metadata = {k: v for k, v in raw_candidate.items() if k not in reserved_keys}

    return BinaryCandidateEntry(
        sequence=sequence,
        sequences=sequences,
        clip_path=str(clip_path or ""),
        present=present,
        clip_start=clip_start,
        clip_end=clip_end,
        metadata=metadata,
    )


def _parse_native_binary_question(
    video_index: int,
    question_position: int,
    raw_question: Dict[str, Any],
    *,
    asset_root: Optional[str],
) -> QuestionEntry:
    prompt = str(raw_question.get("question", ""))
    raw_question_id = raw_question.get("question_id")
    question_id = (
        str(raw_question_id)
        if raw_question_id is not None
        else f"video{video_index}_q{question_position}"
    )

    question_time = _coerce_float(
        raw_question.get("question_time"),
        f"question_time for {question_id}",
    )

    raw_candidate = raw_question.get("candidate", {})
    candidate = _parse_binary_candidate(raw_candidate, asset_root)

    raw_answer = raw_question.get("answer", "")
    binary_answer = str(raw_answer).strip().lower() if raw_answer else None

    clip_start_time = _coerce_optional_float(raw_question.get("clip_start_time"))
    clip_end_time = _coerce_optional_float(raw_question.get("clip_end_time"))

    reserved_keys = {
        "question", "question_id", "question_time", "candidate", "answer",
        "clip_start_time", "clip_end_time", "options", "correct_index",
    }
    metadata = {k: v for k, v in raw_question.items() if k not in reserved_keys}

    question_mode = raw_question.get("question_mode")
    if isinstance(question_mode, str):
        question_mode = question_mode.strip() or None
    else:
        question_mode = None

    sequence_prefixes = _normalize_sequence_mapping(raw_question.get("prefix"))

    return QuestionEntry(
        question_id=question_id,
        prompt=prompt,
        question_time=question_time,
        options=[],
        correct_answer_index=-1,
        dont_know_index=-1,
        clip_start_time=clip_start_time,
        clip_end_time=clip_end_time,
        metadata=metadata,
        question_order=question_position + 1,
        question_mode=question_mode,
        sequence_prefixes=sequence_prefixes,
        is_native_binary=True,
        candidate=candidate,
        binary_answer=binary_answer,
    )


def _parse_question(
    video_index: int,
    question_position: int,
    raw_question: Dict[str, Any],
    *,
    require_option_clips: bool,
    asset_root: Optional[str],
) -> QuestionEntry:
    if _is_native_binary_format(raw_question):
        return _parse_native_binary_question(
            video_index, question_position, raw_question, asset_root=asset_root
        )

    prompt = str(raw_question.get("question", ""))
    raw_question_id = raw_question.get("question_id")
    question_id = (
        str(raw_question_id)
        if raw_question_id is not None
        else f"video{video_index}_q{question_position}"
    )

    question_time = _coerce_float(
        raw_question.get("question_time"),
        f"question_time for {question_id}",
    )

    raw_options = raw_question.get("options")
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError(f"Question {question_id} must include at least one option entry.")

    real_options: List[OptionEntry] = []
    raw_to_eval_index: Dict[int, int] = {}
    uncertain_option: Optional[Dict[str, Any]] = None
    uncertain_raw_index: Optional[int] = None

    for raw_idx, raw_option in enumerate(raw_options):
        if _is_uncertain_option(raw_option):
            uncertain_option = raw_option
            uncertain_raw_index = raw_idx
            continue

        clip_path = _resolve_asset_path(raw_option.get("clip_path"), asset_root)
        if require_option_clips and not clip_path:
            raise ValueError(
                f"Question {question_id} option {raw_idx} is missing 'clip_path'."
            )
        normalized_sequence = _normalize_sequence(raw_option.get("sequence"))

        label = raw_option.get("label") or f"Option {raw_idx}"
        option_metadata = {
            key: value
            for key, value in raw_option.items()
            if key not in {"label", "clip_path"}
        }
        eval_index = len(real_options)
        raw_to_eval_index[raw_idx] = eval_index
        real_options.append(
            OptionEntry(
                source_index=raw_idx,
                label=str(label),
                clip_path=str(clip_path or ""),
                metadata=option_metadata,
                token_sequence=normalized_sequence if normalized_sequence else None,
            )
        )

    if not real_options:
        raise ValueError(f"Question {question_id} does not contain any playable options.")

    dont_know_index = len(real_options)
    if uncertain_raw_index is not None:
        raw_to_eval_index[uncertain_raw_index] = dont_know_index

    raw_correct_index = raw_question.get("correct_index")
    if raw_correct_index is None:
        raise ValueError(f"Question {question_id} is missing 'correct_index'.")
    try:
        raw_correct_index = int(raw_correct_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Question {question_id} has invalid 'correct_index': {raw_correct_index}"
        ) from exc

    if raw_correct_index not in raw_to_eval_index:
        raise ValueError(
            f"Question {question_id} correct_index {raw_correct_index} does not map to a known option."
        )
    correct_answer_index = raw_to_eval_index[raw_correct_index]

    clip_start_time = _coerce_optional_float(raw_question.get("clip_start_time"))
    clip_end_time = _coerce_optional_float(raw_question.get("clip_end_time"))

    metadata_keys = {
        "question",
        "question_id",
        "question_time",
        "options",
        "correct_index",
        "clip_start_time",
        "clip_end_time",
    }
    metadata = {key: value for key, value in raw_question.items() if key not in metadata_keys}
    if uncertain_option is not None:
        metadata["uncertain_option"] = uncertain_option

    question_mode = raw_question.get("question_mode")
    if isinstance(question_mode, str):
        question_mode = question_mode.strip() or None
    else:
        question_mode = None

    sequence_prefixes = _normalize_sequence_mapping(raw_question.get("prefix"))

    return QuestionEntry(
        question_id=question_id,
        prompt=prompt,
        question_time=question_time,
        options=real_options,
        correct_answer_index=correct_answer_index,
        dont_know_index=dont_know_index,
        clip_start_time=clip_start_time,
        clip_end_time=clip_end_time,
        metadata=metadata,
        question_order=question_position + 1,
        question_mode=question_mode,
        sequence_prefixes=sequence_prefixes,
        is_native_binary=False,
        candidate=None,
        binary_answer=None,
    )


def _is_uncertain_option(raw_option: Dict[str, Any]) -> bool:
    label = str(raw_option.get("label", "")).lower()
    if "uncertain" in label or "idk" in label:
        return True
    sequence = raw_option.get("sequence")
    clip_path = raw_option.get("clip_path")
    return sequence is None and not clip_path


def _normalize_sequence(raw_sequence: Any) -> List[str]:
    if not isinstance(raw_sequence, list):
        return []
    return [str(token) for token in raw_sequence]


def _normalize_sequence_mapping(raw_mapping: Any) -> Dict[str, List[str]]:
    if not isinstance(raw_mapping, dict):
        return {}
    normalized: Dict[str, List[str]] = {}
    for key, value in raw_mapping.items():
        if not isinstance(key, str):
            continue
        tokens = _normalize_sequence(value)
        if tokens:
            normalized[key] = tokens
    return normalized


def _coerce_float(value: Any, context: str) -> float:
    if value is None:
        raise ValueError(f"Missing required float value for {context}.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {context}: {value}") from exc


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid optional float value: {value}") from exc


def _resolve_asset_path(raw_path: Any, asset_root: Optional[str]) -> str:
    """Join a relative manifest path to ``asset_root``."""
    if raw_path is None:
        return ""

    candidate = str(raw_path).strip()
    if not candidate:
        return ""

    if os.path.isabs(candidate) or not asset_root:
        return candidate

    normalized_root = os.path.expanduser(asset_root)
    return os.path.normpath(os.path.join(normalized_root, candidate))

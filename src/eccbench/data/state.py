"""Resumable run state.

The state file is JSON: which questions of each stream are done, and every
per-question result. It is rewritten atomically after each question, so
``--resume`` continues a stopped run without repeating work.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class RunStateManager:
    """Reads and writes the state file of one run; does nothing if ``state_path`` is empty."""

    def __init__(self, state_path: Optional[str], resume: bool) -> None:
        self.enabled = bool(state_path)
        self.path = Path(state_path) if state_path else None
        self.state: Dict[str, Dict[str, Any]] = {"videos": {}, "question_results": []}
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.path.exists():
            try:
                self.state = json.loads(self.path.read_text())
                self.state.setdefault("question_results", [])
            except Exception:
                logging.warning("Failed to load existing state from %s; starting fresh.", self.path)
                self.state = {"videos": {}, "question_results": []}

    def _video_key(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> str:
        # Video indices repeat across buckets, so the key includes the bucket.
        key = f"{bucket}:{video_index}" if bucket else str(video_index)
        if variant is not None:
            key = f"{key}_v{variant}"
        return key

    def _ensure_entry(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> Dict[str, Any]:
        videos = self.state.setdefault("videos", {})
        key = self._video_key(video_index, variant, bucket)
        entry = videos.get(key)
        if entry is None:
            entry = {"questions": {}, "completed": False}
            videos[key] = entry
        entry.setdefault("questions", {})
        entry.setdefault("completed", False)
        entry.setdefault("bucket", bucket)
        entry.setdefault("video_index", video_index)
        entry.setdefault("variant", variant)
        return entry

    def get_completed_questions(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> Set[str]:
        if not self.enabled:
            return set()
        entry = self.state.get("videos", {}).get(self._video_key(video_index, variant, bucket))
        if not entry:
            return set()
        questions = entry.get("questions", {})
        return {qid for qid, status in questions.items() if status}

    def mark_question_complete(self, video_index: int, question_id: str, result: Optional[Dict[str, Any]] = None, variant: Optional[int] = None, bucket: Optional[str] = None) -> None:
        if not self.enabled:
            return
        entry = self._ensure_entry(video_index, variant, bucket)
        entry["questions"][question_id] = True
        if result is not None:
            result_data = {
                'video_id': result.get('video_index'),
                'video_index': video_index,
                'variant': variant if variant is not None else 0,
                'bucket': bucket,
                'question_id': result.get('question_id'),
                'question_order': result.get('question_order'),
                'predicted': result.get('predicted'),
                'correct': result.get('correct'),
                'is_correct': result.get('is_correct'),
                'is_dont_know': result.get('is_dont_know'),
                'response': result.get('response', ''),
                'video_entropy': result.get('entropy_prefix_mean'),
                'entropy_prefix_mean': result.get('entropy_prefix_mean'),
                'entropy_empirical_bits': result.get('entropy_empirical_bits'),
                'entropy_empirical_mean': result.get('entropy_empirical_mean'),
                'entropy_analytic_bits': result.get('entropy_analytic_bits'),
                'entropy_analytic_mean': result.get('entropy_analytic_mean'),
                'prefix_match_fraction': result.get('prefix_match_fraction'),
                'correct_option_likelihood': result.get('correct_option_likelihood'),
                'num_options': result.get('num_options'),
                'frames_seen_before_question': result.get('frames_seen_before_question'),
                'latency_ask_question': result.get('latency_ask_question'),
                'peak_gpu_mem_ask_question': result.get('peak_gpu_mem_ask_question'),
                'question_time': result.get('question_time'),
                'question_mode': result.get('question_mode'),
                'sequence_format': result.get('sequence_format'),
                'video_only_mode': result.get('video_only_mode'),
                'eval_mode': result.get('eval_mode'),
                'input_mode': result.get('input_mode'),
                'question_variant': result.get('question_variant'),
                'question_type': result.get('question_type'),
                'max_tokens': result.get('max_tokens'),
                'max_frames': result.get('max_frames'),
                'response_token_count': result.get('response_token_count'),
                'output_was_truncated': result.get('output_was_truncated'),
                'saw_all_frames': result.get('saw_all_frames'),
            }
            question_results = self.state.setdefault("question_results", [])
            sig = (
                result_data.get('question_id'),
                result_data.get('video_index'),
                result_data.get('bucket'),
                result_data.get('question_time'),
                result_data.get('entropy_prefix_mean'),
            )
            already_exists = any(
                sig == (
                    existing.get('question_id'),
                    existing.get('video_index'),
                    existing.get('bucket'),
                    existing.get('question_time'),
                    existing.get('entropy_prefix_mean'),
                )
                for existing in question_results
            )
            if not already_exists:
                question_results.append(result_data)
        self._save()

    def get_saved_results(self) -> List[Dict[str, Any]]:
        """Return all previously saved question results, for resume."""
        if not self.enabled:
            return []
        return self.state.get("question_results", [])

    def mark_video_complete(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> None:
        if not self.enabled:
            return
        entry = self._ensure_entry(video_index, variant, bucket)
        entry["completed"] = True
        self._save()

    def is_video_completed(self, video_index: int, variant: Optional[int] = None, bucket: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        entry = self.state.get("videos", {}).get(self._video_key(video_index, variant, bucket))
        return bool(entry and entry.get("completed"))

    def flush(self) -> None:
        if not self.enabled:
            return
        self._save()

    def _save(self) -> None:
        if not self.enabled or self.path is None:
            return
        tmp_path = self.path.with_suffix('.tmp')
        tmp_path.write_text(json.dumps(self.state, indent=2))
        os.replace(tmp_path, self.path)

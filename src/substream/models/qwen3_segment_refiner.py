"""Segment-refining wrapper around Qwen3-VL (8B), default checkpoint
Qwen/Qwen3-VL-8B-Instruct.

This adapter runs an iterative localisation pass over the main video before
asking the final question. The localisation repeatedly queries the base
Qwen3-VL instruct model on short slices of the main video (paired with the
full option/binary clips) and keeps only the slices that the model marks as
containing the relevant option. The retained slices are stitched together,
resampled with a smaller frame budget, and queried again until the clip set
collapses to a single candidate or the combined clip length drops below a
threshold. The final answer is generated on the filtered clip.

The class inherits only :class:`VideoLanguageModelInterface`, but internally
mirrors the behaviour of the dense Qwen3-VL wrapper for additions to the
context.
"""

from __future__ import annotations

import copy
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata

from substream.models.base import VideoLanguageModelInterface
from substream import config


config.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = str(config.MODEL_CACHE)
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.makedirs(CACHE_ROOT, exist_ok=True)


@dataclass
class _DetectionOutcome:
    """Structured result for a localisation query."""

    positive: bool
    raw_value: str
    candidates: List[int]
    dont_know: bool
    response: str
    confidence: Optional[float] = None


@dataclass
class _PromptPayload:
    """Material required to generate a single model response."""

    text: str
    videos: List[torch.Tensor] = field(default_factory=list)
    metadata: List[VideoMetadata] = field(default_factory=list)
    fps: Optional[float] = None


class Qwen3SegmentRefiner(VideoLanguageModelInterface):
    """Iterative localisation wrapper around Qwen3-VL Instruct."""

    INSTRUCT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    DEFAULT_FRAME_COUNT = 50
    DEFAULT_MIN_FRAMES = 150
    DEFAULT_FRAME_DECAY = 0.95
    DEFAULT_MAX_REFINEMENT_ROUNDS = 6
    DEFAULT_REFINEMENT_BATCH_SIZE = 1
    PRINT_FULL_RESPONSES = False
    _OPTION_LABEL_PATTERN = re.compile(r"^option\s*\d+", re.IGNORECASE)

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        enable_metrics: bool = False,
        verbose: bool = False,
        frame_count: Optional[int] = None,
        min_frames_threshold: Optional[int] = None,
        frame_count_decay: Optional[float] = None,
        max_refinement_rounds: Optional[int] = None,
        binary_questions: bool = False,
        refinement_batch_size: Optional[int] = None,
        max_gpu_mem: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        resolved_id = model_id or self.INSTRUCT_MODEL_ID
        init_kwargs = dict(kwargs)
        init_kwargs["verbose"] = verbose
        if frame_count is not None:
            init_kwargs["frame_count"] = frame_count
        if min_frames_threshold is not None:
            init_kwargs["min_frames_threshold"] = min_frames_threshold
        if frame_count_decay is not None:
            init_kwargs["frame_count_decay"] = frame_count_decay
        if max_refinement_rounds is not None:
            init_kwargs["max_refinement_rounds"] = max_refinement_rounds
        init_kwargs["binary_questions"] = binary_questions
        if refinement_batch_size is not None:
            init_kwargs["refinement_batch_size"] = refinement_batch_size
        super().__init__(
            resolved_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **init_kwargs,
        )

    # --------------------------- setup & context ops ---------------------------

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs: Any) -> None:  # type: ignore[override]
        self.verbose: bool = bool(kwargs.pop("verbose", False))
        self.binary_mode: bool = bool(kwargs.pop("binary_questions", False))

        self.initial_frame_count: int = int(
            kwargs.pop("frame_count", self.DEFAULT_FRAME_COUNT)
        )
        self.min_total_frames: int = int(
            kwargs.pop("min_frames_threshold", self.DEFAULT_MIN_FRAMES)
        )
        self.frame_count_decay: float = float(
            kwargs.pop("frame_count_decay", self.DEFAULT_FRAME_DECAY)
        )
        self.max_refinement_rounds: int = int(
            kwargs.pop("max_refinement_rounds", self.DEFAULT_MAX_REFINEMENT_ROUNDS)
        )
        refinement_batch_size = int(
            kwargs.pop("refinement_batch_size", self.DEFAULT_REFINEMENT_BATCH_SIZE)
        )
        if refinement_batch_size <= 0:
            raise ValueError("refinement_batch_size must be a positive integer")
        self.refinement_batch_size: int = refinement_batch_size
        self.verbose_full_responses: bool = bool(self.PRINT_FULL_RESPONSES)

        min_pixels: Optional[int] = kwargs.get("min_pixels")
        max_pixels: Optional[int] = kwargs.get("max_pixels")

        model_kwargs: Dict[str, Any] = {
            "dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "cache_dir": CACHE_ROOT,
        }
        # Qwen3VL loads rely on automatic device placement and ignore max_gpu_mem.

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        processor_kwargs: Dict[str, Any] = {"cache_dir": CACHE_ROOT}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            **processor_kwargs,
        )

        self.latest_time: float = 0.0
        self.video_segments: Dict[int, List[Dict[str, Any]]] = {}
        self.text_entries: List[Tuple[str, float]] = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

        if hasattr(self, "_reset_state_memory_tracking"):
            self._reset_state_memory_tracking()

    # Add video -----------------------------------------------------------------

    def add_video(
        self,
        video_frames: Union[np.ndarray, torch.Tensor],
        time_start: float,
        time_end: float,
        video_id: Optional[int] = None,
    ) -> None:
        metrics_start: Optional[float] = None
        baseline_mem: float = 0.0
        if self.enable_metrics:
            metrics_start = time.perf_counter()
            if torch.cuda.is_available():
                baseline_mem = float(torch.cuda.memory_allocated())
                torch.cuda.reset_peak_memory_stats()

        if time_start >= time_end:
            raise ValueError("time_end must be greater than time_start")
        if time_start < self.latest_time:
            raise ValueError("time_start must follow the previously added segment")

        self.latest_time = time_end

        if isinstance(video_frames, np.ndarray):
            frames_tensor = torch.from_numpy(video_frames)
        else:
            frames_tensor = video_frames

        if not torch.is_tensor(frames_tensor):
            raise TypeError("video_frames must be a numpy array or torch tensor")

        frames_tensor = frames_tensor.clone()
        # Use input embeddings device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device
        frames_tensor = frames_tensor.to(input_device)

        num_frames = int(frames_tensor.shape[0])
        if num_frames <= 0:
            raise ValueError("video_frames must contain at least one frame")

        duration = float(time_end - time_start)
        fps = float(num_frames / duration)

        if video_id is None:
            video_id = 0

        if video_id not in self.video_segments:
            self.video_segments[video_id] = []

        segment = {
            "frames": frames_tensor,
            "time_start": float(time_start),
            "time_end": float(time_end),
            "duration": duration,
            "num_frames": num_frames,
            "fps": fps,
        }
        self.video_segments[video_id].append(segment)

        if self.enable_metrics:
            assert metrics_start is not None  # for mypy
            latency = time.perf_counter() - metrics_start
            peak_bytes = (
                float(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else baseline_mem
            )
            peak_mem_increase_mb = max(0.0, peak_bytes - baseline_mem) / (1024 * 1024)
            peak_mem_absolute_mb = (
                peak_bytes / (1024 * 1024)
                if torch.cuda.is_available()
                else 0.0
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            state_mem = self._get_state_memory_floats()
            self._record_add_video_metrics(
                latency,
                0.0,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                time_end,
                state_mem,
            )

    # Add text ------------------------------------------------------------------

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        metrics_start: Optional[float] = None
        baseline_mem: float = 0.0
        if self.enable_metrics:
            metrics_start = time.perf_counter()
            if torch.cuda.is_available():
                baseline_mem = float(torch.cuda.memory_allocated())
                torch.cuda.reset_peak_memory_stats()

        timestamp = float(self.latest_time)
        self.text_entries.append((text, timestamp))

        if self.enable_metrics:
            assert metrics_start is not None  # for mypy
            latency = time.perf_counter() - metrics_start
            peak_bytes = (
                float(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else baseline_mem
            )
            peak_mem_increase_mb = max(0.0, peak_bytes - baseline_mem) / (1024 * 1024)
            peak_mem_absolute_mb = (
                peak_bytes / (1024 * 1024)
                if torch.cuda.is_available()
                else 0.0
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            token_count = self._count_text_tokens(text)
            _ = token_count  # explicitly unused but kept for symmetry

            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                0.0,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                state_mem,
            )

    # ------------------------------ ask question ------------------------------

    def ask_question(
        self,
        question: str,
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> str:
        self._video_was_truncated = False  # Reset flag
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        aggregate_data = self._aggregate_video_segments(self.video_segments)
        main_data = aggregate_data.get(0)
        option_ids = [vid for vid in aggregate_data.keys() if vid != 0]

        if not main_data or main_data["frames"] is None:
            # Fallback: no localisation possible
            response = self._generate_response(
                question,
                self.video_segments,
                self.text_entries,
                max_tokens,
                max_frames_in_video,
                sample_method,
            )
            if self.enable_metrics:
                latency = time.perf_counter() - start_time
                peak_mem = (
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                )
                peak_mem_increase_mb = max(0, peak_mem - baseline_mem) / (1024 * 1024)
                peak_mem_absolute_mb = (
                    peak_mem / (1024 * 1024)
                    if torch.cuda.is_available()
                    else 0.0
                )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                state_mem = self._get_state_memory_floats()
                self._record_ask_question_metrics(
                    latency,
                    0.0,
                    peak_mem_increase_mb,
                    peak_mem_absolute_mb,
                    current_video_time,
                    state_mem,
                )
            return response

        option_data = {
            vid: aggregate_data[vid]
            for vid in option_ids
            if aggregate_data[vid]["frames"] is not None
        }

        option_labels, extra_texts = self._extract_text_templates(len(option_data))

        retained_segments = self._initial_localisation(
            main_data,
            option_data,
            option_labels,
            extra_texts,
            max_tokens,
            max_frames_in_video,
            sample_method,
        )

        refined_segments = self._refine_segments(
            retained_segments,
            main_data,
            option_data,
            option_labels,
            extra_texts,
            max_tokens,
            max_frames_in_video,
            sample_method,
        )

        video_context, text_context = self._build_context(
            refined_segments,
            option_data,
            option_labels,
            extra_texts,
        )

        response = self._generate_response(
            question,
            video_context,
            text_context,
            max_tokens,
            max_frames_in_video,
            sample_method,
        )

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            peak_mem_increase_mb = max(0, peak_mem - baseline_mem) / (1024 * 1024)
            peak_mem_absolute_mb = (
                peak_mem / (1024 * 1024)
                if torch.cuda.is_available()
                else 0.0
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            state_mem = self._get_state_memory_floats()
            self._record_ask_question_metrics(
                latency,
                0.0,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                state_mem,
            )

        return response

    # ------------------------- localisation helpers -------------------------

    def _initial_localisation(
        self,
        main_data: Dict[str, Any],
        option_data: Dict[int, Dict[str, Any]],
        option_labels: List[str],
        extra_texts: List[str],
        max_tokens: int,
        max_frames: int,
        sample_method: str,
    ) -> List[Dict[str, Any]]:
        frames_tensor: torch.Tensor = main_data["frames"]
        fps: Optional[float] = main_data.get("fps")
        required_window = self._max_option_frame_length(option_data)
        window_size = max(self.initial_frame_count, required_window)
        initial_segments = self._split_frames_into_segments(
            frames_tensor,
            window_size,
            fps,
            base_offset=0,
        )

        if not initial_segments:
            return self._build_segments_from_ranges(
                frames_tensor,
                fps,
                [(0, int(frames_tensor.shape[0]))],
            )

        if self.verbose:
            print(
                f"[round 0] begin: segments={len(initial_segments)} "
                f"window={window_size} required>={required_window}"
            )

        detection_question = self._build_detection_prompt(len(option_data))

        outcomes = self._evaluate_segments_batch(
            initial_segments,
            option_data,
            option_labels,
            extra_texts,
            detection_question,
            max_tokens,
            max_frames,
            sample_method,
        )

        retained_segments: List[Dict[str, Any]] = []
        for index, (segment, outcome) in enumerate(zip(initial_segments, outcomes)):
            if self.verbose:
                self._log_detection(index, segment, outcome)
            if outcome.positive:
                retained_segments.append(segment)

        if not retained_segments:
            if self.verbose:
                print("[round 0] end: retained 0 segments; using full clip")
            return self._build_segments_from_ranges(
                frames_tensor,
                fps,
                [(0, int(frames_tensor.shape[0]))],
            )

        if self.verbose:
            merged_preview = self._merge_frame_ranges(retained_segments)
            print(
                f"[round 0] end: retained {len(retained_segments)} segments -> ranges={merged_preview}"
            )

        merged_ranges = self._merge_frame_ranges(retained_segments)
        return self._build_segments_from_ranges(frames_tensor, fps, merged_ranges)

    def _refine_segments(
        self,
        retained_segments: List[Dict[str, Any]],
        main_data: Dict[str, Any],
        option_data: Dict[int, Dict[str, Any]],
        option_labels: List[str],
        extra_texts: List[str],
        max_tokens: int,
        max_frames: int,
        sample_method: str,
    ) -> List[Dict[str, Any]]:
        frames_tensor = main_data["frames"]
        fps = main_data.get("fps")
        if not retained_segments:
            return self._build_segments_from_ranges(
                frames_tensor,
                fps,
                [(0, int(frames_tensor.shape[0]))],
            )

        current_ranges = self._merge_frame_ranges(retained_segments)
        current_segments = self._build_segments_from_ranges(frames_tensor, fps, current_ranges)
        required_window = self._max_option_frame_length(option_data)
        current_frame_target = max(required_window, self.initial_frame_count)
        iteration = 0

        while True:
            total_frames = sum(end - start for start, end in current_ranges)
            if len(current_ranges) <= 1 or total_frames <= self.min_total_frames:
                break
            if iteration >= self.max_refinement_rounds:
                break

            iteration += 1
            proposed_target = int(math.floor(current_frame_target * self.frame_count_decay))
            if proposed_target >= current_frame_target and current_frame_target > required_window:
                proposed_target = current_frame_target - 1
            current_frame_target = max(required_window, proposed_target)

            new_segments: List[Dict[str, Any]] = []
            for start, end in current_ranges:
                chunk = frames_tensor[start:end]
                new_segments.extend(
                    self._split_frames_into_segments(
                        chunk,
                        current_frame_target,
                        fps,
                        base_offset=start,
                    )
                )

            if not new_segments:
                current_segments = self._build_segments_from_ranges(
                    frames_tensor,
                    fps,
                    current_ranges,
                )
                break

            if self.verbose:
                print(
                    f"[round {iteration}] begin: segments={len(new_segments)} "
                    f"window={current_frame_target} required>={required_window}"
                )

            detection_question = self._build_detection_prompt(len(option_data))
            outcomes = self._evaluate_segments_batch(
                new_segments,
                option_data,
                option_labels,
                extra_texts,
                detection_question,
                max_tokens,
                max_frames,
                sample_method,
            )

            retained_this_round: List[Dict[str, Any]] = []
            for index, (segment, outcome) in enumerate(zip(new_segments, outcomes)):
                if self.verbose:
                    self._log_detection(iteration * 100 + index, segment, outcome)
                if outcome.positive:
                    retained_this_round.append(segment)

            if not retained_this_round:
                current_ranges = self._merge_frame_ranges(new_segments)
                current_segments = self._build_segments_from_ranges(
                    frames_tensor,
                    fps,
                    current_ranges,
                )
                if self.verbose:
                    print(
                        f"[round {iteration}] end: retained 0 segments; "
                        "carrying latest ranges forward"
                    )
                break

            current_ranges = self._merge_frame_ranges(retained_this_round)
            current_segments = self._build_segments_from_ranges(
                frames_tensor,
                fps,
                current_ranges,
            )

            if self.verbose:
                print(
                    f"[round {iteration}] end: retained {len(current_segments)} segments -> "
                    f"ranges={current_ranges}"
                )

        if self.verbose:
            final_ranges = self._merge_frame_ranges(current_segments) if current_segments else []
            print(
                f"[round final] result: segments={len(current_segments)} -> ranges={final_ranges}"
            )

        return current_segments

    # -------------------------- evaluation helpers --------------------------

    def _evaluate_segment(
        self,
        segment: Dict[str, Any],
        option_data: Dict[int, Dict[str, Any]],
        option_labels: List[str],
        extra_texts: List[str],
        question: str,
        max_tokens: int,
        max_frames: int,
        sample_method: str,
    ) -> _DetectionOutcome:
        outcomes = self._evaluate_segments_batch(
            [segment],
            option_data,
            option_labels,
            extra_texts,
            question,
            max_tokens,
            max_frames,
            sample_method,
        )
        return outcomes[0]

    def _evaluate_segments_batch(
        self,
        segments: List[Dict[str, Any]],
        option_data: Dict[int, Dict[str, Any]],
        option_labels: List[str],
        extra_texts: List[str],
        question: str,
        max_tokens: int,
        max_frames: int,
        sample_method: str,
    ) -> List[_DetectionOutcome]:
        if not segments:
            return []

        outcomes: List[_DetectionOutcome] = []
        batch_size = self.refinement_batch_size
        for start in range(0, len(segments), batch_size):
            batch_segments = segments[start : start + batch_size]
            payloads: List[_PromptPayload] = []
            for segment in batch_segments:
                video_context, text_context = self._build_context(
                    [segment],
                    option_data,
                    option_labels,
                    extra_texts,
                )
                payload = self._prepare_prompt_payload(
                    question,
                    video_context,
                    text_context,
                    max_frames,
                    sample_method,
                )
                payloads.append(payload)

            responses = self._generate_batch_responses(payloads, max_tokens)
            for response in responses:
                outcomes.append(
                    self._parse_detection_response(
                        response,
                        len(option_data),
                    )
                )
        return outcomes

    def _build_detection_prompt(self, option_count: int) -> str:
        if self.binary_mode and option_count <= 1:
            return (
                "You will watch a short segment of the main video followed by a single "
                "option clip. Most segments will not contain this option. First write at "
                "least two complete sentences describing concrete similarities and "
                "differences between the segment and the option. Only accept a match when "
                "the option appears somewhere in the segment with identical details (objects, "
                "colours, motion, ordering, and timing). After your explanation, write `Answer: {yes}` "
                "if and only if the option is an exact match; otherwise write `Answer: {no}`. "
                "Immediately append `[confidence: X/10]` where X is a number from 0 (no confidence) to 10 "
                "(absolute certainty)."
            )

        dont_know_option = option_count
        return (
            "You will watch a short segment of the main video followed by all option videos. "
            "It is unlikely that any option appears. For each comparison, describe what you see "
            "in the segment versus the option, explicitly calling out the decisive similarities "
            "and differences in at least two full sentences before giving your decision. Only "
            "accept a match when the option clip appears somewhere in the segment with identical "
            "details (including ordering, colours, objects, and timing)—partial similarity is not enough. "
            "After your explanation, end with `Answer: {k}` where {k} is the matching option index "
            "(for example {{0}}) when you are certain of an exact match. If none of the options match, end "
            f"with `Answer: {{{dont_know_option}}}`. Immediately append `[confidence: X/10]` where X is a number "
            "from 0 (no confidence) to 10 (absolute certainty)."
        )

    def _append_confidence_instruction(self, question: str) -> str:
        normalized = question.lower()
        if "[confidence" in normalized:
            return question

        addition = (
            "After presenting your final answer inside curly braces, append `[confidence: X/10]` "
            "where X is a number from 0 (no confidence) to 10 (absolute certainty). Decimals are allowed."
        )

        stripped = question.rstrip()
        if not stripped:
            return addition

        if stripped.endswith((".", "?", "!")):
            return f"{stripped} {addition}"

        return f"{stripped}. {addition}"

    def _build_context(
        self,
        main_segments: List[Dict[str, Any]],
        option_data: Dict[int, Dict[str, Any]],
        option_labels: List[str],
        extra_texts: List[str],
    ) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Tuple[str, float]]]:
        video_map: Dict[int, List[Dict[str, Any]]] = {}
        text_entries: List[Tuple[str, float]] = []

        running_time = 0.0
        arranged_segments: List[Dict[str, Any]] = []
        for segment in main_segments:
            duration = segment.get("duration")
            fps = segment.get("fps")
            if duration is None or duration <= 0:
                if fps is None or fps <= 0:
                    raise ValueError("Main video segment is missing duration/fps metadata")
                duration = segment["num_frames"] / float(fps)
            seg_copy = self._create_segment_dict(
                segment["frames"],
                running_time,
                duration=float(duration),
                fps=fps,
                frame_start_index=segment.get("frame_start_index"),
                frame_end_index=segment.get("frame_end_index"),
            )
            arranged_segments.append(seg_copy)
            running_time = seg_copy["time_end"]

        video_map[0] = arranged_segments

        current_time = running_time + 0.25

        # Place extra text (non option labels) first to preserve instructions.
        for text in extra_texts:
            text_entries.append((text, current_time))
            current_time += 0.25

        sorted_options = sorted(option_data.items(), key=lambda kv: kv[0])
        for idx, (video_id, data) in enumerate(sorted_options):
            label = (
                option_labels[idx]
                if idx < len(option_labels)
                else f"Option {idx}:"
            )
            text_entries.append((label, current_time))
            current_time += 0.1

            frames = data["frames"]
            fps = data.get("fps")
            duration = data.get("duration")
            if duration is None or duration <= 0:
                if fps is None or fps <= 0:
                    raise ValueError("Option video segment is missing duration/fps metadata")
                duration = frames.shape[0] / float(fps)

            segment_dict = self._create_segment_dict(
                frames,
                current_time,
                duration=float(duration),
                fps=fps,
                frame_start_index=data.get("frame_start_index"),
                frame_end_index=data.get("frame_end_index"),
            )
            video_map[video_id] = [segment_dict]
            current_time = segment_dict["time_end"] + 0.35

        return video_map, text_entries

    # --------------------------- generation helper ---------------------------

    def _prepare_prompt_payload(
        self,
        question: str,
        video_segments: Dict[int, List[Dict[str, Any]]],
        text_entries: List[Tuple[str, float]],
        max_frames_in_video: int,
        sample_method: str,
    ) -> _PromptPayload:
        timeline_events: List[Dict[str, Any]] = []
        for video_id, segments in video_segments.items():
            for segment in segments:
                timeline_events.append(
                    {
                        "type": "video",
                        "time": segment.get("time_start", 0.0),
                        "video_id": video_id,
                        "segment": segment,
                    }
                )

        for text, timestamp in text_entries:
            timeline_events.append({"type": "text", "time": timestamp, "text": text})

        timeline_events.sort(key=lambda x: (x["time"], x["type"] == "video"))

        content: List[Dict[str, Any]] = []
        video_inputs: List[torch.Tensor] = []
        video_metadata_entries: List[VideoMetadata] = []
        video_fps: List[Optional[float]] = []

        video_data: Dict[int, Dict[str, Any]] = {}
        for event in timeline_events:
            if event["type"] != "video":
                continue
            video_id = event["video_id"]
            segment = event["segment"]
            store = video_data.setdefault(
                video_id,
                {
                    "frames": None,
                    "total_duration": 0.0,
                    "total_frames": 0,
                    "fps_values": [],
                },
            )

            frames = segment["frames"]
            if store["frames"] is None:
                store["frames"] = frames
            else:
                store["frames"] = torch.cat([store["frames"], frames], dim=0)

            duration = float(segment.get("duration", 0.0))
            num_frames = int(segment.get("num_frames", frames.shape[0]))
            if num_frames <= 0:
                raise ValueError(f"Video {video_id} contains an empty segment")

            fps_value = segment.get("fps")
            if duration <= 0:
                if fps_value is None or fps_value <= 0:
                    raise ValueError(
                        f"Video {video_id} segment is missing duration and fps metadata"
                    )
                duration = num_frames / float(fps_value)

            if duration <= 0:
                raise ValueError(f"Video {video_id} segment reported non-positive duration")

            store["total_duration"] += duration
            store["total_frames"] += num_frames
            if fps_value is None or fps_value <= 0:
                fps_value = num_frames / duration
            store["fps_values"].append(float(fps_value))

        for video_id, data in video_data.items():
            frames = data["frames"]
            if frames is None or frames.shape[0] == 0:
                continue

            frame_count = int(frames.shape[0])
            selection = np.arange(frame_count, dtype=int)
            if frame_count > max_frames_in_video:
                self._video_was_truncated = True  # Track truncation
                if sample_method == "RANDOM":
                    indices = np.sort(
                        np.random.choice(frame_count, max_frames_in_video, replace=False)
                    )
                else:
                    indices = np.linspace(0, frame_count - 1, max_frames_in_video, dtype=int)
                frames = frames[indices]
                selection = selection[indices]
            trimmed_frame_count = int(frames.shape[0])
            if trimmed_frame_count <= 0:
                raise ValueError(f"Video {video_id} lost all frames during sampling")

            fps_reference = self._coalesce_video_fps(data.get("fps_values", []))
            if fps_reference is None or fps_reference <= 0:
                raise ValueError(f"Video {video_id} is missing fps metadata")

            safe_duration = trimmed_frame_count / fps_reference

            data["frames"] = frames
            data["selection"] = selection
            data["total_duration"] = safe_duration
            data["fps"] = float(fps_reference)

        processed_video_ids: set[int] = set()
        for event in timeline_events:
            if event["type"] == "video":
                video_id = event["video_id"]
                if video_id in processed_video_ids:
                    continue

                frames_tensor = video_data[video_id]["frames"]
                if frames_tensor is None or frames_tensor.shape[0] == 0:
                    processed_video_ids.add(video_id)
                    continue

                video_inputs.append(frames_tensor)
                frame_count = int(frames_tensor.shape[0])
                duration = float(video_data[video_id].get("total_duration", 0.0))
                if duration <= 0:
                    raise ValueError(f"Video {video_id} is missing duration metadata")
                fps_value = video_data[video_id].get("fps")
                if fps_value is None or fps_value <= 0:
                    fps_value = frame_count / duration
                fps_value = float(fps_value)
                height = int(frames_tensor.shape[2]) if frames_tensor.ndim >= 3 else None
                width = int(frames_tensor.shape[3]) if frames_tensor.ndim >= 4 else None
                frame_indices = (
                    video_data[video_id].get("selection")
                    if video_data[video_id].get("selection") is not None
                    else list(range(frame_count))
                )

                metadata = VideoMetadata(
                    total_num_frames=frame_count,
                    fps=fps_value,
                    width=width,
                    height=height,
                    duration=duration,
                    video_backend="tensor",
                    frames_indices=[int(idx) for idx in frame_indices],
                )
                video_metadata_entries.append(metadata)
                video_fps.append(fps_value)

                content.append({"type": "video", "video": len(video_inputs) - 1})
                processed_video_ids.add(video_id)

            elif event["type"] == "text":
                content.append({"type": "text", "text": event["text"]})

        final_question = self._append_confidence_instruction(question)
        content.append({"type": "text", "text": final_question})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        coalesced_fps = self._coalesce_video_fps(video_fps)
        return _PromptPayload(
            text=text,
            videos=video_inputs,
            metadata=video_metadata_entries,
            fps=coalesced_fps,
        )

    def _generate_batch_responses(
        self,
        payloads: List[_PromptPayload],
        max_tokens: int,
    ) -> List[str]:
        if not payloads:
            return []

        texts = [payload.text for payload in payloads]
        has_video_payloads = any(payload.videos for payload in payloads)
        if has_video_payloads and not all(payload.videos for payload in payloads):
            raise ValueError(
                "Mixed video/text prompts detected; batch inputs must be homogeneous."
            )

        processor_kwargs: Dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }

        if len(payloads) == 1:
            payload = payloads[0]
            if payload.videos:
                processor_kwargs["videos"] = payload.videos
                if payload.metadata:
                    processor_kwargs["video_metadata"] = payload.metadata
                if payload.fps is not None:
                    processor_kwargs["fps"] = payload.fps
        else:
            if has_video_payloads:
                processor_kwargs["videos"] = [payload.videos for payload in payloads]
                if any(payload.metadata for payload in payloads):
                    processor_kwargs["video_metadata"] = [
                        payload.metadata for payload in payloads
                    ]

                fps_values = [payload.fps for payload in payloads if payload.fps is not None]
                if len(fps_values) != len(payloads):
                    raise ValueError("Mixed missing FPS metadata across batched prompts")
                coalesced = self._coalesce_video_fps(fps_values)
                if coalesced is None:
                    raise ValueError("Unable to coalesce FPS values across batched prompts")
                processor_kwargs["fps"] = coalesced
        
        inputs = self.processor(**processor_kwargs)
        # Use input embeddings device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def _generate_response(
        self,
        question: str,
        video_segments: Dict[int, List[Dict[str, Any]]],
        text_entries: List[Tuple[str, float]],
        max_tokens: int,
        max_frames_in_video: int,
        sample_method: str,
    ) -> str:
        payload = self._prepare_prompt_payload(
            question,
            video_segments,
            text_entries,
            max_frames_in_video,
            sample_method,
        )
        responses = self._generate_batch_responses([payload], max_tokens)
        return responses[0] if responses else ""

    # ----------------------------- parsing helpers -----------------------------

    def _extract_confidence_score(self, response: str) -> Optional[float]:
        bracket_chunks = re.findall(r"\[([^\]]+)\]", response)
        if not bracket_chunks:
            return None

        def _clamp(value: float) -> float:
            if value < 0.0:
                return 0.0
            if value > 10.0:
                return 10.0
            return value

        for chunk in reversed(bracket_chunks):
            match = re.search(
                r"confidence[^0-9]*(\d+(?:\.\d+)?)\s*(?:/\s*10)?",
                chunk,
                re.IGNORECASE,
            )
            if match:
                try:
                    return _clamp(float(match.group(1)))
                except ValueError:
                    continue

            numeric_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:/\s*10)?", chunk)
            if numeric_match:
                try:
                    return _clamp(float(numeric_match.group(1)))
                except ValueError:
                    continue

        return None

    def _parse_detection_response(
        self,
        response: str,
        option_count: int,
    ) -> _DetectionOutcome:
        confidence = self._extract_confidence_score(response)
        matches = re.findall(r"\{([^}]+)\}", response)
        if not matches:
            return _DetectionOutcome(False, "", [], False, response, confidence)

        raw_value = matches[-1].strip().lower()
        if self.binary_mode and option_count <= 1:
            if raw_value in {"yes", "y", "true", "1"}:
                return _DetectionOutcome(True, raw_value, [0], False, response, confidence)
            if raw_value in {"no", "n", "false", "0"}:
                return _DetectionOutcome(False, raw_value, [], False, response, confidence)
            return _DetectionOutcome(False, raw_value, [], False, response, confidence)

        numeric_matches = re.findall(r"-?\d+", raw_value)
        candidates = []
        dont_know = False
        for token in numeric_matches:
            idx = int(token)
            if 0 <= idx < option_count:
                candidates.append(idx)
            elif idx == option_count:
                dont_know = True

        positive = bool(candidates)
        if not positive and any(word in raw_value for word in ["none", "no", "neither"]):
            dont_know = True

        return _DetectionOutcome(
            positive,
            raw_value,
            sorted(set(candidates)),
            dont_know,
            response,
            confidence,
        )

    def _log_detection(
        self,
        index: int,
        segment: Dict[str, Any],
        outcome: _DetectionOutcome,
    ) -> None:
        duration = segment.get("duration", 0.0)
        frame_count = segment.get("num_frames", 0)
        if outcome.positive:
            status = "match"
        elif outcome.dont_know:
            status = "uncertain"
        else:
            status = "miss"
        candidates = ", ".join(str(c) for c in outcome.candidates) if outcome.candidates else "-"
        confidence_display = (
            f"{outcome.confidence:.1f}/10" if outcome.confidence is not None else "-"
        )
        print(
            f"[refine] segment {index}: {frame_count} frames / {duration:.2f}s -> {status} "
            f"(candidates: {candidates}; confidence: {confidence_display}; raw='{outcome.raw_value}')"
        )
        if self.verbose_full_responses:
            response_text = outcome.response.strip()
            if response_text:
                for line in response_text.splitlines():
                    print(f"         response: {line}")

    # ----------------------------- data utilities -----------------------------

    def _aggregate_video_segments(
        self, video_segments: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[int, Dict[str, Any]]:
        aggregate: Dict[int, Dict[str, Any]] = {}
        for video_id, segments in video_segments.items():
            frames_list: List[torch.Tensor] = []
            total_duration = 0.0
            total_frames = 0
            fps_samples: List[Optional[float]] = []
            for segment in segments:
                frames = segment.get("frames")
                if frames is None:
                    continue
                frames_list.append(frames)
                frame_count = int(segment.get("num_frames", frames.shape[0]))
                if frame_count <= 0:
                    raise ValueError(f"Segment for video {video_id} has no frames")

                duration = float(segment.get("duration", 0.0))
                if duration <= 0:
                    fps_value = segment.get("fps")
                    if fps_value is not None and fps_value > 0:
                        duration = frame_count / float(fps_value)
                    else:
                        raise ValueError(
                            f"Segment for video {video_id} is missing positive duration"
                            " and fps metadata"
                        )
                total_duration += duration
                total_frames += frame_count
                fps_value = segment.get("fps")
                if fps_value is None or fps_value <= 0:
                    fps_value = frame_count / duration
                fps_samples.append(float(fps_value))

            if not frames_list:
                aggregate[video_id] = {
                    "frames": None,
                    "duration": total_duration,
                    "num_frames": total_frames,
                    "fps": None,
                }
                continue

            if total_duration <= 0 or total_frames <= 0:
                raise ValueError(f"Invalid aggregate metadata for video {video_id}")

            coalesced_fps = self._coalesce_video_fps(fps_samples)
            if coalesced_fps is None:
                raise ValueError(f"Missing FPS metadata for video {video_id}")

            frames_tensor = torch.cat(frames_list, dim=0)
            aggregate[video_id] = {
                "frames": frames_tensor,
                "duration": total_duration,
                "num_frames": total_frames,
                "fps": coalesced_fps,
            }
        return aggregate

    def _split_frames_into_segments(
        self,
        frames: torch.Tensor,
        frame_target: int,
        fps: Optional[float],
        *,
        base_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if frame_target <= 0:
            frame_target = frames.shape[0]
        total_frames = int(frames.shape[0])
        segments: List[Dict[str, Any]] = []
        if total_frames == 0:
            return segments

        if fps is None or fps <= 0:
            raise ValueError("Cannot split frames without positive FPS metadata")
        fps_value = float(fps)

        if frame_target < 1:
            raise ValueError("frame_target must be positive")

        step = frame_target // 2 if frame_target > 1 else 1
        step = max(1, step)

        start_positions: List[int]
        if total_frames <= frame_target:
            start_positions = [0]
        else:
            start_positions = []
            cursor = 0
            while cursor + frame_target < total_frames:
                start_positions.append(cursor)
                cursor += step
            final_start = max(total_frames - frame_target, 0)
            if not start_positions or start_positions[-1] != final_start:
                start_positions.append(final_start)

        for local_start in start_positions:
            local_end = min(local_start + frame_target, total_frames)
            chunk = frames[local_start:local_end]
            if chunk.shape[0] == 0:
                continue
            segment = self._create_segment_dict(
                chunk,
                time_start=None,
                fps=fps_value,
                frame_start_index=base_offset + local_start,
                frame_end_index=base_offset + local_end,
            )
            segments.append(segment)

        return segments

    def _create_segment_dict(
        self,
        frames: torch.Tensor,
        time_start: Optional[float] = None,
        *,
        duration: Optional[float] = None,
        fps: Optional[float] = None,
        frame_start_index: Optional[int] = None,
        frame_end_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        num_frames = int(frames.shape[0])
        if num_frames <= 0:
            raise ValueError("Segments must contain at least one frame")

        fps_value = float(fps) if fps is not None else None
        if fps_value is not None and fps_value <= 0:
            raise ValueError("Segment fps must be positive")

        if frame_start_index is not None and frame_end_index is None:
            frame_end_index = frame_start_index + num_frames
        if frame_start_index is not None:
            frame_start_index = int(frame_start_index)
        if frame_end_index is not None:
            frame_end_index = int(frame_end_index)

        inferred_duration = float(duration) if duration is not None else None
        if inferred_duration is None:
            if fps_value is None:
                raise ValueError(
                    "Segment creation requires a duration or positive fps value"
                )
            inferred_duration = num_frames / fps_value

        if inferred_duration <= 0:
            raise ValueError("Segment duration must be positive")

        if fps_value is None:
            fps_value = num_frames / inferred_duration
        if fps_value <= 0:
            raise ValueError("Segment fps must be positive")

        resolved_time_start: float
        if time_start is not None:
            resolved_time_start = float(time_start)
        elif frame_start_index is not None:
            resolved_time_start = frame_start_index / fps_value
        else:
            resolved_time_start = 0.0

        time_end = resolved_time_start + inferred_duration

        return {
            "frames": frames,
            "time_start": resolved_time_start,
            "time_end": time_end,
            "duration": float(inferred_duration),
            "num_frames": num_frames,
            "fps": float(fps_value),
            "frame_start_index": frame_start_index,
            "frame_end_index": frame_end_index
            if frame_end_index is not None
            else (
                frame_start_index + num_frames if frame_start_index is not None else None
            ),
        }

    def _merge_frame_ranges(
        self, segments: List[Dict[str, Any]]
    ) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        for segment in segments:
            start = segment.get("frame_start_index")
            end = segment.get("frame_end_index")
            if start is None or end is None:
                raise ValueError("Segment is missing frame index metadata")
            if end <= start:
                raise ValueError("Segment frame index range is empty")
            ranges.append((int(start), int(end)))

        if not ranges:
            return []

        ranges.sort(key=lambda item: item[0])
        merged: List[Tuple[int, int]] = []
        current_start, current_end = ranges[0]
        for start, end in ranges[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
                continue
            merged.append((current_start, current_end))
            current_start, current_end = start, end
        merged.append((current_start, current_end))
        return merged

    def _build_segments_from_ranges(
        self,
        frames: torch.Tensor,
        fps: Optional[float],
        ranges: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        if fps is None or fps <= 0:
            raise ValueError("FPS metadata is required to rebuild segments")
        segments: List[Dict[str, Any]] = []
        for start, end in ranges:
            if end <= start:
                continue
            chunk = frames[start:end]
            if chunk.shape[0] == 0:
                continue
            segment = self._create_segment_dict(
                chunk,
                time_start=None,
                fps=fps,
                frame_start_index=start,
                frame_end_index=end,
            )
            segments.append(segment)
        return segments

    def _max_option_frame_length(
        self,
        option_data: Dict[int, Dict[str, Any]],
    ) -> int:
        max_length = 0
        for data in option_data.values():
            frames = data.get("frames")
            if frames is None:
                continue
            max_length = max(max_length, int(frames.shape[0]))
        return max(1, max_length)

    def _extract_text_templates(
        self, option_count: int
    ) -> Tuple[List[str], List[str]]:
        option_labels: List[str] = []
        extras: List[str] = []
        for text, _ in self.text_entries:
            if self._OPTION_LABEL_PATTERN.match(text.strip()):
                option_labels.append(text)
            else:
                extras.append(text)

        if len(option_labels) < option_count:
            start_idx = len(option_labels)
            option_labels.extend(
                [f"Option {i}:" for i in range(start_idx, option_count)]
            )

        return option_labels, extras

    # ----------------------------- state handling -----------------------------

    def get_state(self) -> Dict[str, Any]:
        timeline = []
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline.append(
                    {
                        "type": "video",
                        "timestamp": segment["time_start"],
                        "video_id": video_id,
                        "time_start": segment["time_start"],
                        "time_end": segment["time_end"],
                        "duration": segment["duration"],
                        "num_frames": segment["num_frames"],
                    }
                )
        for text, timestamp in self.text_entries:
            timeline.append({"type": "text", "timestamp": timestamp, "text": text})
        timeline.sort(key=lambda x: (x["timestamp"], x["type"] == "video"))

        return {
            "video_segments": copy.deepcopy(self.video_segments),
            "text_entries": copy.deepcopy(self.text_entries),
            "latest_time": self.latest_time,
            "timeline": timeline,
        }

    def clear_context(self) -> None:
        self.latest_time = 0.0
        self.video_segments = {}
        self.text_entries = []
        self._video_was_truncated = None
        gc = __import__("gc")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def was_video_truncated(self) -> Optional[bool]:
        """Check if video frames were truncated in the last ask_question call."""
        return self._video_was_truncated

    def save_state(self) -> Dict[str, Any]:
        saved_segments: Dict[int, List[Dict[str, Any]]] = {}
        for video_id, segments in self.video_segments.items():
            stored = []
            for segment in segments:
                frames = segment["frames"].detach().cpu().clone()
                fps_value = segment.get("fps")
                if fps_value is None or fps_value <= 0:
                    duration = float(segment.get("duration", 0.0))
                    num_frames = int(segment.get("num_frames", 0))
                    if duration > 0 and num_frames > 0:
                        fps_value = num_frames / duration
                if fps_value is not None:
                    fps_value = float(fps_value)
                frame_start_index = segment.get("frame_start_index")
                frame_end_index = segment.get("frame_end_index")
                if frame_start_index is not None:
                    frame_start_index = int(frame_start_index)
                if frame_end_index is not None:
                    frame_end_index = int(frame_end_index)
                stored.append(
                    {
                        "frames": frames,
                        "time_start": segment["time_start"],
                        "time_end": segment["time_end"],
                        "duration": segment["duration"],
                        "num_frames": segment["num_frames"],
                        "fps": fps_value,
                        "frame_start_index": frame_start_index,
                        "frame_end_index": frame_end_index,
                    }
                )
            saved_segments[video_id] = stored

        state = {
            "video_segments": saved_segments,
            "text_entries": copy.deepcopy(self.text_entries),
            "latest_time": self.latest_time,
            "enable_metrics": self.enable_metrics,
            "_metrics": None,
        }

        if self.enable_metrics and self._metrics is not None:
            metrics = self._metrics
            state["_metrics"] = {
                "latency_add_video": copy.deepcopy(metrics.latency_add_video),
                "latency_add_text": copy.deepcopy(metrics.latency_add_text),
                "state_memory_floats": copy.deepcopy(metrics.state_memory_floats),
                "state_memory_after_add_video": copy.deepcopy(metrics.state_memory_after_add_video),
                "state_memory_after_add_text": copy.deepcopy(metrics.state_memory_after_add_text),
                "state_memory_delta_add_video": copy.deepcopy(metrics.state_memory_delta_add_video),
                "state_memory_delta_add_text": copy.deepcopy(metrics.state_memory_delta_add_text),
                "peak_gpu_mem_increase_add_video": copy.deepcopy(metrics.peak_gpu_mem_increase_add_video),
                "peak_gpu_mem_increase_add_text": copy.deepcopy(metrics.peak_gpu_mem_increase_add_text),
                "peak_gpu_mem_absolute_add_video": copy.deepcopy(metrics.peak_gpu_mem_absolute_add_video),
                "peak_gpu_mem_absolute_add_text": copy.deepcopy(metrics.peak_gpu_mem_absolute_add_text),
                "video_timestamps_add_video": copy.deepcopy(metrics.video_timestamps_add_video),
                "video_timestamps_add_text": copy.deepcopy(metrics.video_timestamps_add_text),
                "first_oom_timestamp": metrics.first_oom_timestamp,
            }
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        # Use input embeddings device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device
        restored_segments: Dict[int, List[Dict[str, Any]]] = {}
        for video_id, segments in state["video_segments"].items():
            restored = []
            for segment in segments:
                frames = segment["frames"].to(input_device)
                duration = float(segment.get("duration", 0.0))
                num_frames = int(segment.get("num_frames", 0))
                fps_value = segment.get("fps")
                if (fps_value is None or fps_value <= 0) and duration > 0 and num_frames > 0:
                    fps_value = num_frames / duration
                if fps_value is not None:
                    fps_value = float(fps_value)
                frame_start_index = segment.get("frame_start_index")
                frame_end_index = segment.get("frame_end_index")
                if frame_start_index is not None:
                    frame_start_index = int(frame_start_index)
                if frame_end_index is not None:
                    frame_end_index = int(frame_end_index)
                restored.append(
                    {
                        "frames": frames,
                        "time_start": segment["time_start"],
                        "time_end": segment["time_end"],
                        "duration": segment["duration"],
                        "num_frames": segment["num_frames"],
                        "fps": fps_value,
                        "frame_start_index": frame_start_index,
                        "frame_end_index": frame_end_index,
                    }
                )
            restored_segments[video_id] = restored
        self.video_segments = restored_segments
        self.text_entries = state.get("text_entries", []).copy()
        self.latest_time = float(state.get("latest_time", 0.0))

        if self.enable_metrics and state.get("_metrics") is not None:
            from substream.models.base import PerformanceMetrics

            if self._metrics is None:
                self._metrics = PerformanceMetrics()
            metrics_data = state["_metrics"]
            metrics = self._metrics
            metrics.latency_add_video = metrics_data.get("latency_add_video", []).copy()
            metrics.latency_add_text = metrics_data.get("latency_add_text", []).copy()
            metrics.state_memory_floats = metrics_data.get("state_memory_floats", []).copy()
            metrics.state_memory_after_add_video = metrics_data.get("state_memory_after_add_video", []).copy()
            metrics.state_memory_after_add_text = metrics_data.get("state_memory_after_add_text", []).copy()
            metrics.state_memory_delta_add_video = metrics_data.get("state_memory_delta_add_video", []).copy()
            metrics.state_memory_delta_add_text = metrics_data.get("state_memory_delta_add_text", []).copy()
            metrics.peak_gpu_mem_increase_add_video = metrics_data.get("peak_gpu_mem_increase_add_video", []).copy()
            metrics.peak_gpu_mem_increase_add_text = metrics_data.get("peak_gpu_mem_increase_add_text", []).copy()
            metrics.peak_gpu_mem_absolute_add_video = metrics_data.get("peak_gpu_mem_absolute_add_video", []).copy()
            metrics.peak_gpu_mem_absolute_add_text = metrics_data.get("peak_gpu_mem_absolute_add_text", []).copy()
            metrics.video_timestamps_add_video = metrics_data.get("video_timestamps_add_video", []).copy()
            metrics.video_timestamps_add_text = metrics_data.get("video_timestamps_add_text", []).copy()
            metrics.first_oom_timestamp = metrics_data.get("first_oom_timestamp")
            if hasattr(self, "_sync_state_memory_tracking_from_metrics"):
                self._sync_state_memory_tracking_from_metrics()

    # ----------------------------- token utilities -----------------------------

    def _get_tokenizer(self):  # type: ignore[override]
        if self._tokenizer_ref is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Processor does not expose a tokenizer")
            self._tokenizer_ref = tokenizer
        return self._tokenizer_ref

    def _count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        cached = self._text_token_cache.get(text)
        if cached is not None:
            return cached
        tokenizer = self._get_tokenizer()
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = (
            encoding.get("input_ids") if isinstance(encoding, dict) else encoding.input_ids
        )
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                count = len(input_ids[0])
            else:
                count = len(input_ids)
        elif hasattr(input_ids, "__len__"):
            count = len(input_ids)
        else:
            shape = getattr(input_ids, "shape", None)
            count = int(shape[0]) if shape else 0
        self._text_token_cache[text] = count
        return count

    def _get_state_memory_floats(self) -> float:
        total = 0.0
        for segments in self.video_segments.values():
            for segment in segments:
                frames = segment.get("frames")
                if torch.is_tensor(frames):
                    total += float(frames.numel())
        for text, _ in self.text_entries:
            total += float(self._count_text_tokens(text))
        return total


__all__ = ["Qwen3SegmentRefiner"]

"""Interface that every evaluated model implements, plus shared metrics bookkeeping."""

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def _coalesce_video_fps(
    fps_values: List[Optional[float]],
    tolerance: float = 0.2,
) -> Optional[float]:
    """Return the single FPS shared by ``fps_values``, or None if there are none.

    Raises ``ValueError`` if two values differ by more than ``tolerance``.
    """
    sanitized = [float(value) for value in fps_values if value is not None]
    if not sanitized:
        return None

    reference = sanitized[0]
    for value in sanitized[1:]:
        if abs(value - reference) > tolerance:
            raise ValueError(
                "Inconsistent FPS values detected; transformer processors expect a"
                f" scalar but received {sanitized}."
            )
    return reference


@dataclass
class PerformanceMetrics:
    """Per-operation latency, state-size and GPU-memory measurements."""
    # Latency for each operation in seconds
    latency_add_video: List[float] = field(default_factory=list)
    latency_add_text: List[float] = field(default_factory=list)
    latency_ask_question: List[float] = field(default_factory=list)

    # Total state memory (in floats) after each operation
    state_memory_floats: List[float] = field(default_factory=list)
    state_memory_after_add_video: List[float] = field(default_factory=list)
    state_memory_after_add_text: List[float] = field(default_factory=list)
    state_memory_after_ask_question: List[float] = field(default_factory=list)
    state_memory_delta_add_video: List[float] = field(default_factory=list)
    state_memory_delta_add_text: List[float] = field(default_factory=list)
    state_memory_delta_ask_question: List[float] = field(default_factory=list)

    # Peak GPU memory increase (in MB) during each operation
    peak_gpu_mem_increase_add_video: List[float] = field(default_factory=list)
    peak_gpu_mem_increase_add_text: List[float] = field(default_factory=list)
    peak_gpu_mem_increase_ask_question: List[float] = field(default_factory=list)

    # Absolute peak GPU memory observed during each operation
    peak_gpu_mem_absolute_add_video: List[float] = field(default_factory=list)
    peak_gpu_mem_absolute_add_text: List[float] = field(default_factory=list)
    peak_gpu_mem_absolute_ask_question: List[float] = field(default_factory=list)

    # Video time (seconds) of each operation, and per-question outcomes
    video_timestamps_add_video: List[float] = field(default_factory=list)
    video_timestamps_add_text: List[float] = field(default_factory=list)
    video_timestamps_ask_question: List[float] = field(default_factory=list)
    question_correctness_rate: List[float] = field(default_factory=list)
    question_dont_know_rate: List[float] = field(default_factory=list)
    question_answered_mask: List[float] = field(default_factory=list)
    video_timestamps_question_outcome: List[float] = field(default_factory=list)

    first_oom_timestamp: Optional[float] = None


class VideoLanguageModelInterface(ABC):
    """Streaming video-language model: add context, ask questions, save and restore state."""

    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ):
        self.model_id = model_id
        self.enable_metrics = enable_metrics
        self._metrics = PerformanceMetrics() if enable_metrics else None
        self._configured_max_gpu_mem = max_gpu_mem
        self._last_state_memory_total: float = 0.0
        self._setup_model(max_gpu_mem=max_gpu_mem, **kwargs)
        self._reset_state_memory_tracking()

    @staticmethod
    def _coalesce_video_fps(
        fps_values: List[Optional[float]],
        tolerance: float = 0.2,
    ) -> Optional[float]:
        return _coalesce_video_fps(fps_values, tolerance)

    @abstractmethod
    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        """Load the checkpoint or client. ``kwargs`` are model-specific options."""

    @abstractmethod
    def add_video(self, video_frames: Union[np.ndarray, List], time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
        """Append frames covering ``[time_start, time_end)`` seconds to the context."""

    @abstractmethod
    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        """Append text to the context."""

    def get_last_response_token_stats(self) -> Optional[Dict[str, Any]]:
        """Token counts of the last response, if the model reports them."""
        return None

    @abstractmethod
    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256, max_frames_in_video: int = 768, sample_method: str = "TIME") -> str:
        """Answer ``question`` from the current context without changing it.

        ``max_frames_in_video`` caps the frames per video; ``sample_method``
        ("TIME", "RANDOM" or "SEGMENT") picks which frames are kept.
        """

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME"
    ) -> List[str]:
        """Answer several questions about the same context. Defaults to one call each."""
        return [
            self.ask_question(q, current_video_time, max_tokens, max_frames_in_video, sample_method)
            for q in questions
        ]

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Describe the current context."""

    @abstractmethod
    def clear_context(self) -> None:
        """Remove all video and text from the context."""

    @abstractmethod
    def save_state(self) -> Any:
        """Return an object that ``load_state`` can restore the context from."""

    @abstractmethod
    def load_state(self, state: Any) -> None:
        """Restore the context saved by ``save_state``."""

    def _flush_cuda_cache(self) -> None:
        """Safely release cached CUDA memory across all visible devices."""
        if not torch.cuda.is_available():
            return

        try:
            device_count = torch.cuda.device_count()
        except Exception:
            device_count = 0

        def _empty_for_device(device_index: Optional[int] = None) -> None:
            try:
                if device_index is None:
                    torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache(device=device_index)
            except TypeError:
                if device_index is not None:
                    try:
                        with torch.cuda.device(device_index):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            except Exception:
                pass

        if device_count and device_count > 1:
            for device_index in range(device_count):
                _empty_for_device(device_index)
        else:
            _empty_for_device()

        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except Exception:
                pass

        reset_peak = getattr(torch.cuda, "reset_peak_memory_stats", None)
        if callable(reset_peak):
            try:
                if device_count and device_count > 1:
                    for device_index in range(device_count):
                        reset_peak(device_index)
                else:
                    reset_peak()
            except Exception:
                pass

    def teardown_after_oom(self) -> None:
        """Clear model context and cached GPU memory after an OOM event."""
        try:
            self.clear_context()
        finally:
            gc.collect()
            self._flush_cuda_cache()

    def recover_from_oom(self, state: Any) -> None:
        """Restore the model to *state* after performing OOM cleanup."""
        self.teardown_after_oom()
        self.load_state(state)

    def get_metrics(self) -> Optional[PerformanceMetrics]:
        return self._metrics

    def was_video_truncated(self) -> Optional[bool]:
        """Whether the last question saw fewer frames than were added; None if not tracked."""
        return None

    def _record_add_video_metrics(
        self,
        latency: float,
        peak_gpu_mem_increase: float,
        peak_gpu_mem_absolute: float,
        video_time: float,
        state_memory_total: Optional[float] = None,
    ) -> None:
        """Record metrics for add_video operation with video timestamp."""
        if self.enable_metrics and self._metrics:
            self._metrics.latency_add_video.append(latency)
            self._metrics.peak_gpu_mem_increase_add_video.append(peak_gpu_mem_increase)
            self._metrics.peak_gpu_mem_absolute_add_video.append(peak_gpu_mem_absolute)
            self._metrics.video_timestamps_add_video.append(video_time)
            if state_memory_total is not None:
                self._update_state_memory_tracking('add_video', state_memory_total)

    def _record_add_text_metrics(
        self,
        latency: float,
        peak_gpu_mem_increase: float,
        peak_gpu_mem_absolute: float,
        video_time: float,
        state_memory_total: Optional[float] = None,
    ) -> None:
        """Record metrics for add_text operation with video timestamp."""
        if self.enable_metrics and self._metrics:
            self._metrics.latency_add_text.append(latency)
            self._metrics.peak_gpu_mem_increase_add_text.append(peak_gpu_mem_increase)
            self._metrics.peak_gpu_mem_absolute_add_text.append(peak_gpu_mem_absolute)
            self._metrics.video_timestamps_add_text.append(video_time)
            if state_memory_total is not None:
                self._update_state_memory_tracking('add_text', state_memory_total)

    def _record_ask_question_metrics(
        self,
        latency: float,
        peak_gpu_mem_increase: float,
        peak_gpu_mem_absolute: float,
        video_time: float,
        state_memory_total: Optional[float] = None,
    ) -> None:
        """Record metrics for ask_question operation with video timestamp."""
        if self.enable_metrics and self._metrics:
            self._metrics.latency_ask_question.append(latency)
            self._metrics.peak_gpu_mem_increase_ask_question.append(peak_gpu_mem_increase)
            self._metrics.peak_gpu_mem_absolute_ask_question.append(peak_gpu_mem_absolute)
            self._metrics.video_timestamps_ask_question.append(video_time)
            if state_memory_total is not None:
                self._update_state_memory_tracking('ask_question', state_memory_total)

    def record_question_outcome(
        self,
        *,
        current_video_time: Optional[float],
        is_correct: bool,
        is_dont_know: bool,
        is_answered: bool,
    ) -> None:
        """Track per-question outcomes for downstream offline analysis."""
        if not (self.enable_metrics and self._metrics):
            return

        timestamp = 0.0 if current_video_time is None else float(current_video_time)
        self._metrics.question_correctness_rate.append(1.0 if is_correct else 0.0)
        self._metrics.question_dont_know_rate.append(1.0 if is_dont_know else 0.0)
        self._metrics.question_answered_mask.append(1.0 if is_answered else 0.0)
        self._metrics.video_timestamps_question_outcome.append(timestamp)

    def record_first_oom_snapshot(self, timestamp: float) -> None:
        """Capture metric lengths at the first observed OOM event."""
        if not (self.enable_metrics and self._metrics):
            return

        metrics = self._metrics
        if metrics.first_oom_timestamp is None or float(timestamp) < metrics.first_oom_timestamp:
            metrics.first_oom_timestamp = float(timestamp)

    def _reset_state_memory_tracking(self) -> None:
        """Reset internal trackers for state memory totals and deltas."""
        self._last_state_memory_total = 0.0

    def _sync_state_memory_tracking_from_metrics(self) -> None:
        """Synchronize internal tracking with existing metrics data."""
        if self._metrics and self._metrics.state_memory_floats:
            self._last_state_memory_total = self._metrics.state_memory_floats[-1]
        else:
            self._last_state_memory_total = 0.0

    def _update_state_memory_tracking(self, operation: str, total: float) -> None:
        """Update stored totals and deltas for a completed operation."""
        if not self._metrics:
            return

        previous_total = getattr(self, '_last_state_memory_total', 0.0)
        delta = total - previous_total
        if delta < 0:
            # Context was likely reset; treat delta as absolute total.
            delta = total

        self._last_state_memory_total = total
        self._metrics.state_memory_floats.append(total)

        if operation == 'add_video':
            self._metrics.state_memory_after_add_video.append(total)
            self._metrics.state_memory_delta_add_video.append(delta)
        elif operation == 'add_text':
            self._metrics.state_memory_after_add_text.append(total)
            self._metrics.state_memory_delta_add_text.append(delta)
        elif operation == 'ask_question':
            self._metrics.state_memory_after_ask_question.append(total)
            self._metrics.state_memory_delta_ask_question.append(delta)

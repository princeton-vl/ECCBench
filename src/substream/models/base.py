"""
Generic interface for video-language models with benchmarking capabilities.
All models must implement this interface for consistent comparison.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import gc
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
from dataclasses import dataclass, field
import torch


def _coalesce_video_fps(
    fps_values: List[Optional[float]],
    tolerance: float = 0.2,
) -> Optional[float]:
    """Collapse consistent FPS readings to a single scalar.

    Transformers' video processors now validate ``fps`` as either a float or an
    int. We often collect one FPS per queued video, so this utility produces the
    representative scalar when all values align, and otherwise raises to surface
    unsupported heterogeneous sampling.

    Args:
        fps_values: Collection of frames-per-second readings from the pipeline.
        tolerance: Maximum allowed absolute difference between values.

    Returns:
        A single FPS value when available, otherwise ``None`` if no values were
        provided.

    Raises:
        ValueError: If conflicting FPS values are supplied.
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


@dataclass(frozen=True)
class QuestionMetricsSummary:
    """Aggregate view of per-question outcomes."""

    total: int
    answered: int
    unanswered: int
    correct: int
    dont_know: int
    overall_accuracy: float
    answered_accuracy: Optional[float]
    dont_know_rate: float
    answered_dont_know_rate: Optional[float]


@dataclass
class PerformanceMetrics:
    """A data structure to hold detailed performance metrics."""
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

    # Actual video timestamps corresponding to each operation (seconds from video start)
    video_timestamps_add_video: List[float] = field(default_factory=list)
    video_timestamps_add_text: List[float] = field(default_factory=list)
    video_timestamps_ask_question: List[float] = field(default_factory=list)
    question_correctness_rate: List[float] = field(default_factory=list)
    question_dont_know_rate: List[float] = field(default_factory=list)
    question_answered_mask: List[float] = field(default_factory=list)
    video_timestamps_question_outcome: List[float] = field(default_factory=list)

    # First OOM diagnostics
    first_oom_timestamp: Optional[float] = None

    def summarize_questions(self) -> Optional[QuestionMetricsSummary]:
        """Return aggregate counters/percentages for question outcomes."""

        totals = self.question_correctness_rate
        if not totals:
            return None

        total_questions = len(totals)

        answered_mask = self.question_answered_mask
        if not answered_mask or len(answered_mask) != total_questions:
            answered_mask = [1.0] * total_questions

        dont_know_series = self.question_dont_know_rate
        if not dont_know_series or len(dont_know_series) != total_questions:
            dont_know_series = [0.0] * total_questions

        def _as_bool(series: List[float]) -> List[bool]:
            return [float(value) >= 0.5 for value in series]

        answered_flags = _as_bool(answered_mask)
        correctness_flags = _as_bool(totals)
        dont_know_flags = _as_bool(dont_know_series)

        answered_count = sum(answered_flags)
        unanswered_count = total_questions - answered_count
        correct_total = sum(correctness_flags)
        dont_know_total = sum(dont_know_flags)

        answered_correct = sum(
            1 for correct, answered in zip(correctness_flags, answered_flags)
            if answered and correct
        )
        answered_dont_know = sum(
            1 for dont_know, answered in zip(dont_know_flags, answered_flags)
            if answered and dont_know
        )

        overall_accuracy = correct_total / total_questions if total_questions else 0.0
        answered_accuracy = (
            answered_correct / answered_count if answered_count else None
        )
        dont_know_rate = dont_know_total / total_questions if total_questions else 0.0
        answered_dont_know_rate = (
            answered_dont_know / answered_count if answered_count else None
        )

        return QuestionMetricsSummary(
            total=total_questions,
            answered=answered_count,
            unanswered=unanswered_count,
            correct=correct_total,
            dont_know=dont_know_total,
            overall_accuracy=overall_accuracy,
            answered_accuracy=answered_accuracy,
            dont_know_rate=dont_know_rate,
            answered_dont_know_rate=answered_dont_know_rate,
        )
    
class VideoLanguageModelInterface(ABC):
    """
    Generic interface for video-language models.
    
    This interface defines the core methods that all video-language models
    must implement for consistent benchmarking and evaluation.
    """
    
    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the model.
        
        Args:
            model_id: Identifier for the model (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
            enable_metrics: Whether to collect performance metrics
            **kwargs: Model-specific initialization parameters
        """
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
        """Expose shared FPS coalescing helper to subclasses."""

        return _coalesce_video_fps(fps_values, tolerance)

    @abstractmethod
    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        """
        Model-specific initialization logic.
        
        Args:
            **kwargs: Model-specific parameters
        """
        pass
    
    @abstractmethod
    def add_video(self, video_frames: Union[np.ndarray, List], time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
        """
        Add video data to the model's context.
        
        This method should mutate the internal context state and return nothing.
        The video data should be integrated into the model's understanding.
        
        Args:
            video_frames: Video data - either:
                         - np.ndarray: Video frames with shape (num_frames, height, width, channels)
                         - List: Pre-processed content (e.g., ["<unit>", image, "<unit>", image, ...])
            time_start: Start time of the video segment
            time_end: End time of the video segment  
            video_id: Optional identifier for the video (used by some implementations)
        """
        pass
    
    @abstractmethod
    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        """
        Add text to the model's context.

        This method should mutate the internal context state and return nothing.
        The text should be integrated into the model's understanding.

        Args:
            text: Text string to add to context
            current_video_time: Current timestamp in video (seconds from start)
        """
        pass

    def get_last_response_token_stats(self) -> Optional[Dict[str, Any]]:
        """Return model-specific token statistics for the most recent response."""
        return None
    
    @abstractmethod
    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256, max_frames_in_video: int = 768, sample_method: str = "TIME") -> str:
        """
        Ask a question based on the current context.

        This method should generate a response based on all previously added
        video and text content without modifying the context.

        Args:
            question: Question to ask
            current_video_time: Current timestamp in video when question is asked (seconds from start)
            max_tokens: Maximum number of tokens to generate
            max_frames_in_video: Maximum frames per video (used by some implementations)
            sample_method: Sampling method for frames ("TIME", "RANDOM", "SEGMENT")

        Returns:
            Generated response as string
        """
        pass

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME"
    ) -> List[str]:
        """
        Ask multiple questions in a batched inference call.

        Default implementation: Fall back to individual calls.
        Models can override this method for true GPU batching.

        Args:
            questions: List of question texts
            current_video_time: Current timestamp when questions are asked
            max_tokens: Maximum tokens to generate per question
            max_frames_in_video: Maximum frames per video
            sample_method: Sampling method for frames

        Returns:
            List of responses, one per question
        """
        return [
            self.ask_question(q, current_video_time, max_tokens, max_frames_in_video, sample_method)
            for q in questions
        ]

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the model.
        
        Returns:
            Dictionary containing current context information
        """
        pass
    
    @abstractmethod
    def clear_context(self) -> None:
        """
        Clear all context (video and text) from the model.

        This should reset the model to its initial state.
        """
        pass

    @abstractmethod
    def save_state(self) -> Any:
        """
        Save the current model state to memory.

        This should capture all context (video segments, text entries, timing)
        so it can be restored later with load_state.

        Returns:
            State object that can be passed to load_state
        """
        pass

    @abstractmethod
    def load_state(self, state: Any) -> None:
        """
        Load a previously saved model state.

        This should restore the model to the exact state it was in when
        save_state was called.

        Args:
            state: State object returned by save_state
        """
        pass

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
        """
        Return the collected performance metrics.
        """
        return self._metrics

    def was_video_truncated(self) -> Optional[bool]:
        """
        Check if video frames were truncated in the last ask_question call.

        Returns:
            True if frames were truncated due to max_frames_in_video limit,
            False if all frames were used,
            None if tracking is not implemented for this model
        """
        return None  # Default: not tracked

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


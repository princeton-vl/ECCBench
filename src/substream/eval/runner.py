#!/usr/bin/env python3
"""Evaluation runner for the Substream Recollection benchmark.

Streams each benchmark video (or its text token-sequence rendering) into a
model, replays the media up to every question's timestamp, presents the
candidate clip or sequence, and scores the model's answer. Progress is written
incrementally to ``<out>/<name>.state`` (JSON, consumed by ``--resume``) and a
per-question log to ``<out>/<name>.csv``.

Example:
    substream-eval --data text --model qwen_full \
        --mode text --variant sequential --out runs --name qwen_text_seq
"""

import argparse
import atexit
import copy
from collections import defaultdict
import gc
import hashlib
import logging
import math
import os
import random
import re
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from substream import config
from substream.data import hf_loader
from substream.data.csv_log import write_question_log_csv
from substream.data.manifest import (
    QuestionEntry,
    VideoEntry,
    load_patternvideos_manifest,
)
from substream.data.state import RunStateManager
from substream.eval.processors.question_processor import QuestionProcessor
from substream.eval.processors.sequence_processor import (
    CommaSeparatedSequenceFormatter,
    SequenceFormatter,
    SequenceProcessor,
    SpatialSequenceFormatter,
)
from substream.eval.processors.text_to_video_processor import TextToVideoProcessor
from substream.eval.processors.video_processor import VideoProcessor
from substream.eval.wandb_log import WandbLogger
from substream.models.base import PerformanceMetrics
from substream.registry import REGISTRY, load_model_class
from substream.utils.gpu_monitor import GPUMonitor
from substream.utils.memory_utils import calculate_max_gpu_mem


logging.getLogger("httpx").setLevel(logging.WARNING)

# Batching configuration. Whether a given model may batch at all comes from its
# registry entry (ModelSpec.text_batching / video_batching, plus
# ModelSpec.video_batch_size); the env vars below gate batching globally and
# set the default batch sizes.
ENABLE_TEXT_BATCHING = os.environ.get('ENABLE_TEXT_BATCHING', '1') == '1'
ENABLE_VIDEO_BATCHING = os.environ.get('ENABLE_VIDEO_BATCHING', '1') == '1'
DEFAULT_BATCH_SIZE = int(os.environ.get('TEXT_BATCH_SIZE', '16'))
VIDEO_BATCH_SIZE = int(os.environ.get('VIDEO_BATCH_SIZE', '16'))


def group_questions_into_batches(questions: List[QuestionEntry], batch_size: int) -> List[List[QuestionEntry]]:
    """Group questions into batches of specified size."""
    if batch_size <= 0:
        batch_size = 1
    return [questions[i:i+batch_size] for i in range(0, len(questions), batch_size)]


class DynamicBatchSizer:
    """Manages batch size with OOM-aware reduction."""

    def __init__(self, initial_size: int = 8, min_size: int = 1, max_size: int = 32):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.oom_count = 0

    def reduce_on_oom(self) -> int:
        """Halve batch size on OOM."""
        self.current_size = max(self.min_size, self.current_size // 2)
        self.oom_count += 1
        return self.current_size

    def increase_on_success(self) -> int:
        """Gradually increase batch size on successful runs."""
        if self.current_size < self.max_size and self.oom_count == 0:
            self.current_size = min(self.max_size, self.current_size + 1)
        return self.current_size

    def get_size(self) -> int:
        return self.current_size


_TERMINATION_REQUESTED = False

def _handle_termination(signum, frame):
    """Handle process termination signals and clean up GPU memory."""
    del frame
    global _TERMINATION_REQUESTED
    _TERMINATION_REQUESTED = True

    # Clean up GPU memory to avoid zombie processes holding GPU resources
    print(f"\nReceived termination signal {signum}, cleaning up GPU memory...")
    try:
        import torch
        if torch.cuda.is_available():
            # Clear CUDA cache on all devices
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                try:
                    torch.cuda.empty_cache()
                    # Also synchronize to ensure all operations are complete
                    torch.cuda.synchronize(device_id)
                except Exception as e:
                    print(f"  Warning: Could not clean GPU {device_id}: {e}")

            print(f"  GPU memory cleared on {device_count} device(s)")

        # Force garbage collection
        import gc
        gc.collect()
        print("  Garbage collection completed")

    except Exception as e:
        print(f"  Warning: GPU cleanup failed: {e}")

    print("Cleanup complete, exiting...")

def termination_requested() -> bool:
    return _TERMINATION_REQUESTED

def _install_signal_handlers() -> None:
    """Register termination handlers; called from main(), not at import."""
    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, _handle_termination)
    atexit.register(_cleanup_on_exit)


def _cleanup_on_exit():
    """Clean up GPU memory when process exits normally."""
    if _TERMINATION_REQUESTED:
        # Already cleaned up via signal handler
        return

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except Exception:
        pass  # Silent cleanup on normal exit


def _extract_bucket_from_video_data(video_data: Any) -> Optional[str]:
    """Extract bucket name from video_data.video_path.

    Returns the bucket name (e.g., 'UNIFORM_EVAL_L008_ELOW') or None if not found.
    This is crucial for handling non-unique video indices across buckets.
    """
    path = getattr(video_data, 'video_path', None) or getattr(video_data, 'main_video_path', None)
    if not path:
        return None
    path_str = str(path)
    parts = path_str.split('/')
    for part in parts:
        if 'UNIFORM_EVAL' in part or any(x in part for x in ['L008', 'L016', 'L032', 'L064', 'L128', 'L256', 'L512', 'L1024', 'L2048', 'L4096']):
            return part
    return None


def extract_bucket_length(bucket: str) -> Optional[int]:
    """Extract numeric length from bucket name (e.g., 'L064' -> 64).

    Used for sorting buckets by frame count and checking per-model limits.
    Returns None if bucket name doesn't contain a valid L{N} pattern.
    """
    if not bucket:
        return None
    match = re.search(r'L(\d+)', bucket)
    if match:
        return int(match.group(1))
    return None


def count_valid_completed_questions(
    completed_set: Set[str],
    video_data: Any,
    eval_mode: str,
) -> int:
    """
    Count how many completed questions are VALID for eval_mode.
    Cross-references completed_set with video_data to filter by question_variant.

    In 'sequential' mode: only counts sequential questions (skips spatial)
    Otherwise: counts all questions
    """
    valid_count = 0
    for i, question_data in enumerate(video_data.questions):
        question_id = question_data.question_id or f"video{video_data.video_index}_q{i}"
        if question_id not in completed_set:
            continue

        # Check if valid for eval_mode
        if eval_mode == 'sequential':
            variant = question_data.metadata.get('question_variant', '').lower()
            if variant == 'spatial':
                continue  # Skip spatial in sequential mode

        valid_count += 1
    return valid_count


def count_total_valid_questions(
    video_data: Any,
    eval_mode: str,
) -> int:
    """
    Count total questions valid for eval_mode.

    In 'sequential' mode: only counts sequential questions (skips spatial)
    Otherwise: counts all questions
    """
    valid_count = 0
    for question_data in video_data.questions:
        if eval_mode == 'sequential':
            variant = question_data.metadata.get('question_variant', '').lower()
            if variant == 'spatial':
                continue
        valid_count += 1
    return valid_count



def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True if exception indicates a CUDA out-of-memory condition."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True

    message = str(exc).lower()
    if "out of memory" in message and "cuda" in message:
        return True
    if "cuda error" in message and "memory" in message:
        return True
    return False


def _clear_exception_traceback(exc: BaseException) -> None:
    """Detach traceback frames so large locals (e.g. CUDA tensors) can be reclaimed."""
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        try:
            frame.f_locals.clear()
        except Exception:
            pass
        tb = tb.tb_next

    try:
        import traceback  # Lazy import keeps startup fast

        traceback.clear_frames(exc.__traceback__)
    except Exception:
        pass

    try:
        exc.__traceback__ = None  # type: ignore[assignment]
    except Exception:
        pass
    exc.__cause__ = None
    exc.__context__ = None


def _lightweight_exception(exc: BaseException) -> BaseException:
    """Return an exception of the same type without traceback references."""
    try:
        clone = exc.__class__(*getattr(exc, "args", ()))
    except Exception:
        clone = RuntimeError(str(exc))
    _clear_exception_traceback(clone)
    return clone


def _reduced_frame_budget(current_max_frames: int) -> int:
    """Compute reduced frame budget after an OOM event (90% of previous, at least 1)."""
    if current_max_frames <= 1:
        return 1

    reduced = max(int(math.floor(current_max_frames * 0.90)), 1)
    if reduced == current_max_frames and current_max_frames > 1:
        reduced = current_max_frames - 1
    return max(reduced, 1)

class QuestionOOMRetry(RuntimeError):
    """Signal that question processing needs to be retried after reducing resources."""

    def __init__(
        self,
        question_index: int,
        partial_results: List[Dict[str, Any]],
        original_exception: BaseException,
        oom_timestamp: float,
        partial_metrics: Optional[PerformanceMetrics],
    ):
        super().__init__("CUDA OOM during question processing")
        self.question_index = question_index
        self.partial_results = partial_results
        self.original_exception_message = repr(original_exception)
        self.original_exception = _lightweight_exception(original_exception)
        _clear_exception_traceback(original_exception)
        self.oom_timestamp = float(oom_timestamp)
        self.partial_metrics = partial_metrics


def process_sequence_native_binary_batched_isolated(
    video_data: VideoEntry,
    model: Any,
    sequence_processor: SequenceProcessor,
    question_processor: QuestionProcessor,
    max_tokens: int,
    batch_size: int,
    eval_mode: str,
    verbose: bool,
    state_manager: Optional[RunStateManager],
    question_log_path: Optional[str],
    question_log_rows: Optional[List[Dict[str, Any]]],
    limit_questions: Optional[int],
    start_question_index: int,
    completed_set: Optional[set],
) -> List[Dict[str, Any]]:
    """
    TRUE parallel batching for sequence mode native binary questions.
    Each question gets its own isolated context (main sequence + ONE candidate).
    """
    video_index = video_data.video_index
    bucket = _extract_bucket_from_video_data(video_data)
    questions = sorted(video_data.questions, key=lambda q: float(q.question_time))

    if verbose:
        print(f"\nProcessing Video {video_index} (bucket: {bucket or 'None'}) with TRUE sequence mode batching (isolated contexts)")

    # Check if model supports isolated batching
    if not hasattr(model, 'ask_question_batch_isolated'):
        raise RuntimeError(
            f"Model {type(model).__name__} does not support isolated batching. "
            f"Disable ENABLE_TEXT_BATCHING or implement ask_question_batch_isolated()."
        )

    # Count valid completed questions (respects eval_mode filtering)
    valid_completed = count_valid_completed_questions(completed_set or set(), video_data, eval_mode)
    total_valid = count_total_valid_questions(video_data, eval_mode)

    # Check if bucket is already at/over limit
    if limit_questions is not None and valid_completed >= limit_questions:
        if verbose:
            print(f"  [SKIP] Bucket already complete: {valid_completed}/{limit_questions} valid questions done")
        return []

    # Filter questions
    questions_to_process = []
    new_valid_count = 0

    for i, question_data in enumerate(questions):
        if i < start_question_index:
            continue

        question_identifier = question_data.question_id or f"video{video_index}_q{i}"
        if completed_set and question_identifier in completed_set:
            continue

        # Filter by eval_mode
        if eval_mode == 'sequential':
            question_variant = question_data.metadata.get('question_variant', '').lower()
            if question_variant == 'spatial':
                continue

        # Check if this is native binary
        if not question_data.is_native_binary:
            continue

        # Check TOTAL count (valid_completed + new_valid_count)
        if limit_questions is not None and (valid_completed + new_valid_count) >= limit_questions:
            if verbose:
                print(f"  [LIMIT] Reached limit_questions={limit_questions} (completed={valid_completed}, new={new_valid_count}), stopping")
            break

        questions_to_process.append((i, question_data, question_identifier))
        new_valid_count += 1

    if not questions_to_process:
        return []

    # Log progress with eval_mode-aware counts
    if verbose and valid_completed > 0:
        print(f"  [RESUME] Skipped {valid_completed}/{total_valid} valid questions (eval_mode={eval_mode}), processing {len(questions_to_process)} remaining")

    # Get main sequence (shared reference, not added to context yet)
    sequences_used = video_data.metadata.get('sequences_used', {})
    main_sequence_tokens = sequences_used.get('S_tokens', [])
    main_sequence_text = ', '.join(str(t) for t in main_sequence_tokens)

    # Group into batches
    batches = group_questions_into_batches(
        [q for _, q, _ in questions_to_process],
        batch_size
    )

    all_results = []

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(print_every_n_batches=5, prefix="  ")

    if verbose:
        print(f"Processing {len(questions_to_process)} native binary questions in {len(batches)} batches (batch_size={batch_size})")

    for batch_idx, batch in enumerate(batches):
        if verbose:
            print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch)} questions")

        # Build separate isolated context for EACH question
        question_contexts = []
        batch_metadata = []

        for q_idx, question_data in enumerate(batch):
            global_idx, original_question, question_identifier = questions_to_process[batch_idx * batch_size + q_idx]

            # Build isolated context for THIS question only
            candidate = question_data.candidate
            candidate_text = ', '.join(candidate.sequence) if candidate and candidate.sequence else ""

            question_prompt = question_processor.build_native_binary_question_text(
                question_data,
                include_uncertain=True
            )

            # Create context dict for this specific question
            ctx = {
                'main_sequence': main_sequence_text,
                'candidate_sequence': candidate_text,
                'question_text': question_prompt,
                'mode': 'sequence'
            }

            question_contexts.append(ctx)
            batch_metadata.append({
                'global_idx': global_idx,
                'question_data': question_data,
                'question_identifier': question_identifier,
                'candidate_text': candidate_text
            })

        # Call model with isolated contexts
        responses = model.ask_question_batch_isolated(
            contexts=question_contexts,
            max_tokens=max_tokens,
        )

        # Process responses
        for q_idx, (response, meta) in enumerate(zip(responses, batch_metadata)):
            question_data = meta['question_data']
            question_identifier = meta['question_identifier']
            global_idx = meta['global_idx']

            raw_answer = question_processor.extract_answer(response)
            predicted_answer = question_processor.normalize_native_binary_answer(raw_answer)

            raw_correct = (question_data.binary_answer or "").strip().lower()
            correct_answer = '0' if raw_correct == 'yes' else ('1' if raw_correct == 'no' else raw_correct)

            is_correct = predicted_answer == correct_answer
            is_dont_know = predicted_answer == '2'

            if verbose:
                status = "correct" if is_correct else "wrong"
                print(f"    Q{global_idx+1}: pred={predicted_answer} correct={correct_answer} ({status}) - Candidate: [{meta['candidate_text']}]")

            result = {
                'video_index': video_index,
                'question_index': global_idx,
                'question_id': question_identifier,
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'is_dont_know': is_dont_know,
                'num_options': 3,
                'response': response,
                'frames_seen_before_question': 0,
                'question_order': question_data.question_order,
                'question_mode': question_data.question_mode,
                'eval_mode': eval_mode,
                'input_mode': 'sequence',
                'sequence_format': 'comma',
            }

            # Extract token statistics
            token_stats = _extract_token_stats(model, response)
            result.update(token_stats)

            # Add max_tokens and calculate truncation
            result['max_tokens'] = max_tokens
            response_token_count = result.get('response_token_count')
            if response_token_count is not None and max_tokens:
                result['output_was_truncated'] = response_token_count >= (max_tokens - 5)
            else:
                result['output_was_truncated'] = None

            # max_frames and saw_all_frames not applicable in sequence mode
            result['max_frames'] = None
            result['saw_all_frames'] = None

            # latency_ask_question and peak_gpu_mem_ask_question
            result['latency_ask_question'] = None
            result['peak_gpu_mem_ask_question'] = None

            # Add entropy features from question metadata
            if hasattr(question_data, 'metadata') and question_data.metadata:
                # entropy_prefix
                entropy_map = question_data.metadata.get('entropy_prefix')
                if isinstance(entropy_map, dict) and entropy_map:
                    try:
                        values = [float(v) for v in entropy_map.values()]
                        if values:
                            mean_entropy = sum(values) / len(values)
                            result['entropy_prefix_values'] = entropy_map
                            result['entropy_prefix_mean'] = mean_entropy
                    except (TypeError, ValueError):
                        pass

                if 'question_variant' in question_data.metadata:
                    result['question_variant'] = question_data.metadata['question_variant']
                if 'question_type' in question_data.metadata:
                    result['question_type'] = question_data.metadata['question_type']
                if 'has_unique_answer' in question_data.metadata:
                    result['has_unique_answer'] = question_data.metadata['has_unique_answer']
                if 'scenario' in question_data.metadata:
                    result['scenario'] = question_data.metadata['scenario']

            # Add video-level entropy from video_data
            if hasattr(video_data, 'metadata') and isinstance(video_data.metadata, dict):
                entropy_overall = video_data.metadata.get('entropy_overall')
                if isinstance(entropy_overall, dict):
                    empirical = entropy_overall.get('empirical_bits')
                    if isinstance(empirical, dict):
                        result['entropy_empirical_bits'] = empirical
                        try:
                            empirical_values = [float(v) for v in empirical.values()]
                            if empirical_values:
                                result['entropy_empirical_mean'] = sum(empirical_values) / len(empirical_values)
                        except (TypeError, ValueError):
                            pass
                    analytic = entropy_overall.get('analytic_bits')
                    if isinstance(analytic, dict):
                        result['entropy_analytic_bits'] = analytic
                        try:
                            analytic_values = [float(v) for v in analytic.values()]
                            if analytic_values:
                                result['entropy_analytic_mean'] = sum(analytic_values) / len(analytic_values)
                        except (TypeError, ValueError):
                            pass
                # Add template field from video metadata (for easy_human dataset)
                if 'template' in video_data.metadata:
                    result['template'] = video_data.metadata['template']

            # Add bucket (already extracted at function start)
            result['bucket'] = bucket

            if question_data.candidate:
                result['candidate_present'] = question_data.candidate.present
                result['candidate_clip_start'] = question_data.candidate.clip_start
                result['candidate_clip_end'] = question_data.candidate.clip_end

            try:
                result['question_time'] = float(question_data.question_time)
            except (TypeError, ValueError):
                result['question_time'] = None

            all_results.append(result)

            csv_row = {
                'video_id': video_index,
                'variant': 0,
                'bucket': result.get('bucket'),
                'question_id': question_identifier,
                'question_order': result.get('question_order'),
                'video_entropy': result.get('entropy_prefix_mean'),
                'correct_answer': result.get('correct'),
                'model_answer': result.get('predicted'),
                'is_correct': result.get('is_correct'),
                'is_dont_know': result.get('is_dont_know'),
                'response': result.get('response', ''),
                'num_options': result.get('num_options'),
                'is_native_binary': True,
                'question_type': result.get('question_type'),
                'question_variant': result.get('question_variant'),
                'question_time': result.get('question_time'),
                'clip_start_time': question_data.clip_start_time,
                'clip_end_time': question_data.clip_end_time,
                'candidate_present': result.get('candidate_present'),
                'candidate_clip_start': result.get('candidate_clip_start'),
                'candidate_clip_end': result.get('candidate_clip_end'),
                'has_unique_answer': result.get('has_unique_answer'),
                'scenario': result.get('scenario'),
                'response_token_count': result.get('response_token_count'),
                'output_was_truncated': result.get('output_was_truncated'),
                'prompt_tokens': result.get('input_tokens'),
                'completion_tokens': result.get('output_tokens'),
                'eval_mode': eval_mode,
                'input_mode': 'sequence',
            }

            if question_log_path and question_log_rows is not None:
                question_log_rows.append(csv_row)

            if state_manager is not None:
                state_manager.mark_question_complete(video_index, question_identifier, result, variant=0, bucket=bucket)
                completed_set.add(question_identifier)

        # Record batch completion for GPU monitoring
        gpu_monitor.record_batch(batch_idx + 1)

    # Print final GPU usage
    if verbose and len(batches) > 0:
        gpu_monitor.print_final()

    return all_results


def process_video_native_binary_batched_isolated(
    video_data: VideoEntry,
    model: Any,
    video_processor: VideoProcessor,
    question_processor: QuestionProcessor,
    max_tokens: int,
    batch_size: int,
    eval_mode: str,
    verbose: bool,
    state_manager: Optional[RunStateManager],
    question_log_path: Optional[str],
    question_log_rows: Optional[List[Dict[str, Any]]],
    limit_questions: Optional[int],
    start_question_index: int,
    completed_set: Optional[set],
) -> List[Dict[str, Any]]:
    """
    TRUE parallel batching for video mode native binary questions.
    Each question gets its own isolated context (main video + ONE candidate clip as separate videos).
    """
    video_index = video_data.video_index
    bucket = _extract_bucket_from_video_data(video_data)
    questions = sorted(video_data.questions, key=lambda q: float(q.question_time))

    if verbose:
        bucket_display = _extract_bucket_from_video_data(video_data)
        print(f"\nProcessing Video {video_index} (bucket: {bucket_display or 'None'}) with TRUE video mode batching (isolated contexts)")

    # Check if model supports isolated batching
    if not hasattr(model, 'ask_question_batch_isolated'):
        raise RuntimeError(
            f"Model {type(model).__name__} does not support isolated batching. "
            f"Disable ENABLE_VIDEO_BATCHING or implement ask_question_batch_isolated()."
        )

    # Load main video once
    print("Loading main video...")
    main_video_path = str(video_data.video_path)
    video_processor.load_main_video(main_video_path)
    basename = os.path.basename(main_video_path)
    print(f"Main video loaded: {basename}")

    # Get all main video frames from processor
    main_video_frames = video_processor.main_video_frames

    # Get frame count for frames_seen_before_question field
    main_video_frame_count = video_processor.frame_sampler.get_frame_count(main_video_frames)

    # Count valid completed questions (respects eval_mode filtering)
    valid_completed = count_valid_completed_questions(completed_set or set(), video_data, eval_mode)
    total_valid = count_total_valid_questions(video_data, eval_mode)

    # Check if bucket is already at/over limit
    if limit_questions is not None and valid_completed >= limit_questions:
        if verbose:
            print(f"  [SKIP] Bucket already complete: {valid_completed}/{limit_questions} valid questions done")
        return []

    # Filter questions
    questions_to_process = []
    new_valid_count = 0

    for i, question_data in enumerate(questions):
        if i < start_question_index:
            continue

        question_identifier = question_data.question_id or f"video{video_index}_q{i}"
        if completed_set and question_identifier in completed_set:
            continue

        # Filter by eval_mode
        if eval_mode == 'sequential':
            question_variant = question_data.metadata.get('question_variant', '').lower()
            if question_variant == 'spatial':
                continue

        # Check if this is native binary
        if not question_data.is_native_binary:
            continue

        # Check TOTAL count (valid_completed + new_valid_count)
        if limit_questions is not None and (valid_completed + new_valid_count) >= limit_questions:
            if verbose:
                print(f"  [LIMIT] Reached limit_questions={limit_questions} (completed={valid_completed}, new={new_valid_count}), stopping")
            break

        questions_to_process.append((i, question_data, question_identifier))
        new_valid_count += 1

    # Log progress with eval_mode-aware counts
    if verbose and valid_completed > 0:
        print(f"  [RESUME] Skipped {valid_completed}/{total_valid} valid questions (eval_mode={eval_mode}), processing {len(questions_to_process)} remaining")

    if not questions_to_process:
        return []

    # Group into batches
    batches = group_questions_into_batches(
        [q for _, q, _ in questions_to_process],
        batch_size
    )

    all_results = []

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(print_every_n_batches=5, prefix="  ")

    if verbose:
        print(f"Processing {len(questions_to_process)} native binary questions in {len(batches)} batches (batch_size={batch_size})")

    for batch_idx, batch in enumerate(batches):
        if verbose:
            print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch)} questions")

        # Build separate isolated context for EACH question
        question_contexts = []
        batch_metadata = []

        for q_idx, question_data in enumerate(batch):
            global_idx, original_question, question_identifier = questions_to_process[batch_idx * batch_size + q_idx]

            # Load candidate clip for this question
            candidate = question_data.candidate
            candidate_frames = None
            if candidate and candidate.clip_path:
                candidate_clips = video_processor.load_option_videos([candidate.clip_path])
                if candidate_clips:
                    candidate_frames = candidate_clips[0]

            question_prompt = question_processor.build_native_binary_question_text(
                question_data,
                include_uncertain=True
            )

            # Create context dict - keep main and candidate as SEPARATE videos
            # Clone frames to avoid sharing references across contexts
            # Handle different frame types properly
            from substream.samplers.internvl_sampler import InternVLSampledVideo
            import numpy as np

            # Clone main video frames
            if isinstance(main_video_frames, InternVLSampledVideo):
                cloned_main = InternVLSampledVideo(
                    pixel_values=main_video_frames.pixel_values.clone(),
                    num_patches_list=list(main_video_frames.num_patches_list),
                    fps=main_video_frames.fps
                )
            elif torch.is_tensor(main_video_frames):
                cloned_main = main_video_frames.clone()
            elif isinstance(main_video_frames, np.ndarray):
                cloned_main = main_video_frames.copy()
            else:
                cloned_main = main_video_frames

            # Clone candidate frames
            if candidate_frames is not None:
                if isinstance(candidate_frames, InternVLSampledVideo):
                    cloned_candidate = InternVLSampledVideo(
                        pixel_values=candidate_frames.pixel_values.clone(),
                        num_patches_list=list(candidate_frames.num_patches_list),
                        fps=candidate_frames.fps
                    )
                elif torch.is_tensor(candidate_frames):
                    cloned_candidate = candidate_frames.clone()
                elif isinstance(candidate_frames, np.ndarray):
                    cloned_candidate = candidate_frames.copy()
                else:
                    cloned_candidate = candidate_frames
            else:
                cloned_candidate = None

            ctx = {
                'main_video_frames': cloned_main,
                'candidate_video_frames': cloned_candidate,
                'question_text': question_prompt,
                'mode': 'video'
            }

            question_contexts.append(ctx)
            # Extract clip name from actual loaded frames, not just metadata
            candidate_clip_name = os.path.basename(candidate.clip_path) if (candidate and candidate_frames is not None) else "None"
            batch_metadata.append({
                'global_idx': global_idx,
                'question_data': question_data,
                'question_identifier': question_identifier,
                'candidate_clip': candidate_clip_name
            })

        # Call model with isolated contexts
        responses = model.ask_question_batch_isolated(
            contexts=question_contexts,
            max_tokens=max_tokens,
            max_frames_in_video=video_processor.model_max_frames,
        )

        # Process responses
        for q_idx, (response, meta) in enumerate(zip(responses, batch_metadata)):
            question_data = meta['question_data']
            question_identifier = meta['question_identifier']
            global_idx = meta['global_idx']

            raw_answer = question_processor.extract_answer(response)
            predicted_answer = question_processor.normalize_native_binary_answer(raw_answer)

            raw_correct = (question_data.binary_answer or "").strip().lower()
            correct_answer = '0' if raw_correct == 'yes' else ('1' if raw_correct == 'no' else raw_correct)

            is_correct = predicted_answer == correct_answer
            is_dont_know = predicted_answer == '2'

            if verbose:
                status = "correct" if is_correct else "wrong"
                print(f"    Q{global_idx+1}: pred={predicted_answer} correct={correct_answer} ({status}) - Clip: {meta['candidate_clip']}")

            result = {
                'video_index': video_index,
                'question_index': global_idx,
                'question_id': question_identifier,
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'is_dont_know': is_dont_know,
                'num_options': 3,
                'response': response,
                'frames_seen_before_question': main_video_frame_count,
                'question_order': question_data.question_order,
                'question_mode': question_data.question_mode,
                'eval_mode': 'spatial',
                'input_mode': 'video',
            }

            # Extract token statistics
            token_stats = _extract_token_stats(model, response)
            result.update(token_stats)

            # Add max_tokens and calculate truncation
            result['max_tokens'] = max_tokens
            response_token_count = result.get('response_token_count')
            if response_token_count is not None and max_tokens:
                result['output_was_truncated'] = response_token_count >= (max_tokens - 5)
            else:
                result['output_was_truncated'] = None

            # Get max_frames and saw_all_frames from video_processor if available
            if video_processor:
                result['max_frames'] = getattr(video_processor, 'max_frames', None)
                result['saw_all_frames'] = getattr(video_processor, 'saw_all_frames', None)
            else:
                result['max_frames'] = None
                result['saw_all_frames'] = None

            # latency_ask_question and peak_gpu_mem_ask_question
            result['latency_ask_question'] = None
            result['peak_gpu_mem_ask_question'] = None

            # Add entropy features from question metadata
            if hasattr(question_data, 'metadata') and question_data.metadata:
                # entropy_prefix
                entropy_map = question_data.metadata.get('entropy_prefix')
                if isinstance(entropy_map, dict) and entropy_map:
                    try:
                        values = [float(v) for v in entropy_map.values()]
                        if values:
                            mean_entropy = sum(values) / len(values)
                            result['entropy_prefix_values'] = entropy_map
                            result['entropy_prefix_mean'] = mean_entropy
                    except (TypeError, ValueError):
                        pass

                if 'question_variant' in question_data.metadata:
                    result['question_variant'] = question_data.metadata['question_variant']
                if 'question_type' in question_data.metadata:
                    result['question_type'] = question_data.metadata['question_type']
                if 'has_unique_answer' in question_data.metadata:
                    result['has_unique_answer'] = question_data.metadata['has_unique_answer']
                if 'scenario' in question_data.metadata:
                    result['scenario'] = question_data.metadata['scenario']

            # Add video-level entropy from video_data
            if hasattr(video_data, 'metadata') and isinstance(video_data.metadata, dict):
                entropy_overall = video_data.metadata.get('entropy_overall')
                if isinstance(entropy_overall, dict):
                    empirical = entropy_overall.get('empirical_bits')
                    if isinstance(empirical, dict):
                        result['entropy_empirical_bits'] = empirical
                        try:
                            empirical_values = [float(v) for v in empirical.values()]
                            if empirical_values:
                                result['entropy_empirical_mean'] = sum(empirical_values) / len(empirical_values)
                        except (TypeError, ValueError):
                            pass
                    analytic = entropy_overall.get('analytic_bits')
                    if isinstance(analytic, dict):
                        result['entropy_analytic_bits'] = analytic
                        try:
                            analytic_values = [float(v) for v in analytic.values()]
                            if analytic_values:
                                result['entropy_analytic_mean'] = sum(analytic_values) / len(analytic_values)
                        except (TypeError, ValueError):
                            pass
                # Add template field from video metadata (for easy_human dataset)
                if 'template' in video_data.metadata:
                    result['template'] = video_data.metadata['template']

            # Add bucket (already extracted at function start)
            result['bucket'] = bucket

            if question_data.candidate:
                result['candidate_present'] = question_data.candidate.present
                result['candidate_clip_start'] = question_data.candidate.clip_start
                result['candidate_clip_end'] = question_data.candidate.clip_end

            try:
                result['question_time'] = float(question_data.question_time)
            except (TypeError, ValueError):
                result['question_time'] = None

            all_results.append(result)

            csv_row = {
                'video_id': video_index,
                'variant': 0,
                'bucket': result.get('bucket'),
                'question_id': question_identifier,
                'question_order': result.get('question_order'),
                'video_entropy': result.get('entropy_prefix_mean'),
                'correct_answer': result.get('correct'),
                'model_answer': result.get('predicted'),
                'is_correct': result.get('is_correct'),
                'is_dont_know': result.get('is_dont_know'),
                'response': result.get('response', ''),
                'num_options': result.get('num_options'),
                'is_native_binary': True,
                'question_type': result.get('question_type'),
                'question_variant': result.get('question_variant'),
                'question_time': result.get('question_time'),
                'clip_start_time': question_data.clip_start_time,
                'clip_end_time': question_data.clip_end_time,
                'candidate_present': result.get('candidate_present'),
                'candidate_clip_start': result.get('candidate_clip_start'),
                'candidate_clip_end': result.get('candidate_clip_end'),
                'has_unique_answer': result.get('has_unique_answer'),
                'scenario': result.get('scenario'),
                'response_token_count': result.get('response_token_count'),
                'output_was_truncated': result.get('output_was_truncated'),
                'prompt_tokens': result.get('input_tokens'),
                'completion_tokens': result.get('output_tokens'),
                'eval_mode': 'spatial',
                'input_mode': 'video',
            }

            if question_log_path and question_log_rows is not None:
                question_log_rows.append(csv_row)

            if state_manager is not None:
                state_manager.mark_question_complete(video_index, question_identifier, result, variant=0, bucket=bucket)
                completed_set.add(question_identifier)

        # Record batch completion for GPU monitoring
        gpu_monitor.record_batch(batch_idx + 1)

    # Print final GPU usage
    if verbose and len(batches) > 0:
        gpu_monitor.print_final()

    return all_results


def process_video_with_questions(
    video_data: VideoEntry,
    model: Any,
    video_processor: Optional[VideoProcessor],
    question_processor: QuestionProcessor,
    max_tokens: int,
    max_frames: int,
    text_to_video_processor: TextToVideoProcessor = None,
    no_option_text: bool = False,
    start_question_index: int = 0,
    existing_results: Optional[List[Dict[str, Any]]] = None,
    binary_questions: bool = False,
    wandb_logger: Optional[WandbLogger] = None,
    input_mode: str = "video",
    sequence_processor: Optional[SequenceProcessor] = None,
    sequence_format_label: Optional[str] = None,
    eval_mode: str = "spatial",
    completed_questions: Optional[Set[str]] = None,
    state_manager: Optional[RunStateManager] = None,
    question_log_path: Optional[str] = None,
    question_log_rows: Optional[List[Dict[str, Any]]] = None,
    limit_questions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Process multiple questions for a single video with progressive main video addition."""

    def _extract_bucket_from_path(video_path: str) -> Optional[str]:
        """Extract bucket name from video path like 'videos/UNIFORM_EVAL_L008_ELOW/video_1_v0.mp4'."""
        if not video_path:
            return None
        # Split by / and look for pattern like UNIFORM_EVAL_*
        parts = video_path.split('/')
        for part in parts:
            if 'UNIFORM_EVAL' in part or 'L008' in part or 'L016' in part or 'L032' in part or 'L064' in part or 'L128' in part or 'L1024' in part:
                return part
        return None

    def _compute_entropy_features(question: QuestionEntry) -> Dict[str, Any]:
        metadata = getattr(question, 'metadata', None)
        if not metadata:
            return {}
        entropy_map = metadata.get('entropy_prefix')
        if not isinstance(entropy_map, dict) or not entropy_map:
            return {}
        try:
            values = [float(value) for value in entropy_map.values()]
        except (TypeError, ValueError):
            return {}
        if not values:
            return {}
        mean_entropy = sum(values) / len(values)
        return {
            'entropy_prefix_values': entropy_map,
            'entropy_prefix_mean': mean_entropy,
        }

    def _compute_video_entropy_features() -> Dict[str, Any]:
        """Extract video-level entropy from video_data.metadata.entropy_overall."""
        metadata = getattr(video_data, 'metadata', None)
        if not metadata or not isinstance(metadata, dict):
            return {}
        entropy_overall = metadata.get('entropy_overall')
        if not isinstance(entropy_overall, dict):
            return {}
        result = {}
        # Extract empirical_bits
        empirical = entropy_overall.get('empirical_bits')
        if isinstance(empirical, dict):
            result['entropy_empirical_bits'] = empirical
            try:
                empirical_values = [float(v) for v in empirical.values()]
                if empirical_values:
                    result['entropy_empirical_mean'] = sum(empirical_values) / len(empirical_values)
            except (TypeError, ValueError):
                pass
        # Extract analytic_bits
        analytic = entropy_overall.get('analytic_bits')
        if isinstance(analytic, dict):
            result['entropy_analytic_bits'] = analytic
            try:
                analytic_values = [float(v) for v in analytic.values()]
                if analytic_values:
                    result['entropy_analytic_mean'] = sum(analytic_values) / len(analytic_values)
            except (TypeError, ValueError):
                pass
        return result

    def _normalize_token_sequence(sequence: Sequence[Any]) -> List[str]:
        return [str(token) for token in sequence]

    def _count_sequence_occurrences(source: List[str], pattern: List[str]) -> int:
        if not source or not pattern or len(pattern) > len(source):
            return 0
        count = 0
        limit = len(source) - len(pattern) + 1
        for idx in range(limit):
            if source[idx: idx + len(pattern)] == pattern:
                count += 1
        return count

    def _compute_prefix_stats(question: QuestionEntry) -> Dict[str, Any]:
        sequences_used = video_data.metadata.get('sequences_used') if isinstance(video_data.metadata, dict) else None
        if not isinstance(sequences_used, dict) or not sequences_used:
            return {}

        option_count = len(question.options)
        if option_count == 0:
            return {}
        correct_index = question.correct_answer_index
        if not (0 <= correct_index < option_count):
            return {}
        option = question.options[correct_index]
        option_sequences = option.metadata.get('sequences') if isinstance(option.metadata, dict) else None
        fallback_sequence = option.metadata.get('sequence') if isinstance(option.metadata, dict) else None

        candidates: List[Tuple[str, List[Any]]] = []
        if isinstance(option_sequences, dict) and option_sequences:
            for key, value in option_sequences.items():
                if isinstance(value, list):
                    candidates.append((key, value))
        if not candidates and isinstance(fallback_sequence, list):
            candidates.append(('default', fallback_sequence))

        for seq_key, prefix_values in candidates:
            source_sequence = sequences_used.get(seq_key)
            if not isinstance(source_sequence, list):
                continue
            normalized_source = _normalize_token_sequence(source_sequence)
            normalized_prefix = _normalize_token_sequence(prefix_values)
            prefix_length = len(normalized_prefix)
            if prefix_length == 0 or prefix_length > len(normalized_source):
                continue
            max_positions = max(len(normalized_source) - prefix_length + 1, 0)
            if max_positions <= 0:
                continue
            occurrences = _count_sequence_occurrences(normalized_source, normalized_prefix)
            fraction = occurrences / max_positions if max_positions > 0 else 0.0
            return {
                'prefix_sequence_key': seq_key,
                'prefix_sequence_length': prefix_length,
                'prefix_total_positions': max_positions,
                'prefix_occurrences': occurrences,
                'prefix_match_fraction': fraction,
            }
        return {}

    def _compute_correct_option_likelihood(question: QuestionEntry) -> Optional[float]:
        options = question.options
        if not options:
            return None
        try:
            correct_index = int(question.correct_answer_index)
        except (TypeError, ValueError):
            return None
        if not (0 <= correct_index < len(options)):
            return None
        option_metadata = options[correct_index].metadata if isinstance(options[correct_index].metadata, dict) else None
        if not option_metadata:
            return None
        likelihood = option_metadata.get('likelihood_combined')
        if likelihood is not None:
            try:
                return float(likelihood)
            except (TypeError, ValueError):
                return None
        likelihoods = option_metadata.get('likelihoods')
        if isinstance(likelihoods, dict) and likelihoods:
            try:
                return float(max(likelihoods.values()))
            except (TypeError, ValueError):
                return None
        return None

    def _snapshot_metric_lengths(metrics_obj: Optional[PerformanceMetrics]) -> Optional[Dict[str, int]]:
        if metrics_obj is None:
            return None
        snapshot: Dict[str, int] = {}
        fields = getattr(metrics_obj, '__dataclass_fields__', {})
        for name in fields:
            value = getattr(metrics_obj, name, None)
            if isinstance(value, list):
                snapshot[name] = len(value)
        return snapshot

    def _restore_metrics_to_snapshot(
        metrics_obj: Optional[PerformanceMetrics],
        snapshot: Optional[Dict[str, int]],
    ) -> None:
        if metrics_obj is None or not snapshot:
            return
        for name, length in snapshot.items():
            value = getattr(metrics_obj, name, None)
            if isinstance(value, list) and len(value) > length:
                del value[length:]

    def _latest_metric_value(metric_name: str) -> Optional[float]:
        metrics_obj = getattr(model, '_metrics', None)
        if metrics_obj is None:
            return None
        values = getattr(metrics_obj, metric_name, None)
        if isinstance(values, list) and values:
            try:
                return float(values[-1])
            except (TypeError, ValueError):
                return None
        return None

    use_sequence_mode = input_mode == "sequence"
    if use_sequence_mode and sequence_processor is None:
        raise ValueError("Sequence mode requires a SequenceProcessor instance.")
    if not use_sequence_mode and video_processor is None:
        raise ValueError("Video mode requires a VideoProcessor instance.")

    video_index = video_data.video_index
    variant = getattr(video_data, 'variant', 0)  # Default to 0 to match batching save calls
    bucket = _extract_bucket_from_video_data(video_data)
    questions = sorted(video_data.questions, key=lambda q: float(q.question_time))
    for order, question in enumerate(questions, start=1):
        question.question_order = order
    completed_set = set(completed_questions or set())

    print(f"\nProcessing Video {video_index} with {len(questions)} questions")
    print("Pre clear context", flush=True)
    # Clear context completely for fresh start
    model.clear_context()
    print("Post clear context", flush=True)
    if not use_sequence_mode:
        # Load main video frames (but don't add to model yet)
        assert video_processor is not None
        main_video = video_processor.load_main_video(video_data.main_video_path)
        print("Post  load video", flush=True)
    else:
        main_video = None

    results: List[Dict[str, Any]] = [] if existing_results is None else list(existing_results)
    sequence_base_state = None
    base_sequence_time = 0.0
    base_sequence_statements: List[str] = []
    if use_sequence_mode and sequence_processor is not None:
        base_sequence_time, base_sequence_statements = sequence_processor.stream_full_sequences(
            model,
            base_time=0.0,
        )
        sequence_base_state = model.save_state()

    # Count valid completed questions (respects eval_mode filtering)
    valid_completed = count_valid_completed_questions(completed_set or set(), video_data, eval_mode)
    total_valid = count_total_valid_questions(video_data, eval_mode)

    # Check if bucket is already at/over limit
    if limit_questions is not None and valid_completed >= limit_questions:
        print(f"  [SKIP] Bucket already complete: {valid_completed}/{limit_questions} valid questions done")
        return results, True  # True = hit limit

    # Process each question progressively
    new_valid_count = 0  # Track new valid questions processed in THIS session

    for i, question_data in enumerate(questions):
        if i < start_question_index:
            continue

        question_identifier = question_data.question_id or f"video{video_index}_q{i}"
        if completed_set and question_identifier in completed_set:
            print(f"Skipping previously completed question {question_identifier}")
            continue

        # Filter questions based on eval_mode
        if eval_mode == 'sequential':
            question_variant = question_data.metadata.get('question_variant', '').lower()
            if question_variant == 'spatial':
                print(f"Skipping spatial question {question_identifier} in sequential mode")
                continue

        # Check TOTAL count (valid_completed + new_valid_count)
        if limit_questions is not None and (valid_completed + new_valid_count) >= limit_questions:
            print(f"Reached question limit ({limit_questions}) for video {video_index} (completed={valid_completed}, new={new_valid_count}), moving to next video")
            return results, True  # True = hit limit

        new_valid_count += 1

        metrics_snapshot: Optional[Dict[str, int]] = None
        metrics_obj = getattr(model, '_metrics', None)
        if metrics_obj is not None:
            metrics_snapshot = _snapshot_metric_lengths(metrics_obj)

        question_mode = (question_data.question_mode or "").strip().lower()

        print("Starting question", flush=True)
        question_time = question_data.question_time
        print(f"\nProcessing Question {i+1} at time {question_time}s")

        base_state = None
        oom_record_time: Optional[float] = None

        # =====================================================================
        # Native Binary Question Handling
        # =====================================================================
        # Check if this is a native binary format question (single candidate with yes/no answer)
        is_native_binary = getattr(question_data, 'is_native_binary', False)
        if is_native_binary:
            try:
                print(f"[native-binary] Processing native binary question {question_identifier}")

                # Restore base state for sequence mode or clear context for video mode
                sequence_context_statements: List[str] = []
                if use_sequence_mode:
                    assert sequence_processor is not None
                    if sequence_base_state is not None:
                        model.load_state(sequence_base_state)
                    actual_time = base_sequence_time
                    sequence_context_statements = list(base_sequence_statements)

                    # For continuation mode, stream prefix
                    if question_mode == "continuation":
                        actual_time, prefix_statements = sequence_processor.stream_question_prefix(
                            model,
                            question_data,
                            base_time=actual_time,
                        )
                        if prefix_statements:
                            sequence_context_statements.extend(prefix_statements)

                    # Add candidate sequence
                    candidate = question_data.candidate
                    if candidate and candidate.sequence:
                        candidate_text = f"\nCandidate sequence: {', '.join(candidate.sequence)}\n"
                        model.add_text(candidate_text, current_video_time=actual_time)
                        actual_time += 1.0
                    frames_seen_before_question = 0
                else:
                    # Video mode: load main video and candidate clip
                    assert video_processor is not None
                    model.clear_context()
                    video_processor.reset_video_position()  # Reset cursor after clearing context
                    prefix_text = "Here is a main video to remember:"
                    model.add_text(prefix_text, current_video_time=0.0)
                    actual_time = video_processor.add_main_video_up_to_time(model, question_time)
                    frames_seen_before_question = video_processor.get_main_frames_streamed()

                    # Add candidate clip
                    candidate = question_data.candidate
                    if candidate and candidate.clip_path:
                        # Handle no_option_text mode: render label as video frames
                        if no_option_text and text_to_video_processor:
                            # Create "Candidate:" label as video frames
                            text_frames = text_to_video_processor.create_option_label_frames_for_model(
                                "Candidate",  # Label for native binary
                                video_processor.frame_sampler,
                                duration_seconds=2.0,
                            )
                            text_frame_count = text_to_video_processor.get_frame_count_for_model(
                                text_frames,
                                video_processor.frame_sampler,
                            )
                            if text_frame_count > 0:
                                text_start_time = actual_time + 1
                                text_end_time = text_start_time + text_frame_count / video_processor.fps
                                actual_time = text_end_time
                                model.add_video(
                                    video_frames=text_frames,
                                    time_start=text_start_time,
                                    time_end=text_end_time,
                                    video_id="candidate_label"
                                )
                        else:
                            # Normal mode: add text label
                            candidate_intro = "\nHere is a candidate clip:\n"
                            model.add_text(candidate_intro, current_video_time=actual_time)

                        # Load and add the candidate video clip
                        candidate_clips = video_processor.load_option_videos([candidate.clip_path])
                        if candidate_clips:
                            candidate_video = candidate_clips[0]
                            frame_count = video_processor.frame_sampler.get_frame_count(candidate_video)
                            if frame_count > 0:
                                candidate_start = actual_time + 1
                                candidate_end = candidate_start + frame_count / video_processor.fps
                                model.add_video(
                                    video_frames=candidate_video,
                                    time_start=candidate_start,
                                    time_end=candidate_end,
                                    video_id="candidate_clip"
                                )
                                actual_time = candidate_end

                oom_record_time = float(actual_time)

                # Process the native binary question.
                # LongVILA over-commits to the "{uncertain}" option on NO clips.
                # Drop the uncertain option for LongVILA only so it is forced
                # into {0}/{1}.
                # LongVILA has text and video batching disabled in the registry,
                # so this single-question path is the only live one for it.
                _use_uncertain_option = type(model).__name__ != 'LongVILAModel'
                result = question_processor.process_native_binary_question(
                    model,
                    question=question_data,
                    video_index=video_index,
                    max_tokens=max_tokens,
                    max_frames=max_frames,
                    current_video_time=actual_time,
                    include_uncertain=_use_uncertain_option,
                )

                # Add entropy features
                entropy_features = _compute_entropy_features(question_data)
                if entropy_features:
                    result.update(entropy_features)
                video_entropy_features = _compute_video_entropy_features()
                if video_entropy_features:
                    result.update(video_entropy_features)
                prefix_stats = _compute_prefix_stats(question_data)
                if prefix_stats:
                    result.update(prefix_stats)
                likelihood_value = _compute_correct_option_likelihood(question_data)
                if likelihood_value is not None:
                    result['correct_option_likelihood'] = likelihood_value

                # Add token metrics
                token_metrics = _extract_token_stats(model, result.get('response', ''))
                if token_metrics:
                    result.update(token_metrics)

                # Check if output was truncated
                response_tokens = result.get('response_token_count')
                if response_tokens is not None and max_tokens:
                    result['output_was_truncated'] = response_tokens >= (max_tokens - 5)
                    result['max_tokens'] = max_tokens
                else:
                    result['output_was_truncated'] = None

                # Add bucket information
                bucket_name = _extract_bucket_from_path(str(video_data.video_path))
                if bucket_name:
                    result['bucket'] = bucket_name

                # Add metadata to result
                result['frames_seen_before_question'] = frames_seen_before_question
                result['question_order'] = question_data.question_order
                result['question_mode'] = question_data.question_mode
                result['eval_mode'] = eval_mode
                result['input_mode'] = input_mode
                result['video_only_mode'] = no_option_text

                if use_sequence_mode:
                    result['sequence_context'] = list(sequence_context_statements)
                    result['sequence_format'] = sequence_format_label or 'comma'

                if hasattr(question_data, 'metadata') and question_data.metadata:
                    if 'question_variant' in question_data.metadata:
                        result['question_variant'] = question_data.metadata['question_variant']
                    if 'question_type' in question_data.metadata:
                        result['question_type'] = question_data.metadata['question_type']
                    # New fields from eval_membership dataset
                    if 'has_unique_answer' in question_data.metadata:
                        result['has_unique_answer'] = question_data.metadata['has_unique_answer']
                    if 'scenario' in question_data.metadata:
                        result['scenario'] = question_data.metadata['scenario']

                # Add candidate-level fields for native binary questions
                if question_data.candidate:
                    result['candidate_present'] = question_data.candidate.present
                    result['candidate_clip_start'] = question_data.candidate.clip_start
                    result['candidate_clip_end'] = question_data.candidate.clip_end

                try:
                    result['question_time'] = float(question_time)
                except (TypeError, ValueError):
                    result['question_time'] = None

                results.append(result)

                # CSV logging
                csv_row = {
                    'video_id': video_data.video_index,
                    'variant': variant if variant is not None else 0,
                    'bucket': result.get('bucket'),
                    'question_id': question_identifier,
                    'question_order': result.get('question_order'),
                    'video_entropy': result.get('entropy_prefix_mean'),
                    'correct_answer': result.get('correct'),
                    'model_answer': result.get('predicted'),
                    'is_correct': result.get('is_correct'),
                    'is_dont_know': result.get('is_dont_know'),
                    'response': result.get('response', ''),
                    'num_options': result.get('num_options'),
                    'is_native_binary': True,
                    'question_type': result.get('question_type'),
                    'question_variant': result.get('question_variant'),
                    'question_time': result.get('question_time'),
                    'clip_start_time': question_data.clip_start_time,
                    'clip_end_time': question_data.clip_end_time,
                    'candidate_present': question_data.candidate.present if question_data.candidate else None,
                    'candidate_clip_start': question_data.candidate.clip_start if question_data.candidate else None,
                    'candidate_clip_end': question_data.candidate.clip_end if question_data.candidate else None,
                    'has_unique_answer': result.get('has_unique_answer'),
                    'scenario': result.get('scenario'),
                    'response_token_count': result.get('response_token_count'),
                    'output_was_truncated': result.get('output_was_truncated'),
                    'prompt_tokens': result.get('input_tokens'),
                    'completion_tokens': result.get('output_tokens'),
                    'eval_mode': eval_mode,
                    'input_mode': input_mode,
                }

                if question_log_path and question_log_rows is not None:
                    question_log_rows.append(csv_row)

                if state_manager is not None:
                    state_manager.mark_question_complete(video_index, question_identifier, result, variant=variant, bucket=bucket)
                    completed_set.add(question_identifier)

                if wandb_logger:
                    wandb_logger.log_question_record(result)

                if termination_requested():
                    print("Termination requested after completing question; exiting question loop.")
                    break

                # Continue to next question
                continue

            except Exception as native_err:
                print(f"[native-binary] Error processing native binary question: {native_err}")
                if metrics_snapshot is not None:
                    _restore_metrics_to_snapshot(metrics_obj, metrics_snapshot)
                raise
        # =====================================================================
        # End Native Binary Question Handling
        # =====================================================================

        try:
            sequence_context_statements: List[str] = []
            if use_sequence_mode:
                assert sequence_processor is not None
                if sequence_base_state is not None:
                    model.load_state(sequence_base_state)
                actual_time = base_sequence_time
                sequence_context_statements = list(base_sequence_statements)
                if question_mode == "continuation":
                    actual_time, prefix_statements = sequence_processor.stream_question_prefix(
                        model,
                        question_data,
                        base_time=actual_time,
                    )
                    if prefix_statements:
                        sequence_context_statements.extend(prefix_statements)
                frames_seen_before_question = 0
                print("Finished streaming sequence context", flush=True)
            else:
                assert video_processor is not None
                # Add prefix text for new prompt format
                prefix_text = (
                    "Here is a main video to remember:"
                    if not use_sequence_mode
                    else "Here is a main sequence to remember:"
                )
                model.add_text(prefix_text, current_video_time=0.0)
                # Add main video up to this question's time point
                actual_time = video_processor.add_main_video_up_to_time(model, question_time)
                print("Finished one adding video", flush=True)
                frames_seen_before_question = video_processor.get_main_frames_streamed()
            oom_record_time = float(actual_time)

            # Save state after adding main video portion (before instruction/options)
            base_state = model.save_state()

            # Add question intro text after main video, before options
            if use_sequence_mode:
                # Keep sequence mode instruction as-is (complex logic for continuation vs exists)
                if question_mode == "continuation":
                    prefix_text = question_processor._format_primary_prefix(question_data)  # pylint: disable=protected-access
                    if prefix_text:
                        instruction_text = (
                            "You just saw the token sequence. Determine which option continues the prefix "
                            f"[{prefix_text}] with the exact next tokens in the original sequence, preserving the original order with no skipped or inserted tokens. More than one continuation may exist; select the continuation that matches one of the provided options."
                        )
                    else:
                        instruction_text = (
                            "You just saw the token sequence with the highlighted prefix. Determine which option continues the prefix with the exact next tokens in the original sequence, in the same order and without adding extra tokens."
                        )
                else:
                    instruction_text = (
                        "You just saw the entire token sequence. Determine which option's subsequence appears somewhere in that sequence."
                    )
                adjust_context = getattr(question_processor, 'adjust_text_for_context', None)
                if callable(adjust_context):
                    instruction_text = adjust_context(instruction_text)
                instruction_payload = f"\n\n{instruction_text}\n\n"
                model.add_text(instruction_payload, current_video_time=actual_time)
            else:
                # New concise format for spatial/video mode
                question_intro = "\nQuestion: Of the following options, which one appeared in the main video?\n"
                model.add_text(question_intro, current_video_time=actual_time)
            if use_sequence_mode:
                print(f"[sequence-mode] Instruction: {instruction_text}", flush=True)

            # Load and add this question's option videos with labels
            running_time = actual_time

            selected_binary_answer: Optional[str] = None
            selected_option_index: Optional[int] = None

            if binary_questions:
                seed_components = [
                    str(video_index),
                    str(i),
                    str(question_data.question_id),
                ]
                seed_material = "|".join(seed_components)
                seed_value = int(hashlib.sha256(seed_material.encode('utf-8')).hexdigest(), 16)
                rng = random.Random(seed_value)

                correct_option_index = int(question_data.correct_answer_index)
                if correct_option_index == question_data.dont_know_index:
                    raise ValueError(
                        "Binary question mode cannot be used when the ground truth is the uncertain option."
                    )

                should_answer_yes = rng.getrandbits(1) == 0
                if should_answer_yes or len(question_data.options) <= 1:
                    selected_option_index = correct_option_index
                    selected_binary_answer = "yes"
                else:
                    incorrect_indices = [
                        idx for idx in range(len(question_data.options)) if idx != correct_option_index
                    ]
                    if incorrect_indices:
                        selected_option_index = incorrect_indices[rng.randrange(len(incorrect_indices))]
                        selected_binary_answer = "no"
                    else:
                        selected_option_index = correct_option_index
                        selected_binary_answer = "yes"

                if selected_option_index is None:
                    raise ValueError("Binary question mode requires at least one playable option.")

                selected_option = question_data.options[selected_option_index]
                if use_sequence_mode:
                    option_sequence = [(selected_option_index, selected_option, None)]
                else:
                    assert video_processor is not None
                    option_videos = video_processor.load_option_videos([selected_option.clip_path])
                    if not option_videos:
                        raise ValueError("Failed to load the selected binary option clip.")
                    option_sequence = [(selected_option_index, selected_option, option_videos[0])]
            else:
                if use_sequence_mode:
                    option_sequence = [
                        (idx, option, None)
                        for idx, option in enumerate(question_data.options)
                    ]
                else:
                    assert video_processor is not None
                    option_paths = [option.clip_path for option in question_data.options]
                    option_videos = video_processor.load_option_videos(option_paths)
                    if len(option_videos) != len(option_paths):
                        raise ValueError(
                            "Mismatch between loaded option clips and manifest entries; expected "
                            f"{len(option_paths)} clips but received {len(option_videos)}."
                        )
                    option_sequence = [
                        (idx, option, option_videos[idx])
                        for idx, option in enumerate(question_data.options)
                    ]

            for option_label_index, option_entry, option_payload in option_sequence:
                label_suffix = ""

                if use_sequence_mode:
                    assert sequence_processor is not None
                    option_text = sequence_processor.build_option_statement(
                        option_label_index,
                        option_entry,
                        label_suffix,
                    )
                    option_payload = f"\n{option_text}\n"
                    model.add_text(option_payload, current_video_time=running_time)
                    running_time += 1.0
                    oom_record_time = float(running_time)
                    continue

                assert video_processor is not None
                option_text = f"\nOption {option_label_index}:\n"

                if no_option_text and text_to_video_processor:
                    text_frames = text_to_video_processor.create_option_label_frames_for_model(
                        option_label_index,
                        video_processor.frame_sampler,
                        duration_seconds=2.0,
                    )
                    text_frame_count = text_to_video_processor.get_frame_count_for_model(
                        text_frames,
                        video_processor.frame_sampler,
                    )

                    if text_frame_count > 0:
                        text_start_time = running_time + 1  # 1 second gap
                        text_end_time = text_start_time + text_frame_count / video_processor.fps
                        running_time = text_end_time

                        model.add_video(
                            video_frames=text_frames,
                            time_start=text_start_time,
                            time_end=text_end_time,
                            video_id=f"option_label_{option_label_index}"
                        )
                else:
                    model.add_text(option_text, current_video_time=running_time)

                option_video = option_payload
                frame_count = video_processor.frame_sampler.get_frame_count(option_video)
                if frame_count > 0:
                    option_start_time = running_time + 1  # 1 second gap
                    option_end_time = option_start_time + frame_count / video_processor.fps
                    running_time = option_end_time
                else:
                    raise ValueError("Option video has zero frames after sampling.")

                model.add_video(
                    video_frames=option_video,
                    time_start=option_start_time,
                    time_end=option_end_time,
                    video_id=option_label_index + 1
                )

                if option_end_time > 0:
                    oom_record_time = float(option_end_time)

            # Process the question with complex format
            oom_record_time = float(actual_time)
            result = question_processor.process_single_question(
                model,
                question=question_data,
                video_index=video_index,
                max_tokens=max_tokens,
                max_frames=max_frames,
                current_video_time=actual_time,
                binary_selected_option_index=selected_option_index,
                binary_correct_answer=selected_binary_answer,
            )
            entropy_features = _compute_entropy_features(question_data)
            if entropy_features:
                result.update(entropy_features)
            video_entropy_features = _compute_video_entropy_features()
            if video_entropy_features:
                result.update(video_entropy_features)
            prefix_stats = _compute_prefix_stats(question_data)
            if prefix_stats:
                result.update(prefix_stats)
            likelihood_value = _compute_correct_option_likelihood(question_data)
            if likelihood_value is not None:
                result['correct_option_likelihood'] = likelihood_value
            result['frames_seen_before_question'] = frames_seen_before_question
            result['question_order'] = question_data.question_order
            if use_sequence_mode:
                result['sequence_context'] = list(sequence_context_statements)
                result['sequence_format'] = sequence_format_label or 'comma'
            if question_data.question_mode:
                result['question_mode'] = question_data.question_mode

            # Add mode tracking fields
            result['video_only_mode'] = no_option_text
            result['eval_mode'] = eval_mode
            result['input_mode'] = input_mode

            # Add per-question type fields from JSON
            if hasattr(question_data, 'metadata'):
                metadata = question_data.metadata
                if metadata:
                    if 'question_variant' in metadata:
                        result['question_variant'] = metadata['question_variant']
                    if 'question_type' in metadata:
                        result['question_type'] = metadata['question_type']
                    # New fields from eval_membership dataset
                    if 'has_unique_answer' in metadata:
                        result['has_unique_answer'] = metadata['has_unique_answer']
                    if 'scenario' in metadata:
                        result['scenario'] = metadata['scenario']

            # Add candidate-level fields if available (for binary questions)
            if hasattr(question_data, 'candidate') and question_data.candidate:
                result['candidate_present'] = question_data.candidate.present
                result['candidate_clip_start'] = question_data.candidate.clip_start
                result['candidate_clip_end'] = question_data.candidate.clip_end

            token_metrics = _extract_token_stats(model, result.get('response', ''))
            if token_metrics:
                result.update(token_metrics)

            # Check if output was truncated (hit max_tokens limit)
            response_tokens = result.get('response_token_count')
            if response_tokens is not None and 'max_tokens' in result:
                # Consider truncated if within 5 tokens of limit (some models may overshoot slightly)
                result['output_was_truncated'] = response_tokens >= (result['max_tokens'] - 5)
            else:
                result['output_was_truncated'] = None

            latency_value = _latest_metric_value('latency_ask_question')
            if latency_value is not None:
                result['latency_ask_question'] = latency_value
            peak_gpu_value = _latest_metric_value('peak_gpu_mem_increase_ask_question')
            if peak_gpu_value is not None:
                result['peak_gpu_mem_ask_question'] = peak_gpu_value
            try:
                result['question_time'] = float(question_time)
            except (TypeError, ValueError):
                result['question_time'] = None
            results.append(result)

            # Prepare CSV log row with enhanced data
            csv_row = {
                'video_id': video_data.video_index,
                'variant': variant if variant is not None else 0,
                'question_id': question_identifier,
                'question_order': result.get('question_order'),
                'video_entropy': result.get('entropy_prefix_mean'),
                'correct_answer': result.get('correct'),
                'model_answer': result.get('predicted'),
                'is_correct': result.get('is_correct'),
                'is_dont_know': result.get('is_dont_know'),
                'response': result.get('response', ''),
                'num_options': result.get('num_options'),
                'question_type': result.get('question_type'),
                'question_variant': result.get('question_variant'),
                'question_time': result.get('question_time'),
                'clip_start_time': question_data.clip_start_time,
                'clip_end_time': question_data.clip_end_time,
                'candidate_present': result.get('candidate_present'),
                'candidate_clip_start': result.get('candidate_clip_start'),
                'candidate_clip_end': result.get('candidate_clip_end'),
                'has_unique_answer': result.get('has_unique_answer'),
                'scenario': result.get('scenario'),
                'prompt_tokens': result.get('input_tokens'),
                'completion_tokens': result.get('output_tokens'),
                'eval_mode': eval_mode,
                'input_mode': input_mode,
            }

            if question_log_path and question_log_rows is not None:
                question_log_rows.append(csv_row)

            if state_manager is not None:
                state_manager.mark_question_complete(video_index, question_identifier, result, variant=variant, bucket=bucket)
                completed_set.add(question_identifier)

            if wandb_logger:
                wandb_logger.log_question_record(result)

            if termination_requested():
                print("Termination requested after completing question; exiting question loop.")
                break

            # Load state back to just main video (before instruction/options) for next question
            if i < len(questions) - 1:
                if not use_sequence_mode:
                    model.load_state(base_state)
                    if video_processor is not None:
                        video_processor.reset_to_main_video_state()  # Sync tracking

        except Exception as exc:
            if _is_cuda_oom_error(exc):
                oom_timestamp = (
                    float(question_data.question_time)
                    if oom_record_time is None
                    else oom_record_time
                )
                if hasattr(model, "record_first_oom_snapshot") and hasattr(model, "_metrics"):
                    model.record_first_oom_snapshot(oom_timestamp)
                # Attempt to revert to clean state before propagating retry request
                if base_state is not None:
                    try:
                        if hasattr(model, "recover_from_oom"):
                            model.recover_from_oom(base_state)
                        else:
                            model.clear_context()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                reset_peak = getattr(torch.cuda, "reset_peak_memory_stats", None)
                                if callable(reset_peak):
                                    reset_peak()
                            model.load_state(base_state)
                    except Exception:
                        pass
                else:
                    try:
                        if hasattr(model, "teardown_after_oom"):
                            model.teardown_after_oom()
                        else:
                            model.clear_context()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                reset_peak = getattr(torch.cuda, "reset_peak_memory_stats", None)
                                if callable(reset_peak):
                                    reset_peak()
                    except Exception:
                        pass
                if metrics_snapshot is not None:
                    _restore_metrics_to_snapshot(getattr(model, '_metrics', None), metrics_snapshot)
                    if hasattr(model, '_sync_state_memory_tracking_from_metrics'):
                        model._sync_state_memory_tracking_from_metrics()

                partial_metrics = None
                if getattr(model, '_metrics', None) is not None:
                    partial_metrics = copy.deepcopy(model._metrics)
                raise QuestionOOMRetry(
                    i,
                    copy.deepcopy(results),
                    exc,
                    oom_timestamp,
                    partial_metrics,
                ) from exc
            raise

    # Clear context only after ALL questions for this video
    model.clear_context()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return results and whether we stopped due to question limit
    hit_question_limit = (limit_questions is not None and
                          len(questions) > limit_questions)
    return results, hit_question_limit


def get_canonical_question_list(
    video_data: Any,
    eval_mode: str,
    limit_questions: Optional[int],
) -> List[str]:
    """
    Get the canonical list of question IDs that should be processed for this video/bucket.

    This is DETERMINISTIC - always returns the same list for the same inputs.
    The list represents the "first N" questions that qualify based on eval_mode.

    Args:
        video_data: Video entry with questions
        eval_mode: 'sequential' or 'spatial'
        limit_questions: Max questions to include (None = all)

    Returns:
        Ordered list of question_ids that should be processed
    """
    canonical_list = []

    for i, question_data in enumerate(video_data.questions):
        question_id = question_data.question_id or f"video{video_data.video_index}_q{i}"

        # Filter by eval_mode
        if eval_mode == 'sequential':
            variant = question_data.metadata.get('question_variant', '').lower()
            if variant == 'spatial':
                continue  # Skip spatial in sequential mode

        canonical_list.append(question_id)

        # Stop once we reach the limit
        if limit_questions is not None and len(canonical_list) >= limit_questions:
            break

    return canonical_list


def complete_remaining_questions(
    videos: List[Any],
    model: Any,
    state_manager: Any,
    args: Any,
    question_processor: Any,
    sequence_formatter: Optional[Any],
    use_sequence_mode: bool,
) -> int:
    """
    Fill in gaps for incomplete buckets after main pass.

    Goes through each video/bucket and processes any missing questions
    from the CANONICAL list (first N questions as determined by get_canonical_question_list).

    This is idempotent - safe to call multiple times.
    Questions are added to the bucket's state as they're completed.

    Returns: Number of additional questions processed
    """
    print(f"\n{'='*60}")
    print("COMPLETION PASS - Filling in gaps for incomplete buckets")
    print(f"{'='*60}\n")

    completion_count = 0

    for video_data in videos:
        variant = getattr(video_data, 'variant', 0)  # Default to 0 to match batching save calls
        bucket = _extract_bucket_from_video_data(video_data)

        # Skip if already marked complete
        if state_manager.is_video_completed(video_data.video_index, variant, bucket):
            continue

        # Get the CANONICAL list of questions for this bucket (deterministic)
        canonical_questions = get_canonical_question_list(
            video_data,
            args.eval_mode,
            args.limit_questions
        )

        # Get completed questions
        completed_set = state_manager.get_completed_questions(video_data.video_index, variant, bucket)

        # Find missing questions from the canonical list
        missing_question_ids = [qid for qid in canonical_questions if qid not in completed_set]

        if not missing_question_ids:
            # No missing questions, mark complete
            print(f"Video {video_data.video_index} bucket {bucket}: All {len(canonical_questions)} canonical questions complete, marking bucket complete")
            state_manager.mark_video_complete(video_data.video_index, variant, bucket)
            continue

        print(f"Video {video_data.video_index} bucket {bucket}: {len(completed_set)}/{len(canonical_questions)} complete, processing {len(missing_question_ids)} remaining")

        # Build list of questions to process (in canonical order)
        questions_to_process = []
        for question_id in missing_question_ids:
            # Find the question data
            for i, question_data in enumerate(video_data.questions):
                qid = question_data.question_id or f"video{video_data.video_index}_q{i}"
                if qid == question_id:
                    questions_to_process.append((i, question_data, question_id))
                    break

        if not questions_to_process:
            # This shouldn't happen if missing_question_ids was non-empty, but safety check
            print(f"  WARNING: Could not find question data for missing questions, skipping")
            continue

        # Process missing questions (fill gaps to complete the canonical list)
        try:
            video_results = 0

            # Load video/sequence context
            if use_sequence_mode:
                if sequence_formatter is None:
                    print(f"  ERROR: Sequence formatter not available, skipping")
                    continue

                # Create processor for this video (same pattern as main loop)
                raw_sequences = video_data.metadata.get('sequences_used')
                if not isinstance(raw_sequences, dict) or not raw_sequences:
                    print(f"  ERROR: Video {video_data.video_index} missing 'sequences_used' metadata, skipping")
                    continue

                video_sequence_processor = SequenceProcessor(
                    sequences_used=raw_sequences,
                    formatter=sequence_formatter,
                    print_chunks=True,
                )

                # Stream sequences
                video_sequence_processor.stream_full_sequences(model, base_time=0.0)
            else:
                # Load video (simplified - just use question processor)
                pass

            for i, question_data, question_id in questions_to_process:
                try:
                    # Process single question
                    result = question_processor.process_single_question(
                        model=model,
                        question=question_data,
                        video_index=video_data.video_index,
                        max_tokens=args.max_tokens,
                    )

                    video_results += 1
                    completion_count += 1

                    # Save progress immediately to bucket state
                    state_manager.mark_question_complete(
                        video_data.video_index,
                        question_id,
                        result,
                        variant=variant,
                        bucket=bucket
                    )

                except Exception as e:
                    print(f"  Failed to process question {question_id}: {e}")
                    # Skip this question, continue with others
                    continue

            # Check if bucket is now complete
            # Re-fetch completed set to include newly completed questions
            completed_set_after = state_manager.get_completed_questions(video_data.video_index, variant, bucket)
            remaining = [qid for qid in canonical_questions if qid not in completed_set_after]

            if not remaining:
                print(f"  OK: Bucket complete: all {len(canonical_questions)} canonical questions done")
                state_manager.mark_video_complete(video_data.video_index, variant, bucket)
            else:
                print(f"  WARNING: Processed {video_results} questions, {len(remaining)} still missing from canonical list")

        except Exception as e:
            print(f"  ERROR: Could not complete bucket {bucket}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nCompletion pass finished. Processed {completion_count} additional questions.\n")
    return completion_count


def _resolve_data(data: str, asset_root_flag: Optional[str]) -> Tuple[str, Optional[str], str]:
    """Resolve --data into (manifest_path, asset_root, label).

    Accepts a local manifest path (media resolved against --asset-root), a
    published Hub config name, or '<repo_id>:<config>' for a copy of the
    dataset repo.
    """
    if os.path.isfile(data):
        return data, asset_root_flag, Path(data).stem
    if data in hf_loader.CONFIGS:
        manifest, root = hf_loader.fetch(data)
        return str(manifest), str(root), data
    if data.count(":") == 1:
        repo, config_name = data.split(":")
        if config_name in hf_loader.CONFIGS:
            manifest, root = hf_loader.fetch(config_name, repo_id=repo)
            return str(manifest), str(root), config_name
    raise SystemExit(
        f"--data {data!r} is neither an existing manifest file, a Hub config name "
        f"({', '.join(hf_loader.CONFIGS)}), nor a '<repo_id>:<config>' reference."
    )


def main() -> None:
    """Main evaluation function."""
    _install_signal_handlers()
    parser = argparse.ArgumentParser(
        description="Substream Recollection evaluation runner",
        epilog="""
Examples:
  %(prog)s --data text --model qwen_full --mode text --variant sequential
  %(prog)s --data synthetic_video --model internvl-3-5
  %(prog)s --data /path/to/manifest.json --asset-root /path/to/media --model dummy_eval
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        help=(
            "Dataset to evaluate: a local manifest JSON path, a Hub config name "
            f"({', '.join(hf_loader.CONFIGS)}), or '<repo_id>:<config>'."
        ),
    )
    parser.add_argument("--model", required=True, choices=sorted(REGISTRY),
                        help="Registry alias of the model to evaluate.")
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default=None,
        help="OpenRouter model slug (e.g. 'google/gemini-3.1-pro-preview'). Required when --model openrouter.",
    )
    parser.add_argument("--mode", choices=["text", "video"], default="video",
                        help="Present the stream as rendered video (default) or as text token sequences.")
    parser.add_argument("--variant", choices=["sequential", "spatial"], default="spatial",
                        help="Question variant: 'spatial' asks all questions; 'sequential' keeps only order-based ones.")
    parser.add_argument("--buckets", action="append", default=None,
                        help="Only process videos whose path contains this bucket name "
                             "(e.g. UNIFORM_EVAL_L008_ELOW). Repeatable; a video matches if ANY filter matches.")
    parser.add_argument("--arm", choices=["A", "B"], default=None,
                        help="Calibration arm; exported as CALIB_ARM before any prompt is built.")
    parser.add_argument("--rag", action="store_true",
                        help="Prefix prompts with the RAG preamble (exports RAG_PREAMBLE=1).")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--limit-questions", type=int, default=None,
                        help="Limit number of questions to process per video (in sorted order).")
    parser.add_argument("--fps", type=float, default=1, help="Frames per second to sample")
    parser.add_argument("--max-frames", type=int, default=5000, help="Maximum frames per video")
    parser.add_argument("--chunk-size", type=float, default=15,
                        help="Stream video into the model in chunks of this many seconds.")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Maximum tokens to generate per response")
    parser.add_argument(
        "--asset-root",
        type=str,
        default=None,
        help="Root directory that prefixes relative media paths of a local manifest (ignored for Hub configs).",
    )
    parser.add_argument("--out", type=str, default=str(config.RUN_ROOT),
                        help="Directory for the run's state file and CSV log (default: SUBSTREAM_RUN_ROOT).")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name; outputs go to <out>/<name>.state and <out>/<name>.csv "
                             "(default: <model>_<data>_<mode>_<variant>).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing state file instead of refusing to overwrite it.")
    parser.add_argument("--enable-metrics", action="store_true",
                        help="Enable performance metrics collection, reporting, and wandb logging.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full questions and responses for debugging")
    parser.add_argument("--max-gpu-mem", type=float, default=None,
                        help="Per-GPU memory budget in GiB for VLM checkpoints. Auto-derived from model size if unset.")
    parser.add_argument(
        "--restart-on-oom",
        action="store_true",
        help="Reinitialize the model and retry the current video when a CUDA OOM is encountered.",
    )
    parser.add_argument(
        "--max-oom-retries",
        type=int,
        default=35,
        help="Maximum number of retry attempts per video after a CUDA OOM before the run aborts.",
    )

    args = parser.parse_args()

    # Prompt-affecting env vars must be exported before any question is built.
    if args.arm:
        os.environ["CALIB_ARM"] = args.arm
    if args.rag:
        os.environ["RAG_PREAMBLE"] = "1"

    manifest_path, resolved_asset_root, data_label = _resolve_data(args.data, args.asset_root)

    # The processing loop below predates this CLI; expose the names it expects.
    args.json_path = manifest_path
    args.asset_root = resolved_asset_root
    args.input_mode = "sequence" if args.mode == "text" else "video"
    args.eval_mode = args.variant
    args.bucket_filter = args.buckets

    if args.name is None:
        args.name = f"{args.model}_{data_label}_{args.mode}_{args.variant}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.state_file = str(out_dir / f"{args.name}.state")
    args.question_log_csv = str(out_dir / f"{args.name}.csv")

    if not args.resume and Path(args.state_file).exists():
        parser.error(
            f"State file {args.state_file} already exists. Use --resume to continue or remove the file."
        )

    state_manager = RunStateManager(args.state_file, resume=args.resume)

    spec = REGISTRY[args.model]
    # A model batches only when its registry entry allows it AND the env gate is on.
    text_batching_enabled = ENABLE_TEXT_BATCHING and spec.text_batching
    video_batching_enabled = ENABLE_VIDEO_BATCHING and spec.video_batching

    use_sequence_mode = args.input_mode == "sequence"

    # Load manifest entries and optionally limit the workload
    videos = load_patternvideos_manifest(
        args.json_path,
        require_video_assets=not use_sequence_mode,
        asset_root=args.asset_root,
    )

    # Filter by bucket if specified (supports multiple --buckets flags; ANY-match)
    if args.bucket_filter:
        filters = args.bucket_filter if isinstance(args.bucket_filter, list) else [args.bucket_filter]
        original_count = len(videos)
        videos = [
            v for v in videos
            if any(bf in str(getattr(v, 'video_path', getattr(v, 'main_video_path', ''))) for bf in filters)
        ]
        print(f"Bucket filter {filters}: {len(videos)}/{original_count} videos match")
        if len(videos) == 0:
            print(f"ERROR: No videos found matching bucket filters {filters}")
            print(f"Available buckets in dataset:")
            buckets = set()
            for v in load_patternvideos_manifest(args.json_path, require_video_assets=False, asset_root=args.asset_root):
                path = str(getattr(v, 'video_path', getattr(v, 'main_video_path', '')))
                if '/' in path:
                    bucket = path.split('/')[1] if path.startswith('videos/') else path.split('/')[0]
                    buckets.add(bucket)
            for bucket in sorted(buckets):
                print(f"  - {bucket}")
            sys.exit(1)

    if args.limit:
        videos = videos[:args.limit]

    # Sort videos by bucket frame count (increasing order: L008, L016, L032, ...)
    # This ensures small buckets process before large ones for efficient resource usage
    def get_video_sort_key(video_data):
        bucket = _extract_bucket_from_video_data(video_data)
        bucket_length = extract_bucket_length(bucket)
        if bucket_length is not None:
            return bucket_length
        # Videos without buckets process last
        return float('inf')

    videos = sorted(videos, key=get_video_sort_key)
    print(f"Sorted {len(videos)} videos by frame count (L008, L016, L032, ...)")

    # Load the model class through the registry
    model_class = load_model_class(args.model)
    model_name = spec.display_name

    print(f"Starting evaluation with {model_name}")
    print(f"Processing {len(videos)} videos")
    print(f"Settings: fps={args.fps}, max_frames={args.max_frames}, verbose={args.verbose}")

    # Show GPU memory configuration
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s)")

    print("Initializing model...")

    def instantiate_model(max_frames: int) -> Any:
        """Create a fresh model instance from the registry spec.

        ``max_frames`` is accepted for call-site symmetry with the OOM restart
        path; the frame budget itself lives in the VideoProcessor.
        """
        del max_frames
        kwargs = dict(spec.init_kwargs)
        kwargs["enable_metrics"] = args.enable_metrics
        if spec.gpu_mem_aware:
            max_gpu_mem_val = calculate_max_gpu_mem(args.model, override=args.max_gpu_mem)
            kwargs["max_gpu_mem"] = max_gpu_mem_val
            if args.verbose:
                print(f"Using max_gpu_mem={max_gpu_mem_val:.1f} GB per GPU")
        if spec.takes_max_tokens_at_init:
            kwargs["generation_max_tokens"] = args.max_tokens
        if args.model == "qwen3_segment_refiner":
            # The refiner adapts its prompting per question style; every
            # current dataset uses native binary questions.
            kwargs["verbose"] = args.verbose
            kwargs["binary_questions"] = True
        if args.model == "openrouter":
            slug = args.openrouter_model
            if not slug:
                raise SystemExit("--model openrouter requires --openrouter-model <slug>")
            kwargs["model_id"] = slug
            kwargs.pop("max_gpu_mem", None)
        if spec.kind == "api" and spec.checkpoint and args.model != "openrouter":
            kwargs["model_id"] = spec.checkpoint
        if spec.checkpoint and spec.kind != "api":
            return model_class(spec.checkpoint, **kwargs)
        return model_class(**kwargs)

    model = instantiate_model(args.max_frames)

    aggregate_metrics = None
    if args.enable_metrics and getattr(model, '_metrics', None) is not None:
        metrics_cls = model._metrics.__class__  # type: ignore[attr-defined]
        aggregate_metrics = metrics_cls()

    def _metric_attr_names(metrics_obj: Any) -> List[str]:
        """Return list attributes that should be cleared or extended."""
        if metrics_obj is None:
            return []

        dataclass_fields = getattr(metrics_obj, '__dataclass_fields__', None)
        if dataclass_fields:
            return list(dataclass_fields.keys())

        return [name for name, value in vars(metrics_obj).items() if isinstance(value, list)]

    def _clear_metrics(metrics_obj: Any) -> None:
        """Clear all list-based metrics in place."""
        if metrics_obj is None:
            return

        for attr in _metric_attr_names(metrics_obj):
            attr_value = getattr(metrics_obj, attr, None)
            if isinstance(attr_value, list):
                attr_value.clear()

    def _extend_metrics(destination: Any, source: Any) -> None:
        """Append metrics from source into destination."""
        if destination is None or source is None:
            return

        dest_oom_timestamp = getattr(destination, 'first_oom_timestamp', None)
        src_oom_timestamp = getattr(source, 'first_oom_timestamp', None)
        if src_oom_timestamp is not None:
            if dest_oom_timestamp is None or src_oom_timestamp < dest_oom_timestamp:
                destination.first_oom_timestamp = float(src_oom_timestamp)

        for attr in _metric_attr_names(destination):
            dest_list = getattr(destination, attr, None)
            src_list = getattr(source, attr, None)
            if isinstance(dest_list, list) and isinstance(src_list, list):
                dest_list.extend(src_list)

    def _capture_first_oom(aggregate: Optional[PerformanceMetrics], timestamp: float) -> None:
        """Record the earliest OOM timestamp across the entire evaluation."""
        if aggregate is None:
            return

        current = getattr(aggregate, 'first_oom_timestamp', None)
        if current is None or float(timestamp) < current:
            aggregate.first_oom_timestamp = float(timestamp)

    def _has_metric_data(metrics_obj: Any) -> bool:
        """Return True if any tracked metric list contains data."""
        if metrics_obj is None:
            return False

        for attr in _metric_attr_names(metrics_obj):
            attr_value = getattr(metrics_obj, attr, None)
            if isinstance(attr_value, list) and attr_value:
                return True
        return False

    # Create processors. QuestionProcessor's remaining constructor flags map to
    # retired CLI options; they are pinned to the defaults every published run
    # used (descriptions off, multi-option prompts, non-predictive questions).
    question_processor = QuestionProcessor(
        args.verbose,
        binary_questions=False,
        predictive_questions=False,
        sequence_mode=use_sequence_mode,
    )
    sequence_formatter: Optional[SequenceFormatter] = None
    if use_sequence_mode:
        # Use SpatialSequenceFormatter for spatial mode, CommaSeparatedSequenceFormatter for sequential mode
        if args.eval_mode == "spatial":
            sequence_formatter = SpatialSequenceFormatter()
        else:
            sequence_formatter = CommaSeparatedSequenceFormatter()
    # Option labels are always presented as text (the render-labels-as-video
    # mode was retired), so no TextToVideoProcessor is needed.
    text_to_video_processor = None

    # Initialize wandb logger if metrics are enabled
    wandb_logger = WandbLogger(enabled=args.enable_metrics) if args.enable_metrics else None

    # Process all videos
    all_results = []
    correct_count = 0
    dont_know_count = 0
    question_outcomes: List[Dict[str, Any]] = []
    entropy_accuracy_records: List[Tuple[float, bool, bool]] = []  # (value, is_correct, is_dont_know)
    prefix_fraction_records: List[Tuple[float, bool, bool]] = []  # (value, is_correct, is_dont_know)
    correct_likelihood_records: List[Tuple[float, bool, bool]] = []  # (value, is_correct, is_dont_know)
    # Initialize with saved results from previous runs (resume support)
    question_log_rows: List[Dict[str, Any]] = []
    if state_manager.enabled:
        saved_results = state_manager.get_saved_results()
        if saved_results:
            print(f"Loaded {len(saved_results)} question results from previous run(s)")
            question_log_rows.extend(saved_results)

    frame_budget = args.max_frames
    missing_media_count = 0

    for i, video_data in enumerate(videos):
        variant = getattr(video_data, 'variant', 0)  # Default to 0 to match batching functions
        bucket = _extract_bucket_from_video_data(video_data)
        # Always print bucket info for clarity
        print(f"\n{'='*60}")
        print(f"Video {video_data.video_index} | Bucket: {bucket or 'None'} | Variant: {variant}")
        print(f"{'='*60}")


        # Some published natural_video rows are provenance-only: the manifest
        # entry exists but the media file is not distributed
        # (media_included: false). Skip those in video mode instead of failing.
        if args.input_mode == "video":
            _main_media_path = str(video_data.main_video_path)
            if not _main_media_path or not os.path.exists(_main_media_path):
                print(f"WARNING: Skipping video {video_data.video_index}: media file not found ({_main_media_path})")
                missing_media_count += 1
                continue

        # Check if model cannot process this bucket (VIDEO MODE ONLY - skip check entirely for sequence mode)
        # This allows per-model bucket limits (see `max_video_bucket` in the registry)
        # CRITICAL: Only applies when input_mode == "video" (sequence mode processes all buckets)
        if bucket and args.input_mode == "video":
            bucket_length = extract_bucket_length(bucket)
            model_limit = spec.max_video_bucket

            if bucket_length is not None and model_limit is not None and bucket_length > model_limit:
                print(f"WARNING: SKIPPING: {args.model} cannot process bucket '{bucket}' "
                      f"({bucket_length} frames > {model_limit} frame limit for VIDEO mode)")
                print(f"    (Sequence mode would process this bucket without limits)")

                # Mark all questions as complete to skip on resume
                if state_manager.enabled:
                    question_ids = [
                        q.question_id or f"video{video_data.video_index}_q{qi}"
                        for qi, q in enumerate(video_data.questions)
                    ]
                    for qid in question_ids:
                        state_manager.mark_question_complete(
                            video_data.video_index, qid, result=None, variant=variant, bucket=bucket
                        )
                    state_manager.flush()
                continue
        # If input_mode == "sequence", the entire block above is skipped and all buckets process

        if args.verbose and i < 5:  # Report the first 5 videos
            state_key = state_manager._video_key(video_data.video_index, variant, bucket) if state_manager.enabled else None
            print(f"State key: {state_key}")
        if state_manager.enabled and state_manager.is_video_completed(video_data.video_index, variant, bucket):
            print(f"Skipping completed video {video_data.video_index} variant {variant} bucket {bucket}")
            continue
        completed_questions = state_manager.get_completed_questions(video_data.video_index, variant, bucket) if state_manager.enabled else None
        if args.verbose and i < 5:  # Report the first 5 videos
            print(f"Video {video_data.video_index}: completed_questions={completed_questions if completed_questions else 'EMPTY/None'}")
            if completed_questions:
                print(f"  Sample: {list(completed_questions)[:5]}")
        current_max_frames = frame_budget
        oom_retry_count = 0
        video_results = None
        start_question_idx = 0
        partial_results: Optional[List[Dict[str, Any]]] = None
        carried_metrics: Optional[PerformanceMetrics] = None

        video_hit_question_limit = False  # Track if we stopped due to question limit

        while True:
            video_processor = None
            if not use_sequence_mode:
                video_processor = VideoProcessor(
                    args.model,
                    args.fps,
                    current_max_frames,
                    args.chunk_size,
                    model_max_frames=current_max_frames,
                )

            sequence_processor_instance: Optional[SequenceProcessor] = None
            if use_sequence_mode:
                raw_sequences = video_data.metadata.get('sequences_used')
                if not isinstance(raw_sequences, dict) or not raw_sequences:
                    raise ValueError(
                        f"Video {video_data.video_index} is missing 'sequences_used' details required for sequence mode."
                    )
                if sequence_formatter is None:
                    raise ValueError("Sequence formatter is not configured.")
                sequence_processor_instance = SequenceProcessor(
                    sequences_used=raw_sequences,
                    formatter=sequence_formatter,
                    print_chunks=True,
                )

            if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                if carried_metrics is not None:
                    model._metrics = copy.deepcopy(carried_metrics)
                    if hasattr(model, '_sync_state_memory_tracking_from_metrics'):
                        model._sync_state_memory_tracking_from_metrics()
                else:
                    _clear_metrics(model._metrics)
                    if hasattr(model, '_reset_state_memory_tracking'):
                        model._reset_state_memory_tracking()

            # Initialize dynamic batch sizer for text mode
            batch_sizer = None
            if args.input_mode == "sequence" and text_batching_enabled and sequence_processor_instance:
                batch_sizer = DynamicBatchSizer(initial_size=DEFAULT_BATCH_SIZE)
            # Initialize dynamic batch sizer for video mode (persists across videos
            # so batch reductions from earlier OOMs carry over).
            video_batch_sizer = None
            if args.input_mode == "video" and video_batching_enabled and video_processor:
                # Registry batch sizes are sized for the longest buckets;
                # VIDEO_BATCH_SIZE_FORCE=1 overrides. Safe because
                # DynamicBatchSizer still halves and retries on OOM.
                _initial = (VIDEO_BATCH_SIZE
                            if os.environ.get('VIDEO_BATCH_SIZE_FORCE') == '1'
                            else (spec.video_batch_size or VIDEO_BATCH_SIZE))
                video_batch_sizer = DynamicBatchSizer(initial_size=_initial, min_size=1, max_size=_initial)

            try:
                # Use batched processing for text/sequence mode when enabled
                if args.input_mode == "sequence" and text_batching_enabled and sequence_processor_instance and batch_sizer:
                    # Try with current batch size, reduce on OOM
                    max_batch_retries = 3
                    for batch_retry in range(max_batch_retries):
                        try:
                            current_batch_size = batch_sizer.get_size()
                            if args.verbose:
                                print(f"Attempting batch processing with batch_size={current_batch_size}")

                            # Use TRUE isolated batching for sequence mode
                            candidate_results = process_sequence_native_binary_batched_isolated(
                                video_data=video_data,
                                model=model,
                                sequence_processor=sequence_processor_instance,
                                question_processor=question_processor,
                                max_tokens=args.max_tokens,
                                batch_size=current_batch_size,
                                eval_mode=args.eval_mode,
                                verbose=args.verbose,
                                state_manager=state_manager if state_manager.enabled else None,
                                question_log_path=args.question_log_csv,
                                question_log_rows=question_log_rows,
                                limit_questions=args.limit_questions,
                                start_question_index=start_question_idx,
                                completed_set=completed_questions,
                            )
                            hit_limit = len(candidate_results) >= (args.limit_questions or float('inf'))

                            # Success! Maybe increase batch size for next video
                            batch_sizer.increase_on_success()
                            break

                        except RuntimeError as batch_exc:
                            # Check if it's a CUDA OOM error
                            if "out of memory" in str(batch_exc).lower() or "oom" in str(batch_exc).lower():
                                if current_batch_size == 1:
                                    # Can't reduce further, re-raise
                                    raise

                                new_size = batch_sizer.reduce_on_oom()
                                print(f"WARNING: OOM during batching with batch_size={current_batch_size}. Reducing to {new_size} and retrying...")

                                # Clear GPU memory
                                model.clear_context()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                                if batch_retry == max_batch_retries - 1:
                                    raise  # Max retries reached
                            else:
                                raise  # Non-OOM error, don't retry
                elif args.input_mode == "video" and video_batching_enabled and video_processor:
                    # Use batched processing for video mode native binary questions.
                    # Auto-halve on OOM (mirrors the text-mode pattern above).
                    # video_batch_sizer was initialized above the try-block.
                    initial_video_batch = video_batch_sizer.max_size
                    max_batch_retries = max(3, int(math.log2(initial_video_batch)) + 2)
                    for batch_retry in range(max_batch_retries):
                        try:
                            current_batch_size = video_batch_sizer.get_size()
                            if args.verbose:
                                print(f"Attempting video batch processing with batch_size={current_batch_size}")

                            # Use TRUE isolated batching for video mode
                            if args.verbose and completed_questions:
                                print(f"Passing {len(completed_questions)} completed questions to batching function")
                            candidate_results = process_video_native_binary_batched_isolated(
                                video_data=video_data,
                                model=model,
                                video_processor=video_processor,
                                question_processor=question_processor,
                                max_tokens=args.max_tokens,
                                batch_size=current_batch_size,
                                eval_mode=args.eval_mode,
                                verbose=args.verbose,
                                state_manager=state_manager if state_manager.enabled else None,
                                question_log_path=args.question_log_csv,
                                question_log_rows=question_log_rows,
                                limit_questions=args.limit_questions,
                                start_question_index=start_question_idx,
                                completed_set=completed_questions,
                            )
                            hit_limit = len(candidate_results) >= (args.limit_questions or float('inf'))
                            break

                        except RuntimeError as batch_exc:
                            if "out of memory" in str(batch_exc).lower() or "oom" in str(batch_exc).lower():
                                if current_batch_size == 1:
                                    # Can't reduce further, re-raise
                                    raise
                                new_size = video_batch_sizer.reduce_on_oom()
                                print(f"WARNING: OOM during video batching with batch_size={current_batch_size}. Reducing to {new_size} and retrying...")
                                # Clear GPU memory before retry
                                if hasattr(model, 'clear_context'):
                                    model.clear_context()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                if batch_retry == max_batch_retries - 1:
                                    raise  # Max retries reached
                            else:
                                raise  # Non-OOM error, don't retry

                else:
                    # Use original processing (video mode or batching disabled)
                    candidate_results, hit_limit = process_video_with_questions(
                        video_data,
                        model,
                        video_processor,
                        question_processor,
                        args.max_tokens,
                        current_max_frames,
                        text_to_video_processor,
                        False,
                        start_question_index=start_question_idx,
                        existing_results=partial_results,
                        binary_questions=False,
                        wandb_logger=wandb_logger,
                        input_mode=args.input_mode,
                        sequence_processor=sequence_processor_instance,
                        sequence_format_label="comma",
                        eval_mode=args.eval_mode,
                        completed_questions=completed_questions,
                        state_manager=state_manager if state_manager.enabled else None,
                        question_log_path=args.question_log_csv,
                        question_log_rows=question_log_rows,
                        limit_questions=args.limit_questions,
                    )
                video_results = candidate_results
                video_hit_question_limit = hit_limit
                frame_budget = current_max_frames
                start_question_idx = 0
                partial_results = None
                oom_retry_count = 0
                carried_metrics = None
                break
            except QuestionOOMRetry as retry_exc:
                exc = retry_exc.original_exception
                handled = False

                if args.enable_metrics:
                    _capture_first_oom(aggregate_metrics, retry_exc.oom_timestamp)

                if args.restart_on_oom:
                    print(
                        "Restarting model after CUDA OOM to ensure clean state.",
                        flush=True,
                    )
                    try:
                        if hasattr(model, 'teardown_after_oom'):
                            model.teardown_after_oom()
                        elif hasattr(model, 'clear_context'):
                            model.clear_context()
                    except Exception:
                        pass
                    try:
                        if hasattr(model, 'close'):
                            model.close()
                    except Exception:
                        pass
                    try:
                        if hasattr(model, 'shutdown'):
                            model.shutdown()
                    except Exception:
                        pass

                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                    model = instantiate_model(current_max_frames)
                    handled = True
                    # Metrics instance belongs to the new model; ensure it is clean before retry.
                    if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                        _clear_metrics(model._metrics)
                        if hasattr(model, '_reset_state_memory_tracking'):
                            model._reset_state_memory_tracking()

                if handled:
                    carried_metrics = retry_exc.partial_metrics
                    start_question_idx = retry_exc.question_index
                    partial_results = retry_exc.partial_results
                    oom_retry_count += 1
                    if oom_retry_count > args.max_oom_retries:
                        raise RuntimeError(
                            f"Exceeded maximum CUDA OOM retries ({args.max_oom_retries}) for video index {i} while using max_frames={current_max_frames}."
                        ) from exc
                    if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                        _clear_metrics(model._metrics)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    if state_manager.enabled:
                        completed_questions = state_manager.get_completed_questions(video_data.video_index, variant, bucket)
                    continue

                # Not handled; re-raise original OOM exception chain.
                # Print the clear stderr line first so batch logs make the
                # failure mode obvious without scrolling through the trace.
                print(
                    "[OOM-on-eval] Refusing to silently reduce frame budget. Exit. "
                    "Bump the GPU allocation or lower --max-frames.",
                    file=sys.stderr,
                    flush=True,
                )
                raise exc

            except Exception as exc:
                if _is_cuda_oom_error(exc):
                    handled = False

                    if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                        timestamp = getattr(model._metrics, 'first_oom_timestamp', None)
                        if timestamp is not None:
                            _capture_first_oom(aggregate_metrics, timestamp)

                    if args.restart_on_oom:
                        print(
                            "Restarting model after CUDA OOM to ensure clean state.",
                            flush=True,
                        )
                        try:
                            if hasattr(model, 'teardown_after_oom'):
                                model.teardown_after_oom()
                            elif hasattr(model, 'clear_context'):
                                model.clear_context()
                        except Exception:
                            pass
                        try:
                            if hasattr(model, 'close'):
                                model.close()
                        except Exception:
                            pass
                        try:
                            if hasattr(model, 'shutdown'):
                                model.shutdown()
                        except Exception:
                            pass

                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()

                        model = instantiate_model(current_max_frames)
                        handled = True
                        # Metrics instance belongs to the new model; ensure it is clean before retry.
                        if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                            _clear_metrics(model._metrics)
                            if hasattr(model, '_reset_state_memory_tracking'):
                                model._reset_state_memory_tracking()

                    if handled:
                        start_question_idx = 0
                        partial_results = None
                        oom_retry_count += 1
                        if oom_retry_count > args.max_oom_retries:
                            raise RuntimeError(
                                f"Exceeded maximum CUDA OOM retries ({args.max_oom_retries}) for video index {i} while using max_frames={current_max_frames}."
                            ) from exc
                        if args.enable_metrics and getattr(model, '_metrics', None) is not None:
                            _clear_metrics(model._metrics)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        continue

                if _is_cuda_oom_error(exc):
                    # Print the clear stderr line before re-raising.
                    print(
                        "[OOM-on-eval] Refusing to silently reduce frame budget. Exit. "
                        "Bump the GPU allocation or lower --max-frames.",
                        file=sys.stderr,
                        flush=True,
                    )
                raise

        # Only mark video as complete if we didn't stop due to question limit
        # This allows resuming with a higher limit to process additional questions
        if state_manager.enabled and not termination_requested() and not video_hit_question_limit:
            state_manager.mark_video_complete(video_data.video_index, variant=variant, bucket=bucket)
        all_results.extend(video_results)

        if termination_requested():
            print("Termination requested; stopping after current video.")
            break

        # Calculate video-level stats
        video_correct = sum(1 for result in video_results if result['is_correct'])
        video_accuracy = video_correct / len(video_results) if video_results else 0

        # Log to wandb if enabled
        if wandb_logger:
            additional_metrics = _summarize_video_question_stats(video_results)
            wandb_logger.log_video_completion(
                num_questions=len(video_results),
                video_accuracy=video_accuracy,
                model_metrics=model._metrics if hasattr(model, '_metrics') else None,
                model_name=model_name,
                extra_metrics=additional_metrics,
            )

        for result in video_results:
            if result['is_correct']:
                correct_count += 1
            if result['is_dont_know']:
                dont_know_count += 1

            question_time = result.get('question_time')
            if question_time is not None:
                question_outcomes.append(
                    {
                        'timestamp_sec': float(question_time),
                        'is_correct': 1 if result.get('is_correct') else 0,
                        'video_index': video_data.video_index,
                        'question_id': result.get('question_id'),
                    }
                )

            entropy_value = result.get('entropy_prefix_mean')
            if entropy_value is not None:
                entropy_accuracy_records.append((float(entropy_value), bool(result.get('is_correct')), bool(result.get('is_dont_know'))))

            prefix_fraction = result.get('prefix_match_fraction')
            if prefix_fraction is not None:
                prefix_fraction_records.append((float(prefix_fraction), bool(result.get('is_correct')), bool(result.get('is_dont_know'))))

            likelihood_value = result.get('correct_option_likelihood')
            if likelihood_value is not None:
                correct_likelihood_records.append((float(likelihood_value), bool(result.get('is_correct')), bool(result.get('is_dont_know'))))

        if args.enable_metrics and aggregate_metrics is not None and getattr(model, '_metrics', None) is not None:
            _extend_metrics(aggregate_metrics, model._metrics)

    extra_final_metrics = _aggregate_global_question_stats(
        entropy_accuracy_records,
        prefix_fraction_records,
        correct_likelihood_records,
    )

    summary_metrics = None
    if args.enable_metrics:
        if aggregate_metrics is not None and _has_metric_data(aggregate_metrics):
            summary_metrics = aggregate_metrics
        elif getattr(model, '_metrics', None) is not None and _has_metric_data(model._metrics):
            summary_metrics = model._metrics

    if summary_metrics is not None:
        summary_cls = type(model)
        if hasattr(summary_cls, 'render_metrics_summary'):
            summary_cls.render_metrics_summary(summary_metrics)
        elif hasattr(model, 'print_metrics_summary'):
            model.print_metrics_summary()

        if hasattr(summary_metrics, 'video_timestamps_add_video') and hasattr(summary_cls, 'analyze_metrics'):
            try:
                summary_cls.analyze_metrics(summary_metrics, print_results=True)
            except Exception as exc:
                print(f"WARNING: Curve fitting analysis failed: {exc}")

    if state_manager.enabled:
        state_manager.flush()

    # Completion pass: Fill in gaps for incomplete buckets
    # This happens AFTER the limited main pass, so these questions don't count toward limit
    if state_manager.enabled and not termination_requested():
        try:
            completion_count = complete_remaining_questions(
                videos=videos,
                model=model,
                state_manager=state_manager,
                args=args,
                question_processor=question_processor,
                sequence_formatter=sequence_formatter,
                use_sequence_mode=use_sequence_mode,
            )
            print(f"Completion pass processed {completion_count} additional questions (not included in stats)")
        except Exception as e:
            print(f"WARNING: Completion pass failed: {e}")
            import traceback
            traceback.print_exc()

    # Apply limit_questions PER BUCKET for stats
    if args.limit_questions is not None:

        # If no new results (all buckets complete), use saved results from state for stats
        results_for_stats = all_results
        if len(all_results) == 0 and state_manager.enabled:
            saved_for_stats = state_manager.get_saved_results()
            if saved_for_stats:
                print(f"\nNo new results this run. Using {len(saved_for_stats)} saved results for stats calculation.")
                results_for_stats = saved_for_stats

        # Group by bucket
        bucket_results = defaultdict(list)
        for result in results_for_stats:
            bucket = result.get('bucket')
            if bucket:
                bucket_results[bucket].append(result)

        # Limit per bucket (respecting eval_mode)
        stats_results = []
        over_limit_buckets = []

        for bucket in sorted(bucket_results.keys()):
            bucket_data = bucket_results[bucket]

            # Filter by eval_mode
            valid_results = []
            for res in bucket_data:
                if args.eval_mode == 'sequential':
                    variant = res.get('question_variant', '').lower()
                    if variant == 'spatial':
                        continue
                valid_results.append(res)

            # Check if over limit
            if len(valid_results) > args.limit_questions:
                over_limit_buckets.append((bucket, len(valid_results)))

            # Take first limit_questions
            limited = valid_results[:args.limit_questions]
            stats_results.extend(limited)

        print(f"\nApplied limit of {args.limit_questions} per bucket (eval_mode={args.eval_mode}):")
        print(f"  Total results available: {len(results_for_stats)}")
        total_valid_all = sum(len([r for r in bucket_results[b]
                                    if args.eval_mode != 'sequential'
                                    or r.get('question_variant','').lower() != 'spatial'])
                              for b in bucket_results)
        print(f"  Valid for eval_mode: {total_valid_all}")
        print(f"  Used for stats: {len(stats_results)} ({len(bucket_results)} buckets)")

        # Warn about buckets whose saved state holds more questions than the limit
        if over_limit_buckets:
            print(f"\nWARNING: {len(over_limit_buckets)} buckets have more than {args.limit_questions} questions:")
            for bucket, count in over_limit_buckets:
                print(f"    {bucket}: {count} questions (using first {args.limit_questions} for stats, keeping all in state)")
    else:
        stats_results = all_results

    # Final cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print final results
    # ALWAYS recalculate correct_count and dont_know_count from stats_results
    # (stats_results may be limited per-bucket, whereas the running counts covered all_results)
    correct_count = sum(1 for r in stats_results if r.get('is_correct', False))
    dont_know_count = sum(1 for r in stats_results if r.get('is_dont_know', False))

    total_questions = len(stats_results)
    accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
    idk_rate = (dont_know_count / total_questions * 100) if total_questions > 0 else 0
    answered_questions = total_questions - dont_know_count
    answered_accuracy = (correct_count / answered_questions * 100) if answered_questions > 0 else 0

    # Calculate spatial vs sequential breakdown
    spatial_stats = {'total': 0, 'correct': 0, 'dont_know': 0}
    sequential_stats = {'total': 0, 'correct': 0, 'dont_know': 0}

    for result in stats_results:
        variant = result.get('question_variant', '').lower()
        is_correct = result.get('is_correct', False)
        is_dont_know = result.get('is_dont_know', False)

        if variant == 'spatial':
            spatial_stats['total'] += 1
            if is_correct:
                spatial_stats['correct'] += 1
            if is_dont_know:
                spatial_stats['dont_know'] += 1
        elif variant == 'sequential':
            sequential_stats['total'] += 1
            if is_correct:
                sequential_stats['correct'] += 1
            if is_dont_know:
                sequential_stats['dont_know'] += 1

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {model_name}")
    print(f"{'='*60}")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_count}")
    print(f"Don't Know Responses: {dont_know_count}")
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print(f"IDK Rate: {idk_rate:.1f}%")
    print(f"Accuracy (excluding IDK): {answered_accuracy:.1f}%")

    # Print spatial vs sequential breakdown
    if spatial_stats['total'] > 0 or sequential_stats['total'] > 0:
        print(f"\nBreakdown by Question Type:")
        if spatial_stats['total'] > 0:
            spatial_acc = (spatial_stats['correct'] / spatial_stats['total'] * 100)
            spatial_answered = spatial_stats['total'] - spatial_stats['dont_know']
            spatial_acc_excl = (spatial_stats['correct'] / spatial_answered * 100) if spatial_answered > 0 else 0
            print(f"  Spatial: {spatial_stats['total']} questions, {spatial_acc:.1f}% accurate ({spatial_acc_excl:.1f}% excl. IDK)")
        if sequential_stats['total'] > 0:
            seq_acc = (sequential_stats['correct'] / sequential_stats['total'] * 100)
            seq_answered = sequential_stats['total'] - sequential_stats['dont_know']
            seq_acc_excl = (sequential_stats['correct'] / seq_answered * 100) if seq_answered > 0 else 0
            print(f"  Sequential: {sequential_stats['total']} questions, {seq_acc:.1f}% accurate ({seq_acc_excl:.1f}% excl. IDK)")

    if extra_final_metrics:
        _print_question_stat_summaries(extra_final_metrics)
    print(f"{'='*60}")

    # Compute and display per-bucket stats
    def compute_per_bucket_stats(results):
        bucket_stats = defaultdict(lambda: {
            'spatial': {'correct': 0, 'total': 0, 'dont_know': 0},
            'sequential': {'correct': 0, 'total': 0, 'dont_know': 0},
        })

        for result in results:
            bucket = result.get('bucket')
            if not bucket:
                continue

            variant = result.get('question_variant', '').lower()
            category = 'spatial' if variant == 'spatial' else 'sequential'

            stats = bucket_stats[bucket][category]
            stats['total'] += 1
            if result.get('is_correct', False):
                stats['correct'] += 1
            if result.get('is_dont_know', False):
                stats['dont_know'] += 1

        return dict(bucket_stats)

    bucket_stats = compute_per_bucket_stats(stats_results)
    if bucket_stats:
        print(f"\n{'='*60}")
        print("PER-BUCKET METRICS")
        print(f"{'='*60}")

        for bucket in sorted(bucket_stats.keys()):
            print(f"\n{bucket}:")
            for category in ['spatial', 'sequential']:
                stats = bucket_stats[bucket][category]
                if stats['total'] == 0:
                    continue

                correct, total, dont_know = stats['correct'], stats['total'], stats['dont_know']
                answered = total - dont_know
                raw_acc = (correct / total * 100) if total > 0 else 0
                adj_acc = (correct / answered * 100) if answered > 0 else 0

                print(f"  {category.capitalize():12s}: Acc={raw_acc:5.1f}% (adj={adj_acc:5.1f}%) | answered={answered}/{total}")
        print(f"{'='*60}")

    # Compute and display temporal distribution for L032+ buckets
    def extract_length_from_bucket(bucket: str):
        """Extract length value from bucket name (e.g., 'L008' -> 8)."""
        match = re.search(r'L(\d+)', bucket)
        if match:
            return int(match.group(1))
        return None

    def compute_temporal_distribution(results):
        """Compute accuracy and recall by temporal position for L032+ buckets."""
        # Structure: length -> fourth -> {correct, total, dont_know, tp, fn}
        length_fourth_stats = defaultdict(lambda: [
            {'correct': 0, 'total': 0, 'dont_know': 0, 'tp': 0, 'fn': 0}
            for _ in range(4)
        ])

        for result in results:
            bucket = result.get('bucket')
            if not bucket:
                continue

            length = extract_length_from_bucket(bucket)
            if not length or length < 32:
                continue  # Only L032+

            # Get clip timing
            clip_start = result.get('clip_start_time')
            clip_end = result.get('clip_end_time')
            if clip_start is None or clip_end is None:
                continue

            # Check if answer was yes (clip is present)
            # For native binary, check if correct answer is "0" (yes)
            correct_answer = result.get('correct')
            if correct_answer != '0':  # Not a "yes" question
                continue

            # Video duration equals the length (L032 = 32 seconds, etc.)
            video_duration = float(length)
            fourth_duration = video_duration / 4.0

            # Calculate which fourths the clip overlaps
            start_fourth = int(clip_start / fourth_duration)
            end_fourth = int(clip_end / fourth_duration)

            # Clamp to valid range [0, 3]
            start_fourth = max(0, min(3, start_fourth))
            end_fourth = max(0, min(3, end_fourth))

            # Determine fourths covered
            fourths_covered = list(range(start_fourth, end_fourth + 1))
            if len(fourths_covered) == 0:
                continue

            # Check correctness and predictions
            is_correct = result.get('is_correct', False)
            is_dont_know = result.get('is_dont_know', False)

            # For recall calculation
            predicted_yes = (result.get('predicted') == '0')
            actual_yes = True  # Filtered for answer=yes above

            # Increment each fourth this clip touches
            weight = 1.0 / len(fourths_covered) if len(fourths_covered) > 1 else 1.0

            for fourth_idx in fourths_covered:
                stats = length_fourth_stats[length][fourth_idx]
                stats['total'] += weight

                if is_correct:
                    stats['correct'] += weight
                if is_dont_know:
                    stats['dont_know'] += weight

                # For recall (precision omitted - always 100% for yes-only questions)
                if actual_yes and predicted_yes:
                    stats['tp'] += weight
                elif actual_yes and not predicted_yes and not is_dont_know:
                    stats['fn'] += weight

        return dict(length_fourth_stats)

    temporal_dist = compute_temporal_distribution(stats_results)
    if temporal_dist:
        print(f"\n{'='*60}")
        print("TEMPORAL METRICS BY FOURTH/QUARTER (L032+ buckets)")
        print(f"{'='*60}")
        print("For each fourth: metrics for questions where answer=yes (clip is present)")
        print("NOTE: 'No' questions excluded (undefined temporal position)")
        print("      Precision omitted (always 100% for yes-only questions)")

        for length in sorted(temporal_dist.keys()):
            fourth_stats = temporal_dist[length]

            # Calculate totals
            total_questions = sum(stats['total'] for stats in fourth_stats)

            print(f"\nL{length:04d} (total true questions: {total_questions:.1f}):")

            # Row 1: Fourth numbers
            print("  Fourth:         ", end="")
            for i in range(4):
                print(f"{i+1:12d}", end="  ")
            print()

            # Row 2: Question counts
            print("  Questions:      ", end="")
            for stats in fourth_stats:
                if stats['total'] > 0:
                    print(f"{stats['total']:12.1f}", end="  ")
                else:
                    print(f"         N/A", end="  ")
            print()

            # Row 3: Accuracy (raw)
            print("  Accuracy:       ", end="")
            for stats in fourth_stats:
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    print(f"{accuracy:11.1f}%", end="  ")
                else:
                    print(f"         N/A", end="  ")
            print()

            # Row 4: Accuracy (adjusted for uncertainty)
            print("  Acc (adj IDK):  ", end="")
            for stats in fourth_stats:
                answered = stats['total'] - stats['dont_know']
                if answered > 0:
                    adj_acc = (stats['correct'] / answered) * 100
                    print(f"{adj_acc:11.1f}%", end="  ")
                else:
                    print(f"         N/A", end="  ")
            print()

            # Row 5: Recall (precision omitted - always 100%)
            print("  Recall:         ", end="")
            for stats in fourth_stats:
                tp_fn = stats['tp'] + stats['fn']
                if tp_fn > 0:
                    recall = (stats['tp'] / tp_fn) * 100
                    print(f"{recall:11.1f}%", end="  ")
                else:
                    print(f"         N/A", end="  ")
            print()
        print(f"{'='*60}")

    # Log final results to wandb
    if wandb_logger:
        wandb_logger.log_final_results(
            total_videos=len(videos),
            total_questions=total_questions,
            overall_accuracy=accuracy,
            idk_rate=idk_rate,
            model_name=model_name,
            extra_metrics=extra_final_metrics,
        )
        if question_outcomes:
            wandb_logger.log_question_accuracy_scatter(question_outcomes)
        wandb_logger.finish()

    if args.question_log_csv:
        try:
            destination = write_question_log_csv(args.question_log_csv, question_log_rows)
            print(f"Question log CSV written to {destination}", flush=True)
        except Exception as exc:
            print(
                f"Failed to write question log CSV to {args.question_log_csv}: {exc}",
                flush=True,
            )

    if missing_media_count:
        print(
            f"Skipped {missing_media_count} video(s) with missing media files (provenance-only rows).",
            flush=True,
        )


def _summarize_video_question_stats(video_results: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    def _collect_means(result_key: str, metric_prefix: str) -> None:
        values = [float(result[result_key]) for result in video_results if result.get(result_key) is not None]
        if not values:
            return
        metrics[f'{metric_prefix}_mean'] = sum(values) / len(values)
        correct_values = [
            float(result[result_key])
            for result in video_results
            if result.get(result_key) is not None and result.get('is_correct')
        ]
        if correct_values:
            metrics[f'{metric_prefix}_correct_mean'] = sum(correct_values) / len(correct_values)

    _collect_means('entropy_prefix_mean', 'video_entropy')
    _collect_means('prefix_match_fraction', 'video_prefix_fraction')
    _collect_means('correct_option_likelihood', 'video_correct_likelihood')

    return metrics


def _aggregate_global_question_stats(
    entropy_records: List[Tuple[float, bool, bool]],
    prefix_records: List[Tuple[float, bool, bool]],
    likelihood_records: List[Tuple[float, bool, bool]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    for records, key, label in (
        (entropy_records, 'entropy', 'Entropy'),
        (prefix_records, 'prefix_fraction', 'Prefix Fraction'),
        (likelihood_records, 'correct_likelihood', 'Correct Option Likelihood'),
    ):
        metric_summary = _summarize_metric_records(records, label)
        if metric_summary:
            summary[key] = metric_summary

    return summary


def _summarize_metric_records(
    records: List[Tuple[float, bool, bool]],
    label: str,
) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    values = [value for value, _, _ in records]
    summary = {
        'label': label,
        'mean': sum(values) / len(values),
    }
    correct_values = [value for value, is_correct, _ in records if is_correct]
    incorrect_values = [value for value, is_correct, _ in records if not is_correct]
    if correct_values:
        summary['mean_correct'] = sum(correct_values) / len(correct_values)
    if incorrect_values:
        summary['mean_incorrect'] = sum(incorrect_values) / len(incorrect_values)
    summary['bins'] = _build_accuracy_bins(records, label)
    return summary


def _build_accuracy_bins(records: List[Tuple[float, bool, bool]], label: str, bins: int = 3) -> List[Dict[str, Any]]:
    if not records:
        return []
    sorted_records = sorted(records, key=lambda item: item[0])
    bin_count = min(bins, len(sorted_records))
    if bin_count <= 0:
        return []
    base_size = max(1, len(sorted_records) // bin_count)
    result: List[Dict[str, Any]] = []
    start_index = 0
    for idx in range(bin_count):
        if idx == bin_count - 1:
            end_index = len(sorted_records)
        else:
            end_index = min(len(sorted_records), start_index + base_size)
        subset = sorted_records[start_index:end_index]
        start_index = end_index
        if not subset:
            continue

        # Calculate raw accuracy
        correct_count = sum(1 for _, is_correct, _ in subset if is_correct)
        accuracy = (correct_count / len(subset)) * 100.0

        # Calculate uncertainty-adjusted accuracy
        answered_count = sum(1 for _, _, is_dont_know in subset if not is_dont_know)
        adj_accuracy = (correct_count / answered_count * 100.0) if answered_count > 0 else 0.0

        value_range = (subset[0][0], subset[-1][0])
        result.append(
            {
                'label': f"{label} Bin {idx + 1}",
                'range': value_range,
                'accuracy': accuracy,
                'accuracy_adj': adj_accuracy,
                'count': len(subset),
            }
        )
    return result


def _print_question_stat_summaries(summary: Dict[str, Any]) -> None:
    for metric_key, data in summary.items():
        label = data.get('label', metric_key.replace('_', ' ').title())
        mean_value = data.get('mean')
        if mean_value is None:
            continue
        print(f"{label} Mean: {mean_value:.3f}")
        correct_mean = data.get('mean_correct')
        incorrect_mean = data.get('mean_incorrect')
        if correct_mean is not None or incorrect_mean is not None:
            print(
                f"  {label} (correct/incorrect): "
                f"{(correct_mean or 0.0):.3f} / {(incorrect_mean or 0.0):.3f}"
            )
        for bucket in data.get('bins', []):
            rng = bucket.get('range', (0.0, 0.0))
            acc = bucket.get('accuracy', 0.0)
            adj_acc = bucket.get('accuracy_adj', 0.0)
            print(
                f"  {bucket['label']}: accuracy {acc:.1f}% (adj {adj_acc:.1f}%)"
                f" (range {rng[0]:.3f}–{rng[1]:.3f}, n={bucket['count']})"
            )


def _find_tokenizer(model: Any) -> Any:
    """Return the model's text tokenizer, looking through common attributes."""
    for attr in ("tokenizer", "_tokenizer"):
        tokenizer = getattr(model, attr, None)
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            return tokenizer
    processor = getattr(model, "processor", None) or getattr(model, "_processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return tokenizer
    return None


def _count_response_tokens(model: Any, response: str) -> int:
    """Count response tokens with the model's own tokenizer.

    Falls back to a whitespace split only when the model exposes no tokenizer
    (the dummy model; API models overwrite this with the provider's count).
    """
    if not response:
        return 0
    tokenizer = _find_tokenizer(model)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(response, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(response))
        except Exception:
            pass
    return len(response.strip().split())


def _extract_token_stats(model: Any, response: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    stats['response_token_count'] = _count_response_tokens(model, response)

    provider = getattr(model, 'get_last_response_token_stats', None)
    if callable(provider):
        try:
            token_data = provider() or {}
        except Exception:
            token_data = {}
        if token_data:
            stats.update(token_data)
            total_tokens = token_data.get('total_output_tokens')
            if isinstance(total_tokens, (int, float)):
                stats['response_token_count'] = int(total_tokens)
    return stats


if __name__ == "__main__":
    main()

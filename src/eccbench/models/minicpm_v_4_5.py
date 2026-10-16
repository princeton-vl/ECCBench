"""MiniCPM-V 4.5 (openbmb/MiniCPM-V-4_5) wrapper around the checkpoint's ``chat`` method.

The MiniCPM 4.5 frame sampler supplies the temporal ids for the 3D resampler.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from eccbench.models.base import PerformanceMetrics, VideoLanguageModelInterface
from eccbench.models.device_map_utils import build_max_memory_map
from eccbench.samplers.minicpm45_sampler import MiniCPM45Sample


@dataclass
class _MiniCPMSegment:
    frames: List[Image.Image]
    temporal_ids: List[List[int]]
    video_id: str
    time_start: float
    time_end: float


class MiniCPM45Model(VideoLanguageModelInterface):
    """MiniCPM-V 4.5 with interleaved video and text context."""

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        attn_impl = kwargs.pop("attn_implementation", "sdpa")
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        device_map = kwargs.pop("device_map", "auto")

        # A max_memory budget too small for the model (about 17 GB) makes
        # Accelerate offload weights to the meta device, which breaks batching.
        # Such a budget is ignored.
        load_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            total_mem_gb = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(gpu_count))

            if max_gpu_mem is not None and device_map == "auto":
                estimated_total = max_gpu_mem * gpu_count

                if estimated_total >= 20.0:
                    load_kwargs["max_memory"] = build_max_memory_map(max_gpu_mem)
                    print(f"[MiniCPM-4.5] Using {gpu_count} GPU(s) with max_memory={max_gpu_mem:.1f}GB each")
                else:
                    print(f"[MiniCPM-4.5] WARNING: max_gpu_mem={max_gpu_mem:.1f}GB too small, ignoring to avoid offloading")
                    print(f"[MiniCPM-4.5] Using device_map='auto' with full GPU memory ({total_mem_gb:.1f}GB total)")
            else:
                print(f"[MiniCPM-4.5] Using device_map='{device_map}' across {gpu_count} GPU(s) ({total_mem_gb:.1f}GB total)")

        load_kwargs.update(kwargs)
        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        self._timeline: List[Dict[str, Any]] = []
        self._text_token_cache: Dict[str, int] = {}
        self._history: Optional[Any] = None
        self._video_was_truncated: Optional[bool] = None
        self._default_generation_config = {
            "use_image_id": False,
            "max_slice_nums": 1,
            "max_new_tokens": 512,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def add_video(
        self,
        video_frames: Union[MiniCPM45Sample, Sequence[Image.Image]],
        time_start: float,
        time_end: float,
        video_id: Optional[int] = None,
    ) -> None:
        metrics_start, baseline_mem = self._metrics_begin()
        segment = self._coerce_segment(video_frames, time_start, time_end, video_id)
        self._timeline.append({"type": "video", "segment": segment})
        if self.enable_metrics:
            latency, peak_increase_mb, peak_absolute_mb = self._metrics_end(metrics_start, baseline_mem)
            state_mem = self._estimate_state_memory_floats()
            self._record_metrics_event(
                "add_video",
                latency=latency,
                peak_increase_mb=peak_increase_mb,
                peak_absolute_mb=peak_absolute_mb,
                video_time=time_end,
                state_memory_total=state_mem,
            )

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        metrics_start, baseline_mem = self._metrics_begin()
        self._timeline.append({"type": "text", "text": text, "time": current_video_time})
        if self.enable_metrics:
            latency, peak_increase_mb, peak_absolute_mb = self._metrics_end(metrics_start, baseline_mem)
            state_mem = self._estimate_state_memory_floats()
            self._record_metrics_event(
                "add_text",
                latency=latency,
                peak_increase_mb=peak_increase_mb,
                peak_absolute_mb=peak_absolute_mb,
                video_time=current_video_time,
                state_memory_total=state_mem,
            )

    def ask_question(
        self,
        question: str,
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> str:
        self._video_was_truncated = False
        metrics_start, baseline_mem = self._metrics_begin()
        frames, temporal_ids, content = self._build_chat_inputs(question)

        msgs = [{"role": "user", "content": content}]
        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = max(1, int(max_tokens))

        start = time.perf_counter()
        response = None
        last_error = None

        # Raise the input limit with the number of video segments so the
        # context is not truncated.
        video_segments = len([e for e in self._timeline if e.get("type") == "video"])
        max_inp_length = max(16384, video_segments * 1000)

        if temporal_ids is not None:
            try:
                response = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    temporal_ids=temporal_ids,
                    max_inp_length=max_inp_length,
                    **generation_config,
                )
            except RuntimeError as e:
                if "Sizes of tensors must match" in str(e):
                    print(f"Warning: Tensor shape mismatch with temporal_ids, trying without: {e}")
                    last_error = e
                else:
                    raise

        if response is None:
            try:
                response = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    temporal_ids=None,
                    max_inp_length=max_inp_length,
                    **generation_config,
                )
            except RuntimeError as e:
                print("Error: Failed even without temporal_ids")
                print(f"  Current error: {e}")
                if last_error:
                    print(f"  First error (with temporal_ids): {last_error}")
                print(f"  Video segments in context: {len([e for e in self._timeline if e.get('type') == 'video'])}")
                print(f"  Used max_inp_length: {max_inp_length}")
                raise

        latency = time.perf_counter() - start
        if self.enable_metrics:
            _, peak_increase_mb, peak_absolute_mb = self._metrics_end(metrics_start, baseline_mem)
            state_mem = self._estimate_state_memory_floats()
            self._record_metrics_event(
                "ask_question",
                latency=latency,
                peak_increase_mb=peak_increase_mb,
                peak_absolute_mb=peak_absolute_mb,
                video_time=current_video_time,
                state_memory_total=state_mem,
            )
        return response

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> List[str]:
        """Answer several questions about the same context with one batched ``chat`` call.

        Falls back to one question at a time after an OOM.
        """
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        self._video_was_truncated = False

        frames, temporal_ids, base_content = self._build_chat_inputs("")
        if base_content and base_content[-1] == "":
            base_content = base_content[:-1]

        batch_msgs = []
        for question in questions:
            content = list(base_content)
            content.append(question)
            batch_msgs.append([{"role": "user", "content": content}])

        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = max(1, int(max_tokens))

        video_segments = len([e for e in self._timeline if e.get("type") == "video"])
        max_inp_length = max(16384, video_segments * 1000)

        all_responses: List[str] = []
        batch_succeeded = False

        batch_size = len(questions)
        while batch_size >= 1 and not batch_succeeded:
            try:
                if batch_size < len(questions):
                    break

                if temporal_ids is not None:
                    try:
                        responses = self.model.chat(
                            msgs=batch_msgs,
                            tokenizer=self.tokenizer,
                            temporal_ids=temporal_ids,
                            max_inp_length=max_inp_length,
                            **generation_config,
                        )
                        # A string instead of a list means batched chat is unavailable.
                        if isinstance(responses, str):
                            raise ValueError("Batch inference not supported, falling back")
                        all_responses = responses if isinstance(responses, list) else [responses]
                        batch_succeeded = True
                    except (RuntimeError, ValueError) as e:
                        if "Sizes of tensors must match" not in str(e) and "Batch inference not supported" not in str(e):
                            raise
                        responses = None

                if not all_responses and not batch_succeeded:
                    responses = self.model.chat(
                        msgs=batch_msgs,
                        tokenizer=self.tokenizer,
                        temporal_ids=None,
                        max_inp_length=max_inp_length,
                        **generation_config,
                    )
                    if isinstance(responses, str):
                        raise ValueError("Batch inference not supported")
                    all_responses = responses if isinstance(responses, list) else [responses]
                    batch_succeeded = True

                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[MiniCPM] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    all_responses = []
                    break
                else:
                    raise

        if not batch_succeeded:
            print("[MiniCPM] Batched inference failed; answering one question at a time")
            all_responses = []
            for question in questions:
                try:
                    result = self.ask_question(question, current_video_time, max_tokens, max_frames_in_video, sample_method)
                    all_responses.append(result)
                except RuntimeError as e2:
                    if "out of memory" in str(e2).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("[MiniCPM] OOM, appending empty response")
                        all_responses.append("")
                    else:
                        raise

        return all_responses

    def ask_question_batch_isolated(
        self,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
    ) -> List[str]:
        """Answer one question per context, each with its own input.

        A sequence context has ``main_sequence``, ``candidate_sequence`` and
        ``question_text``; a video context has ``main_video_frames``,
        ``candidate_video_frames`` (MiniCPM45Sample or image lists) and
        ``question_text``. A single context also goes through the batched path.
        """
        if not contexts:
            return []

        is_sequence_mode = contexts[0].get('mode', 'sequence') == 'sequence'

        batch_messages = []

        for ctx in contexts:
            content = []

            if is_sequence_mode:
                full_text = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )
                content.append(full_text)
            else:
                main_frames = self._frame_list(ctx.get('main_video_frames'))
                candidate_frames = self._frame_list(ctx.get('candidate_video_frames'))

                if main_frames:
                    content.append("Here is a main video to remember:")
                    for frame in main_frames[:max_frames_in_video // 2]:
                        content.append(frame)

                if candidate_frames:
                    content.append("\nHere is a candidate clip:\n")
                    for frame in candidate_frames[:max_frames_in_video // 2]:
                        content.append(frame)

                content.append(ctx['question_text'])

            batch_messages.append([{"role": "user", "content": content}])

        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = max(1, int(max_tokens))

        batch_size = len(contexts)
        all_responses = []

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(contexts))
                    chunk_msgs = batch_messages[chunk_start:chunk_end]

                    responses = self.model.chat(
                        msgs=chunk_msgs,
                        tokenizer=self.tokenizer,
                        temporal_ids=None,
                        max_inp_length=max(16384, len(chunk_msgs) * 1000),
                        **generation_config,
                    )

                    if isinstance(responses, str):
                        all_responses.append(responses)
                    elif isinstance(responses, list):
                        all_responses.extend(responses)
                    else:
                        raise ValueError(f"Unexpected response type: {type(responses)}")

                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[MiniCPM] OOM: Reducing batch size {old_batch_size} -> {batch_size}")
                    all_responses = []
                else:
                    raise

        return all_responses

    @staticmethod
    def _frame_list(frames: Any) -> List[Any]:
        """Frames of a MiniCPM45Sample or a frame list; empty for anything else."""
        if isinstance(frames, MiniCPM45Sample):
            return frames.frames
        if isinstance(frames, (list, tuple)):
            return frames
        return []

    def get_state(self) -> Dict[str, Any]:
        return {
            "timeline": copy.deepcopy(self._timeline),
        }

    def clear_context(self) -> None:
        self._timeline.clear()
        self._video_was_truncated = None

    def was_video_truncated(self) -> Optional[bool]:
        return self._video_was_truncated

    def save_state(self) -> Any:
        return copy.deepcopy(self.get_state())

    def load_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("Invalid MiniCPM-V state payload")
        self._timeline = copy.deepcopy(state.get("timeline", []))

    def _coerce_segment(
        self,
        video_frames: Union[MiniCPM45Sample, Sequence[Image.Image]],
        time_start: float,
        time_end: float,
        video_id: Optional[int],
    ) -> _MiniCPMSegment:
        vid_key = str(video_id if video_id is not None else 0)
        if isinstance(video_frames, MiniCPM45Sample):
            frames = [self._ensure_image(frame) for frame in video_frames.frames]
            temporal = [list(ids) for ids in video_frames.temporal_ids]
        else:
            frames = [self._ensure_image(frame) for frame in video_frames]
            temporal = []
        if temporal and sum(len(ids) for ids in temporal) != len(frames):
            raise ValueError("MiniCPM temporal ids must align with frames")
        return _MiniCPMSegment(
            frames=frames,
            temporal_ids=temporal,
            video_id=vid_key,
            time_start=time_start,
            time_end=time_end,
        )

    @staticmethod
    def _ensure_image(frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame
        if isinstance(frame, np.ndarray):
            return Image.fromarray(frame)
        raise TypeError("MiniCPM frames must be PIL.Image or numpy arrays")

    def _build_chat_inputs(
        self, question: str
    ) -> tuple[List[Image.Image], Optional[List[List[int]]], List[Any]]:
        frames: List[Image.Image] = []
        temporal: List[List[int]] = []
        temporal_available = True
        content: List[Any] = []

        for entry in self._timeline:
            if entry.get("type") == "video":
                segment = entry["segment"]
                frames.extend(segment.frames)
                if segment.temporal_ids:
                    temporal.extend(segment.temporal_ids)
                else:
                    temporal_available = False
                content.extend(segment.frames)
            elif entry.get("type") == "text":
                content.append(entry["text"])

        content.append(question)
        temporal_ids = temporal if temporal and temporal_available else None
        return frames, temporal_ids, content

    def _record_metrics_event(
        self,
        event: str,
        *,
        latency: float,
        peak_increase_mb: float,
        peak_absolute_mb: float,
        video_time: float,
        state_memory_total: float,
    ) -> None:
        if not self.enable_metrics or not isinstance(self._metrics, PerformanceMetrics):
            return
        if event == "add_video":
            self._record_add_video_metrics(
                latency,
                peak_increase_mb,
                peak_absolute_mb,
                video_time,
                state_memory_total,
            )
        elif event == "add_text":
            self._record_add_text_metrics(
                latency,
                peak_increase_mb,
                peak_absolute_mb,
                video_time,
                state_memory_total,
            )
        elif event == "ask_question":
            self._record_ask_question_metrics(
                latency,
                peak_increase_mb,
                peak_absolute_mb,
                video_time,
                state_memory_total,
            )

    def _metrics_begin(self) -> Tuple[Optional[float], float]:
        if not self.enable_metrics:
            return None, 0.0
        baseline_mem = 0.0
        if torch.cuda.is_available():
            baseline_mem = float(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter(), baseline_mem

    def _metrics_end(self, start_time: Optional[float], baseline_mem: float) -> Tuple[float, float, float]:
        if not self.enable_metrics or start_time is None:
            return 0.0, 0.0, 0.0
        latency = time.perf_counter() - start_time
        peak_bytes = baseline_mem
        if torch.cuda.is_available():
            peak_bytes = float(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        peak_increase_mb = max(0.0, peak_bytes - baseline_mem) / (1024 * 1024)
        peak_absolute_mb = (
            peak_bytes / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        return latency, peak_increase_mb, peak_absolute_mb

    def _estimate_state_memory_floats(self) -> float:
        total = 0.0
        for entry in self._timeline:
            if entry.get("type") == "video":
                segment: _MiniCPMSegment = entry["segment"]
                for frame in segment.frames:
                    total += self._frame_float_count(frame)
            elif entry.get("type") == "text":
                total += self._count_text_tokens(entry.get("text", ""))
        return total

    @staticmethod
    def _frame_float_count(frame: Any) -> float:
        if isinstance(frame, Image.Image):
            array_view = np.asarray(frame)
            return float(array_view.size)
        return 0.0

    def _count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        cached = self._text_token_cache.get(text)
        if cached is not None:
            return cached
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoding.get("input_ids") if isinstance(encoding, dict) else encoding.input_ids
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

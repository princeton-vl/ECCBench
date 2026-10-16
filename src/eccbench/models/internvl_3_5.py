"""InternVL3.5 wrapper (OpenGVLab/InternVL3_5-*), standard and thinking variants."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from eccbench.models.base import VideoLanguageModelInterface
from eccbench.models.device_map_utils import build_max_memory_map
from eccbench.samplers.internvl_sampler import InternVLSampledVideo

# Thinking-mode system prompt from the InternVL3.5 model card.
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


@dataclass
class _VideoSegment:
    clip: InternVLSampledVideo
    video_id: str
    frame_labels: List[str]
    time_start: float
    time_end: float


class InternVL35Model(VideoLanguageModelInterface):
    """InternVL3.5 through the checkpoint's ``chat`` and ``batch_chat`` methods."""

    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        *,
        is_thinking: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.is_thinking = is_thinking
        generation_override = kwargs.pop("generation_max_tokens", None)
        if generation_override is None:
            generation_override = kwargs.pop("max_tokens", None)
        self._generation_override = (
            int(generation_override) if generation_override is not None else None
        )
        self._model_kwargs = dict(kwargs)
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **self._model_kwargs,
        )

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        torch.set_float32_matmul_precision("high")
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if self._has_flash_attention():
            load_kwargs["use_flash_attn"] = True
        load_kwargs.update(self._model_kwargs)
        if max_gpu_mem is not None and torch.cuda.is_available():
            load_kwargs["max_memory"] = build_max_memory_map(max_gpu_mem)
        # Thinking mode uses the same weights with another system prompt and sampling.
        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=False,
            padding_side="left",
        )
        if self.is_thinking:
            self.model.system_message = R1_SYSTEM_PROMPT
        self._text_token_cache: Dict[str, int] = {}
        self._primary_device = next(self.model.parameters()).device
        self._timeline: List[Dict[str, Any]] = []
        self._video_frame_counters: Dict[str, int] = {}
        self._video_was_truncated: Optional[bool] = None
        default_max = self._generation_override or 512
        if self.is_thinking:
            self._default_generation_config = {
                "do_sample": True,
                "temperature": 0.6,
                "max_new_tokens": max(1, default_max),
            }
        else:
            self._default_generation_config = {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": max(1, default_max),
            }

    def add_video(
        self,
        video_frames: Union[InternVLSampledVideo, torch.Tensor, np.ndarray],
        time_start: float,
        time_end: float,
        video_id: Optional[int] = None,
    ) -> None:
        metrics_start: Optional[float] = None
        baseline_mem = 0.0
        if self.enable_metrics:
            metrics_start = time.perf_counter()
            if torch.cuda.is_available():
                baseline_mem = float(torch.cuda.memory_allocated())
        clip = self._coerce_clip(video_frames)
        vid_key = str(video_id if video_id is not None else 0)
        start_counter = self._video_frame_counters.get(vid_key, 0)
        frame_labels = [
            f"Video{vid_key} Frame{start_counter + idx + 1}: <image>"
            for idx in range(len(clip.num_patches_list))
        ]
        self._video_frame_counters[vid_key] = start_counter + len(clip.num_patches_list)
        segment = _VideoSegment(
            clip=clip,
            video_id=vid_key,
            frame_labels=frame_labels,
            time_start=time_start,
            time_end=time_end,
        )
        self._timeline.append({"type": "video", "segment": segment})

        if self.enable_metrics and metrics_start is not None:
            latency = time.perf_counter() - metrics_start
            peak_mem = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
            peak_mem_increase_mb = max(0.0, (peak_mem - baseline_mem) / (1024 * 1024))
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            state_mem = self._get_state_memory_floats()
            self._record_add_video_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                time_end,
                state_mem,
            )

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        metrics_start: Optional[float] = None
        baseline_mem = 0.0
        if self.enable_metrics:
            metrics_start = time.perf_counter()
            if torch.cuda.is_available():
                baseline_mem = float(torch.cuda.memory_allocated())

        timestamp = float(current_video_time)
        self._timeline.append({"type": "text", "text": text, "time": timestamp})

        if self.enable_metrics and metrics_start is not None:
            latency = time.perf_counter() - metrics_start
            peak_mem = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
            peak_mem_increase_mb = max(0.0, (peak_mem - baseline_mem) / (1024 * 1024))
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                timestamp,
                state_mem,
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
        clip = self._build_combined_clip()
        prompt = self._build_prompt(question)

        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = int(max(1, max_tokens))
        pixel_values = None
        num_patches_list: Optional[List[int]] = None
        if clip is not None:
            pixel_values = clip.pixel_values.to(
                dtype=torch.bfloat16,
                device=self._primary_device,
                non_blocking=True,
            )
            num_patches_list = clip.num_patches_list

        metrics_start: Optional[float] = None
        baseline_mem = 0.0
        if self.enable_metrics:
            metrics_start = time.perf_counter()
            if torch.cuda.is_available():
                baseline_mem = float(torch.cuda.memory_allocated())

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list,
        )

        if self.enable_metrics and metrics_start is not None:
            latency = time.perf_counter() - metrics_start
            peak_mem = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
            peak_mem_increase_mb = max(0.0, (peak_mem - baseline_mem) / (1024 * 1024))
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            latest_context_time = self._latest_context_timestamp()
            question_timestamp = float(current_video_time) if current_video_time > 0 else latest_context_time
            state_mem = self._get_state_memory_floats()
            self._record_ask_question_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                question_timestamp,
                state_mem,
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
        """Answer several questions about the same context with ``batch_chat``."""
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        self._video_was_truncated = False
        clip = self._build_combined_clip()
        prompts = [self._build_prompt(q) for q in questions]

        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = int(max(1, max_tokens))

        batch_size = len(questions)
        all_responses: List[Optional[str]] = [None] * len(questions)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(questions), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(questions))
                    chunk_prompts = prompts[chunk_start:chunk_end]
                    chunk_size = len(chunk_prompts)

                    # batch_chat needs len(pixel_values) == sum(num_patches_list)
                    # and one num_patches entry per question.
                    chunk_pixel_values = None
                    chunk_num_patches = None

                    if clip is not None:
                        single_pixel_values = clip.pixel_values.to(
                            dtype=torch.bfloat16,
                            device=self._primary_device,
                            non_blocking=True,
                        )
                        chunk_pixel_values = torch.cat([single_pixel_values] * chunk_size, dim=0)
                        num_patches_per_question = sum(clip.num_patches_list)
                        chunk_num_patches = [num_patches_per_question] * chunk_size
                    else:
                        chunk_num_patches = [0] * chunk_size

                    responses = self.model.batch_chat(
                        self.tokenizer,
                        chunk_pixel_values,
                        num_patches_list=chunk_num_patches,
                        questions=chunk_prompts,
                        generation_config=generation_config,
                    )

                    for i, resp in enumerate(responses):
                        all_responses[chunk_start + i] = resp

                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[InternVL] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    all_responses = [None] * len(questions)
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
        ``candidate_video_frames`` (InternVLSampledVideo) and ``question_text``.
        """
        if not contexts:
            return []

        # A single context goes through chat(), which labels every frame.
        if len(contexts) == 1:
            return [self._process_single_context_isolated(contexts[0], max_tokens, max_frames_in_video)]

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        prompts = []
        all_pixel_values = []
        all_num_patches = []
        all_total_patches = []   # per-question totals, for pixel slicing only

        for ctx in contexts:
            if is_sequence_mode:
                prompt = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )
                prompts.append(prompt)
                all_num_patches.append(0)
                all_total_patches.append(0)

            else:
                main_clip = ctx.get('main_video_frames')
                candidate_clip = ctx.get('candidate_video_frames')

                # batch_chat() expands only the first '<image>' of each prompt.
                # Expand every other frame here into the same
                # <img><IMG_CONTEXT>...</img> block that chat() builds, so each
                # frame's embeddings land at its own label, as in chat().
                _nit = self.model.num_image_token

                def _expand(n):
                    return '<img>' + '<IMG_CONTEXT>' * (_nit * n) + '</img>'

                prompt_parts = []
                current_patches_for_question = []
                first_frame_patches = None

                def _frame_marker(num_patches):
                    nonlocal first_frame_patches
                    if first_frame_patches is None:
                        first_frame_patches = num_patches
                        return '<image>'
                    return _expand(num_patches)

                if main_clip is not None:
                    prompt_parts.append("Here is a main video to remember:")
                    for idx, np_ in enumerate(main_clip.num_patches_list):
                        prompt_parts.append(f"Main Frame{idx + 1}: {_frame_marker(np_)}")
                    all_pixel_values.append(main_clip.pixel_values)
                    current_patches_for_question.extend(main_clip.num_patches_list)

                if candidate_clip is not None:
                    prompt_parts.append("Here is a candidate clip:")
                    for idx, np_ in enumerate(candidate_clip.num_patches_list):
                        prompt_parts.append(f"Candidate Frame{idx + 1}: {_frame_marker(np_)}")
                    all_pixel_values.append(candidate_clip.pixel_values)
                    current_patches_for_question.extend(candidate_clip.num_patches_list)

                prompt_parts.append(f"\n{ctx['question_text']}")

                prompt = "\n".join(prompt_parts)
                prompts.append(prompt)

                # batch_chat sizes the one literal '<image>' from the first
                # frame's patch count; slicing pixel values needs the total.
                all_num_patches.append(first_frame_patches or 0)
                all_total_patches.append(sum(current_patches_for_question))

        generation_config = dict(self._default_generation_config)
        generation_config["max_new_tokens"] = int(max(1, max_tokens))

        combined_pixel_values = None
        if not is_sequence_mode and all_pixel_values:
            combined_pixel_values = torch.cat(all_pixel_values, dim=0).to(
                dtype=torch.bfloat16,
                device=self._primary_device,
                non_blocking=True,
            )

        batch_size = len(contexts)
        all_responses: List[Optional[str]] = [None] * len(contexts)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(contexts))
                    chunk_prompts = prompts[chunk_start:chunk_end]
                    chunk_size = len(chunk_prompts)

                    chunk_pixel_values = None
                    chunk_num_patches = None

                    if is_sequence_mode:
                        chunk_num_patches = [0] * chunk_size
                    else:
                        patches_before_chunk = sum(all_total_patches[:chunk_start])
                        patches_in_chunk = sum(all_total_patches[chunk_start:chunk_end])

                        if combined_pixel_values is not None:
                            chunk_pixel_values = combined_pixel_values[patches_before_chunk:patches_before_chunk + patches_in_chunk]
                        chunk_num_patches = all_num_patches[chunk_start:chunk_end]

                    responses = self.model.batch_chat(
                        self.tokenizer,
                        chunk_pixel_values,
                        num_patches_list=chunk_num_patches,
                        questions=chunk_prompts,
                        generation_config=generation_config,
                    )

                    for i, resp in enumerate(responses):
                        all_responses[chunk_start + i] = resp

                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[InternVL] OOM: Reducing batch size {old_batch_size} -> {batch_size}")
                else:
                    raise

        return all_responses

    def _process_single_context_isolated(self, ctx: Dict[str, Any], max_tokens: int, max_frames_in_video: int) -> str:
        mode = ctx.get('mode', 'sequence')

        if mode == 'sequence':
            prompt = (
                f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                f"{ctx['question_text']}"
            )
            return self.ask_question(prompt, current_video_time=0.0, max_tokens=max_tokens)
        else:
            self.clear_context()

            main_clip = ctx.get('main_video_frames')
            candidate_clip = ctx.get('candidate_video_frames')

            # Frames are at 1 fps, so a video lasts as many seconds as it has frames.
            main_dur = 0.0
            if main_clip is not None:
                num_main = len(main_clip.num_patches_list)
                main_dur = float(num_main)
                main_frame_labels = [
                    f"Main Frame{idx + 1}: <image>"
                    for idx in range(num_main)
                ]
                self._timeline.append({
                    "type": "video",
                    "segment": _VideoSegment(
                        clip=main_clip,
                        video_id="main_video",
                        frame_labels=main_frame_labels,
                        time_start=0.0,
                        time_end=main_dur
                    )
                })

            cand_start = main_dur + 1.0
            cand_dur = 0.0
            if candidate_clip is not None:
                num_cand = len(candidate_clip.num_patches_list)
                cand_dur = float(num_cand)
                candidate_frame_labels = [
                    f"Candidate Frame{idx + 1}: <image>"
                    for idx in range(num_cand)
                ]
                self._timeline.append({
                    "type": "video",
                    "segment": _VideoSegment(
                        clip=candidate_clip,
                        video_id="candidate_clip",
                        frame_labels=candidate_frame_labels,
                        time_start=cand_start,
                        time_end=cand_start + cand_dur
                    )
                })

            return self.ask_question(ctx['question_text'], current_video_time=cand_start + cand_dur, max_tokens=max_tokens)

    def get_state(self) -> Dict[str, Any]:
        return {
            "timeline": copy.deepcopy(self._timeline),
            "counters": dict(self._video_frame_counters),
        }

    def clear_context(self) -> None:
        self._timeline.clear()
        self._video_frame_counters.clear()
        self._video_was_truncated = None

    def was_video_truncated(self) -> Optional[bool]:
        return self._video_was_truncated

    def save_state(self) -> Any:
        return copy.deepcopy(self.get_state())

    def load_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("Invalid InternVL state")
        self._timeline = copy.deepcopy(state.get("timeline", []))
        self._video_frame_counters = dict(state.get("counters", {}))

    def _latest_context_timestamp(self) -> float:
        latest = 0.0
        for entry in self._timeline:
            if entry.get("type") == "video":
                latest = max(latest, float(entry["segment"].time_end))
            elif entry.get("type") == "text":
                latest = max(latest, float(entry.get("time", 0.0)))
        return latest

    def _get_state_memory_floats(self) -> float:
        total = 0.0
        for entry in self._timeline:
            if entry.get("type") == "video":
                clip = entry["segment"].clip
                total += float(clip.pixel_values.numel())
            elif entry.get("type") == "text":
                total += float(self._count_text_tokens(entry.get("text", "")))
        return total

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

    def _coerce_clip(
        self,
        video_frames: Union[InternVLSampledVideo, torch.Tensor, np.ndarray],
    ) -> InternVLSampledVideo:
        if isinstance(video_frames, InternVLSampledVideo):
            return InternVLSampledVideo(
                pixel_values=video_frames.pixel_values.detach().cpu(),
                num_patches_list=list(video_frames.num_patches_list),
                fps=video_frames.fps,
            )
        tensor = self._ensure_tensor(video_frames)
        if tensor.ndim != 4:
            raise ValueError("InternVL requires 4D tensors (frames, C, H, W)")
        if tensor.shape[1] not in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        num_frames = tensor.shape[0]
        counts = [1] * num_frames
        return InternVLSampledVideo(pixel_values=tensor, num_patches_list=counts, fps=1.0)

    @staticmethod
    def _ensure_tensor(data: Union[torch.Tensor, np.ndarray, InternVLSampledVideo]) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            return tensor
        raise TypeError("Unsupported video frame type for InternVL model")

    def _build_combined_clip(self) -> Optional[InternVLSampledVideo]:
        if not self._timeline:
            return None
        pixel_values: List[torch.Tensor] = []
        counts: List[int] = []
        for entry in self._timeline:
            if entry.get("type") != "video":
                continue
            segment = entry["segment"]
            pixel_values.append(segment.clip.pixel_values)
            counts.extend(segment.clip.num_patches_list)
        if not pixel_values:
            return None

        # Resize segments whose frame size differs from the first segment.
        if len(pixel_values) > 1:
            reference_shape = pixel_values[0].shape
            if any(pv.shape != reference_shape for pv in pixel_values[1:]):
                resized_pixel_values = [pixel_values[0]]
                for pv in pixel_values[1:]:
                    if pv.shape != reference_shape:
                        resized = torch.nn.functional.interpolate(
                            pv,
                            size=(reference_shape[-2], reference_shape[-1]),
                            mode='bilinear',
                            align_corners=False
                        )
                        resized_pixel_values.append(resized)
                    else:
                        resized_pixel_values.append(pv)
                pixel_values = resized_pixel_values

        combined = torch.cat(pixel_values, dim=0)
        return InternVLSampledVideo(pixel_values=combined, num_patches_list=counts, fps=1.0)

    def _build_prompt(self, question: str) -> str:
        lines: List[str] = []
        for entry in self._timeline:
            if entry.get("type") == "video":
                lines.extend(entry["segment"].frame_labels)
            elif entry.get("type") == "text":
                lines.append(entry["text"])
        if lines:
            lines.append(question)
            return "\n".join(lines)
        return question

    @staticmethod
    def _has_flash_attention() -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("flash_attn") is not None
        except Exception:
            return False


class InternVL35ThinkingModel(InternVL35Model):
    """Thin wrapper that enables thinking mode."""

    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            is_thinking=True,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )

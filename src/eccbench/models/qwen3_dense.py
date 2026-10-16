"""Qwen3-VL-8B Instruct and Thinking, built on the Qwen2.5-VL wrapper."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.video_utils import VideoMetadata

from eccbench.models.qwen2_5_vl import CACHE_ROOT, QwenFullVideo


class Qwen3Dense(QwenFullVideo):
    """Qwen3-VL-8B; passes explicit video metadata to the processor."""

    INSTRUCT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    THINKING_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"

    def __init__(
        self,
        model_id: Optional[str] = None,
        *,
        thinking: bool = False,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ) -> None:
        resolved_model_id = model_id or (
            self.THINKING_MODEL_ID if thinking else self.INSTRUCT_MODEL_ID
        )
        self.uses_thinking_variant = resolved_model_id == self.THINKING_MODEL_ID

        super().__init__(
            resolved_model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        min_pixels: Optional[int] = kwargs.get("min_pixels")
        max_pixels: Optional[int] = kwargs.get("max_pixels")

        model_kwargs: Dict[str, object] = {
            "dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "cache_dir": CACHE_ROOT,
        }

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        processor_kwargs: Dict[str, object] = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        processor_kwargs["cache_dir"] = CACHE_ROOT
        processor_kwargs["padding_side"] = "left"

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            **processor_kwargs,
        )

        # The default longest_edge caps the total video pixels and would shrink
        # frames of long videos; keep 448x448 frames up to 1024 frames.
        self.processor.video_processor.size["longest_edge"] = 1024 * 448 * 448

        self.latest_time = 0.0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

        if hasattr(self, "_reset_state_memory_tracking"):
            self._reset_state_memory_tracking()

    def ask_question(
        self,
        question: str,
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> str:
        self._video_was_truncated = False
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        timeline_events = []
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline_events.append(
                    {
                        "type": "video",
                        "time": segment["time_start"],
                        "video_id": video_id,
                        "segment": segment,
                    }
                )

        for text, timestamp in self.text_entries:
            timeline_events.append({"type": "text", "time": timestamp, "text": text})

        timeline_events.sort(key=lambda x: (x["time"], x["type"] == "video"))

        content: List[Dict[str, object]] = []
        video_inputs: List[torch.Tensor] = []
        video_metadata_entries: List[VideoMetadata] = []
        video_fps: List[Optional[float]] = []

        video_data: Dict[int, Dict[str, object]] = {}
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
                },
            )

            if store["frames"] is None:
                store["frames"] = segment["frames"]
            else:
                segment_frames = segment["frames"]
                if isinstance(segment_frames, torch.Tensor) and isinstance(store["frames"], torch.Tensor):
                    if segment_frames.device != store["frames"].device:
                        segment_frames = segment_frames.to(store["frames"].device)
                store["frames"] = torch.cat([store["frames"], segment_frames], dim=0)

            store["total_duration"] = float(store["total_duration"]) + float(segment["duration"])
            store["total_frames"] = int(store["total_frames"]) + int(
                segment.get("num_frames", segment["frames"].shape[0])
            )

        for video_id, data in video_data.items():
            frames: torch.Tensor = data["frames"]
            if frames is None:
                data["frame_indices"] = []
                continue

            frame_count = int(frames.shape[0])
            selection = np.arange(frame_count, dtype=int)
            if frame_count > max_frames_in_video:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    indices = np.sort(
                        np.random.choice(frame_count, max_frames_in_video, replace=False)
                    )
                else:
                    indices = np.linspace(0, frame_count - 1, max_frames_in_video, dtype=int)

                data["frames"] = frames[indices]
                selection = selection[indices]

            trimmed_frame_count = int(data["frames"].shape[0]) if data["frames"] is not None else 0
            original_frame_count = int(data.get("total_frames", 0))
            original_duration = float(data.get("total_duration", 0.0))
            if (
                original_frame_count
                and original_duration
                and trimmed_frame_count
                and trimmed_frame_count != original_frame_count
            ):
                original_fps = original_frame_count / original_duration
                if original_fps > 0:
                    data["total_duration"] = trimmed_frame_count / original_fps

            if isinstance(selection, np.ndarray):
                data["frame_indices"] = selection.astype(int).tolist()
            else:
                data["frame_indices"] = list(selection)

        processed_video_ids = set()
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

                duration = float(video_data[video_id].get("total_duration", 0.0))
                frame_count = int(frames_tensor.shape[0])
                safe_duration = duration if duration > 0 else max(frame_count / 24.0, 1e-3)
                fps_value = frame_count / safe_duration if safe_duration > 0 else None
                height = int(frames_tensor.shape[2]) if frames_tensor.ndim >= 3 else None
                width = int(frames_tensor.shape[3]) if frames_tensor.ndim >= 4 else None
                frame_indices = video_data[video_id].get("frame_indices") or list(range(frame_count))

                metadata = VideoMetadata(
                    total_num_frames=frame_count,
                    fps=fps_value,
                    width=width,
                    height=height,
                    duration=safe_duration,
                    video_backend="tensor",
                    frames_indices=[int(idx) for idx in frame_indices],
                )
                video_metadata_entries.append(metadata)
                video_fps.append(fps_value)

                content.append({"type": "video", "video": len(video_inputs) - 1})
                processed_video_ids.add(video_id)

            elif event["type"] == "text":
                content.append({"type": "text", "text": event["text"]})

        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        processor_kwargs: Dict[str, object] = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            video_inputs_cpu = [v.cpu() if isinstance(v, torch.Tensor) else v for v in video_inputs]
            processor_kwargs["videos"] = video_inputs_cpu
            if video_metadata_entries:
                processor_kwargs["video_metadata"] = video_metadata_entries

            coalesced_fps = self._coalesce_video_fps(video_fps)
            if coalesced_fps is not None:
                processor_kwargs["fps"] = coalesced_fps

        inputs = self.processor(**processor_kwargs)
        input_device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            latest_context_time = 0.0
            for event in timeline_events:
                if event["type"] == "video":
                    latest_context_time = max(
                        latest_context_time, float(event["segment"]["time_end"])
                    )
                else:
                    latest_context_time = max(latest_context_time, float(event["time"]))

            question_timestamp = float(current_video_time) if current_video_time is not None else 0.0
            if question_timestamp <= 0 and latest_context_time > 0:
                question_timestamp = latest_context_time

            state_mem = self._get_state_memory_floats()
            self._record_ask_question_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                question_timestamp,
                state_mem,
            )

        return output_text[0] if output_text else ""

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> List[str]:
        """Answer several questions about the same context in one batch."""
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        self._video_was_truncated = False

        timeline_events = []
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline_events.append({
                    "type": "video",
                    "time": segment["time_start"],
                    "video_id": video_id,
                    "segment": segment,
                })

        for text, timestamp in self.text_entries:
            timeline_events.append({"type": "text", "time": timestamp, "text": text})

        timeline_events.sort(key=lambda x: (x["time"], x["type"] == "video"))

        video_data: Dict[int, Dict[str, object]] = {}
        for event in timeline_events:
            if event["type"] != "video":
                continue
            video_id = event["video_id"]
            segment = event["segment"]
            store = video_data.setdefault(
                video_id,
                {"frames": None, "total_duration": 0.0, "total_frames": 0},
            )

            if store["frames"] is None:
                store["frames"] = segment["frames"]
            else:
                segment_frames = segment["frames"]
                if isinstance(segment_frames, torch.Tensor) and isinstance(store["frames"], torch.Tensor):
                    if segment_frames.device != store["frames"].device:
                        segment_frames = segment_frames.to(store["frames"].device)
                store["frames"] = torch.cat([store["frames"], segment_frames], dim=0)

            store["total_duration"] = float(store["total_duration"]) + float(segment["duration"])
            store["total_frames"] = int(store["total_frames"]) + int(
                segment.get("num_frames", segment["frames"].shape[0])
            )

        for video_id, data in video_data.items():
            frames: torch.Tensor = data["frames"]
            if frames is None:
                data["frame_indices"] = []
                continue

            frame_count = int(frames.shape[0])
            selection = np.arange(frame_count, dtype=int)
            if frame_count > max_frames_in_video:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    indices = np.sort(
                        np.random.choice(frame_count, max_frames_in_video, replace=False)
                    )
                else:
                    indices = np.linspace(0, frame_count - 1, max_frames_in_video, dtype=int)

                data["frames"] = frames[indices]
                selection = selection[indices]

            trimmed_frame_count = int(data["frames"].shape[0]) if data["frames"] is not None else 0
            original_frame_count = int(data.get("total_frames", 0))
            original_duration = float(data.get("total_duration", 0.0))
            if (
                original_frame_count
                and original_duration
                and trimmed_frame_count
                and trimmed_frame_count != original_frame_count
            ):
                original_fps = original_frame_count / original_duration
                if original_fps > 0:
                    data["total_duration"] = trimmed_frame_count / original_fps

            if isinstance(selection, np.ndarray):
                data["frame_indices"] = selection.astype(int).tolist()
            else:
                data["frame_indices"] = list(selection)

        shared_content: List[Dict[str, object]] = []
        video_inputs: List[torch.Tensor] = []
        video_metadata_entries: List[VideoMetadata] = []
        video_fps: List[Optional[float]] = []
        processed_video_ids = set()

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

                duration = float(video_data[video_id].get("total_duration", 0.0))
                frame_count = int(frames_tensor.shape[0])
                safe_duration = duration if duration > 0 else max(frame_count / 24.0, 1e-3)
                fps_value = frame_count / safe_duration if safe_duration > 0 else None
                height = int(frames_tensor.shape[2]) if frames_tensor.ndim >= 3 else None
                width = int(frames_tensor.shape[3]) if frames_tensor.ndim >= 4 else None
                frame_indices = video_data[video_id].get("frame_indices") or list(range(frame_count))

                metadata = VideoMetadata(
                    total_num_frames=frame_count,
                    fps=fps_value,
                    width=width,
                    height=height,
                    duration=safe_duration,
                    video_backend="tensor",
                    frames_indices=[int(idx) for idx in frame_indices],
                )
                video_metadata_entries.append(metadata)
                video_fps.append(fps_value)

                shared_content.append({"type": "video", "video": len(video_inputs) - 1})
                processed_video_ids.add(video_id)

            elif event["type"] == "text":
                shared_content.append({"type": "text", "text": event["text"]})

        batch_messages = []
        for question_text in questions:
            content = list(shared_content)
            content.append({"type": "text", "text": question_text})
            batch_messages.append([{"role": "user", "content": content}])

        batch_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        input_device = self.model.get_input_embeddings().weight.device

        video_inputs_cpu = None
        if video_inputs:
            video_inputs_cpu = [v.cpu() if isinstance(v, torch.Tensor) else v for v in video_inputs]

        # Halve the batch on CUDA OOM and retry.
        batch_size = len(questions)
        all_output_texts = [None] * len(questions)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(questions), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(questions))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    processor_kwargs: Dict[str, object] = {
                        "text": chunk_texts,
                        "padding": True,
                        "return_tensors": "pt",
                    }
                    if video_inputs_cpu:
                        # One list of videos and one list of metadata per prompt.
                        chunk_size = len(chunk_texts)
                        processor_kwargs["videos"] = [video_inputs_cpu for _ in range(chunk_size)]
                        if video_metadata_entries:
                            processor_kwargs["video_metadata"] = [video_metadata_entries for _ in range(chunk_size)]

                        coalesced_fps = self._coalesce_video_fps(video_fps)
                        if coalesced_fps is not None:
                            processor_kwargs["fps"] = coalesced_fps

                    inputs = self.processor(**processor_kwargs)
                    inputs = inputs.to(input_device)

                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    chunk_outputs = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    for i, output in enumerate(chunk_outputs):
                        all_output_texts[chunk_start + i] = output

                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[Qwen3Dense] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    all_output_texts = [None] * len(questions)
                else:
                    raise

        if self.enable_metrics and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        return all_output_texts

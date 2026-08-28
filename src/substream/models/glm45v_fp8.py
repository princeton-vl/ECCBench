"""GLM-4.5V FP8 video-language model wrapper.

Wraps the zai-org/GLM-4.5V-FP8 checkpoint (FP8-quantized GLM-4.5V) for the
shared streaming video-QA benchmarking interface.
"""

import sys
import os
import time
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from substream.models.base import VideoLanguageModelInterface
from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration
from substream import config


class GLM45VFP8(VideoLanguageModelInterface):
    """GLM-4.5V-FP8 video-language model implementation."""

    def __init__(
        self,
        model_id: str,
        enable_metrics: bool = False,
        max_gpu_mem: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        """
        Robust loader for GLM-4.5V-FP8 that:
        - Preferentially loads from the latest local snapshot if present,
        - Else downloads from Hugging Face,
        - Uses device_map='auto' for placement (no manual skeleton),
        - Avoids torch_dtype='auto', which some transformers versions reject.
        """
        from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration

        # --- config knobs ---
        min_pixels = kwargs.get("min_pixels")
        max_pixels = kwargs.get("max_pixels")
        config.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        cache_dir = str(config.MODEL_CACHE)
        os.makedirs(cache_dir, exist_ok=True)

        self._min_pixels = min_pixels
        self._max_pixels = max_pixels

        if not torch.cuda.is_available():
            raise RuntimeError("GLM-4.5V-FP8 requires CUDA devices for multi-GPU placement.")

        # --- find latest local snapshot (if any) ---
        repo_dir = os.path.join(cache_dir, "models--zai-org--GLM-4.5V-FP8")
        snapshots_dir = os.path.join(repo_dir, "snapshots")
        load_path = None
        if os.path.isdir(snapshots_dir):
            snapshot_hashes = [
                h for h in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, h))
            ]
            if snapshot_hashes:
                snapshot_hashes.sort(key=lambda h: os.path.getmtime(os.path.join(snapshots_dir, h)))
                load_path = os.path.join(snapshots_dir, snapshot_hashes[-1])

        # --- load model (local snapshot if present, else HF repo) ---
        model_src = load_path if load_path is not None else self.model_id
        if load_path is None:
            print("[GLM45V-FP8] No local snapshot found; downloading model from Hugging Face.")

        # Optional offload directory for larger models
        offload_dir = os.environ.get("GLM_OFFLOAD_DIR", str(config.SCRATCH_ROOT / "glm_fp8_offload"))
        os.makedirs(offload_dir, exist_ok=True)

        # Cap memory per GPU to avoid poor distribution by Accelerate
        max_memory = {}
        # Use max_gpu_mem if provided, otherwise default to 60 GiB
        per_gpu_limit = int(max_gpu_mem) if max_gpu_mem is not None else 60
        for i in range(torch.cuda.device_count()):
            max_memory[i] = f"{per_gpu_limit}GiB"
        max_memory["cpu"] = "0GiB"  # Disable CPU offload

        model_kwargs = {
            "device_map": "auto",
            "max_memory": max_memory,
            "trust_remote_code": True,
            "cache_dir": cache_dir,
            "offload_folder": offload_dir,
            "offload_state_dict": True,
        }

        self.model = Glm4vMoeForConditionalGeneration.from_pretrained(
            model_src,
            **model_kwargs,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        # --- processor ---
        self.processor = AutoProcessor.from_pretrained(
            "zai-org/GLM-4.5V-FP8",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # --- bookkeeping used elsewhere ---
        self._model_devices = self._resolve_model_devices()
        self.latest_time = 0
        self.video_segments: Dict[int, List[Dict[str, Any]]] = {}
        self.text_entries: List[tuple[str, float]] = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

    def add_video(self, video_frames: np.ndarray, time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
        if time_start >= time_end:
            raise ValueError("time_end must be greater than time_start")
        if time_start < self.latest_time:
            raise ValueError("time_start must be after the last added video segment")

        self.latest_time = time_end

        if isinstance(video_frames, np.ndarray):
            video_frames = torch.from_numpy(video_frames)

        video_frames = video_frames.clone()

        if torch.is_tensor(video_frames) and video_frames.ndim == 4:
            height = video_frames.shape[2]
            width = video_frames.shape[3]
            area = height * width
            target_area = None
            if self._max_pixels and area > self._max_pixels:
                target_area = self._max_pixels
            elif self._min_pixels and area < self._min_pixels:
                target_area = self._min_pixels
            if target_area and height > 0 and width > 0:
                scale = math.sqrt(target_area / float(area))
                new_h = max(1, int(round(height * scale)))
                new_w = max(1, int(round(width * scale)))
                if new_h != height or new_w != width:
                    dtype = video_frames.dtype
                    needs_conversion = not torch.is_floating_point(video_frames)
                    frames_for_resize = (
                        video_frames.float() / 255.0 if needs_conversion else video_frames
                    )
                    video_frames = F.interpolate(
                        frames_for_resize,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if needs_conversion:
                        video_frames = (video_frames * 255.0).to(dtype)

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = self._cuda_memory_allocated()

        if video_id is None:
            video_id = 0
        if video_id not in self.video_segments:
            self.video_segments[video_id] = []

        segment = {
            "frames": video_frames,
            "time_start": time_start,
            "time_end": time_end,
            "duration": time_end - time_start,
            "num_frames": video_frames.shape[0],
        }
        self.video_segments[video_id].append(segment)

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = self._cuda_peak_memory()
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            self._reset_cuda_peak_memory()
            state_mem = self._get_state_memory_floats()
            self._record_add_video_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                time_end,
                state_memory_total=state_mem,
            )

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = self._cuda_memory_allocated()

        timestamp = max(0.0, float(current_video_time))
        self.text_entries.append((text, timestamp))

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = self._cuda_peak_memory()
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            self._reset_cuda_peak_memory()
            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                timestamp,
                state_memory_total=state_mem,
            )

    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256,
                     max_frames_in_video: int = 768, sample_method: str = "TIME") -> str:
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = self._cuda_memory_allocated()

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
            timeline_events.append({
                "type": "text",
                "time": timestamp,
                "text": text,
            })
        timeline_events.sort(key=lambda x: (x["time"], x["type"] == "video"))

        content = []
        video_inputs = []
        video_data: Dict[int, Dict[str, Any]] = {}
        video_fps: List[float] = []

        for event in timeline_events:
            if event["type"] == "video":
                vid = event["video_id"]
                segment = event["segment"]
                if vid not in video_data:
                    video_data[vid] = {"frames": None, "total_duration": 0.0}
                if video_data[vid]["frames"] is None:
                    video_data[vid]["frames"] = segment["frames"]
                else:
                    video_data[vid]["frames"] = torch.cat(
                        [video_data[vid]["frames"], segment["frames"]], dim=0
                    )
                video_data[vid]["total_duration"] += segment["duration"]

        self._video_was_truncated = False  # Reset flag
        for vid, data in video_data.items():
            frames = data["frames"]
            original_frame_count = frames.shape[0]
            if original_frame_count > max_frames_in_video:
                self._video_was_truncated = True  # Track truncation
                if sample_method == "RANDOM":
                    indices = np.sort(np.random.choice(frames.shape[0], max_frames_in_video, replace=False))
                else:
                    indices = np.linspace(0, frames.shape[0] - 1, max_frames_in_video, dtype=int)
                video_data[vid]["frames"] = frames[indices]

                # Adjust duration proportionally to maintain consistent FPS after subsampling
                subsample_ratio = max_frames_in_video / original_frame_count
                video_data[vid]["total_duration"] *= subsample_ratio

        processed_videos = set()
        for event in timeline_events:
            if event["type"] == "video":
                vid = event["video_id"]
                if vid not in processed_videos:
                    content.append({"type": "video"})
                    frames = video_data[vid]["frames"]
                    video_inputs.append(frames)
                    fps = frames.shape[0] / max(video_data[vid]["total_duration"], 1e-6)
                    video_fps.append(fps)
                    processed_videos.add(vid)
            elif event["type"] == "text":
                content.append({"type": "text", "text": event["text"]})

        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            do_sample_frames=False,
        )

        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
            "do_sample_frames": False,
        }
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
            coalesced_fps = self._coalesce_video_fps(video_fps)
            if coalesced_fps is not None:
                processor_kwargs["fps"] = coalesced_fps
        inputs = self.processor(**processor_kwargs)

        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        # Move inputs to the device of the first embedding layer (proper multi-GPU support)
        # Using next(parameters()) would only get cuda:0 even if model spans multiple GPUs
        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is not None:
            model_device = input_embeddings.weight.device
            inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = self._cuda_peak_memory()
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            self._reset_cuda_peak_memory()

            latest_context_time = 0.0
            for event in timeline_events:
                if event["type"] == "video":
                    latest_context_time = max(latest_context_time, float(event["segment"]["time_end"]))
                else:
                    latest_context_time = max(latest_context_time, float(event["time"]))
            question_timestamp = float(current_video_time) if current_video_time is not None else 0.0
            if question_timestamp <= 0 and latest_context_time > 0:
                question_timestamp = latest_context_time
            state_mem = self._get_state_memory_floats()
            self._record_ask_question_metrics(
                latency,
                0.0,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                question_timestamp,
                state_memory_total=state_mem,
            )

        return output_text[0]

    def get_state(self) -> Dict[str, Any]:
        timeline = []
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline.append({
                    "type": "video",
                    "timestamp": segment["time_start"],
                    "video_id": video_id,
                    "time_start": segment["time_start"],
                    "time_end": segment["time_end"],
                    "duration": segment["duration"],
                    "num_frames": segment["num_frames"],
                })
        for text, timestamp in self.text_entries:
            timeline.append({
                    "type": "text",
                    "timestamp": timestamp,
                    "text": text,
            })
        timeline.sort(key=lambda x: (x["timestamp"], x["type"] == "video"))
        return {
            "video_segments": dict(self.video_segments),
            "text_entries": list(self.text_entries),
            "latest_time": self.latest_time,
            "timeline": timeline,
        }

    def clear_context(self) -> None:
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._video_was_truncated = None
        if hasattr(self, '_text_token_cache'):
            self._text_token_cache.clear()
        else:
            self._text_token_cache = {}
        self._tokenizer_ref = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            devices = getattr(self, "_model_devices", [])
            if devices:
                for device in devices:
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.empty_cache(device)
            else:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

    def was_video_truncated(self) -> Optional[bool]:
        """Check if video frames were truncated in the last ask_question call."""
        return self._video_was_truncated

    def save_state(self) -> Dict[str, Any]:
        import copy
        saved_video_segments = {}
        for video_id, segments in self.video_segments.items():
            saved_segments = []
            for segment in segments:
                frames_cpu = segment["frames"].detach().cpu().clone() if torch.is_tensor(segment["frames"]) else segment["frames"]
                saved_segment = {
                    "frames": frames_cpu,
                    "time_start": segment["time_start"],
                    "time_end": segment["time_end"],
                    "duration": segment["duration"],
                    "num_frames": segment["num_frames"],
                }
                saved_segments.append(saved_segment)
            saved_video_segments[video_id] = saved_segments

        state = {
            "video_segments": saved_video_segments,
            "text_entries": copy.deepcopy(self.text_entries),
            "latest_time": self.latest_time,
            "enable_metrics": self.enable_metrics,
            "_metrics": None,
        }

        if self.enable_metrics and self._metrics is not None:
            metrics_dict = {
                "latency_add_video": copy.deepcopy(self._metrics.latency_add_video),
                "latency_add_text": copy.deepcopy(self._metrics.latency_add_text),
                "latency_ask_question": copy.deepcopy(self._metrics.latency_ask_question),
                "state_memory_floats": copy.deepcopy(self._metrics.state_memory_floats),
                "peak_gpu_mem_increase_add_video": copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_video),
                "peak_gpu_mem_increase_add_text": copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_text),
                "peak_gpu_mem_increase_ask_question": copy.deepcopy(self._metrics.peak_gpu_mem_increase_ask_question),
                "peak_gpu_mem_absolute_add_video": copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_video),
                "peak_gpu_mem_absolute_add_text": copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_text),
                "peak_gpu_mem_absolute_ask_question": copy.deepcopy(self._metrics.peak_gpu_mem_absolute_ask_question),
                "video_timestamps_add_video": copy.deepcopy(self._metrics.video_timestamps_add_video),
                "video_timestamps_add_text": copy.deepcopy(self._metrics.video_timestamps_add_text),
                "video_timestamps_ask_question": copy.deepcopy(self._metrics.video_timestamps_ask_question),
                "question_correctness_rate": copy.deepcopy(self._metrics.question_correctness_rate),
                "question_dont_know_rate": copy.deepcopy(self._metrics.question_dont_know_rate),
                "question_answered_mask": copy.deepcopy(self._metrics.question_answered_mask),
                "video_timestamps_question_outcome": copy.deepcopy(self._metrics.video_timestamps_question_outcome),
            }
            state["_metrics"] = metrics_dict

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        if state.get("enable_metrics", False) != self.enable_metrics:
            print("Warning: loading state with different metrics configuration.")

        self.video_segments = {}
        # Use input embeddings device for proper multi-GPU support
        input_embeddings = self.model.get_input_embeddings()
        model_device = input_embeddings.weight.device if input_embeddings is not None else next(self.model.parameters()).device
        for video_id, segments in state["video_segments"].items():
            restored = []
            for segment in segments:
                frames = segment["frames"]
                frames = frames.to(model_device) if torch.is_tensor(frames) else frames
                restored.append({
                    "frames": frames,
                    "time_start": segment["time_start"],
                    "time_end": segment["time_end"],
                    "duration": segment["duration"],
                    "num_frames": segment["num_frames"],
                })
            self.video_segments[video_id] = restored

        self.text_entries = state["text_entries"].copy()
        self.latest_time = state["latest_time"]
        if hasattr(self, '_text_token_cache'):
            self._text_token_cache.clear()

        if self.enable_metrics and "_metrics" in state and state["_metrics"] is not None:
            from substream.models.base import PerformanceMetrics
            if self._metrics is None:
                self._metrics = PerformanceMetrics()
            metrics_data = state["_metrics"]
            self._metrics.latency_add_video = metrics_data.get("latency_add_video", []).copy()
            self._metrics.latency_add_text = metrics_data.get("latency_add_text", []).copy()
            self._metrics.latency_ask_question = metrics_data.get("latency_ask_question", []).copy()
            self._metrics.state_memory_floats = metrics_data.get("state_memory_floats", []).copy()
            self._metrics.peak_gpu_mem_increase_add_video = metrics_data.get("peak_gpu_mem_increase_add_video", []).copy()
            self._metrics.peak_gpu_mem_increase_add_text = metrics_data.get("peak_gpu_mem_increase_add_text", []).copy()
            self._metrics.peak_gpu_mem_increase_ask_question = metrics_data.get("peak_gpu_mem_increase_ask_question", []).copy()
            self._metrics.peak_gpu_mem_absolute_add_video = metrics_data.get("peak_gpu_mem_absolute_add_video", []).copy()
            self._metrics.peak_gpu_mem_absolute_add_text = metrics_data.get("peak_gpu_mem_absolute_add_text", []).copy()
            self._metrics.peak_gpu_mem_absolute_ask_question = metrics_data.get("peak_gpu_mem_absolute_ask_question", []).copy()
            self._metrics.video_timestamps_add_video = metrics_data.get("video_timestamps_add_video", []).copy()
            self._metrics.video_timestamps_add_text = metrics_data.get("video_timestamps_add_text", []).copy()
            self._metrics.video_timestamps_ask_question = metrics_data.get("video_timestamps_ask_question", []).copy()
            self._metrics.question_correctness_rate = metrics_data.get("question_correctness_rate", []).copy()
            self._metrics.question_dont_know_rate = metrics_data.get("question_dont_know_rate", []).copy()
            self._metrics.question_answered_mask = metrics_data.get("question_answered_mask", []).copy()
            self._metrics.video_timestamps_question_outcome = metrics_data.get("video_timestamps_question_outcome", []).copy()

    def _get_state_memory_floats(self) -> float:
        total = 0
        for video_id in self.video_segments:
            for segment in self.video_segments[video_id]:
                frames_tensor = segment.get("frames")
                if torch.is_tensor(frames_tensor):
                    total += frames_tensor.numel()
        for text, _ in self.text_entries:
            total += self._count_text_tokens(text)
        return total

    def _get_tokenizer(self):
        if getattr(self, "_tokenizer_ref", None) is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Processor does not expose a tokenizer; cannot measure text state size.")
            self._tokenizer_ref = tokenizer
        return self._tokenizer_ref

    def _count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        cache = getattr(self, "_text_token_cache", None)
        if cache is None:
            cache = {}
            self._text_token_cache = cache
        cached = cache.get(text)
        if cached is not None:
            return cached

        tokenizer = self._get_tokenizer()
        encoding = tokenizer(
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
        cache[text] = count
        return count

    def _resolve_model_devices(self) -> List[torch.device]:
        device_map = getattr(self.model, "hf_device_map", None)
        devices: List[torch.device] = []
        if device_map:
            for placement in device_map.values():
                if isinstance(placement, torch.device):
                    target = placement
                elif isinstance(placement, str):
                    if placement.startswith("cpu") or placement == "disk":
                        raise RuntimeError(f"Model placement includes non-CUDA device: {placement}")
                    target = torch.device(placement)
                elif isinstance(placement, int):
                    target = torch.device("cuda", placement)
                else:
                    continue
                if target.type != "cuda":
                    raise RuntimeError(f"Model placement includes non-CUDA device: {target}")
                if target not in devices:
                    devices.append(target)
        if not devices:
            param = next(self.model.parameters())
            if param.device.type != "cuda":
                raise RuntimeError(f"Unexpected parameter device {param.device}; CUDA required.")
            devices.append(param.device)
        return devices

    def _cuda_memory_allocated(self) -> int:
        if not torch.cuda.is_available():
            return 0
        devices = getattr(self, "_model_devices", [])
        if not devices:
            return torch.cuda.memory_allocated()
        return sum(torch.cuda.memory_allocated(device) for device in devices)

    def _cuda_peak_memory(self) -> int:
        if not torch.cuda.is_available():
            return 0
        devices = getattr(self, "_model_devices", [])
        if not devices:
            return torch.cuda.max_memory_allocated()
        return sum(torch.cuda.max_memory_allocated(device) for device in devices)

    def _reset_cuda_peak_memory(self) -> None:
        if not torch.cuda.is_available():
            return
        devices = getattr(self, "_model_devices", [])
        if not devices:
            torch.cuda.reset_peak_memory_stats()
            return
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)

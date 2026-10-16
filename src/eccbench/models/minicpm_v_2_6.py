"""MiniCPM-V 2.6 (openbmb/MiniCPM-V-2_6) wrapper using streaming_prefill and streaming_generate."""

import copy
import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import transformers.models.whisper.modeling_whisper as mw
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from eccbench import config
from eccbench.models.base import VideoLanguageModelInterface


@dataclass
class PromptTokenState:
    """Track the accumulated language and vision tokens in the streaming prompt."""

    text_tokens: int = 0
    vision_tokens: int = 0

    def total_tokens(self) -> int:
        return self.text_tokens + self.vision_tokens

    def add_text(self, count: int) -> None:
        if count < 0:
            raise ValueError("Token count cannot be negative")
        self.text_tokens += count

    def add_vision(self, count: int) -> None:
        if count < 0:
            raise ValueError("Token count cannot be negative")
        self.vision_tokens += count

    def reset(self) -> None:
        self.text_tokens = 0
        self.vision_tokens = 0


# The MiniCPM remote code looks up WHISPER_ATTENTION_CLASSES, including an
# "sdpa" entry, which some transformers releases do not define.
WhisperAttention = mw.WhisperAttention

if not hasattr(mw, "WHISPER_ATTENTION_CLASSES"):
    mw.WHISPER_ATTENTION_CLASSES = {"default": WhisperAttention}
elif not isinstance(mw.WHISPER_ATTENTION_CLASSES, dict):
    mw.WHISPER_ATTENTION_CLASSES = {"default": WhisperAttention}

mw.WHISPER_ATTENTION_CLASSES.setdefault("default", WhisperAttention)
mw.WHISPER_ATTENTION_CLASSES.setdefault("sdpa", mw.WHISPER_ATTENTION_CLASSES["default"])


class MiniCPMVideo(VideoLanguageModelInterface):
    """MiniCPM-V 2.6 that keeps the context in the model's streaming session."""

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
        # Frame count and size of each added video segment.
        self._vision_segments: List[Dict[str, int]] = []
        self._text_token_cache: Dict[str, int] = {}
        self._prompt_tokens = PromptTokenState()
        self._video_was_truncated: Optional[bool] = None

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        logging.getLogger("transformers_modules.openbmb.MiniCPM-V-2_6").setLevel(logging.WARNING)
        config.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        cache_dir = str(config.MODEL_CACHE)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, cache_dir=cache_dir)
        if hasattr(self.model, "get_sys_prompt"):
            self.sys_msg = self.model.get_sys_prompt(mode='omni', language='en')
        else:
            self.sys_msg = {"role": "system", "content": "You are a helpful multimodal assistant."}

        if not hasattr(self.model, "streaming_prefill"):
            raise RuntimeError(
                f"MiniCPM model '{self.model_id}' does not expose streaming_prefill; "
                "pin a revision of openbmb/MiniCPM-V-2_6 that supports streaming."
            )
        self.session_id = '0'
        self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=[self.sys_msg],
            tokenizer=self.tokenizer
        )

    def add_video(self, video_frames, time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
        """Prefill a video segment.

        ``video_frames`` is either sampler output (``["<unit>", image, ...]``)
        or an array of frames, from which two evenly spaced frames are kept.
        """
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        frames_as_images: list = []

        if isinstance(video_frames, list) and len(video_frames) > 0 and video_frames[0] == "<unit>":
            content = video_frames
            i = 0
            while i < len(video_frames):
                if video_frames[i] == "<unit>" and i + 1 < len(video_frames):
                    item = video_frames[i + 1]
                    if hasattr(item, 'size'):
                        frames_as_images.append(item)
                    i += 2
                else:
                    i += 1
        else:
            num_frames = len(video_frames)
            sampled_indices = np.linspace(0, num_frames - 1, num=2, dtype=int)

            for idx in sampled_indices:
                frame = video_frames[idx]
                if torch.is_tensor(frame):
                    frame = frame.cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                if len(frame.shape) == 3:
                    if frame.shape[0] == 3:
                        frame = np.transpose(frame, (1, 2, 0))
                    if frame.shape[-1] == 3:
                        frames_as_images.append(Image.fromarray(frame))

            content = []
            for image in frames_as_images:
                content.extend(["<unit>", image])

        msgs = [{"role": "user", "content": content}]
        self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=msgs,
            tokenizer=self.tokenizer
        )

        if frames_as_images:
            width, height = frames_as_images[0].size
            self._vision_segments.append(
                {
                    "frames": len(frames_as_images),
                    "height": height,
                    "width": width,
                }
            )

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            state_mem = 0.0
            kv_cache = self.model.llm_past_key_values
            for layer in kv_cache:
                for k_or_v in layer:
                    state_mem += k_or_v.numel()
            self._record_add_video_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                time_end,
                state_mem,
            )

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        new_text_tokens = self._count_text_tokens(text)

        msgs = [{"role": "user", "content": text}]
        self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=msgs,
            tokenizer=self.tokenizer
        )

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            state_mem = 0
            kv_cache = self.model.llm_past_key_values
            for layer in kv_cache:
                for k_or_v in layer:
                    state_mem += k_or_v.numel()

            self._record_add_text_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                float(state_mem),
            )

        if new_text_tokens:
            self._prompt_tokens.add_text(new_text_tokens)

    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256, max_frames_in_video: int = 768, sample_method: str = "TIME") -> str:
        """Prefill the question and generate; the streamed context is never truncated."""
        self._video_was_truncated = False
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        msgs = [{"role": "user", "content": question}]
        self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=msgs,
            tokenizer=self.tokenizer
        )

        res = self.model.streaming_generate(
            session_id=self.session_id,
            tokenizer=self.tokenizer,
            temperature=0.6,
            generate_audio=False,
            max_tokens=max_tokens
        )

        text = ""
        for r in res:
            if hasattr(r, 'text') and r.text:
                text += r.text
            elif isinstance(r, dict) and 'text' in r:
                text += r['text']

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            state_mem = 0
            kv_cache = self.model.llm_past_key_values
            for layer in kv_cache:
                for k_or_v in layer:
                    state_mem += k_or_v.numel()

            self._record_ask_question_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                float(state_mem),
            )

        return text.strip()

    def get_state(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'model_id': self.model_id,
            'status': 'ready'
        }

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> List[str]:
        """Answer each question from a copy of the current session, one at a time."""
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        self._video_was_truncated = False
        shared_state = self.save_state()

        responses = []
        for question_text in questions:
            self.load_state(shared_state)

            msgs = [{"role": "user", "content": question_text}]
            self.model.streaming_prefill(
                session_id=self.session_id,
                msgs=msgs,
                tokenizer=self.tokenizer
            )

            res = self.model.streaming_generate(
                session_id=self.session_id,
                tokenizer=self.tokenizer,
                temperature=0.6,
                generate_audio=False,
                max_tokens=max_tokens
            )

            text = ""
            for r in res:
                if hasattr(r, 'text') and r.text:
                    text += r.text
                elif isinstance(r, dict) and 'text' in r:
                    text += r['text']

            responses.append(text)

        self.load_state(shared_state)
        return responses

    def clear_context(self) -> None:
        """Start a new session that holds only the system prompt."""
        self.model.reset_session()
        self.session_id = str(int(self.session_id) + 1)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=[self.sys_msg],
            tokenizer=self.tokenizer
        )
        self._vision_segments.clear()
        self._prompt_tokens.reset()
        self._video_was_truncated = None
        self._reset_state_memory_tracking()

    def was_video_truncated(self) -> Optional[bool]:
        return self._video_was_truncated

    def _clone_kv_cache(self, cache: Optional[Any], device: Optional[torch.device] = None):
        """Return a deep copy of the KV cache, optionally moved to *device*."""

        def _move(obj, target_device):
            if obj is None:
                return None
            if torch.is_tensor(obj):
                return obj.to(target_device)
            if isinstance(obj, (list, tuple)):
                moved = [_move(item, target_device) for item in obj]
                return type(obj)(moved)
            if isinstance(obj, dict):
                return {k: _move(v, target_device) for k, v in obj.items()}
            if hasattr(obj, "to"):
                return obj.to(device=target_device)
            return obj

        if cache is None:
            return None

        cloned = copy.deepcopy(cache)

        if device is not None:
            cloned = _move(cloned, device)

        return cloned

    def save_state(self) -> Dict[str, Any]:
        """Capture the streaming session, with its KV caches copied to CPU."""
        cpu_device = torch.device('cpu')
        model_session = {
            'session_id': getattr(self.model, 'session_id', None),
            'new_user_msg': getattr(self.model, 'new_user_msg', None),
            'llm_generated': getattr(self.model, 'llm_generated', None),
            'llm_generate_completed': getattr(self.model, 'llm_generate_completed', None),
            'is_first': getattr(self.model, 'is_first', None),
            'llm_past_key_values': self._clone_kv_cache(
                getattr(self.model, 'llm_past_key_values', None), device=cpu_device
            ),
            'audio_past_key_values': self._clone_kv_cache(
                getattr(self.model, 'audio_past_key_values', None), device=cpu_device
            ),
        }

        return {
            'session_id': self.session_id,
            'vision_segments': copy.deepcopy(self._vision_segments),
            'prompt_tokens': {
                'text': self._prompt_tokens.text_tokens,
                'vision': self._prompt_tokens.vision_tokens,
            },
            'model_session': model_session,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore a session captured by save_state."""
        if state is None:
            raise ValueError("Cannot load None state")

        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")

        model_session = state.get('model_session')

        if model_session is None:
            # A state that holds only a KV cache.
            cached_kv = state.get('kv_cache')
            model_session = {
                'session_id': state.get('session_id', None),
                'llm_past_key_values': cached_kv,
            }

        if hasattr(self.model, 'reset_session'):
            self.model.reset_session()

        self.session_id = str(state.get('session_id', '0'))
        self._vision_segments = copy.deepcopy(state.get('vision_segments', []))
        self._prompt_tokens.reset()
        prompt_token_state = state.get('prompt_tokens')
        if isinstance(prompt_token_state, dict):
            self._prompt_tokens.add_text(int(prompt_token_state.get('text', 0)))
            self._prompt_tokens.add_vision(int(prompt_token_state.get('vision', 0)))

        model_device = next(self.model.parameters()).device
        restored_llm_cache = self._clone_kv_cache(model_session.get('llm_past_key_values'), device=model_device)
        restored_audio_cache = self._clone_kv_cache(model_session.get('audio_past_key_values'), device=model_device)

        restored_session_id = model_session.get('session_id', self.session_id)
        if restored_session_id is not None:
            restored_session_id = str(restored_session_id)
        else:
            restored_session_id = self.session_id
        self.model.session_id = restored_session_id
        self.model.new_user_msg = model_session.get('new_user_msg', True)
        self.model.llm_generated = model_session.get('llm_generated', False)
        self.model.llm_generate_completed = model_session.get('llm_generate_completed', False)

        is_first_flag = model_session.get('is_first')
        if is_first_flag is not None:
            self.model.is_first = is_first_flag
        else:
            self.model.is_first = restored_llm_cache is None

        self.model.llm_past_key_values = restored_llm_cache
        if hasattr(self.model, 'audio_past_key_values'):
            self.model.audio_past_key_values = restored_audio_cache

        self._sync_state_memory_tracking_from_metrics()

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

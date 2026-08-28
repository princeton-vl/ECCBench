"""
MiniCPM-V 2.6 (8B) streaming wrapper for the shared benchmarking interface.

Targets openbmb/MiniCPM-V-2_6 via the model's streaming_prefill /
streaming_generate API.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import copy
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math
import logging
from PIL import Image
import tempfile
import librosa
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer
from substream.models.base import VideoLanguageModelInterface
from substream import config
#https://github.com/OpenBMB/MiniCPM-V

import sys
import os


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


# do this once, before AutoModel.from_pretrained(...)
import transformers.models.whisper.modeling_whisper as mw

try:
    from transformers.models.whisper.modeling_whisper import WhisperAttention
except Exception as exc:  # pragma: no cover - defensive import
    raise RuntimeError("Whisper compatibility patch failed") from exc

# Create/normalize the symbol expected by MiniCPM and make sure the new
# attn_implementation="sdpa" option has a valid entry. Some HuggingFace
# releases omit both the mapping and the "sdpa" alias, so patch it here.
if not hasattr(mw, "WHISPER_ATTENTION_CLASSES"):
    mw.WHISPER_ATTENTION_CLASSES = {"default": WhisperAttention}
elif not isinstance(mw.WHISPER_ATTENTION_CLASSES, dict):
    mw.WHISPER_ATTENTION_CLASSES = {"default": WhisperAttention}

mw.WHISPER_ATTENTION_CLASSES.setdefault("default", WhisperAttention)
mw.WHISPER_ATTENTION_CLASSES.setdefault("sdpa", mw.WHISPER_ATTENTION_CLASSES["default"])


    
class MiniCPMVideo(VideoLanguageModelInterface):
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
        """Initialize MiniCPMVideo with proper base class initialization."""
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )
        # Track accumulated vision segments (count + spatial resolution) for exact FLOP accounting
        self._vision_segments: List[Dict[str, int]] = []
        self._text_token_cache: Dict[str, int] = {}
        self._prompt_tokens = PromptTokenState()
        self._video_was_truncated: Optional[bool] = None

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        """
        Model-specific initialization logic.

        Args:
            **kwargs: Model-specific parameters
        """
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
        """
        Add video data to the model's context using streaming prefill.
        
        This method should mutate the internal context state and return nothing.
        The video data should be integrated into the model's understanding.
        
        Args:
            video_frames: Either numpy array of video frames OR pre-chunked list content
            time_start: Start time of the video segment
            time_end: End time of the video segment  
            video_id: Optional identifier for the video (used by some implementations)
        """
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Initialize frames_as_images for metrics calculation
        frames_as_images: list = []

        # Check if video_frames is already chunked content (List with "<unit>" markers)
        if isinstance(video_frames, list) and len(video_frames) > 0 and video_frames[0] == "<unit>":
            # Already chunked content - use directly
            content = video_frames
            # Extract PIL Images from chunked content for metrics
            # Format: ["<unit>", image, "<unit>", image, ...]
            frames_as_images = []
            i = 0
            while i < len(video_frames):
                if video_frames[i] == "<unit>" and i + 1 < len(video_frames):
                    item = video_frames[i + 1]
                    if hasattr(item, 'size'):  # PIL Images have .size attribute
                        frames_as_images.append(item)
                    i += 2
                else:
                    i += 1
        else:
            # Traditional numpy array - process as before
            # Calculate video duration and frames per second
            duration = time_end - time_start
            num_frames = len(video_frames)
            fps = num_frames / duration if duration > 0 else num_frames
            
            # Sample 2 frames per second uniformly
            frames_per_second = max(2, int(fps))
            sampled_indices = np.linspace(0, num_frames - 1, num=2, dtype=int)
            
            # Convert sampled video frames to PIL Images for MiniCPM
            frames_as_images = []
            for idx in sampled_indices:
                frame = video_frames[idx]
                
                # Convert tensor to numpy if needed
                if torch.is_tensor(frame):
                    frame = frame.cpu().numpy()
                
                # Ensure frame is uint8 and in correct shape
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                # Handle both (3, H, W) and (H, W, 3) formats
                if len(frame.shape) == 3:
                    if frame.shape[0] == 3:  # (3, H, W) format
                        frame = np.transpose(frame, (1, 2, 0))  # Convert to (H, W, 3)
                    if frame.shape[-1] == 3:  # Now in (H, W, 3) format
                        image = Image.fromarray(frame)
                        frames_as_images.append(image)
            
            content = []
            for image in frames_as_images:
                content.extend(["<unit>", image]) 
        
        # Stream prefill the video content - This is the core model operation
        msgs = [{"role": "user", "content": content}]
        o = self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=msgs,
            tokenizer=self.tokenizer
        )

        # Record vision segment statistics for downstream FLOP accounting
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
                    state_mem += k_or_v.numel()  # Number of floats in the tensor
            self._record_add_video_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                time_end,
                state_mem,
            )
 
    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        """
        Add text to the model's context using streaming prefill.
        
        This method should mutate the internal context state and return nothing.
        The text should be integrated into the model's understanding.
        
        Args:
            text: Text string to add to context
        """
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        new_text_tokens = self._count_text_tokens(text)

        # Stream prefill the text content - This is the core model operation
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
                    state_mem += k_or_v.numel()  # Number of floats in the tensor

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
        """
        Ask a question based on the current context using streaming generation.

        This method should generate a response based on all previously added
        video and text content without modifying the context.

        Args:
            question: Question to ask
            max_tokens: Maximum number of tokens to generate
            max_frames_in_video: Maximum frames per video (used by some implementations)
            sample_method: Sampling method for frames ("TIME", "RANDOM", "SEGMENT")

        Returns:
            Generated response as string
        """
        self._video_was_truncated = False  # MiniCPM uses streaming, no truncation
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Add question to context via streaming prefill
        msgs = [{"role": "user", "content": question}]
        self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        
        # Generate response using streaming generation - This is the core model operation
        res = self.model.streaming_generate(
            session_id=self.session_id,
            tokenizer=self.tokenizer,
            temperature=0.6,
            generate_audio=False,  # We don't need audio generation
            max_tokens=max_tokens
        )
        
        # Collect all text output
        text = ""
        for r in res:
            if hasattr(r, 'text') and r.text:
                text += r.text
            elif isinstance(r, dict) and 'text' in r:
                text += r['text']
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            print("Latency: ", latency)
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            state_mem = 0
            kv_cache = self.model.llm_past_key_values
            for layer in kv_cache:
                for k_or_v in layer:
                    state_mem += k_or_v.numel()  # Number of floats in the tensor

            self._record_ask_question_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                float(state_mem),
            )
        
        return text.strip()


    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the model.
        
        Returns:
            Dictionary containing current context information
        """
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
        """
        Ask multiple questions in a batched inference call.

        This method processes all questions by:
        1. Saving the current shared context state (videos + text)
        2. For each question, loading the shared state and generating independently
        3. Returning independent results for each question

        Args:
            questions: List of questions to ask
            current_video_time: Current video timestamp
            max_tokens: Maximum tokens to generate per question
            max_frames_in_video: Max frames (unused for MiniCPM)
            sample_method: Sampling method (unused for MiniCPM)

        Returns:
            List of generated responses, one per question
        """
        if not questions:
            return []

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        self._video_was_truncated = False

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Save current state to restore for each question
        shared_state = self.save_state()

        # Process each question independently with the shared context
        responses = []
        for question_text in questions:
            # Restore shared state for this question
            self.load_state(shared_state)

            # Add question via streaming prefill
            msgs = [{"role": "user", "content": question_text}]
            self.model.streaming_prefill(
                session_id=self.session_id,
                msgs=msgs,
                tokenizer=self.tokenizer
            )

            # Generate response
            res = self.model.streaming_generate(
                session_id=self.session_id,
                tokenizer=self.tokenizer,
                temperature=0.6,
                generate_audio=False,
                max_tokens=max_tokens
            )

            # Collect text
            text = ""
            for r in res:
                if hasattr(r, 'text') and r.text:
                    text += r.text
                elif isinstance(r, dict) and 'text' in r:
                    text += r['text']

            responses.append(text)

        # Restore original state
        self.load_state(shared_state)

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            print(f"[MiniCPM Batch] Processed {len(questions)} questions in {latency:.2f}s ({latency/len(questions):.2f}s per question)")

        return responses

    def clear_context(self) -> None:
        """
        Clear all context (video and text) from the model.
        
        This should reset the model to its initial state.
        """
        self.model.reset_session()
        self.session_id = str(int(self.session_id) + 1)
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Re-initialize with system prompt
        self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=[self.sys_msg],
            tokenizer=self.tokenizer
        )
        self._vision_segments.clear()
        self._prompt_tokens.reset()
        self._video_was_truncated = None
        if hasattr(self, '_reset_state_memory_tracking'):
            self._reset_state_memory_tracking()

    def was_video_truncated(self) -> Optional[bool]:
        """Check if video frames were truncated in the last ask_question call."""
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
        """Capture the current MiniCPM streaming session so it can be restored later."""
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
        """Load a previously saved model state, restoring MiniCPM's streaming caches."""
        if state is None:
            raise ValueError("Cannot load None state")

        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")

        # Validate that we can load this state
    

        model_session = state.get('model_session')

        if model_session is None:
            # Backward compatibility with older checkpoints that only stored kv_cache.
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

        if hasattr(self, '_sync_state_memory_tracking_from_metrics'):
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

"""
Qwen2.5-VL (7B) full-video streaming wrapper.
Targets the Qwen/Qwen2.5-VL-7B-Instruct checkpoint.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import numpy as np
from dataclasses import dataclass
from substream.models.base import VideoLanguageModelInterface, PerformanceMetrics
from enum import Enum
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import copy

import sys
import os



import time

from substream import config


config.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
CACHE_ROOT = str(config.MODEL_CACHE)
os.environ.setdefault("HF_HOME", CACHE_ROOT)
os.makedirs(CACHE_ROOT, exist_ok=True)

    
class QwenFullVideo(VideoLanguageModelInterface):
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
        """Initialize QwenFullVideo with proper base class initialization."""
        super().__init__(
            model_id,
            enable_metrics=enable_metrics,
            max_gpu_mem=max_gpu_mem,
            **kwargs,
        )
    
    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        """
        Model-specific initialization logic.
        
        Args:
            **kwargs: Model-specific parameters. Supported keys:
                - min_pixels (Optional[int])
                - max_pixels (Optional[int])
        """
        min_pixels = kwargs.get("min_pixels")
        max_pixels = kwargs.get("max_pixels")

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "cache_dir": CACHE_ROOT,
        }

        # Some callers forwarded max_gpu_mem; the argument is now
        # handled explicitly in the base class, so nothing further to do here.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            **model_kwargs,
        )

        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        processor_kwargs["cache_dir"] = CACHE_ROOT
        processor_kwargs["padding_side"] = "left"  # Required for decoder-only batched generation
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            **processor_kwargs,
        )
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

        if hasattr(self, '_reset_state_memory_tracking'):
            self._reset_state_memory_tracking()

            
    def add_video(self, video_frames: np.ndarray, time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
        """
        Add video frames to the model's context.
        
        This method should mutate the internal context state and return nothing.
        The video frames should be integrated into the model's understanding.
        
        Args:
            video_frames: Video frames as numpy array with shape (num_frames, 3, height, width)
            time_start: The time the video frames starts. Must be after last was added
            time_end: The time the video frames end. Must be after last were added
            video_id: Optional identifier for the video (defaults to 0 if not provided)
        """
        # Validate time ordering
        if time_start >= time_end:
            raise ValueError("time_end must be greater than time_start")
        if time_start < self.latest_time:
            raise ValueError("time_start must be after the last added video segment")
        
        self.latest_time = time_end
        
        if isinstance(video_frames, np.ndarray):
            video_frames = torch.from_numpy(video_frames)

        # Clone tensor to avoid memory sharing issues and keep a CPU copy
        video_frames = video_frames.clone()
        if video_frames.device.type != "cpu":
            video_frames = video_frames.cpu()
        
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Default video_id to 0 if not provided
        if video_id is None:
            video_id = 0
        
        if video_id not in self.video_segments:
            self.video_segments[video_id] = []
        
        segment = {
            'frames': video_frames,
            'time_start': time_start,
            'time_end': time_end,
            'duration': time_end - time_start,
            'num_frames': video_frames.shape[0]
        }
        
        self.video_segments[video_id].append(segment)
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
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
        """
        Add text to the model's context.

        This method should mutate the internal context state and return nothing.
        The text should be integrated into the model's understanding.

        Args:
            text: Text string to add to context
            current_video_time: Current timestamp in video (seconds from start)
        """
        # Add text with current timestamp (text goes first if same time as video)
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        self.text_entries.append((text, self.latest_time))
        
        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            state_mem = self._get_state_memory_floats()
            self._record_add_text_metrics(
                latency,
                peak_mem_increase_mb,
                peak_mem_absolute_mb,
                current_video_time,
                state_mem,
            )
    
    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256, max_frames_in_video: int = 768, sample_method: str = "TIME") -> str:
        """
        Ask a question based on the current context.

        This method should generate a response based on all previously added
        video and text content without modifying the context.

        Args:
            question: Question to ask
            current_video_time: Current timestamp in video when question is asked (seconds from start)
            max_tokens: Maximum number of tokens to generate
            max_frames_in_video: Maximum frames per video
            sample_method: Sampling method for frames ("TIME", "RANDOM", "SEGMENT")

        Returns:
            Generated response as string
        """
        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        # Create timeline of events (videos and text)
        timeline_events = []
        
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline_events.append({
                    'type': 'video',
                    'time': segment['time_start'],
                    'video_id': video_id,
                    'segment': segment
                })
        
        for text, timestamp in self.text_entries:
            timeline_events.append({
                'type': 'text',
                'time': timestamp,
                'text': text
            })
        
        timeline_events.sort(key=lambda x: (x['time'], x['type'] == 'video'))
        
        # Build message content based on timeline
        content = []
        video_inputs = []
        # Group segments by video_id to combine into single videos
        video_data = {}  # video_id -> {'frames': tensor, 'total_duration': float, 'total_frames': int}
        video_fps = []
        
        # First pass: collect all frames for each video
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                segment = event['segment']

                if video_id not in video_data:
                    video_data[video_id] = {
                        'frames': None,
                        'total_duration': 0.0,
                        'total_frames': 0,
                    }

                # Add all frames (we'll sample later)
                if video_data[video_id]['frames'] is None:
                    video_data[video_id]['frames'] = segment['frames']
                else:
                    video_data[video_id]['frames'] = torch.cat(
                        [video_data[video_id]['frames'], segment['frames']], dim=0
                    )

                video_data[video_id]['total_duration'] += segment['duration']
                video_data[video_id]['total_frames'] += int(
                    segment.get('num_frames', segment['frames'].shape[0])
                )

        # Second pass: apply max_frames_in_video limit to each complete video
        self._video_was_truncated = False  # Reset flag
        for video_id, data in video_data.items():
            frames = data['frames']
            if frames.shape[0] > max_frames_in_video:
                self._video_was_truncated = True  # Track that truncation occurred
                if sample_method == "RANDOM":
                    indices = np.random.choice(frames.shape[0], max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:  # TIME or SEGMENT (same logic)
                    indices = np.linspace(0, frames.shape[0] - 1, max_frames_in_video, dtype=int)

                data['frames'] = frames[indices]

            trimmed_frame_count = data['frames'].shape[0]
            original_frame_count = data['total_frames']
            original_duration = data['total_duration']
            if (
                original_frame_count
                and original_duration
                and trimmed_frame_count
                and trimmed_frame_count != original_frame_count
            ):
                original_fps = original_frame_count / original_duration
                if original_fps > 0:
                    data['total_duration'] = trimmed_frame_count / original_fps

        # Process timeline events for content
        processed_video_ids = set()
        
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']

                # Only add video to content once per video_id
                if video_id not in processed_video_ids:
                    frames_tensor = video_data[video_id]['frames']
                    if frames_tensor is None or frames_tensor.shape[0] == 0:
                        continue

                    frames_tensor = frames_tensor.to(device='cpu')
                    video_inputs.append(frames_tensor)

                    # All videos in this pipeline use fps=1.0
                    fps = 1.0
                    video_fps.append(fps)

                    content.append({"type": "video", "video": len(video_inputs) - 1})
                    processed_video_ids.add(video_id)

            elif event['type'] == 'text':
                content.append({"type": "text", "text": event['text']})
        
        # Add the question at the end
        content.append({
            "type": "text",
            "text": question
        })
        
        messages = [{"role": "user", "content": content}]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("\n[model-input] ===== RAW CHAT PROMPT START =====")
        print(text)
        print("[model-input] ===== RAW CHAT PROMPT END =====", flush=True)
        # Process inputs
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
            coalesced_fps = self._coalesce_video_fps(video_fps)
            if coalesced_fps is not None:
                processor_kwargs["fps"] = coalesced_fps
        inputs = self.processor(**processor_kwargs)
        # Move to model's input embedding device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Print raw response for debugging
        print("\n[model-output] ===== RAW MODEL RESPONSE START (Qwen) =====")
        print(output_text[0] if output_text else "")
        print("[model-output] ===== RAW MODEL RESPONSE END (Qwen) =====", flush=True)

        if self.enable_metrics:
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)
            peak_mem_absolute_mb = peak_mem / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            latest_context_time = 0.0
            for event in timeline_events:
                if event['type'] == 'video':
                    latest_context_time = max(latest_context_time, float(event['segment']['time_end']))
                else:
                    latest_context_time = max(latest_context_time, float(event['time']))
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
        
        return output_text[0]

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME"
    ) -> List[str]:
        """
        Ask multiple questions in a batched inference call for improved performance.

        All questions share the same video/text context (timeline).
        """
        if not questions:
            return []

        if len(questions) == 1:
            # Single question - use regular path
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Build timeline events (shared across all questions)
        timeline_events = []

        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline_events.append({
                    'type': 'video',
                    'time': segment['time_start'],
                    'video_id': video_id,
                    'segment': segment
                })

        for text, timestamp in self.text_entries:
            timeline_events.append({
                'type': 'text',
                'time': timestamp,
                'text': text
            })

        timeline_events.sort(key=lambda x: (x['time'], x['type'] == 'video'))

        # Build shared content (video/text context - same for all questions)
        shared_content = []
        video_inputs = []
        video_data = {}
        video_fps = []

        # First pass: collect all frames for each video
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                segment = event['segment']

                if video_id not in video_data:
                    video_data[video_id] = {
                        'frames': None,
                        'total_duration': 0.0,
                        'total_frames': 0,
                    }

                if video_data[video_id]['frames'] is None:
                    video_data[video_id]['frames'] = segment['frames']
                else:
                    video_data[video_id]['frames'] = torch.cat(
                        [video_data[video_id]['frames'], segment['frames']], dim=0
                    )

                video_data[video_id]['total_duration'] += segment['duration']
                video_data[video_id]['total_frames'] += int(
                    segment.get('num_frames', segment['frames'].shape[0])
                )

        # Second pass: apply max_frames_in_video limit
        self._video_was_truncated = False
        for video_id, data in video_data.items():
            frames = data['frames']
            if frames.shape[0] > max_frames_in_video:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    indices = np.random.choice(frames.shape[0], max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:
                    indices = np.linspace(0, frames.shape[0] - 1, max_frames_in_video, dtype=int)

                data['frames'] = frames[indices]

            trimmed_frame_count = data['frames'].shape[0]
            original_frame_count = data['total_frames']
            original_duration = data['total_duration']
            if (
                original_frame_count
                and original_duration
                and trimmed_frame_count
                and trimmed_frame_count != original_frame_count
            ):
                original_fps = original_frame_count / original_duration
                if original_fps > 0:
                    data['total_duration'] = trimmed_frame_count / original_fps

        # Process timeline events for shared content
        processed_video_ids = set()

        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']

                if video_id not in processed_video_ids:
                    frames_tensor = video_data[video_id]['frames']
                    if frames_tensor is None or frames_tensor.shape[0] == 0:
                        continue

                    frames_tensor = frames_tensor.to(device='cpu')
                    video_inputs.append(frames_tensor)

                    # All videos in this pipeline use fps=1.0
                    fps = 1.0
                    video_fps.append(fps)

                    shared_content.append({"type": "video", "video": len(video_inputs) - 1})
                    processed_video_ids.add(video_id)

            elif event['type'] == 'text':
                shared_content.append({"type": "text", "text": event['text']})

        # Build batch of messages (one per question)
        batch_messages = []
        for question_text in questions:
            content = list(shared_content)  # Copy shared content
            content.append({"type": "text", "text": question_text})
            batch_messages.append({"role": "user", "content": content})

        # Apply chat template to all questions
        batch_texts = [
            self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        # Get model device
        input_device = self.model.get_input_embeddings().weight.device

        # Batched inference with OOM retry (halve batch size on OOM)
        batch_size = len(questions)
        all_output_texts = [None] * len(questions)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(questions), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(questions))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    # Process this chunk
                    processor_kwargs = {
                        "text": chunk_texts,
                        "padding": True,
                        "return_tensors": "pt",
                    }
                    if video_inputs:
                        # Qwen2.5-VL processor expects nested lists for batched videos:
                        # videos = [[video1, video2], [video1, video2], ...]  # one list per text
                        chunk_size = len(chunk_texts)
                        processor_kwargs["videos"] = [video_inputs for _ in range(chunk_size)]
                        coalesced_fps = self._coalesce_video_fps(video_fps)
                        if coalesced_fps is not None:
                            processor_kwargs["fps"] = coalesced_fps

                    inputs = self.processor(**processor_kwargs)
                    inputs = inputs.to(input_device)

                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    chunk_outputs = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    for i, output in enumerate(chunk_outputs):
                        all_output_texts[chunk_start + i] = output

                # Success - break out of retry loop
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    # Clear CUDA cache and halve batch size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[QwenFullVideo] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                    # Reset results for retry
                    all_output_texts = [None] * len(questions)
                else:
                    raise

        output_texts = all_output_texts

        # Print debug info for first question
        if output_texts:
            print(f"\n[model-output-batch] ===== BATCHED RESPONSE ({len(output_texts)} questions, batch_size={batch_size}) =====")
            for i, text in enumerate(output_texts[:3]):  # Show first 3
                print(f"Q{i+1}: {text}")
            if len(output_texts) > 3:
                print(f"... and {len(output_texts)-3} more")
            print("[model-output-batch] ===== END =====", flush=True)

        if self.enable_metrics:
            # Record metrics for batch (simplified - could be per-question)
            latency = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_mem_increase_mb = max(0, (peak_mem - baseline_mem)) / (1024 * 1024)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        return output_texts

    def ask_question_batch_isolated(
        self,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
    ) -> List[str]:
        """
        TRUE parallel batching with isolated contexts per question.

        Each question gets its own complete context (main + ONE candidate).

        Args:
            contexts: List of context dicts, one per question.
                     For sequence mode: {'main_sequence': str, 'candidate_sequence': str,
                                        'question_text': str, 'mode': 'sequence'}
                     For video mode: {'main_video_frames': tensor, 'candidate_video_frames': tensor,
                                     'question_text': str, 'mode': 'video'}
            max_tokens: Maximum tokens to generate
            max_frames_in_video: Max frames per video

        Returns:
            List of response strings, one per context
        """
        if not contexts:
            return []

        # NOTE: Do NOT shortcut to _process_single_context_isolated for batch_size=1.
        # That path uses add_video()+ask_question() which has different FPS/content logic.
        # Always use the batch code path for consistency.

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Build separate messages for each question
        batch_messages = []
        all_videos_per_question = []

        for ctx in contexts:
            if is_sequence_mode:
                # Sequence mode: pure text
                prompt = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )
                content = [{"type": "text", "text": prompt}]
                message = {"role": "user", "content": content}
                batch_messages.append(message)
                all_videos_per_question.append([])

            else:
                # Video mode: build separate video list for this question
                main_frames = ctx.get('main_video_frames')
                candidate_frames = ctx.get('candidate_video_frames')

                videos_for_question = []
                content = []

                # Add main video
                if main_frames is not None and main_frames.shape[0] > 0:
                    # Sample to half of budget
                    if main_frames.shape[0] > max_frames_in_video // 2:
                        indices = np.linspace(0, main_frames.shape[0] - 1, max_frames_in_video // 2, dtype=int)
                        sampled_main = main_frames[indices]
                    else:
                        sampled_main = main_frames

                    sampled_main = sampled_main.to(device='cpu')
                    videos_for_question.append(sampled_main)
                    # Add text anchor BEFORE video (matches non-batched)
                    content.append({"type": "text", "text": "Here is a main video to remember:"})
                    content.append({"type": "video", "video": len(videos_for_question) - 1})

                # Add candidate clip - MODIFIED TO MATCH NON-BATCHED MODE
                if candidate_frames is not None and candidate_frames.shape[0] > 0:
                    # Add text anchor BEFORE video (matches non-batched)
                    content.append({"type": "text", "text": "\nHere is a candidate clip:\n"})
                    # Sample to remaining budget
                    total_main = sampled_main.shape[0] if main_frames is not None else 0
                    remaining = max_frames_in_video - total_main
                    if candidate_frames.shape[0] > remaining:
                        indices = np.linspace(0, candidate_frames.shape[0] - 1, max(1, remaining), dtype=int)
                        sampled_cand = candidate_frames[indices]
                    else:
                        sampled_cand = candidate_frames

                    sampled_cand = sampled_cand.to(device='cpu')
                    videos_for_question.append(sampled_cand)
                    content.append({"type": "video", "video": len(videos_for_question) - 1})

                content.append({"type": "text", "text": ctx['question_text']})
                message = {"role": "user", "content": content}
                batch_messages.append(message)
                all_videos_per_question.append(videos_for_question)

        # Apply chat template
        batch_texts = [
            self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        # Get device
        input_device = self.model.get_input_embeddings().weight.device

        # Batch process with OOM retry
        batch_size = len(contexts)
        all_output_texts = [None] * len(contexts)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(contexts))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    # Prepare processor kwargs
                    processor_kwargs = {
                        "text": chunk_texts,
                        "padding": True,
                        "return_tensors": "pt",
                    }

                    if not is_sequence_mode:
                        # Video mode: get video lists for this chunk
                        chunk_videos = all_videos_per_question[chunk_start:chunk_end]
                        # Filter out empty lists
                        valid_videos = [v for v in chunk_videos if v]

                        if valid_videos:
                            processor_kwargs["videos"] = valid_videos
                            # Add fps to avoid warning about missing video metadata
                            processor_kwargs["fps"] = 1.0

                            # Build video_metadata for Qwen3 compatibility (optional for Qwen2.5)
                            try:
                                from transformers.video_utils import VideoMetadata
                                video_metadata_per_question = []
                                for videos_list in valid_videos:
                                    metadata_for_question = []
                                    for video_tensor in videos_list:
                                        if torch.is_tensor(video_tensor):
                                            metadata = VideoMetadata(
                                                total_num_frames=int(video_tensor.shape[0]),
                                                fps=1.0,
                                                width=int(video_tensor.shape[3]),
                                                height=int(video_tensor.shape[2]),
                                                duration=float(video_tensor.shape[0] / 1.0),
                                                video_backend="tensor",
                                                frames_indices=list(range(video_tensor.shape[0])),
                                            )
                                            metadata_for_question.append(metadata)
                                    video_metadata_per_question.append(metadata_for_question)

                                if video_metadata_per_question:
                                    processor_kwargs["video_metadata"] = video_metadata_per_question
                            except ImportError:
                                # VideoMetadata not available in older transformers, skip
                                pass

                    # Tokenize
                    inputs = self.processor(**processor_kwargs)
                    inputs = inputs.to(device=input_device, non_blocking=True)

                    # Generate
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max(1, int(max_tokens))
                        )

                    # Trim prompts
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                    ]

                    # Decode
                    chunk_outputs = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    for i, output in enumerate(chunk_outputs):
                        all_output_texts[chunk_start + i] = output

                # Success
                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[QwenFull-Isolated] OOM: Reducing batch size {old_batch_size} → {batch_size}")
                    # Keep partial responses instead of resetting to None
                else:
                    raise

        # Debug output
        print(f"\n[model-output-batch-isolated] ===== TRUE BATCHED RESPONSE QwenFull "
              f"({len(all_output_texts)} questions, batch_size={batch_size}, mode={mode}) =====")
        for i, text in enumerate(all_output_texts[:3]):
            preview = text[:100] if text else "(empty)"
            print(f"Q{i+1}: {preview}...")
        if len(all_output_texts) > 3:
            print(f"... and {len(all_output_texts)-3} more")
        print("[model-output-batch-isolated] ===== END =====", flush=True)

        return all_output_texts

    def _process_single_context_isolated(self, ctx: Dict[str, Any], max_tokens: int, max_frames_in_video: int) -> str:
        """Process a single isolated context (fallback for batch size 1)."""
        mode = ctx.get('mode', 'sequence')

        if mode == 'sequence':
            # Text-only
            prompt = (
                f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                f"{ctx['question_text']}"
            )
            return self.ask_question(prompt, current_video_time=0.0, max_tokens=max_tokens)
        else:
            # Video mode
            main_frames = ctx.get('main_video_frames')
            candidate_frames = ctx.get('candidate_video_frames')

            # Clear and rebuild context
            self.clear_context()

            # Add main video — duration = num_frames at 1fps
            if main_frames is not None:
                main_dur = float(main_frames.shape[0])
                self.add_video(main_frames, time_start=0.0, time_end=main_dur, video_id="main_video")
            else:
                main_dur = 0.0

            # Add candidate clip after main
            cand_start = main_dur + 1.0
            if candidate_frames is not None:
                cand_dur = float(candidate_frames.shape[0])
                self.add_video(candidate_frames, time_start=cand_start, time_end=cand_start + cand_dur, video_id="candidate_clip")
            else:
                cand_dur = 0.0

            return self.ask_question(ctx['question_text'], current_video_time=cand_start + cand_dur, max_tokens=max_tokens)

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the model.
        
        Returns:
            Dictionary containing current context information
        """
        # Create timeline of all events
        timeline = []
        
        # Add video segments to timeline
        for video_id, segments in self.video_segments.items():
            for segment in segments:
                timeline.append({
                    'type': 'video',
                    'timestamp': segment['time_start'],
                    'video_id': video_id,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames']
                })
        
        # Add text entries to timeline
        for text, timestamp in self.text_entries:
            timeline.append({
                'type': 'text',
                'timestamp': timestamp,
                'text': text
            })
        
        # Sort timeline by timestamp (text first if same time)
        timeline.sort(key=lambda x: (x['timestamp'], x['type'] == 'video'))
        
        return {
            'video_segments': dict(self.video_segments),
            'text_entries': list(self.text_entries),
            'latest_time': self.latest_time,
            'timeline': timeline
        }

    
    def clear_context(self) -> None:
        """
        Clear all context (video and text) from the model.

        This should reset the model to its initial state.
        """
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._video_was_truncated = None

        # Additional memory cleanup
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Note: Don't reset metrics here - they should accumulate across questions

    def was_video_truncated(self) -> Optional[bool]:
        """
        Check if video frames were truncated in the last ask_question call.

        Returns:
            True if frames were truncated due to max_frames_in_video limit,
            False if all frames were used,
            None if no question has been asked yet
        """
        return self._video_was_truncated

    def save_state(self) -> Dict[str, Any]:
        """
        Save the current model state to memory.

        Returns a deep copy of all state so GPU memory can be freed while preserving
        the ability to restore the exact state later. Preserves metrics across save/load.
        """
        # Deep copy video segments to CPU memory
        saved_video_segments = {}
        for video_id, segments in self.video_segments.items():
            saved_segments = []
            for segment in segments:
                # Move tensor to CPU and detach
                frames_cpu = segment['frames'].detach().cpu().clone() if torch.is_tensor(segment['frames']) else segment['frames']
                saved_segment = {
                    'frames': frames_cpu,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames']
                }
                saved_segments.append(saved_segment)
            saved_video_segments[video_id] = saved_segments

        state = {
            'video_segments': saved_video_segments,
            'text_entries': copy.deepcopy(self.text_entries),
            'latest_time': self.latest_time,
            'enable_metrics': self.enable_metrics,
            '_metrics': None
        }

        # Save metrics if enabled (preserve across save/load cycles)
        if self.enable_metrics and self._metrics is not None:
            metrics_dict = {
                'latency_add_video': copy.deepcopy(self._metrics.latency_add_video),
                'latency_add_text': copy.deepcopy(self._metrics.latency_add_text),
                'state_memory_floats': copy.deepcopy(self._metrics.state_memory_floats),
                'state_memory_after_add_video': copy.deepcopy(self._metrics.state_memory_after_add_video),
                'state_memory_after_add_text': copy.deepcopy(self._metrics.state_memory_after_add_text),
                'state_memory_delta_add_video': copy.deepcopy(self._metrics.state_memory_delta_add_video),
                'state_memory_delta_add_text': copy.deepcopy(self._metrics.state_memory_delta_add_text),
                'peak_gpu_mem_increase_add_video': copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_video),
                'peak_gpu_mem_increase_add_text': copy.deepcopy(self._metrics.peak_gpu_mem_increase_add_text),
                'peak_gpu_mem_absolute_add_video': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_video),
                'peak_gpu_mem_absolute_add_text': copy.deepcopy(self._metrics.peak_gpu_mem_absolute_add_text),
                'video_timestamps_add_video': copy.deepcopy(self._metrics.video_timestamps_add_video),
                'video_timestamps_add_text': copy.deepcopy(self._metrics.video_timestamps_add_text),
                'first_oom_timestamp': self._metrics.first_oom_timestamp,
            }
            state['_metrics'] = metrics_dict

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved model state.

        Restores video segments (moving them back to model device), text entries, timing,
        and preserves metrics across save/load cycles.
        """
        # Get the model's input device for proper multi-GPU support
        input_device = self.model.get_input_embeddings().weight.device

        # Restore video segments and move tensors back to model device
        self.video_segments = {}
        for video_id, segments in state['video_segments'].items():
            restored_segments = []
            for segment in segments:
                # Move tensor back to model device
                if torch.is_tensor(segment['frames']):
                    frames_gpu = segment['frames'].to(input_device)
                else:
                    frames_gpu = segment['frames']

                restored_segment = {
                    'frames': frames_gpu,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames']
                }
                restored_segments.append(restored_segment)
            self.video_segments[video_id] = restored_segments

        # Restore other state
        self.text_entries = state['text_entries'].copy()
        self.latest_time = state['latest_time']

        # Restore metrics if present and metrics are enabled
        if self.enable_metrics and '_metrics' in state and state['_metrics'] is not None:
            from substream.models.base import PerformanceMetrics

            # If we don't have metrics yet, create new object
            if self._metrics is None:
                self._metrics = PerformanceMetrics()

            preserved_fields = {
                'latency_ask_question',
                'state_memory_after_ask_question',
                'state_memory_delta_ask_question',
                'peak_gpu_mem_increase_ask_question',
                'peak_gpu_mem_absolute_ask_question',
                'video_timestamps_ask_question',
                'question_correctness_rate',
                'question_dont_know_rate',
                'question_answered_mask',
                'video_timestamps_question_outcome',
            }

            preserved = {
                name: copy.deepcopy(getattr(self._metrics, name, []))
                for name in preserved_fields
            }

            metrics_data = state['_metrics']
            self._metrics.latency_add_video = metrics_data.get('latency_add_video', []).copy()
            self._metrics.latency_add_text = metrics_data.get('latency_add_text', []).copy()
            self._metrics.state_memory_floats = metrics_data.get('state_memory_floats', []).copy()
            self._metrics.state_memory_after_add_video = metrics_data.get('state_memory_after_add_video', []).copy()
            self._metrics.state_memory_after_add_text = metrics_data.get('state_memory_after_add_text', []).copy()
            self._metrics.state_memory_delta_add_video = metrics_data.get('state_memory_delta_add_video', []).copy()
            self._metrics.state_memory_delta_add_text = metrics_data.get('state_memory_delta_add_text', []).copy()
            self._metrics.peak_gpu_mem_increase_add_video = metrics_data.get('peak_gpu_mem_increase_add_video', []).copy()
            self._metrics.peak_gpu_mem_increase_add_text = metrics_data.get('peak_gpu_mem_increase_add_text', []).copy()
            self._metrics.peak_gpu_mem_absolute_add_video = metrics_data.get('peak_gpu_mem_absolute_add_video', []).copy()
            self._metrics.peak_gpu_mem_absolute_add_text = metrics_data.get('peak_gpu_mem_absolute_add_text', []).copy()
            self._metrics.video_timestamps_add_video = metrics_data.get('video_timestamps_add_video', []).copy()
            self._metrics.video_timestamps_add_text = metrics_data.get('video_timestamps_add_text', []).copy()
            self._metrics.first_oom_timestamp = metrics_data.get('first_oom_timestamp')

            for name, value in preserved.items():
                setattr(self._metrics, name, value)

            self._sync_state_memory_tracking_from_metrics()

    def _get_tokenizer(self):
        if self._tokenizer_ref is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Processor does not expose a tokenizer; cannot measure text state size.")
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

    def _get_state_memory_floats(self) -> float:
        """Calculates the total memory usage of the stored state."""
        total_floats = 0.0
        for video_id in self.video_segments:
            for segment in self.video_segments[video_id]:
                frames_tensor = segment.get('frames')
                if torch.is_tensor(frames_tensor):
                    total_floats += float(frames_tensor.numel())

        for text, _ in self.text_entries:
            total_floats += float(self._count_text_tokens(text))
        return total_floats
    

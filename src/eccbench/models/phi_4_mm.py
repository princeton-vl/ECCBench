"""Phi-4-Multimodal (microsoft/Phi-4-multimodal-instruct) wrapper.

Frames are passed as images referenced by ``<|image_i|>`` tokens in the prompt.
"""

import copy
import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import dispatch_model
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from eccbench import config
from eccbench.models.base import PerformanceMetrics, VideoLanguageModelInterface

config.MODEL_CACHE.mkdir(parents=True, exist_ok=True)


def split_frame_budget(ctx, max_frames_in_video):
    """Return (main_frames, candidate_frames) for one context within the frame budget.

    The budget is split in half only when there is a candidate clip. Natural
    video contexts carry ``candidate_video_frames=None``. The batched path
    uses this function both to slice frames and to compute chunk offsets, so
    the two always agree.
    """
    main = ctx.get('main_video_frames') or []
    cand = ctx.get('candidate_video_frames') or []
    if cand:
        cap = max_frames_in_video // 2
        return list(main[:cap]), list(cand[:cap])
    return list(main[:max_frames_in_video]), []


# Upper limit on frames per video; more frames overflow the image index and memory.
PHI4_MAX_FRAMES_PER_VIDEO = 768


class PhiMultimodalVideo(VideoLanguageModelInterface):
    """Phi-4-Multimodal with interleaved video and text context."""

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
        model_path = "microsoft/Phi-4-multimodal-instruct"

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        multi_gpu = visible_gpus > 1

        model_kwargs = dict(
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
            low_cpu_mem_usage=True,
        )

        if multi_gpu:
            # Load weights on CPU first, then dispatch evenly across GPUs.
            model_kwargs['device_map'] = 'cpu'
        elif torch.cuda.is_available():
            model_kwargs['device_map'] = 'auto'

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )

        if multi_gpu:
            gpu_indices = list(range(visible_gpus))
            device_map: Dict[str, int] = {}

            def assign(module_name: str, device_idx: int) -> None:
                device_map[module_name] = gpu_indices[device_idx % visible_gpus]

            assign('model.embed_tokens', 0)
            assign('model.embed_dropout', 0)
            assign('lm_head', 0)
            assign('model.embed_tokens_extend', 0)

            layers = getattr(self.model.model, 'layers', [])
            for layer_idx, _ in enumerate(layers):
                assign(f'model.layers.{layer_idx}', layer_idx)

            assign('model.norm', len(layers) - 1 if layers else 0)

            self.model = dispatch_model(self.model, device_map=device_map)
        elif torch.cuda.is_available():
            from accelerate import infer_auto_device_map

            max_memory = {}
            per_gpu_limit = int(max_gpu_mem) if max_gpu_mem is not None else 60
            for i in range(torch.cuda.device_count()):
                max_memory[i] = f"{per_gpu_limit}GiB"
            max_memory["cpu"] = "0GiB"

            device_map = infer_auto_device_map(self.model, max_memory=max_memory)
            self.model = dispatch_model(self.model, device_map=device_map)

        self.temperature = 0.1
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._text_token_cache: Dict[str, int] = {}
        self._tokenizer_ref = None
        self._video_was_truncated: Optional[bool] = None

    def add_video(self, video_frames: List, time_start: float, time_end: float, video_id: Optional[int] = None) -> None:
        """Append frames (tensor, array or list of images) covering [time_start, time_end)."""
        if time_start >= time_end:
            raise ValueError("time_end must be greater than time_start")
        if time_start < self.latest_time:
            raise ValueError("time_start must be after the last added video segment")

        self.latest_time = time_end

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        frames = self._normalize_video_frames(video_frames)
        if not frames:
            raise ValueError("PhiMultimodalVideo requires at least one frame per segment")

        frame_count = len(frames)
        height, width = self._infer_frame_dimensions(frames[0])
        state_floats = self._compute_frame_float_total(frames)

        if video_id is None:
            video_id = 0

        if video_id not in self.video_segments:
            self.video_segments[video_id] = []

        segment = {
            'frames': frames,
            'time_start': time_start,
            'time_end': time_end,
            'duration': time_end - time_start,
            'num_frames': frame_count,
            'state_floats': state_floats,
            'vision_stats': {
                'frames': frame_count,
                'height': height,
                'width': width,
            },
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

    def ask_question(self, question: str, current_video_time: float = 0.0, max_tokens: int = 256, max_frames_in_video: int = PHI4_MAX_FRAMES_PER_VIDEO, sample_method: str = "TIME") -> str:
        max_frames_in_video = min(max_frames_in_video, PHI4_MAX_FRAMES_PER_VIDEO)

        if self.enable_metrics:
            start_time = time.perf_counter()
            baseline_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
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

        video_data = {}  # video_id -> {'frames': list, 'total_duration': float, 'total_frames': int}

        # Join the segments of each video.
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
                    video_data[video_id]['frames'].extend(segment['frames'])

                video_data[video_id]['total_duration'] += segment['duration']
                video_data[video_id]['total_frames'] += int(
                    segment.get('num_frames', len(segment['frames']))
                )

        # Subsample videos longer than max_frames_in_video.
        self._video_was_truncated = False
        for video_id, data in video_data.items():
            frames = data['frames']
            if len(frames) > max_frames_in_video:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    indices = np.random.choice(len(frames), max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:
                    indices = np.linspace(0, len(frames) - 1, max_frames_in_video, dtype=int)

                data['frames'] = [frames[i] for i in indices]

            trimmed_frame_count = len(data['frames'])
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

        processed_video_ids = set()
        video_images = []
        input_text = ""
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']

                if video_id not in processed_video_ids:
                    frames = video_data[video_id]['frames']
                    if frames is None or len(frames) == 0:
                        continue

                    video_images.extend(frames)
                    input_text += "".join(f'<|image_{i}|>' for i in range(len(video_images) - len(frames), len(video_images)))
                    processed_video_ids.add(video_id)

            elif event['type'] == 'text':
                input_text += " " + event['text']

        question_text = question.strip() if isinstance(question, str) else ""
        if question_text:
            if input_text.strip():
                input_text = input_text.strip() + "\n\n" + question_text
            else:
                input_text = question_text

        messages = [{'role': 'user', 'content': input_text}]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs = {
            'text': text,
            'return_tensors': 'pt',
        }
        if video_images:
            processor_kwargs['images'] = video_images
        inputs = self.processor(**processor_kwargs)

        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is not None:
            model_device = input_embeddings.weight.device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        target_max = max(1, int(max_tokens))
        generation_args = {
            'max_new_tokens': target_max,
            'temperature': self.temperature,
            'do_sample': True,
        }

        generation = self.model.generate(
            **inputs,
            **generation_args,
            generation_config=self.generation_config,
        )
        if isinstance(generation, tuple):
            generate_ids = generation[0]
        else:
            generate_ids = generation
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
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

        return response

    def ask_question_batch(
        self,
        questions: List[str],
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = PHI4_MAX_FRAMES_PER_VIDEO,
        sample_method: str = "TIME",
    ) -> List[str]:
        """Answer several questions about the same context in one batch."""
        if not questions:
            return []

        max_frames_in_video = min(max_frames_in_video, PHI4_MAX_FRAMES_PER_VIDEO)

        if len(questions) == 1:
            return [self.ask_question(questions[0], current_video_time, max_tokens, max_frames_in_video, sample_method)]

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

        video_data = {}
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                segment = event['segment']
                if video_id not in video_data:
                    video_data[video_id] = {'frames': None, 'total_duration': 0.0, 'total_frames': 0}
                if video_data[video_id]['frames'] is None:
                    video_data[video_id]['frames'] = segment['frames']
                else:
                    video_data[video_id]['frames'].extend(segment['frames'])
                video_data[video_id]['total_duration'] += segment['duration']
                video_data[video_id]['total_frames'] += int(segment.get('num_frames', len(segment['frames'])))

        self._video_was_truncated = False
        for video_id, data in video_data.items():
            frames = data['frames']
            if frames and len(frames) > max_frames_in_video:
                self._video_was_truncated = True
                if sample_method == "RANDOM":
                    indices = np.random.choice(len(frames), max_frames_in_video, replace=False)
                    indices = np.sort(indices)
                else:
                    indices = np.linspace(0, len(frames) - 1, max_frames_in_video, dtype=int)
                data['frames'] = [frames[i] for i in indices]

        processed_video_ids = set()
        video_images = []
        base_text = ""
        for event in timeline_events:
            if event['type'] == 'video':
                video_id = event['video_id']
                if video_id not in processed_video_ids:
                    frames = video_data[video_id]['frames']
                    if frames:
                        video_images.extend(frames)
                        base_text += "".join(f'<|image_{i}|>' for i in range(len(video_images) - len(frames), len(video_images)))
                        processed_video_ids.add(video_id)
            elif event['type'] == 'text':
                base_text += " " + event['text']

        batch_texts = []
        for question in questions:
            question_text = question.strip() if isinstance(question, str) else ""
            if question_text:
                if base_text.strip():
                    input_text = base_text.strip() + "\n\n" + question_text
                else:
                    input_text = question_text
            else:
                input_text = base_text

            messages = [{'role': 'user', 'content': input_text}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        input_embeddings = self.model.get_input_embeddings()
        model_device = input_embeddings.weight.device if input_embeddings is not None else 'cuda'

        all_responses: List[Optional[str]] = [None] * len(questions)

        if video_images:
            # Every sample gets its own copy of the images and numbers them from 0,
            # so image indices do not grow with the batch size.
            batch_size_val = len(questions)
            num_images_per_sample = len(video_images)
            all_images = video_images * batch_size_val

            base_text_parts = []
            for event in timeline_events:
                if event['type'] == 'text':
                    base_text_parts.append(event['text'])
            base_text = " ".join(base_text_parts) if base_text_parts else ""

            rebatched_texts = []
            for question_text in questions:
                sample_input_text = "".join(
                    f'<|image_{i}|>'
                    for i in range(num_images_per_sample)
                )
                if base_text:
                    sample_input_text += " " + base_text

                if question_text:
                    if sample_input_text.strip():
                        sample_input_text = sample_input_text.strip() + "\n\n" + question_text
                    else:
                        sample_input_text = question_text

                messages = [{'role': 'user', 'content': sample_input_text}]
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                rebatched_texts.append(text)

            generation_args = {
                'max_new_tokens': max(1, int(max_tokens)),
            }
            if self.temperature > 0:
                generation_args['temperature'] = self.temperature
                generation_args['do_sample'] = True
            else:
                generation_args['do_sample'] = False

            try:
                processor_kwargs = {
                    'text': rebatched_texts,
                    'images': all_images,
                    'return_tensors': 'pt',
                    'padding': True,
                }

                inputs = self.processor(**processor_kwargs)
                inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                generation = self.model.generate(**inputs, **generation_args, generation_config=self.generation_config)
                if isinstance(generation, tuple):
                    generate_ids = generation[0]
                else:
                    generate_ids = generation

                input_len = inputs['input_ids'].shape[1]
                generate_ids = generate_ids[:, input_len:]

                all_responses = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("[Phi] OOM, falling back to one question at a time")
                    all_responses = [None] * len(questions)
                    for i, text in enumerate(batch_texts):
                        try:
                            processor_kwargs = {'text': text, 'return_tensors': 'pt', 'images': video_images}
                            inputs = self.processor(**processor_kwargs)
                            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                            generation = self.model.generate(**inputs, **generation_args, generation_config=self.generation_config)
                            if isinstance(generation, tuple):
                                generate_ids = generation[0]
                            else:
                                generate_ids = generation
                            input_len = inputs['input_ids'].shape[1]
                            generate_ids = generate_ids[:, input_len:]
                            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                            all_responses[i] = response
                        except RuntimeError:
                            all_responses[i] = ""
                else:
                    raise
        else:
            batch_size = len(questions)
            while batch_size >= 1:
                try:
                    for chunk_start in range(0, len(questions), batch_size):
                        chunk_end = min(chunk_start + batch_size, len(questions))
                        chunk_texts = batch_texts[chunk_start:chunk_end]

                        processor_kwargs = {
                            'text': chunk_texts,
                            'return_tensors': 'pt',
                            'padding': True,
                        }

                        inputs = self.processor(**processor_kwargs)
                        inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                        generation_args = {
                            'max_new_tokens': max(1, int(max_tokens)),
                        }
                        if self.temperature > 0:
                            generation_args['temperature'] = self.temperature
                            generation_args['do_sample'] = True
                        else:
                            generation_args['do_sample'] = False

                        generation = self.model.generate(
                            **inputs,
                            **generation_args,
                            generation_config=self.generation_config,
                        )

                        if isinstance(generation, tuple):
                            generate_ids = generation[0]
                        else:
                            generate_ids = generation

                        input_len = inputs['input_ids'].shape[1]
                        generate_ids = generate_ids[:, input_len:]

                        chunk_responses = self.processor.batch_decode(
                            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )

                        for i, resp in enumerate(chunk_responses):
                            all_responses[chunk_start + i] = resp

                    break

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and batch_size > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        old_batch_size = batch_size
                        batch_size = batch_size // 2
                        print(f"[Phi] OOM: Reducing batch size from {old_batch_size} to {batch_size}")
                        all_responses = [None] * len(questions)
                    else:
                        raise

        return all_responses

    def ask_question_batch_isolated(
        self,
        contexts: List[Dict[str, Any]],
        max_tokens: int = 256,
        max_frames_in_video: int = PHI4_MAX_FRAMES_PER_VIDEO,
    ) -> List[str]:
        """Answer one question per context, each with its own input.

        A sequence context has ``main_sequence``, ``candidate_sequence`` and
        ``question_text``; a video context has ``main_video_frames``,
        ``candidate_video_frames`` (lists of images) and ``question_text``.
        """
        if not contexts:
            return []

        max_frames_in_video = min(max_frames_in_video, PHI4_MAX_FRAMES_PER_VIDEO)

        if len(contexts) == 1:
            return [self._process_single_isolated_context(contexts[0], max_tokens, max_frames_in_video)]

        mode = contexts[0].get('mode', 'sequence')
        is_sequence_mode = (mode == 'sequence')

        input_embeddings = self.model.get_input_embeddings()
        model_device = input_embeddings.weight.device if input_embeddings is not None else 'cuda'

        if is_sequence_mode:
            batch_texts = []
            for ctx in contexts:
                prompt_text = (
                    f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                    f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                    f"{ctx['question_text']}"
                )
                messages = [{'role': 'user', 'content': prompt_text}]
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)

            all_responses = self._execute_text_batch(
                batch_texts=batch_texts,
                max_tokens=max_tokens,
                model_device=model_device,
            )

        else:
            all_responses = self._execute_video_isolated_batch(
                contexts=contexts,
                max_tokens=max_tokens,
                max_frames_in_video=max_frames_in_video,
                model_device=model_device,
            )

        return all_responses

    def _process_single_isolated_context(
        self,
        ctx: Dict[str, Any],
        max_tokens: int,
        max_frames_in_video: int,
    ) -> str:
        mode = ctx.get('mode', 'sequence')

        if mode == 'sequence':
            prompt_text = (
                f"Sequence S_tokens: {ctx['main_sequence']}\n\n"
                f"Candidate sequence: {ctx['candidate_sequence']}\n\n"
                f"{ctx['question_text']}"
            )
            messages = [{'role': 'user', 'content': prompt_text}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_embeddings = self.model.get_input_embeddings()
            model_device = input_embeddings.weight.device if input_embeddings is not None else 'cuda'

            processor_kwargs = {'text': text, 'return_tensors': 'pt'}
            inputs = self.processor(**processor_kwargs)
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            generation_args = {
                'max_new_tokens': max(1, int(max_tokens)),
                'temperature': self.temperature,
                'do_sample': True,
            }

            generation = self.model.generate(
                **inputs,
                **generation_args,
                generation_config=self.generation_config,
            )

            if isinstance(generation, tuple):
                generate_ids = generation[0]
            else:
                generate_ids = generation

            input_len = inputs['input_ids'].shape[1]
            generate_ids = generate_ids[:, input_len:]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response

        else:
            main_frames, candidate_frames = split_frame_budget(
                ctx, max_frames_in_video)
            all_frames = main_frames + candidate_frames

            main_frame_count = len(main_frames)
            cand_frame_count = len(candidate_frames)

            prompt_text = "Here is a main video to remember:"
            prompt_text += "".join(f'<|image_{i}|>' for i in range(main_frame_count))
            if cand_frame_count:
                prompt_text += "\n\nHere is a candidate clip:\n"
                prompt_text += "".join(f'<|image_{i}|>' for i in range(main_frame_count, len(all_frames)))
            prompt_text += "\n\n" + ctx['question_text']

            messages = [{'role': 'user', 'content': prompt_text}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_embeddings = self.model.get_input_embeddings()
            model_device = input_embeddings.weight.device if input_embeddings is not None else 'cuda'

            processor_kwargs = {
                'text': text,
                'images': all_frames,
                'return_tensors': 'pt',
            }
            inputs = self.processor(**processor_kwargs)
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            generation_args = {
                'max_new_tokens': max(1, int(max_tokens)),
                'temperature': self.temperature,
                'do_sample': True,
            }

            generation = self.model.generate(
                **inputs,
                **generation_args,
                generation_config=self.generation_config,
            )

            if isinstance(generation, tuple):
                generate_ids = generation[0]
            else:
                generate_ids = generation

            input_len = inputs['input_ids'].shape[1]
            generate_ids = generate_ids[:, input_len:]
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response

    def _execute_text_batch(
        self,
        batch_texts: List[str],
        max_tokens: int,
        model_device: Any,
    ) -> List[str]:
        """Execute batched text-only inference with OOM retry."""
        batch_size = len(batch_texts)
        all_responses: List[Optional[str]] = [None] * len(batch_texts)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(batch_texts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(batch_texts))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    processor_kwargs = {
                        'text': chunk_texts,
                        'return_tensors': 'pt',
                        'padding': True,
                    }

                    inputs = self.processor(**processor_kwargs)
                    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                    generation_args = {
                        'max_new_tokens': max(1, int(max_tokens)),
                        'temperature': self.temperature,
                        'do_sample': True,
                    }

                    generation = self.model.generate(
                        **inputs,
                        **generation_args,
                        generation_config=self.generation_config,
                    )

                    if isinstance(generation, tuple):
                        generate_ids = generation[0]
                    else:
                        generate_ids = generation

                    input_len = inputs['input_ids'].shape[1]
                    generate_ids = generate_ids[:, input_len:]

                    chunk_responses = self.processor.batch_decode(
                        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    for i, resp in enumerate(chunk_responses):
                        all_responses[chunk_start + i] = resp

                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[Phi] OOM: Reducing batch size {old_batch_size} -> {batch_size}")
                else:
                    raise

        return all_responses

    def _execute_video_isolated_batch(
        self,
        contexts: List[Dict[str, Any]],
        max_tokens: int,
        max_frames_in_video: int,
        model_device: Any,
    ) -> List[str]:
        """Batched video inference with one context per sample.

        Each sample numbers its images from 0, so image indices do not grow
        with the batch size.
        """
        all_frames = []
        batch_texts = []

        for ctx in contexts:
            main_frames, candidate_frames = split_frame_budget(
                ctx, max_frames_in_video)
            all_frames.extend(main_frames + candidate_frames)

            prompt_text = "Here is a main video to remember:"
            for i in range(len(main_frames)):
                prompt_text += f'<|image_{i}|>'
            prompt_text += "\n\nHere is a candidate clip:\n"
            for i in range(len(candidate_frames)):
                prompt_text += f'<|image_{len(main_frames) + i}|>'
            prompt_text += "\n\n"
            prompt_text += ctx['question_text']

            messages = [{'role': 'user', 'content': prompt_text}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_texts.append(text)

        batch_size = len(contexts)
        all_responses: List[Optional[str]] = [None] * len(contexts)

        while batch_size >= 1:
            try:
                for chunk_start in range(0, len(contexts), batch_size):
                    chunk_end = min(chunk_start + batch_size, len(contexts))
                    chunk_texts = batch_texts[chunk_start:chunk_end]

                    # Locate this chunk's frames with the same split as above.
                    frames_before_chunk = 0
                    frames_in_chunk = 0
                    for i in range(chunk_start):
                        m_i, c_i = split_frame_budget(contexts[i], max_frames_in_video)
                        frames_before_chunk += len(m_i) + len(c_i)
                    for i in range(chunk_start, chunk_end):
                        m_i, c_i = split_frame_budget(contexts[i], max_frames_in_video)
                        frames_in_chunk += len(m_i) + len(c_i)

                    chunk_frames = all_frames[frames_before_chunk:frames_before_chunk + frames_in_chunk]

                    processor_kwargs = {
                        'text': chunk_texts,
                        'images': chunk_frames,
                        'return_tensors': 'pt',
                        'padding': True,
                    }

                    inputs = self.processor(**processor_kwargs)
                    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                    generation_args = {
                        'max_new_tokens': max(1, int(max_tokens)),
                        'temperature': self.temperature,
                        'do_sample': True,
                    }

                    generation = self.model.generate(
                        **inputs,
                        **generation_args,
                        generation_config=self.generation_config,
                    )

                    if isinstance(generation, tuple):
                        generate_ids = generation[0]
                    else:
                        generate_ids = generation

                    input_len = inputs['input_ids'].shape[1]
                    generate_ids = generate_ids[:, input_len:]

                    chunk_responses = self.processor.batch_decode(
                        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    for i, resp in enumerate(chunk_responses):
                        all_responses[chunk_start + i] = resp

                break

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    old_batch_size = batch_size
                    batch_size = batch_size // 2
                    print(f"[Phi] OOM: Reducing batch size {old_batch_size} -> {batch_size}")
                else:
                    raise

        return all_responses

    def get_state(self) -> Dict[str, Any]:
        timeline = []
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
        for text, timestamp in self.text_entries:
            timeline.append({
                'type': 'text',
                'timestamp': timestamp,
                'text': text
            })
        # Text goes before video at the same time.
        timeline.sort(key=lambda x: (x['timestamp'], x['type'] == 'video'))

        return {
            'video_segments': dict(self.video_segments),
            'text_entries': list(self.text_entries),
            'latest_time': self.latest_time,
            'timeline': timeline
        }

    def clear_context(self) -> None:
        """Drop all video and text context; metrics are kept."""
        self.latest_time = 0
        self.video_segments = {}
        self.text_entries = []
        self._video_was_truncated = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def was_video_truncated(self) -> Optional[bool]:
        return self._video_was_truncated

    def save_state(self) -> Dict[str, Any]:
        """Return a CPU copy of the context and the add_video/add_text metrics."""
        saved_video_segments = {}
        for video_id, segments in self.video_segments.items():
            saved_segments = []
            for segment in segments:
                frames_obj = segment['frames']
                if torch.is_tensor(frames_obj):
                    frames_cpu = frames_obj.detach().cpu().clone()
                elif isinstance(frames_obj, list):
                    frames_cpu = [self._clone_frame(frame) for frame in frames_obj]
                else:
                    frames_cpu = frames_obj
                saved_segment = {
                    'frames': frames_cpu,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames'],
                    'state_floats': segment.get('state_floats'),
                    'vision_stats': copy.deepcopy(segment.get('vision_stats', {})),
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
        """Restore a state from save_state; current ask_question metrics are kept."""
        input_embeddings = self.model.get_input_embeddings()
        model_device = input_embeddings.weight.device if input_embeddings is not None else next(self.model.parameters()).device

        self.video_segments = {}
        for video_id, segments in state['video_segments'].items():
            restored_segments = []
            for segment in segments:
                if torch.is_tensor(segment['frames']):
                    frames_gpu = segment['frames'].to(model_device)
                else:
                    frames_gpu = segment['frames']

                restored_segment = {
                    'frames': frames_gpu,
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'duration': segment['duration'],
                    'num_frames': segment['num_frames'],
                    'state_floats': segment.get('state_floats'),
                    'vision_stats': copy.deepcopy(segment.get('vision_stats', {})),
                }
                restored_segments.append(restored_segment)
            self.video_segments[video_id] = restored_segments

        self.text_entries = state['text_entries'].copy()
        self.latest_time = state['latest_time']

        if self.enable_metrics and '_metrics' in state and state['_metrics'] is not None:
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

    def _normalize_video_frames(self, video_frames: Any) -> List[Image.Image]:
        raw_frames: List[Any]
        if isinstance(video_frames, list):
            raw_frames = [self._clone_frame(frame) for frame in video_frames]
        elif torch.is_tensor(video_frames):
            tensor = video_frames.detach().cpu()
            if tensor.ndim != 4:
                raise ValueError("Expected 4D tensor for video frames (frames, C, H, W)")
            raw_frames = [tensor[i].clone() for i in range(tensor.shape[0])]
        elif isinstance(video_frames, np.ndarray):
            array = np.array(video_frames, copy=True)
            if array.ndim != 4:
                raise ValueError("Expected 4D ndarray for video frames")
            raw_frames = [np.array(array[i], copy=True) for i in range(array.shape[0])]
        else:
            raise TypeError("Unsupported video_frames type for PhiMultimodalVideo")

        pil_frames: List[Image.Image] = []
        for frame in raw_frames:
            pil_frames.append(self._frame_to_pil(frame))
        return pil_frames

    @staticmethod
    def _clone_frame(frame: Any) -> Any:
        if isinstance(frame, Image.Image):
            return frame.copy()
        if torch.is_tensor(frame):
            return frame.detach().cpu().clone()
        if isinstance(frame, np.ndarray):
            return np.array(frame, copy=True)
        return frame

    def _frame_to_pil(self, frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.copy()
        if torch.is_tensor(frame):
            array = frame.detach().cpu().numpy()
        elif isinstance(frame, np.ndarray):
            array = np.array(frame)
        else:
            raise TypeError("Unsupported frame type for PhiMultimodalVideo")

        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 3 and array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim != 3 or array.shape[2] not in (1, 3):
            raise ValueError("Expected frame data with channel dimension")
        return Image.fromarray(array)

    @staticmethod
    def _infer_frame_dimensions(frame: Any) -> Tuple[int, int]:
        if isinstance(frame, Image.Image):
            width, height = frame.size
            return int(height), int(width)
        if torch.is_tensor(frame):
            tensor = frame.detach().cpu()
            if tensor.ndim == 3:
                channels, h, w = tensor.shape
                if channels in (1, 3):
                    return int(h), int(w)
                return int(channels), int(h)
            if tensor.ndim >= 4:
                return int(tensor.shape[-2]), int(tensor.shape[-1])
            return 0, 0
        if isinstance(frame, np.ndarray):
            array = np.array(frame)
            if array.ndim == 3:
                if array.shape[0] in (1, 3):
                    return int(array.shape[1]), int(array.shape[2])
                return int(array.shape[0]), int(array.shape[1])
            if array.ndim >= 4:
                return int(array.shape[-2]), int(array.shape[-1])
            return 0, 0
        return 0, 0

    def _compute_frame_float_total(self, frames: List[Any]) -> float:
        total = 0.0
        for frame in frames or []:
            if torch.is_tensor(frame):
                total += float(frame.numel())
            elif isinstance(frame, np.ndarray):
                total += float(frame.size)
            elif isinstance(frame, Image.Image):
                array_view = np.asarray(frame)
                total += float(array_view.size)
        return total

    def _get_state_memory_floats(self) -> float:
        """Size of the stored context: frame elements plus text tokens."""
        total_floats = 0.0
        for video_id in self.video_segments:
            for segment in self.video_segments[video_id]:
                segment_total = segment.get('state_floats')
                if segment_total is not None:
                    total_floats += float(segment_total)
                else:
                    frames = segment.get('frames', [])
                    total_floats += self._compute_frame_float_total(frames)

        for text, _ in self.text_entries:
            total_floats += float(self._count_text_tokens(text))
        return total_floats


"""Frame sampler for Qwen3-Omni, through ``qwen_omni_utils``."""

from typing import Iterable, Optional, Union

import cv2
import numpy as np
import torch
from qwen_omni_utils import process_mm_info

from eccbench.samplers.base_sampler import FrameSamplerInterface, render_text_frame


class Qwen3oFrameSampler(FrameSamplerInterface):
    """Samples frames with the Qwen3-Omni preprocessing and returns a tensor."""

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample ``fps`` frames per second, at most ``max_frames``."""
        computed_max_frames = None
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_native_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps and native_fps > 0 and total_native_frames and total_native_frames > 0:
                duration_sec = total_native_frames / native_fps
                computed_max_frames = max(1, int(round(duration_sec * fps)))
        cap.release()

        if computed_max_frames is None:
            final_max_frames = max_frames
        elif max_frames is not None:
            final_max_frames = min(max_frames, computed_max_frames)
        else:
            final_max_frames = computed_max_frames

        content_entry = {"type": "video", "video": video_path}
        if final_max_frames is not None:
            content_entry["max_frames"] = final_max_frames

        messages = [{"role": "user", "content": [content_entry]}]
        _, _, video_inputs = process_mm_info(messages, use_audio_in_video=False)
        if isinstance(video_inputs, list) and len(video_inputs) > 0:
            return video_inputs[0]
        return video_inputs

    def _as_tensor(self, video_frames: Union[torch.Tensor, np.ndarray, Iterable]) -> torch.Tensor:
        """Join a tensor, an array, or a list of either into one ``(N, C, H, W)`` tensor."""
        if isinstance(video_frames, torch.Tensor):
            tensor = video_frames
        elif isinstance(video_frames, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(video_frames))
        elif isinstance(video_frames, (list, tuple)):
            if not video_frames:
                raise ValueError("No frame data provided for chunking.")
            tensor = torch.cat([self._as_tensor(part) for part in video_frames], dim=0)
        else:
            raise TypeError(f"Unsupported video frame container: {type(video_frames)}")

        if tensor.ndim != 4:
            raise ValueError(
                f"Expected tensor of shape (frames, channels, height, width); got {tensor.shape}"
            )
        return tensor.contiguous()

    def make_video_chunks(self, video_frames, fps: float, chunk_size: float,
                          segment_start_time: float):
        frames_tensor = self._as_tensor(video_frames)
        chunks = super().make_video_chunks(frames_tensor, fps, chunk_size, segment_start_time)
        return [(frames.contiguous(), start, end) for frames, start, end in chunks]

    def get_frame_count(self, video_frames) -> int:
        if video_frames is None:
            return 0

        def _count_frames(container) -> int:
            if isinstance(container, (torch.Tensor, np.ndarray)):
                return 0 if container.ndim == 0 else int(container.shape[0])
            if isinstance(container, (list, tuple)):
                return sum(_count_frames(item) for item in container)
            raise TypeError(f"Unsupported frame container: {type(container)}")

        return _count_frames(video_frames)

    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 5.0,
                           frame_width: int = 640, frame_height: int = 480) -> torch.Tensor:
        num_frames = max(1, int(duration_seconds * fps))
        frames = np.stack([render_text_frame(text, frame_width, frame_height)] * num_frames)
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().to(dtype=torch.uint8)

    def slice_frames(self, video_frames, start_idx: int, end_idx: int):
        return self._as_tensor(video_frames)[start_idx:end_idx].contiguous()

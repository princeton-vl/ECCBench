"""Frame sampler that returns a uint8 ``(N, 3, H, W)`` tensor.

Used by the Qwen-family, GLM-4.5V and MiMo-VL wrappers, the API models and
the dummy model.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from eccbench.samplers.base_sampler import (
    FrameSamplerInterface,
    decode_all_frames,
    render_text_frame,
    resolve_video_path,
)


class QwenFrameSampler(FrameSamplerInterface):
    """Reads every frame of a video into one tensor."""

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Decode all frames of the video; ``fps`` is not used."""
        return self._decode_with_decord_all_frames(
            self._resolve_video_path(video_path),
            max_frames=max_frames,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def _resolve_video_path(self, video_path: str) -> str:
        return resolve_video_path(video_path)

    def _decode_with_decord_all_frames(
        self,
        video_path: str,
        max_frames: Optional[int],
        min_pixels: Optional[int],
        max_pixels: Optional[int],
    ) -> torch.Tensor:
        frames = torch.from_numpy(decode_all_frames(video_path, max_frames))
        frames = frames.permute(0, 3, 1, 2).contiguous().to(dtype=torch.uint8)
        if min_pixels is not None or max_pixels is not None:
            frames = self._apply_pixel_constraints(frames, min_pixels, max_pixels)
        return frames

    @staticmethod
    def _apply_pixel_constraints(
        frames: torch.Tensor,
        min_pixels: Optional[int],
        max_pixels: Optional[int],
    ) -> torch.Tensor:
        """Resize frames so that H * W lies within ``[min_pixels, max_pixels]``."""
        if not torch.is_tensor(frames) or frames.ndim != 4:
            return frames

        height = frames.shape[2]
        width = frames.shape[3]
        if height <= 0 or width <= 0:
            return frames

        area = height * width
        target_area: Optional[int] = None
        if max_pixels and area > max_pixels:
            target_area = max_pixels
        elif min_pixels and area < min_pixels:
            target_area = min_pixels
        if not target_area or target_area <= 0 or target_area == area:
            return frames

        scale = math.sqrt(target_area / float(area))
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))
        if new_height == height and new_width == width:
            return frames

        resized = F.interpolate(
            frames.float().div_(255.0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        resized.mul_(255.0)
        return resized.to(dtype=torch.uint8)

    def get_frame_count(self, video_frames) -> int:
        if hasattr(video_frames, 'shape'):
            return video_frames.shape[0]
        return 0

    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 2.0,
                           frame_width: int = 640, frame_height: int = 480) -> torch.Tensor:
        num_frames = max(1, int(duration_seconds * fps))
        frames = np.stack([render_text_frame(text, frame_width, frame_height)] * num_frames)
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().to(dtype=torch.uint8)

    def slice_frames(self, video_frames, start_idx: int, end_idx: int):
        if torch.is_tensor(video_frames):
            return video_frames[start_idx:end_idx].contiguous()
        raise TypeError("Qwen sampler expects torch.Tensor outputs")

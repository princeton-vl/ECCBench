"""Frame sampler for InternVL3.5: normalised 448x448 tiles per frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None

from eccbench.samplers.base_sampler import (
    FrameSamplerInterface,
    decode_all_frames,
    render_text_frame,
    resolve_video_path,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class InternVLSampledVideo:
    """Tiles of all frames, with the number of tiles per frame."""

    pixel_values: torch.Tensor
    num_patches_list: List[int]
    fps: float


class InternVLFrameSampler(FrameSamplerInterface):
    """Follows the preprocessing of the official InternVL video example."""

    DEFAULT_INPUT_SIZE = 448
    DEFAULT_MAX_TILES = 1
    DEFAULT_SEGMENTS = 32

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> InternVLSampledVideo:
        if VideoReader is None or cpu is None:
            raise RuntimeError(
                "InternVLFrameSampler requires decord. Please install it to continue."
            )

        resolved_path = resolve_video_path(video_path)
        num_segments = self._segments_from_request(
            resolved_path,
            requested_fps=fps,
            max_frames=max_frames,
        )
        pixel_values, num_patches_list = load_video(
            resolved_path,
            input_size=self.DEFAULT_INPUT_SIZE,
            max_num=self.DEFAULT_MAX_TILES,
            num_segments=num_segments,
        )
        fps_value = float(fps) if fps and fps > 0 else 1.0
        return InternVLSampledVideo(pixel_values=pixel_values, num_patches_list=num_patches_list, fps=fps_value)

    def make_video_chunks(
        self,
        video_frames: InternVLSampledVideo,
        fps: float,
        chunk_size: Optional[float],
        segment_start_time: float = 0.0,
    ) -> List[tuple]:
        if not isinstance(video_frames, InternVLSampledVideo):
            duration = 0.0
            return [(video_frames, segment_start_time, segment_start_time + duration)]

        frame_count = len(video_frames.num_patches_list)
        if frame_count == 0:
            return []

        frame_rate = fps if fps and fps > 0 else video_frames.fps
        total_duration = frame_count / frame_rate

        if not chunk_size or chunk_size <= 0:
            return [
                (
                    video_frames,
                    segment_start_time,
                    segment_start_time + total_duration,
                )
            ]

        frames_per_chunk = max(1, int(round(chunk_size * frame_rate)))
        chunks = []
        start_frame = 0
        while start_frame < frame_count:
            end_frame = min(frame_count, start_frame + frames_per_chunk)
            chunk_frames = self._slice_clip(video_frames, start_frame, end_frame)
            chunk_start = segment_start_time + (start_frame / frame_rate)
            chunk_end = segment_start_time + (end_frame / frame_rate)
            chunks.append((chunk_frames, chunk_start, chunk_end))
            start_frame = end_frame
        return chunks

    def get_frame_count(self, video_frames: InternVLSampledVideo) -> int:
        if isinstance(video_frames, InternVLSampledVideo):
            return len(video_frames.num_patches_list)
        if isinstance(video_frames, torch.Tensor):
            return int(video_frames.shape[0])
        return 0

    def create_text_frames(
        self,
        text: str,
        fps: float = 1.0,
        duration_seconds: float = 2.0,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> torch.Tensor:
        num_frames = max(1, int(duration_seconds * fps))
        frames = np.stack([render_text_frame(text, frame_width, frame_height)] * num_frames)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        return tensor.to(dtype=torch.uint8)

    def slice_frames(
        self,
        video_frames: InternVLSampledVideo,
        start_idx: int,
        end_idx: int,
    ) -> InternVLSampledVideo:
        if not isinstance(video_frames, InternVLSampledVideo):
            raise TypeError("InternVL sampler expects InternVLSampledVideo containers")
        return self._slice_clip(video_frames, start_idx, end_idx)

    def _slice_clip(
        self,
        clip: InternVLSampledVideo,
        start_frame: int,
        end_frame: int,
    ) -> InternVLSampledVideo:
        prefix = self._frame_offsets(clip.num_patches_list)
        start_idx = prefix[start_frame]
        end_idx = prefix[end_frame]
        pixel_values = clip.pixel_values[start_idx:end_idx]
        num_patches = clip.num_patches_list[start_frame:end_frame]
        return InternVLSampledVideo(pixel_values=pixel_values, num_patches_list=num_patches, fps=clip.fps)

    @staticmethod
    def _frame_offsets(counts: Sequence[int]) -> List[int]:
        offsets = [0]
        total = 0
        for count in counts:
            total += count
            offsets.append(total)
        return offsets

    def _segments_from_request(
        self,
        video_path: str,
        *,
        requested_fps: int,
        max_frames: Optional[int],
    ) -> int:
        target_fps = requested_fps if requested_fps and requested_fps > 0 else None
        if target_fps is not None:
            duration = self._infer_duration(video_path)
            if duration is not None:
                segments = max(1, int(round(duration * target_fps)))
                if max_frames is not None and max_frames > 0:
                    segments = min(segments, max_frames)
                return segments
        if max_frames is not None and max_frames > 0:
            return max(1, int(max_frames))
        return self.DEFAULT_SEGMENTS

    @staticmethod
    def _infer_duration(video_path: str) -> Optional[float]:
        if VideoReader is None or cpu is None:
            return None
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            fps_value = float(vr.get_avg_fps())
            if total_frames > 0 and fps_value > 0:
                return total_frames / fps_value
        except Exception:
            return None
        return None


# Helpers adapted from the InternVL3.5 video example.


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video(video_path, input_size=448, max_num=1, num_segments=32):
    """Return ``(pixel_values, num_patches_list)`` for at most ``num_segments`` frames.

    ``pixel_values`` has shape ``(total_tiles, 3, input_size, input_size)``.
    """
    transform = _build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []
    for frame in decode_all_frames(video_path, num_segments):
        img = Image.fromarray(frame).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list), num_patches_list

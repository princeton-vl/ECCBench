"""Frame sampler for MiniCPM-V 2.6: a flat ``["<unit>", image, "<unit>", image, ...]`` list."""

import math
from typing import List, Optional

from PIL import Image

from eccbench.samplers.base_sampler import (
    FrameSamplerInterface,
    decode_all_frames,
    render_text_frame,
)


def _images(contents: List) -> List:
    """The images of a ``["<unit>", image, ...]`` list, in order."""
    images, i = [], 0
    while i < len(contents):
        if contents[i] == "<unit>" and i + 1 < len(contents):
            images.append(contents[i + 1])
            i += 2
        else:
            i += 1
    return images


def _units(images: List) -> List:
    contents: List = []
    for image in images:
        contents.extend(["<unit>", image])
    return contents


class MiniCPMFrameSampler(FrameSamplerInterface):
    """Reads every frame of a video as a PIL image, each preceded by ``"<unit>"``."""

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> List:
        """Decode all frames of the video; ``fps`` is not used."""
        images = []
        for frame in decode_all_frames(video_path, kwargs.get("max_frames")):
            image = Image.fromarray(frame)
            if min_pixels is not None or max_pixels is not None:
                image = self._downsample_image(image, min_pixels, max_pixels)
            images.append(image)
        return _units(images)

    def make_video_chunks(
        self,
        video_frames: List,
        fps: float,
        chunk_size: Optional[float],
        segment_start_time: float = 0.0,
    ) -> List[tuple]:
        """Split into ``(contents, time_start, time_end)`` chunks of ``chunk_size`` seconds."""
        frames = _images(video_frames) if isinstance(video_frames, list) else []
        total_frames = len(frames)
        if total_frames == 0:
            return []
        if fps is None or fps <= 0:
            fps = 1.0

        if not chunk_size or chunk_size <= 0:
            return [(video_frames, segment_start_time, segment_start_time + total_frames / fps)]

        frames_per_chunk = max(1, int(math.ceil(chunk_size * fps)))
        chunks = []
        for start_idx in range(0, total_frames, frames_per_chunk):
            end_idx = min(total_frames, start_idx + frames_per_chunk)
            chunks.append((
                _units(frames[start_idx:end_idx]),
                segment_start_time + start_idx / fps,
                segment_start_time + end_idx / fps,
            ))
        return chunks

    @staticmethod
    def _downsample_image(image: Image.Image, min_pixels: Optional[int],
                          max_pixels: Optional[int]) -> Image.Image:
        """Resize so that width * height lies within ``[min_pixels, max_pixels]``."""
        width, height = image.size
        current_pixels = width * height
        if max_pixels is not None and current_pixels > max_pixels:
            target_pixels = max_pixels
        elif min_pixels is not None and current_pixels < min_pixels:
            target_pixels = min_pixels
        else:
            return image

        scale_factor = (target_pixels / current_pixels) ** 0.5
        new_width = max(1, int(width * scale_factor))
        new_height = max(1, int(height * scale_factor))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def get_frame_count(self, video_frames) -> int:
        if not isinstance(video_frames, list):
            return 0
        return len(_images(video_frames))

    def slice_frames(self, video_frames, start_idx: int, end_idx: int):
        if not isinstance(video_frames, list):
            raise TypeError("MiniCPM sampler expects flattened frame lists for slicing")
        return _units(_images(video_frames)[start_idx:end_idx])

    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 5.0,
                           frame_width: int = 640, frame_height: int = 480) -> List:
        num_frames = max(1, int(duration_seconds * fps))
        image = Image.fromarray(render_text_frame(text, frame_width, frame_height))
        return _units([image] * num_frames)

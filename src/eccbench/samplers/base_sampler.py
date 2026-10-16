"""Base interface for frame samplers, plus helpers shared by several samplers."""

import logging
import os
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FrameSamplerInterface(ABC):
    """Turns a video file into the frame container one model expects."""

    @abstractmethod
    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Decode ``video_path`` into the model's frame container."""

    @abstractmethod
    def get_frame_count(self, video_frames: Any) -> int:
        """Number of frames in a container returned by ``sample_frames``."""

    @abstractmethod
    def create_text_frames(
        self,
        text: str,
        fps: float = 1.0,
        duration_seconds: float = 2.0,
        frame_width: int = 448,
        frame_height: int = 448,
    ) -> Any:
        """Frames that show ``text``, in the model's frame container."""

    def make_video_chunks(self, video_frames, fps: float, chunk_size: float,
                          segment_start_time: float):
        """Split frames into ``(frames, time_start, time_end)`` chunks of ``chunk_size`` seconds.

        A ``chunk_size`` of ``None`` or ``<= 0`` gives one chunk with the whole video.
        """
        total_frames = len(video_frames)
        if not chunk_size or chunk_size <= 0:
            return [(video_frames, segment_start_time, segment_start_time + total_frames / fps)]

        frames_per_chunk = max(1, int(chunk_size * fps))
        chunks = []
        for i in range(0, total_frames, frames_per_chunk):
            chunk = video_frames[i : i + frames_per_chunk]
            start = segment_start_time + i / fps
            chunks.append((chunk, start, segment_start_time + (i + len(chunk)) / fps))
        return chunks

    def slice_frames(self, video_frames: Any, start_idx: int, end_idx: int) -> Any:
        """Frames ``start_idx`` to ``end_idx`` of a container."""
        if hasattr(video_frames, "__getitem__"):
            return video_frames[start_idx:end_idx]
        raise TypeError("This frame container does not support slicing; override slice_frames.")


def decode_all_frames(video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """Decode every frame with decord as an ``(N, H, W, 3)`` uint8 array.

    If ``max_frames`` is set and smaller than N, keep ``max_frames`` evenly spaced frames.
    """
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise RuntimeError("decord is required to decode videos: pip install decord") from exc

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    n = len(reader)
    frames = reader.get_batch(list(range(n))).asnumpy()
    if max_frames and n > max_frames:
        logger.info("Downsampling %s from %d to %d frames", video_path, n, max_frames)
        frames = frames[np.linspace(0, n - 1, max_frames, dtype=int)]
    return frames


def resolve_video_path(video_path: str) -> str:
    """Return ``video_path``, or else the largest file next to it named ``<stem>*<suffix>``."""
    if not video_path:
        raise ValueError("video_path must be provided for frame sampling")
    if os.path.exists(video_path):
        return video_path

    candidate = Path(video_path)
    if not candidate.parent.exists():
        raise FileNotFoundError(f"Video directory does not exist for requested path: {video_path}")
    suffix = candidate.suffix or ".mp4"
    matches = [p for p in candidate.parent.glob(f"{candidate.stem}*{suffix}") if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"Video file not found: {video_path}")
    chosen = max(matches, key=lambda p: (p.stat().st_size, p.stat().st_mtime))
    logger.warning("Resolved %s to candidate video %s", video_path, chosen)
    return str(chosen)


def render_text_frame(text: str, frame_width: int, frame_height: int) -> np.ndarray:
    """Black ``(H, W, 3)`` uint8 frame with ``text`` centred in white."""
    import cv2

    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    font, scale, thickness, line_height = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2, 30
    lines = textwrap.wrap(text, width=40)[:8]
    start_y = (frame_height - len(lines) * line_height) // 2 + line_height
    for i, line in enumerate(lines):
        (text_width, _), _ = cv2.getTextSize(line, font, scale, thickness)
        x = (frame_width - text_width) // 2
        cv2.putText(frame, line, (x, start_y + i * line_height), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

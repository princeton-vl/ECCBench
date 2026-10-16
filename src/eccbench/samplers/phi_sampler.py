"""Frame sampler for Phi-4-MM: a list of RGB PIL images."""

from typing import List, Optional

from PIL import Image

from eccbench.samplers.base_sampler import (
    FrameSamplerInterface,
    decode_all_frames,
    render_text_frame,
    resolve_video_path,
)


class PhiFrameSampler(FrameSamplerInterface):
    """Reads every frame of a video as a PIL image."""

    def sample_frames(
        self,
        video_path: str,
        fps: int = 2,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Decode all frames of the video; ``fps`` and the pixel limits are not used."""
        frames = decode_all_frames(resolve_video_path(video_path), max_frames)
        return [Image.fromarray(frame).convert("RGB") for frame in frames]

    def slice_frames(self, video_frames, start_idx: int, end_idx: int):
        if isinstance(video_frames, list):
            return video_frames[start_idx:end_idx]
        raise TypeError("Phi sampler expects a list of PIL images")

    def get_frame_count(self, video_frames) -> int:
        return len(video_frames)

    def create_text_frames(self, text: str, fps: float = 1.0, duration_seconds: float = 2.0,
                           frame_width: int = 640, frame_height: int = 480) -> List[Image.Image]:
        num_frames = max(1, int(duration_seconds * fps))
        image = Image.fromarray(render_text_frame(text, frame_width, frame_height))
        return [image.copy() for _ in range(num_frames)]

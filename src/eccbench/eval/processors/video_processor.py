"""Load a stream's video and add it to a model up to a given time."""

import math
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from eccbench.samplers import get_frame_sampler


class VideoProcessor:
    """Loads one video and streams it into a model in chunks.

    ``add_main_video_up_to_time`` keeps a cursor, so repeated calls add only
    the new part of the video. ``load_main_video`` and
    ``reset_video_position`` move the cursor back to 0.
    """

    def __init__(
        self,
        model_type: str,
        fps: int = 1,
        max_frames: Optional[int] = 256,
        chunk_size: float = 30,
        model_max_frames: Optional[int] = None,
    ) -> None:
        self.model_type = model_type
        self.fps = fps
        requested_sample_cap = max_frames
        if requested_sample_cap is not None:
            requested_sample_cap = max(1, int(requested_sample_cap))
        resolved_model_cap = model_max_frames
        if resolved_model_cap is not None:
            resolved_model_cap = max(1, int(resolved_model_cap))
            if requested_sample_cap is None:
                requested_sample_cap = resolved_model_cap
            else:
                requested_sample_cap = min(requested_sample_cap, resolved_model_cap)
        self.sample_max_frames = requested_sample_cap
        self.model_max_frames = resolved_model_cap if resolved_model_cap is not None else requested_sample_cap
        self.chunk_size = chunk_size
        self.frame_sampler = get_frame_sampler(model_type)
        self.current_main_video_time = 0.0
        self.main_video_frames = None
        self.main_video_duration: Optional[float] = None
        self.main_video_frames_added = 0

    def load_main_video(self, video_path: str) -> Union[np.ndarray, List]:
        """Sample the main video's frames and reset the cursor."""
        self.main_video_frames = self.frame_sampler.sample_frames(
            video_path,
            fps=self.fps,
            max_frames=self.sample_max_frames
        )
        self.main_video_duration = self._compute_video_duration(video_path)
        self.current_main_video_time = 0.0
        self.main_video_frames_added = 0
        return self.main_video_frames

    def reset_video_position(self) -> None:
        """Move the cursor to 0 without reloading frames (after ``model.clear_context()``)."""
        self.current_main_video_time = 0.0
        self.main_video_frames_added = 0

    def load_option_videos(self, option_paths: List[str]) -> List[Union[np.ndarray, List]]:
        """Sample the frames of each option or candidate clip."""
        option_videos = []
        for option_path in option_paths:
            option_video = self.frame_sampler.sample_frames(
                option_path,
                fps=self.fps,
                max_frames=self.sample_max_frames
            )
            option_videos.append(option_video)
        return option_videos

    def add_main_video_up_to_time(self, model: Any, target_time: float) -> float:
        """Add the main video from the cursor up to ``target_time``; return the new cursor."""
        if self.main_video_frames is None:
            raise ValueError("Main video not loaded. Call load_main_video() first.")

        if target_time <= self.current_main_video_time:
            print(f"  > Main video already at {self.current_main_video_time}s (target was {target_time}s)")
            return self.current_main_video_time

        all_chunks = self.frame_sampler.make_video_chunks(
            self.main_video_frames,
            self.fps,
            self.chunk_size,
            0.0,
        )
        # Rescale chunk times to the video's real duration.

        if self.main_video_duration and all_chunks:
            total_frames_available = self.frame_sampler.get_frame_count(self.main_video_frames)
            if total_frames_available:
                seconds_per_frame = self.main_video_duration / total_frames_available
                frame_cursor = 0
                adjusted_chunks = []
                for chunk_frames, _, _ in all_chunks:
                    frame_count = self.frame_sampler.get_frame_count(chunk_frames)
                    if frame_count == 0:
                        continue
                    chunk_start = frame_cursor * seconds_per_frame
                    frame_cursor += frame_count
                    chunk_end = min(self.main_video_duration, frame_cursor * seconds_per_frame)
                    adjusted_chunks.append((chunk_frames, chunk_start, chunk_end))
                if adjusted_chunks:
                    all_chunks = adjusted_chunks

        chunks_to_add: List[Tuple[Union[np.ndarray, List], float, float]] = []
        for chunk_frames, chunk_start, chunk_end in all_chunks:
            frame_count = self.frame_sampler.get_frame_count(chunk_frames)
            if frame_count == 0:
                continue

            if chunk_end <= self.current_main_video_time + 1e-9:
                continue

            if chunk_start >= target_time:
                break

            effective_start = max(chunk_start, self.current_main_video_time)
            effective_end = min(chunk_end, target_time)

            if effective_end <= effective_start:
                continue

            trimmed_frames, trimmed_start, trimmed_end = self._trim_chunk_to_range(
                chunk_frames,
                chunk_start,
                chunk_end,
                effective_start,
                effective_end,
            )
            chunks_to_add.append((trimmed_frames, trimmed_start, trimmed_end))

            if trimmed_end >= target_time - 1e-9:
                break

        frames_added = 0
        if chunks_to_add:
            for chunk_frames, _, _ in chunks_to_add:
                frames_added += self.frame_sampler.get_frame_count(chunk_frames)
            print(
                f"  > Adding {len(chunks_to_add)} main-video segments from {self.current_main_video_time}s"
                f" toward target {target_time}s"
            )
            final_time = self.add_video_chunks(model, chunks_to_add, 0)
            self.current_main_video_time = final_time
            self.main_video_frames_added += frames_added
            print(f"  > Main video now at {final_time}s (target was {target_time}s)")
            return final_time

        sampled_frames = self.frame_sampler.get_frame_count(self.main_video_frames)
        print(
            "  > No new chunks scheduled; existing samples already span the requested horizon."
            f" Current cursor {self.current_main_video_time}s, target {target_time}s"
            f" (sampled {sampled_frames} frames)"
        )
        return self.current_main_video_time

    def add_video_chunks(
        self,
        model: Any,
        chunks: List[Tuple[Union[np.ndarray, List], float, float]],
        video_id: int,
    ) -> float:
        """Add ``(frames, start, end)`` chunks to the model; return the last end time."""
        if not chunks:
            return 0.0

        for frames, time_start, time_end in chunks:
            model.add_video(
                video_frames=frames,
                time_start=time_start,
                time_end=time_end,
                video_id=video_id
            )
        return chunks[-1][2]

    def get_main_frames_streamed(self) -> int:
        return self.main_video_frames_added


    def _slice_frames(
        self,
        frames: Any,
        start_idx: int,
        end_idx: int,
        total_frames: int,
    ) -> Any:
        """Return frames ``start_idx:end_idx`` through the sampler."""
        if start_idx <= 0 and end_idx >= total_frames:
            return frames

        try:
            return self.frame_sampler.slice_frames(frames, start_idx, end_idx)
        except TypeError as exc:
            raise TypeError(
                "Frame representation does not support slicing; sampler must handle partial chunks."
            ) from exc

    def _trim_chunk_to_range(
        self,
        frames: Any,
        chunk_start: float,
        chunk_end: float,
        desired_start: float,
        desired_end: float,
    ) -> Tuple[Any, float, float]:
        """Trim a chunk to a time window, keeping at least one frame."""
        frame_count = self.frame_sampler.get_frame_count(frames)
        if frame_count <= 0:
            raise ValueError("Chunk contains no frames to trim.")

        chunk_duration = max(chunk_end - chunk_start, 0.0)
        if chunk_duration <= 0:
            seconds_per_frame = 1.0 / max(float(self.fps), 1e-6)
            chunk_duration = seconds_per_frame * frame_count
            chunk_end = chunk_start + chunk_duration
        else:
            seconds_per_frame = chunk_duration / frame_count

        clamped_start = max(chunk_start, min(desired_start, chunk_end - seconds_per_frame))
        clamped_end = max(clamped_start + seconds_per_frame, min(desired_end, chunk_end))

        start_offset = clamped_start - chunk_start
        end_offset = clamped_end - chunk_start
        start_idx = int(max(0, min(frame_count - 1, math.floor(start_offset / seconds_per_frame + 1e-6))))
        end_idx = int(max(start_idx + 1, math.ceil(end_offset / seconds_per_frame - 1e-6)))
        end_idx = min(frame_count, end_idx)

        trimmed_frames = self._slice_frames(frames, start_idx, end_idx, frame_count)
        actual_start = chunk_start + start_idx * seconds_per_frame
        actual_end = chunk_start + end_idx * seconds_per_frame
        actual_end = min(actual_end, chunk_end)

        return trimmed_frames, actual_start, actual_end

    @staticmethod
    def _compute_video_duration(video_path: str) -> Optional[float]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                return None

            metadata_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            native_fps = cap.get(cv2.CAP_PROP_FPS)

            # Some files report more frames than they contain; count the readable ones.
            actual_frames = 0
            while actual_frames < metadata_frames:
                success, _ = cap.read()
                if not success:
                    break
                actual_frames += 1

            cap.release()
            total_frames = actual_frames if actual_frames < metadata_frames else metadata_frames

            if native_fps and native_fps > 0 and total_frames and total_frames > 0:
                return float(total_frames) / float(native_fps)
        except Exception:
            return None
        return None

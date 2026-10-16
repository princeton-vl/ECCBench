"""Shared context handling for models served over an HTTP API.

The adapters send every frame of every video at the resolution the sampler
returns. They never subsample. If a provider rejects a request because of
its size or image count, the adapter raises :class:`RequestTooLargeError`.
"""
from __future__ import annotations

import base64
import copy
import io
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from eccbench.models.base import VideoLanguageModelInterface

_SIZE_REJECTION = re.compile(r"image|too large|too many|payload|request size|exceed", re.IGNORECASE)


class RequestTooLargeError(RuntimeError):
    """The provider rejected a request because of its size or image count."""


def is_size_rejection(status: Optional[int], message: str) -> bool:
    """True for HTTP 413, or HTTP 400 whose message names an image or size limit."""
    return status == 413 or (status == 400 and bool(_SIZE_REJECTION.search(message)))


def size_rejection_error(
    provider: str, num_images: int, num_bytes: int, exc: BaseException
) -> RequestTooLargeError:
    return RequestTooLargeError(
        f"{provider} rejected a request with {num_images} images ({num_bytes / 1e6:.1f} MB "
        f"of image data). The provider's per-request limit is lower than this video needs; "
        f"frames are not subsampled. Provider message: {exc}"
    )


def status_code(exc: BaseException) -> Optional[int]:
    """HTTP status code of an SDK exception, if it has one."""
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _to_hwc_uint8(frames: torch.Tensor) -> np.ndarray:
    if frames.dim() != 4:
        raise ValueError(f"expected a 4D frame tensor, got shape {tuple(frames.shape)}")
    if frames.shape[1] == 3 and frames.shape[-1] != 3:
        frames = frames.permute(0, 2, 3, 1)
    elif frames.shape[-1] != 3:
        raise ValueError(f"expected 3 colour channels, got shape {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        frames = frames.clamp(0, 255).to(torch.uint8)
    return frames.contiguous().cpu().numpy()


def _resize_max_edge(img: Image.Image, max_edge: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    scale = max_edge / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


class APIVideoModel(VideoLanguageModelInterface):
    """Keeps the video and text context and encodes frames for an API request.

    Subclasses implement ``_setup_client`` and ``ask_question``.
    """

    default_image_format = "png"

    def _setup_model(self, max_gpu_mem: Optional[float] = None, **kwargs) -> None:
        del max_gpu_mem
        self.image_format = str(kwargs.get("image_format", self.default_image_format)).lower()
        if self.image_format not in {"png", "jpeg"}:
            raise ValueError("image_format must be 'png' or 'jpeg'")
        self.image_mime = f"image/{self.image_format}"
        self.image_max_edge: Optional[int] = kwargs.get("image_max_edge")
        self.latest_time = 0.0
        self.video_segments: Dict[int, List[Dict[str, Any]]] = {}
        self.text_entries: List[Tuple[str, float]] = []
        self._last_usage: Optional[Dict[str, Any]] = None
        self._last_request_images = 0
        self._last_request_bytes = 0
        self._setup_client(**kwargs)

    def _setup_client(self, **kwargs) -> None:
        raise NotImplementedError

    def add_video(
        self,
        video_frames: Union[np.ndarray, torch.Tensor],
        time_start: float,
        time_end: float,
        video_id: Optional[int] = None,
    ) -> None:
        if time_start >= time_end:
            raise ValueError("time_end must be greater than time_start")
        if time_start < self.latest_time:
            raise ValueError("time_start must be after the last added video segment")
        self.latest_time = time_end
        if isinstance(video_frames, np.ndarray):
            video_frames = torch.from_numpy(video_frames)
        frames = video_frames.detach().cpu()
        self.video_segments.setdefault(video_id or 0, []).append(
            {"frames": frames, "time_start": float(time_start), "time_end": float(time_end)}
        )

    def add_text(self, text: str, current_video_time: float = 0.0) -> None:
        self.text_entries.append((text, self.latest_time))

    def has_context(self) -> bool:
        return bool(self.video_segments or self.text_entries)

    def timeline(self) -> Iterator[Tuple[str, Any]]:
        """Yield ``("frames", tensor)`` and ``("text", str)`` items in time order.

        All segments of one video are joined and emitted at the video's first
        segment, as the local models do.
        """
        events = [
            (segs[0]["time_start"], True, vid) for vid, segs in self.video_segments.items() if segs
        ]
        events += [(t, False, text) for text, t in self.text_entries]
        events.sort(key=lambda event: (event[0], event[1]))
        for _, is_video, item in events:
            if is_video:
                frames = torch.cat([seg["frames"] for seg in self.video_segments[item]], dim=0)
                if frames.shape[0] > 0:
                    yield "frames", frames
            else:
                yield "text", item

    def encode_frames(self, frames: torch.Tensor) -> List[str]:
        """Return each frame as a base64 string in ``image_format``."""
        encoded = []
        for frame in _to_hwc_uint8(frames):
            img = Image.fromarray(frame, mode="RGB")
            if self.image_max_edge is not None:
                img = _resize_max_edge(img, int(self.image_max_edge))
            buf = io.BytesIO()
            if self.image_format == "jpeg":
                img.save(buf, format="JPEG", quality=85, optimize=True)
            else:
                img.save(buf, format="PNG", optimize=True)
            data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
            self._last_request_images += 1
            self._last_request_bytes += len(data)
            encoded.append(data)
        return encoded

    def _start_request(self) -> float:
        self._last_request_images = 0
        self._last_request_bytes = 0
        return time.perf_counter()

    def _record_latency(self, start: float, current_video_time: float) -> None:
        if not (self.enable_metrics and self._metrics):
            return
        question_time = float(current_video_time or 0.0)
        if question_time <= 0:
            question_time = self.latest_time
        self._record_ask_question_metrics(
            latency=time.perf_counter() - start,
            peak_gpu_mem_increase=0.0,
            peak_gpu_mem_absolute=0.0,
            video_time=question_time,
            state_memory_total=0.0,
        )

    def get_last_response_token_stats(self) -> Optional[Dict[str, Any]]:
        return copy.copy(self._last_usage)

    def was_video_truncated(self) -> Optional[bool]:
        return False

    def get_state(self) -> Dict[str, Any]:
        return {
            "latest_time": self.latest_time,
            "text_entries": list(self.text_entries),
            "video_segments": {
                vid: [dict(s) for s in segs] for vid, segs in self.video_segments.items()
            },
        }

    def clear_context(self) -> None:
        self.latest_time = 0.0
        self.video_segments.clear()
        self.text_entries.clear()
        self._last_usage = None

    def save_state(self) -> Any:
        return self.get_state()

    def load_state(self, state: Any) -> None:
        self.latest_time = float(state.get("latest_time", 0.0))
        self.text_entries = list(state.get("text_entries", []))
        self.video_segments = {
            vid: [dict(s) for s in segs] for vid, segs in state.get("video_segments", {}).items()
        }
        self._last_usage = None

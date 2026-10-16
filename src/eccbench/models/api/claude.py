"""Claude models through the Anthropic Messages API.

Needs ``ANTHROPIC_API_KEY``. Every frame is sent as an image block.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from eccbench.models.api.common import (
    APIVideoModel,
    is_size_rejection,
    size_rejection_error,
    status_code,
)

DEFAULT_MODEL_ID = "claude-opus-4-7"

logger = logging.getLogger(__name__)

# Models that accept only adaptive thinking; omitting the field turns thinking off.
_ADAPTIVE_ONLY_MODELS = frozenset({"claude-opus-4-7"})


class ClaudeAPIModel(APIVideoModel):
    """Claude adapter.

    Options: ``thinking_mode`` ("disabled" or "adaptive"), ``image_format``
    ("png" or "jpeg"), ``image_max_edge``, ``request_timeout``, ``max_retries``.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **kwargs) -> None:
        super().__init__(model_id, **kwargs)

    def _setup_client(self, **kwargs) -> None:
        import anthropic

        api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY to use Claude models.")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=kwargs.get("request_timeout", 120.0),
            max_retries=kwargs.get("max_retries", 3),
        )
        self.thinking_mode = str(kwargs.get("thinking_mode", "disabled")).lower()
        if self.thinking_mode not in {"disabled", "adaptive"}:
            raise ValueError(
                f"thinking_mode must be 'disabled' or 'adaptive', got {self.thinking_mode!r}"
            )

    def _content(self, question: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for kind, item in self.timeline():
            if kind == "text":
                content.append({"type": "text", "text": item})
                continue
            for data in self.encode_frames(item):
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": self.image_mime, "data": data},
                })
        # Cache the image and context prefix across questions about the same video.
        content.append({"type": "text", "text": question, "cache_control": {"type": "ephemeral"}})
        return content

    def ask_question(
        self,
        question: str,
        current_video_time: float = 0.0,
        max_tokens: int = 256,
        max_frames_in_video: int = 768,
        sample_method: str = "TIME",
    ) -> str:
        start = self._start_request()
        model_id = self.model_id
        request: Dict[str, Any] = {
            "model": model_id,
            "max_tokens": max(1, int(max_tokens)),
            "messages": [{"role": "user", "content": self._content(question)}],
        }
        if self.thinking_mode == "adaptive":
            request["thinking"] = {"type": "adaptive"}
        elif model_id not in _ADAPTIVE_ONLY_MODELS:
            request["thinking"] = {"type": "disabled"}

        try:
            response = self._client.messages.create(**request)
        except self._anthropic.APIError as exc:
            if is_size_rejection(status_code(exc), str(exc)):
                raise size_rejection_error(
                    "Anthropic", self._last_request_images, self._last_request_bytes, exc
                ) from exc
            logger.error("Claude request failed: %s", exc)
            return ""

        text = next(
            (block.text or "" for block in response.content or [] if block.type == "text"), ""
        )
        usage = response.usage
        self._last_usage = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            "stop_reason": response.stop_reason,
            "model": response.model,
        }
        self._record_latency(start, current_video_time)
        return text

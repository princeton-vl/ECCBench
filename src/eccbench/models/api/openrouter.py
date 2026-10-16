"""Any model served by OpenRouter, through its OpenAI-compatible chat API.

Use ``--model openrouter --openrouter-model <slug>``, for example
``google/gemini-3.1-pro-preview``. Needs ``OPENROUTER_API_KEY``. Every frame
is sent as an ``image_url`` block. The model list is at
https://openrouter.ai/models.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Union

from eccbench.models.api.common import (
    APIVideoModel,
    is_size_rejection,
    size_rejection_error,
    status_code,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_RETRY_STATUS = (408, 429, 500, 502, 503, 504)

logger = logging.getLogger(__name__)


class OpenRouterAPIModel(APIVideoModel):
    """OpenRouter adapter.

    Options: ``reasoning_effort`` (low, medium or high; default from
    ``OPENROUTER_REASONING_EFFORT``), ``image_format`` ("jpeg" or "png"),
    ``image_max_edge``, ``request_timeout``, ``max_retries``.
    """

    default_image_format = "jpeg"

    def __init__(self, model_id: str, **kwargs) -> None:
        if not model_id:
            raise ValueError("OpenRouterAPIModel needs an OpenRouter model slug as model_id.")
        super().__init__(model_id, **kwargs)

    def _setup_client(self, **kwargs) -> None:
        import openai

        api_key = kwargs.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY to use OpenRouter models.")
        # Retries are done here, so that only transient errors are retried.
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=float(kwargs.get("request_timeout", 120.0)),
            max_retries=0,
        )
        self.max_retries = int(kwargs.get("max_retries", 3))
        self._retry_base_delay = float(kwargs.get("retry_base_delay", 2.0))
        self._extra_headers = {
            "HTTP-Referer": str(kwargs.get(
                "http_referer", "https://huggingface.co/datasets/anonstreammem/substream-recollection"
            )),
            "X-Title": str(kwargs.get("app_title", "eccbench eval")),
        }
        effort = kwargs.get("reasoning_effort") or os.environ.get("OPENROUTER_REASONING_EFFORT")
        self.reasoning_effort = str(effort).lower().strip() if effort else None
        if self.reasoning_effort not in (None, "low", "medium", "high"):
            raise ValueError(f"reasoning_effort must be low, medium or high, got {effort!r}")

    def _content(self, question: str) -> Union[str, List[Dict[str, Any]]]:
        if not self.has_context():
            return question
        content: List[Dict[str, Any]] = []
        for kind, item in self.timeline():
            if kind == "text":
                content.append({"type": "text", "text": item})
                continue
            for data in self.encode_frames(item):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{self.image_mime};base64,{data}"},
                })
        content.append({"type": "text", "text": question})
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
        request: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": self._content(question)}],
            "max_tokens": max(1, int(max_tokens)),
            "extra_headers": self._extra_headers,
        }
        if self.reasoning_effort:
            request["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}

        response = self._create(request)
        if response is None:
            return ""

        text = ""
        if response.choices:
            content = response.choices[0].message.content
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text") if isinstance(part, dict) else getattr(part, "text", ""))
                    for part in content
                )
            text = content or ""
        usage = response.usage
        if usage is not None:
            self._last_usage = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "stop_reason": response.choices[0].finish_reason if response.choices else None,
                "model": response.model,
            }
        self._record_latency(start, current_video_time)
        return text

    def _create(self, request: Dict[str, Any]):
        """Send the request, retrying transient errors with exponential backoff.

        Returns ``None`` after a non-transient error or when retries run out.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.chat.completions.create(**request)
            except Exception as exc:  # the SDK wraps many httpx errors
                status = status_code(exc)
                if is_size_rejection(status, str(exc)):
                    raise size_rejection_error(
                        "OpenRouter", self._last_request_images, self._last_request_bytes, exc
                    ) from exc
                if status not in _RETRY_STATUS or attempt == self.max_retries:
                    logger.error(
                        "OpenRouter request failed (status=%s, attempt %d/%d): %s",
                        status, attempt + 1, self.max_retries + 1, exc,
                    )
                    return None
                delay = self._retry_base_delay * 2**attempt
                logger.warning("OpenRouter status %s; retrying in %.0f s: %s", status, delay, exc)
                time.sleep(delay)
        return None

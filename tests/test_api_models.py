"""API adapters send every frame and fail loudly when a provider rejects the size."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from eccbench.models.api.common import RequestTooLargeError  # noqa: E402

NUM_FRAMES = 300


class _Rejected(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def _frames(n=NUM_FRAMES):
    return torch.zeros((n, 3, 8, 8), dtype=torch.uint8)


def _openrouter(create):
    pytest.importorskip("openai")
    from eccbench.models.api.openrouter import OpenRouterAPIModel

    model = OpenRouterAPIModel("test/model", api_key="test", max_retries=0)
    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return model


def test_openrouter_sends_every_frame():
    sent = {}

    def create(**request):
        sent.update(request)
        message = SimpleNamespace(content="{0}")
        choice = SimpleNamespace(message=message, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return SimpleNamespace(choices=[choice], usage=usage, model="test/model")

    model = _openrouter(create)
    model.add_video(_frames(), 0.0, float(NUM_FRAMES))
    assert model.ask_question("q?") == "{0}"
    content = sent["messages"][0]["content"]
    assert sum(block["type"] == "image_url" for block in content) == NUM_FRAMES
    assert content[-1] == {"type": "text", "text": "q?"}


def test_openrouter_size_rejection_raises():
    def create(**request):
        raise _Rejected(413, "payload too large")

    model = _openrouter(create)
    model.add_video(_frames(), 0.0, float(NUM_FRAMES))
    with pytest.raises(RequestTooLargeError, match=f"{NUM_FRAMES} images"):
        model.ask_question("q?")


def test_openrouter_other_errors_return_empty_answer():
    def create(**request):
        raise _Rejected(400, "invalid model slug")

    model = _openrouter(create)
    assert model.ask_question("q?") == ""


def test_claude_size_rejection_raises():
    pytest.importorskip("anthropic")
    from eccbench.models.api.claude import ClaudeAPIModel

    def create(**kwargs):
        assert sum(b["type"] == "image" for b in kwargs["messages"][0]["content"]) == NUM_FRAMES
        raise _Rejected(400, "too many images in request")

    model = ClaudeAPIModel(api_key="test")
    model._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    model._anthropic = SimpleNamespace(APIError=_Rejected)
    model.add_video(_frames(), 0.0, float(NUM_FRAMES))
    with pytest.raises(RequestTooLargeError, match="Anthropic"):
        model.ask_question("q?")

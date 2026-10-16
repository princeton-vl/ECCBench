"""GPU forward-pass sweep over every registered backend.

Runs only under ``pytest -m gpu`` inside the ``backbones`` environment; the
default test run excludes it. Each backend is constructed from its registry
defaults and must produce a finite output for one small batch.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")

pytestmark = pytest.mark.gpu

# Backends with extra input needs. log_linear_mamba's kernel needs a
# state_size divisible by 64, so its default of 32 is overridden.
SPECIAL = {
    "deltaformer": {"dtype": "bf16"},
    "dnc": {"stream_len": 8},
    "stm": {"stream_len": 8},
    "mom": {"seq_len": 64},
    "log_linear_mamba": {"seq_len": 64, "overrides": {"state_size": 64}},
}

# Requires the ThunderKittens compiled extension; skipped when absent.
OPTIONAL_EXTENSION = {"ttt_fast"}


def _backend_names():
    from eccbench.backbones.training.registries import BACKEND_SPECS

    return sorted(BACKEND_SPECS)


@pytest.mark.parametrize("name", _backend_names())
def test_backend_forward(name):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from eccbench.backbones.backend.models.backend_base import SequenceInputs
    from eccbench.backbones.training.registries import build_backend

    special = SPECIAL.get(name, {})
    try:
        backend, cfg = build_backend(name, special.get("overrides", {}))
    except RuntimeError as exc:
        if name in OPTIONAL_EXTENSION:
            pytest.skip(str(exc))
        raise
    device = "cuda"
    backend = backend.to(device).eval()
    dtype = torch.bfloat16 if special.get("dtype") == "bf16" else torch.float32
    if dtype is torch.bfloat16:
        backend = backend.to(dtype)
    batch_size, seq_len = 2, special.get("seq_len", 16)
    dim = int(cfg.get("embed_dim", 64))
    stream_len = special.get("stream_len")
    batch = SequenceInputs(
        input_ids=torch.randint(0, 16, (batch_size, seq_len), device=device),
        token_embeddings=torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype),
        lengths=torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
        padding_mask=torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        stream_lengths=(
            torch.full((batch_size,), stream_len, dtype=torch.long, device=device)
            if stream_len
            else None
        ),
    )
    with torch.no_grad():
        try:
            out = backend.encode_sequence(batch)
        except (TypeError, ValueError) as exc:
            if "label_mask" not in str(exc):
                if name in OPTIONAL_EXTENSION:
                    pytest.skip(str(exc))
                raise
            label_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            out = backend.encode_sequence(batch, label_mask=label_mask)
    tensor = out[0] if isinstance(out, (tuple, list)) else out
    assert tensor.dim() == 3
    assert torch.isfinite(tensor.float()).all()

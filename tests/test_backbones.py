"""CPU tests for the ``substream.backbones`` package.

Designed to stay green in the ``full-stack`` env, which has real torch but
none of the heavier backbone deps (lightning, fla, mamba-ssm, dnc,
titans-pytorch). Anything beyond plain torch is guarded with
``pytest.importorskip`` / graceful skips so ``pytest tests/`` never breaks:

* registry sanity tests skip when the registry module cannot import (it pulls
  in fla/dnc/titans at module level);
* pure-torch backends (identity, simple_rnn, transformer_pp,
  compressive_transformer, memory_mosaic) are constructed directly from their
  model modules and run a tiny CPU forward pass;
* kernel/dependency-bound backends (dnc, stm via the ``dnc`` package) skip
  cleanly when their import fails;
* a boundary test asserts backbones only imports substream.backbones and
  substream.config from the substream namespace.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "randn"):  # conftest stub on torch-less envs
    pytest.skip("real torch is required for backbone tests", allow_module_level=True)

BACKBONES_ROOT = Path(__file__).resolve().parents[1] / "src" / "substream" / "backbones"


def _import_or_skip(module_name: str):
    """Import a backbones module, skipping when optional deps are absent."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"{module_name} unavailable in this env: {exc}")


# ---------------------------------------------------------------------------
# (a) BACKEND_SPECS / DATASET_SPECS registry sanity
# ---------------------------------------------------------------------------


class TestRegistries:
    """Skips wholesale in envs lacking fla/dnc/titans (module-level imports)."""

    @pytest.fixture(scope="class")
    def registries(self):
        return _import_or_skip("substream.backbones.training.registries")

    def test_backend_specs_well_formed(self, registries):
        assert registries.BACKEND_SPECS, "BACKEND_SPECS must not be empty"
        for key, spec in registries.BACKEND_SPECS.items():
            assert spec.name == key, f"spec name {spec.name!r} != registry key {key!r}"
            assert isinstance(spec.description, str) and spec.description.strip()
            assert callable(spec.builder), f"builder for {key} is not callable"
            assert isinstance(spec.defaults, dict)
            assert isinstance(spec.required_keys, tuple)
            assert all(isinstance(k, str) for k in spec.required_keys)

    def test_dataset_specs_well_formed(self, registries):
        assert set(registries.DATASET_SPECS) >= {"synthetic", "file", "video_features"}
        for key, spec in registries.DATASET_SPECS.items():
            assert spec.name == key
            assert callable(spec.builder)
            assert isinstance(spec.defaults, dict)

    def test_choices_match_specs(self, registries):
        assert set(registries.backend_choices()) == set(registries.BACKEND_SPECS)
        assert set(registries.dataset_choices()) == set(registries.DATASET_SPECS)

    def test_unknown_backend_override_rejected(self, registries):
        spec = registries.BACKEND_SPECS["identity"]
        with pytest.raises(KeyError):
            spec.materialize({"definitely_not_a_real_option": 1})

    def test_unknown_backend_name_rejected(self, registries):
        with pytest.raises(KeyError):
            registries.build_backend("no_such_backend", {})


# ---------------------------------------------------------------------------
# (b) pure-torch backends: tiny CPU construction + forward pass
# ---------------------------------------------------------------------------

# name -> (module, class, tiny constructor kwargs)
_CPU_BACKENDS = {
    "identity": (
        "substream.backbones.backend.models.identity",
        "IdentityBackend",
        {"embed_dim": 16},
    ),
    "simple_rnn": (
        "substream.backbones.backend.models.simple_rnn",
        "SimpleRNNEncoder",
        {"embed_dim": 16, "hidden_dim": 32, "num_layers": 2, "dropout": 0.0},
    ),
    "transformer_pp": (
        "substream.backbones.backend.models.transformer_pp_backend",
        "TransformerPPBackend",
        {"embed_dim": 16, "num_layers": 1, "num_heads": 2, "mlp_ratio": 2, "dropout": 0.0},
    ),
    "compressive_transformer": (
        "substream.backbones.backend.models.lt_ct",
        "CompressiveTransformerBackend",
        {
            "embed_dim": 16,
            "hidden_dim": 16,
            "num_layers": 1,
            "heads": 2,
            "block_length": 4,
            "mem_length": 8,
            "compression_factors": 2,
            "dropout": 0.0,
        },
    ),
    "memory_mosaic": (
        "substream.backbones.backend.models.memory_mosaic_backend",
        "MemoryMosaicBackend",
        {
            "embed_dim": 16,
            "n_layer": 1,
            "n_head": 2,
            "pmem_size": 16,
            "pmem_count": 1,
            "dropout": 0.0,
            "block_size": 32,
        },
    ),
    # mann.py imports the optional `dnc` package at module level, so both of
    # these skip in envs without it (e.g. full-stack).
    "stm": (
        "substream.backbones.backend.models.mann",
        "STMBackend",
        {"embed_dim": 16, "segment_length": 4, "stm_input_dim": 8, "segmentation_method": "avg"},
    ),
    "dnc": (
        "substream.backbones.backend.models.mann",
        "DNCBackend",
        {
            "embed_dim": 16,
            "segment_length": 4,
            "mem_input_dim": 8,
            "segmentation_method": "flat",
            "hidden_size": 16,
            "num_layers": 1,
            "nr_cells": 4,
            "read_heads": 2,
            "cell_size": 8,
        },
    ),
}


def _tiny_sequence_inputs(embed_dim: int):
    base_mod = _import_or_skip("substream.backbones.backend.models.backend_base")
    batch, length = 2, 12
    torch.manual_seed(0)
    return base_mod.SequenceInputs(
        input_ids=torch.randint(1, 8, (batch, length)),
        token_embeddings=torch.randn(batch, length, embed_dim),
        lengths=torch.tensor([length, length - 3]),
        padding_mask=None,
        stream_lengths=torch.tensor([length - 4, length - 5]),
    )


@pytest.mark.parametrize("backend_name", sorted(_CPU_BACKENDS))
def test_backend_cpu_forward(backend_name):
    module_name, class_name, kwargs = _CPU_BACKENDS[backend_name]
    module = _import_or_skip(module_name)
    backend_cls = getattr(module, class_name)
    backend = backend_cls(**kwargs)
    backend.eval()

    inputs = _tiny_sequence_inputs(embed_dim=kwargs["embed_dim"])
    with torch.no_grad():
        outputs = backend.encode_sequence(inputs)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.dim() == 3, f"{backend_name} should emit (batch, slots, dim)"
    assert outputs.size(0) == inputs.input_ids.size(0)
    assert torch.isfinite(outputs).all(), f"{backend_name} produced non-finite outputs"


def test_backend_base_padding_mask_from_lengths():
    base_mod = _import_or_skip("substream.backbones.backend.models.backend_base")
    identity_mod = _import_or_skip("substream.backbones.backend.models.identity")
    backend = identity_mod.IdentityBackend(embed_dim=4)
    inputs = base_mod.SequenceInputs(
        input_ids=torch.ones(2, 5, dtype=torch.long),
        token_embeddings=torch.ones(2, 5, 4),
        lengths=torch.tensor([5, 3]),
        padding_mask=None,
    )
    outputs = backend.encode_sequence(inputs)
    # positions beyond each length must be zeroed out
    assert outputs[1, 3:].abs().sum().item() == 0.0
    assert outputs[0].abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# synthetic data path used by the train smoke (pure torch)
# ---------------------------------------------------------------------------


def test_synthetic_dataset_and_collate():
    seq_gen = _import_or_skip("substream.backbones.data.sequences.sequence_generator")
    collator = _import_or_skip("substream.backbones.data.sequences.collator")

    dataset = seq_gen.SyntheticSequenceDataset(
        num_sequences=4,
        seq_len=16,
        unique_sequences=2,
        vocab_size=8,
        seed=0,
        task="membership",
        cont_len=3,
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert len(sample) >= 3
    seq, labels, stream_len = sample[:3]
    assert seq.dim() == 1 and labels.shape == seq.shape
    assert 0 < int(stream_len.item()) <= seq.numel()

    collate = collator.build_collate(dataset.pad_id)
    batch = collate([dataset[i] for i in range(4)])
    sequence = batch["sequence"]
    assert sequence["input_ids"].size(0) == 4
    assert sequence["padding_mask"].shape == sequence["input_ids"].shape
    assert batch["labels"].shape == sequence["input_ids"].shape
    assert len(batch["metadata"]) == 4
    assert all("stream_length" in entry for entry in batch["metadata"])


# ---------------------------------------------------------------------------
# boundary: backbones must not depend on the rest of substream
# ---------------------------------------------------------------------------

_ALLOWED_SUBSTREAM_IMPORTS = {"backbones", "config"}
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+substream\.(\w+)", re.MULTILINE)


def test_backbones_import_boundary():
    assert BACKBONES_ROOT.is_dir()
    violations: list[str] = []
    for path in BACKBONES_ROOT.rglob("*.py"):
        for match in _IMPORT_RE.finditer(path.read_text(encoding="utf-8")):
            subpackage = match.group(1)
            if subpackage not in _ALLOWED_SUBSTREAM_IMPORTS:
                violations.append(f"{path}: substream.{subpackage}")
    assert not violations, "backbones may only import substream.backbones/config:\n" + "\n".join(violations)

# Backbone Training

This guide shows you how to train and evaluate the alternative-architecture
memory backbones on the benchmark's token sequences.

`eccbench.backbones` trains small memory architectures at matched parameter
budgets. Each backbone turns a token stream into a memory representation; a
shared transformer decoder (`MemoryBankDecoder`) answers membership or
continuation queries against it.

Select a backend (`eccbench.backbones.training.registries`) with
`--backend`:

- Recurrent / state-space: `simple_rnn`, `mamba` (Mamba-2), `log_linear_mamba`
- Linear-attention (flash-linear-attention): `deltanet`, `gated_deltanet`, `deltaformer`, `retnet`, `gla`, `mom`, `rwkv`
- Test-time training: `ttt`, `ttt_fast`
- Memory-augmented: `dnc`, `stm`, `titans`, `titans_external`, `memory_mosaic`, `compressive_transformer`
- Baselines: `identity`, `transformer_pp`

`--dataset` selects `synthetic` (generated streams) or `file`
(manifest-backed streams).

## Training

Start with a minimal synthetic membership run:

```bash
python -m eccbench.backbones.train \
  --backend mamba \
  --dataset synthetic \
  --dataset-option num_sequences=64 \
  --dataset-option seq_len_min=16 \
  --dataset-option seq_len_max=512 \
  --dataset-option vocab_size=16 \
  --dataset-option task=membership \
  --batch-size 8 --max-epochs 200 --val-fraction 0.2 \
  --enable-checkpoints --disable-wandb
```

Override backend hyperparameters with repeated `--backend-option key=value`;
dataset knobs with `--dataset-option key=value`; `--help` lists everything.

## Evaluation

Evaluate a trained checkpoint against a question manifest:

```bash
python -m eccbench.backbones.train --mode eval \
  --eval-run-dir artifacts/checkpoints/<run-name> \
  --eval-manifest path/to/questions.json \
  --eval-write-csv
```

`--eval-random-synthetic N` evaluates on N fresh synthetic questions instead
of a manifest. `--mode benchmark` measures latency without training
(see `--benchmark-seq-lens`). `scripts/generate_val_manifest.py` builds a
balanced validation manifest and `scripts/run_validation.py` evaluates a
checkpoint against it.

Outputs: checkpoints and `config.json` under `--checkpoint-dir` (default
`artifacts/checkpoints/<run-name>/`); CSV/W&B logs under `--log-dir` (default
`artifacts/logs/runs/`); evaluation summaries under `--eval-output-dir`
(default `artifacts/logs/eval/`). The Hugging Face cache follows
`ECCBENCH_MODEL_CACHE` (docs/ENVIRONMENTS.md); per-run temp directories via
`--temp-root`.

## Environment

The `backbones` extra is mutually exclusive with the model-evaluation extras.
The Mamba CUDA kernels must compile against the torch already in the
environment, so install in two phases (a one-step install fails: the kernels'
isolated build resolves its own torch):

```bash
uv venv -p 3.11 "$ECCBENCH_ENVS_ROOT/eccbench-backbones"
export VIRTUAL_ENV="$ECCBENCH_ENVS_ROOT/eccbench-backbones"
uv pip install "torch>=2.9,<2.10" setuptools wheel ninja packaging einops
uv pip install --no-build-isolation causal-conv1d==1.6.1 mamba-ssm==2.3.1
uv pip install -e ".[backbones]"

bin/envctl run backbones -- python -m eccbench.backbones.train --help
```

The flash-linear backends (`deltaformer`, `deltanet`, `gated_deltanet`,
`log_linear_mamba`, `mom`, `retnet`, `rwkv`) are GPU-only (CUDA +
Triton/FlashAttention kernels). `titans_external` uses the `titans-pytorch`
package, which the extra installs.

## Testing

`tests/test_backbones.py` runs anywhere (kernel-needing backends skip). The
GPU sweep constructs every backend and runs one forward pass:

```bash
bin/envctl run backbones -- python -m pytest tests/test_backbones_gpu.py -m gpu -q
```

Caveat: `ttt_fast` needs the ThunderKittens `tk_ttt_linear_prefill`
extension compiled, and skips when absent.

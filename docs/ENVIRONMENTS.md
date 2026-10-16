# Environments and configuration

This guide shows you how to create the virtual environment that each group
of models needs. It also lists every environment variable that the package
reads.

The evaluated models need conflicting `transformers` versions. Each stack is
therefore an optional extra in `pyproject.toml` with its own virtual
environment. The extras are mutually exclusive under `[tool.uv]`. The pins
reproduce the published results, so do not relax them.

| extra | stack, in one line |
|---|---|
| `full-stack` | transformers pinned to a git commit (recent enough for Qwen3-VL), vLLM >= 0.10.2, flash-attention 2.8.3; used by most models |
| `hf450` | transformers 4.50.0 |
| `hf448` | transformers 4.48.2, torch 2.6.0 |
| `api` | `anthropic` and `openai` clients, plus torch, torchvision, pillow, opencv, qwen-vl-utils and decord for frame sampling; no GPU |

The base install (`pip install -e .`) loads the dataset and runs most CPU
tests. The API models and the dummy model need the `api` extra. The pinned
flash-attention wheels are built for CPython 3.11, so use Python 3.11 for
the GPU environments.

## Which model needs which environment

`env` comes from each model's registry entry (`src/eccbench/registry.py`);
the GPU counts are what the published runs used.

| alias | display name | env extra | GPUs | notes |
|---|---|---|---|---|
| `qwen_full` | Qwen2.5-VL (7B) | `full-stack` | 1 | |
| `qwen3_full` / `qwen3_full_thinking` | Qwen3-VL (8B) / -Thinking | `full-stack` | 1 | video batching disabled |
| `qwen3_omni` | Qwen3-Omni (30B, A3B) | `full-stack` | 2 | |
| `mimo-vl` | MiMo-VL (7B) | `full-stack` | 1 | video batch size capped at 4 |
| `glm45v` | GLM-4.5V (104B, A12B) | `full-stack` | 4 | video batch size 1 |
| `internvl-3-5` / `-thinking` | InternVL3.5 (8B) | `full-stack` | 1 | max video bucket L = 512 |
| `internvl-3-5-30b-a3b` / `-thinking` | InternVL3.5 (30B, A3B) | `full-stack` | 2 | |
| `internvl-3-5-38b` / `-thinking` | InternVL3.5 (38B) | `full-stack` | 2 | |
| `minicpm` | MiniCPM-V 2.6 (8B) | `hf450` | 1 | video batching disabled |
| `minicpm-4-5` | MiniCPM-V 4.5 (9B) | `full-stack` | 1 | |
| `longvila` | LongVILA-R1 (7B) | `full-stack` | 1 | all batching disabled; max video bucket L = 256 |
| `phi_multimodal` | Phi-4-MM (6B) | `hf448` | 1 | max video bucket L = 512 |
| `claude-opus-4-7`, `claude-opus-4-6` | Claude Opus | `api` | 0 | needs `ANTHROPIC_API_KEY` |
| `openrouter` | any OpenRouter-served model | `api` | 0 | needs `OPENROUTER_API_KEY` and `--openrouter-model` |
| `dummy_eval` | integration-test stub | `api` | 0 | CPU-only plumbing tests |

"Batching disabled" rows ignore the batch-size variables below. The retrieval
embedder (`docs/ARCHITECTURES.md`) uses the `full-stack` extra.

## Creating an environment

Environments are directories named `<prefix><env_name>` under
`$ECCBENCH_ENVS_ROOT` (default: `envs/` in the repository). The prefix is
`$ECCBENCH_ENVS_PREFIX` (default: `eccbench-`). Create one with
[uv](https://github.com/astral-sh/uv):

```bash
export ECCBENCH_ENVS_ROOT=$PWD/envs
uv venv -p 3.11 "$ECCBENCH_ENVS_ROOT/eccbench-full-stack"
VIRTUAL_ENV="$ECCBENCH_ENVS_ROOT/eccbench-full-stack" uv pip install -e ".[full-stack]"
```

Repeat for `hf450` and `hf448`. Run commands inside an environment with
`bin/envctl`:

```bash
bin/envctl run full-stack -- eccbench-eval --model qwen_full --data text --mode text
bin/envctl run hf450      -- eccbench-eval --model minicpm  --data synthetic_video
```

`envctl run <env_name> -- <command>` exports `VIRTUAL_ENV`, prepends the
environment's `bin/` to `PATH`, and execs the command.

The alternative-architecture backbones ([BACKBONES.md](BACKBONES.md)) train
in their own environment from the `backbones` extra, created the same way.

## Environment variables

Everything the package reads from the process environment; paths in "where
read" are under `src/eccbench/`. Prefer the `eccbench-eval` flags
`--arm A|B` (sets `CALIB_ARM`) and `--rag` (sets `RAG_PREAMBLE=1`); they
export the variable before any prompt is built.

| variable | default | effect | where read |
|---|---|---|---|
| `ECCBENCH_SCRATCH_ROOT` | `~/.cache/eccbench` | root for all scratch space | `config.py` |
| `ECCBENCH_MODEL_CACHE` | `$ECCBENCH_SCRATCH_ROOT/model-cache` | model-weights cache; also the `HF_HOME` default (below) | `config.py` |
| `ECCBENCH_DATA_ROOT` | `$ECCBENCH_SCRATCH_ROOT/data` | root for local dataset copies (RAG pipeline, compose scripts) and for filtered manifests | `config.py` |
| `ECCBENCH_RUN_ROOT` | `runs` | where state files and per-question CSVs go; the `eccbench-eval --out` default | `config.py` |
| `ECCBENCH_ENVS_ROOT` | `<repo>/envs` | directory containing the virtual environments | `bin/envctl` |
| `ECCBENCH_ENVS_PREFIX` | `eccbench-` | name prefix of those directories | `bin/envctl` |
| `ECCBENCH_TEMP_ROOT` | system temp dir | base directory for the per-run temp folders created by backbone training (`--temp-root` default is `$ECCBENCH_TEMP_ROOT/eccbench-temp`) | `backbones/train.py` |
| `ECCBENCH_ALLOW_REMOTE_TEMP` | unset | `1` allows backbone training to place its temp folder on a network filesystem instead of falling back to local temp | `backbones/utils/paths.py` |
| `CALIB_ARM` | unset | `A` or `B` appends the corresponding scoring sentence to every prompt (`docs/PROTOCOL.md`); unset gives the byte-identical control prompt; other values raise | `eval/processors/question_processor.py` |
| `RAG_PREAMBLE` | unset | `1` prepends the retrieval-disclosure preamble to every prompt | `eval/processors/question_processor.py` |
| `USE_SHORT_PROMPT_FOR_SYNTH` | unset | `1` routes synthetic video questions to the short direct prompt instead of the conveyor-belt template (per-model override) | `eval/processors/question_processor.py` |
| `ENABLE_TEXT_BATCHING` | `1` | global gate on batched text processing; a registry entry can still disable batching per model | `eval/runner.py` |
| `ENABLE_VIDEO_BATCHING` | `1` | same, for video | `eval/runner.py` |
| `TEXT_BATCH_SIZE` | `16` | initial text batch size (shrinks automatically on out-of-memory) | `eval/runner.py` |
| `VIDEO_BATCH_SIZE` | `16` | video batch size when the registry sets no per-model cap | `eval/runner.py` |
| `VIDEO_BATCH_SIZE_FORCE` | unset | `1` makes `VIDEO_BATCH_SIZE` override the registry's per-model cap | `eval/runner.py` |
| `FORCE_QWENVL_VIDEO_READER` | `torchvision` (set if unset) | selects the `qwen-vl-utils` video decoding backend; the frame samplers force `torchvision` | `samplers/` |
| `GLM_OFFLOAD_DIR` | `$ECCBENCH_SCRATCH_ROOT/glm_offload` | disk-offload directory for GLM-4.5V weights | `models/glm45v.py` |
| `ANTHROPIC_API_KEY` | unset | credential for the Claude models (`--model claude-opus-4-7` etc.) | `models/api/claude.py` |
| `OPENROUTER_API_KEY` | unset | credential for `--model openrouter` | `models/api/openrouter.py` |
| `OPENROUTER_REASONING_EFFORT` | unset | forwarded as the reasoning-effort parameter on OpenRouter requests | `models/api/openrouter.py` |
| `HF_HOME` | set to `ECCBENCH_MODEL_CACHE` if unset | Hugging Face cache; an existing value wins | `config.py` |
| `TMPDIR`, `TEMP`, `TMP` | set to `$ECCBENCH_SCRATCH_ROOT/tmp` if unset | temporary files of decoded frames and model loading; an existing value wins | `config.py` |
| `SLURM_JOB_ID`, `SLURM_NODELIST`, `SLURMD_NODENAME` | — | read only when present, to label GPU-monitor output; never written | `utils/gpu_monitor.py` |

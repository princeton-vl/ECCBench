# Environments and configuration

This guide shows you how to create the virtual environment each group of
models needs, which model needs which environment, and every environment
variable the package reads.

The evaluated models need conflicting `transformers` versions, so each stack
is an optional extra in `pyproject.toml` with its own virtual environment,
declared mutually exclusive under `[tool.uv]`. The pins reproduce the
published results — do not relax them.

| extra | stack, in one line |
|---|---|
| `full-stack` | transformers pinned to a git commit (recent enough for Qwen3-VL, >= 4.57 behaviour), vLLM >= 0.10.2, flash-attention 2.8.3 — the default, used by most models |
| `hf450` | transformers 4.50.0 |
| `hf448` | transformers 4.48.2, torch 2.6.0 |
| `hf451` | transformers 4.52, vLLM 0.8.4, torch 2.6.0 (the M3-Agent stack) |
| `api` | `anthropic` + `openai` clients, plus torch, pillow, and opencv-python-headless (the runner's shared video plumbing); no GPU required |

The base install (`pip install -e .`) loads the dataset and runs the CPU
tests; running API models needs the `api` extra. The pinned flash-attention
wheels are built for CPython 3.11 — use it for the GPU environments.

## Which model needs which environment

`env` comes from each model's registry entry (`src/substream/registry.py`);
the GPU counts are what the published runs used.

| alias | display name | env extra | GPUs | notes |
|---|---|---|---|---|
| `qwen_full` | Qwen2.5-VL (7B) | `full-stack` | 1 | |
| `qwen3_full` / `qwen3_full_thinking` | Qwen3-VL (8B) / -Thinking | `full-stack` | 1 | video batching disabled |
| `qwen3_omni` | Qwen3-Omni (30B, A3B) | `full-stack` | 2 | |
| `mimo-vl` | MiMo-VL (7B) | `full-stack` | 1 | video batch size capped at 4 |
| `glm45v` / `glm45v_fp8` | GLM-4.5V (104B, A12B) / FP8 | `full-stack` | 4 | video batch size 1 |
| `internvl-3-5` / `-thinking` | InternVL3.5 (8B) | `full-stack` | 1 | max video bucket L = 512 |
| `internvl-3-5-30b-a3b` / `-thinking` | InternVL3.5 (30B, A3B) | `full-stack` | 2 | |
| `internvl-3-5-38b` / `-thinking` | InternVL3.5 (38B) | `full-stack` | 2 | |
| `minicpm` | MiniCPM-V 2.6 (8B) | `hf450` | 1 | video batching disabled |
| `minicpm-4-5` | MiniCPM-V 4.5 (9B) | `full-stack` | 1 | |
| `longvila` | LongVILA-R1 (7B) | `full-stack` | 1 | all batching disabled; max video bucket L = 256 |
| `phi_multimodal` | Phi-4-MM (6B) | `hf448` | 1 | max video bucket L = 512 |
| `timechat` | TimeChat-Online (7B) | `hf450` | 1 | all batching disabled |
| `m3_agent` | M3-Agent (7B) | `hf451` | 3 | memory-graph architecture; all batching disabled |
| `claude-opus-4-7`, `claude-opus-4-6` | Claude Opus | `api` | 0 | needs `ANTHROPIC_API_KEY` |
| `openrouter` | any OpenRouter-served model | `api` | 0 | needs `OPENROUTER_API_KEY` and `--openrouter-model` |
| `dummy_eval` | integration-test stub | `api` | 0 | CPU-only plumbing tests |

"Batching disabled" rows ignore the batch-size variables below. The retrieval
embedders (`docs/ARCHITECTURES.md`) use the same extras: Qwen3-VL embedder →
`full-stack`, GVE-7B → `hf450`.

## Creating an environment

Environments are directories named `<prefix><env_name>` under
`$SUBSTREAM_ENVS_ROOT` (default: `./envs` next to the repository), with prefix
`$SUBSTREAM_ENVS_PREFIX` (default: `substream-`). Create one with
[uv](https://github.com/astral-sh/uv):

```bash
uv venv "$SUBSTREAM_ENVS_ROOT/substream-full-stack" && \
VIRTUAL_ENV="$SUBSTREAM_ENVS_ROOT/substream-full-stack" \
  uv pip install -e ".[full-stack]"
```

Repeat for `hf450`, `hf448`, `hf451`. Run commands inside an environment with
`bin/envctl`:

```bash
bin/envctl run full-stack -- substream-eval --model qwen_full --data text --mode text
bin/envctl run hf450      -- substream-eval --model minicpm  --data synthetic_video
```

`envctl run <env_name> -- <command>` exports `VIRTUAL_ENV`, prepends the
environment's `bin/` to `PATH`, and execs the command.

The alternative-architecture backbones ([BACKBONES.md](BACKBONES.md)) train
in their own environment from the `backbones` extra, created the same way.

## Environment variables

Everything the package reads from the process environment; paths in "where
read" are under `src/substream/`. Prefer the `substream-eval` flags
`--arm A|B` (sets `CALIB_ARM`) and `--rag` (sets `RAG_PREAMBLE=1`); they
export the variable before any prompt is built.

| variable | default | effect | where read |
|---|---|---|---|
| `SUBSTREAM_SCRATCH_ROOT` | `~/.cache/substream` | root for all scratch space | `config.py` |
| `SUBSTREAM_MODEL_CACHE` | `$SUBSTREAM_SCRATCH_ROOT/model-cache` | model-weights cache; also the `HF_HOME` default (below) | `config.py` |
| `SUBSTREAM_DATA_ROOT` | `$SUBSTREAM_SCRATCH_ROOT/data` | root for local dataset copies (RAG pipeline, compose scripts) | `config.py` |
| `SUBSTREAM_RUN_ROOT` | `runs` | where state files and per-question CSVs go; the `substream-eval --out` default | `config.py` |
| `SUBSTREAM_ENVS_ROOT` | `<repo>/envs` | directory containing the virtual environments | `bin/envctl` |
| `SUBSTREAM_ENVS_PREFIX` | `substream-` | name prefix of those directories | `bin/envctl` |
| `SUBSTREAM_TEMP_ROOT` | system temp dir | base directory for the per-run temp folders created by backbone training (`--temp-root` default is `$SUBSTREAM_TEMP_ROOT/substream-temp`) | `backbones/train.py` |
| `SUBSTREAM_ALLOW_REMOTE_TEMP` | unset | `1` allows backbone training to place its temp folder on a network filesystem instead of falling back to local temp | `backbones/utils/paths.py` |
| `CALIB_ARM` | unset | `A` or `B` appends the corresponding scoring sentence to every prompt (`docs/PROTOCOL.md`); unset gives the byte-identical control prompt; other values raise | `eval/processors/question_processor.py` |
| `RAG_PREAMBLE` | unset | `1` prepends the retrieval-disclosure preamble to every prompt | `eval/processors/question_processor.py` |
| `USE_SHORT_PROMPT_FOR_SYNTH` | unset | `1` routes synthetic video questions to the short direct prompt instead of the conveyor-belt template (per-model override) | `eval/processors/question_processor.py` |
| `ENABLE_TEXT_BATCHING` | `1` | global gate on batched text processing; a registry entry can still disable batching per model | `eval/runner.py` |
| `ENABLE_VIDEO_BATCHING` | `1` | same, for video | `eval/runner.py` |
| `TEXT_BATCH_SIZE` | `16` | initial text batch size (shrinks automatically on out-of-memory) | `eval/runner.py` |
| `VIDEO_BATCH_SIZE` | `16` | video batch size when the registry sets no per-model cap | `eval/runner.py` |
| `VIDEO_BATCH_SIZE_FORCE` | unset | `1` makes `VIDEO_BATCH_SIZE` override the registry's per-model cap | `eval/runner.py` |
| `FORCE_QWENVL_VIDEO_READER` | `torchvision` (set if unset) | selects the `qwen-vl-utils` video decoding backend; the frame samplers force `torchvision` | `samplers/` |
| `GLM_OFFLOAD_DIR` | `$SUBSTREAM_SCRATCH_ROOT/glm_offload` (`glm_fp8_offload` for FP8) | disk-offload directory for GLM-4.5V weights | `models/glm45v.py`, `models/glm45v_fp8.py` |
| `WANDB_PROJECT` | `substream` | Weights & Biases project for `substream-eval --enable-metrics` | `eval/wandb_log.py` |
| `ANTHROPIC_API_KEY` | unset | credential for the Claude models (`--model claude-opus-4-7` etc.) | `models/api/claude.py` |
| `OPENROUTER_API_KEY` | unset | credential for `--model openrouter` | `models/api/openrouter.py` |
| `OPENROUTER_REASONING_EFFORT` | unset | forwarded as the reasoning-effort parameter on OpenRouter requests | `models/api/openrouter.py` |
| `HF_HOME` | set to `SUBSTREAM_MODEL_CACHE` if unset | Hugging Face cache; `setdefault` at import, an existing value wins | `config.py` |
| `SLURM_GPUS_ON_NODE`, `SLURM_GPUS`, `SLURM_STEP_GPUS`, `CUDA_VISIBLE_DEVICES` | — | M3-Agent only: read (in that order, never written) to count allocated GPUs | `architectures/m3_agent/gpu_config.py` |
| `SLURM_JOB_ID`, `SLURM_NODELIST`, `SLURMD_NODENAME` | — | read only when present, to label GPU-monitor output; never written | `utils/gpu_monitor.py` |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | unset | M3-Agent only: credentials/endpoint for an OpenAI-compatible text-embedding service; environment beats the config file | `architectures/m3_agent/text_embedding.py` |

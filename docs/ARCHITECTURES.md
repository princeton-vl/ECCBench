# Architectures: retrieval (RAG)

This guide shows you how to run the retrieval probe and how to compose and
evaluate retrieval-augmented evaluation sets.

The registry (`src/substream/registry.py`) distinguishes three subject
`kind`s: `vlm` (a local checkpoint, e.g. `qwen3_full`), `api` (a
hosted model, no GPU, e.g. `claude-opus-4-7`), and
`architecture` (models plus retrieval or memory, e.g. `m3_agent`).

All code lives under `src/substream/architectures/`. RAG is not a registry
entry: it rewrites the *dataset* into shorter retrieval-composed videos,
which any registry model then evaluates.

## The retrieval probe

`src/substream/architectures/rag/probe.py` tests whether an off-the-shelf
video embedder can localise the queried event in a long video:

- Each L = 512 source video is cut into **16-second windows at stride 4**
  (125 per video).
- Queries (text for natural video, candidate clips for synthetic) are
  embedded and ranked against windows by cosine similarity.
- A window is correct only if it **fully contains** a gold event.
- Scoring is rank-based (10 points halving per rank down to rank 10), with
  hit@1/3/10 and a shuffled-ranking baseline.

Run the four idempotent stages in order:

```bash
python -m substream.architectures.rag.probe [--out runs/retrieval_probe] build   [--seed 0] [--repo ID] [--revision REV]
python -m substream.architectures.rag.probe [--out runs/retrieval_probe] extract [--jobs 8]
python -m substream.architectures.rag.probe [--out runs/retrieval_probe] embed   --backend qwen [--batch 8]
python -m substream.architectures.rag.probe [--out runs/retrieval_probe] score   --backends qwen gve
```

`build` downloads the `natural_video` and `synthetic_video` configs of the
published dataset (see "Data" below) and picks 32 L = 512 videos per domain,
up to 2 usable queries each: for natural video the clip's `designed`
yes-question, with the gold span from `event_clip_start` / `event_clip_end`;
for synthetic video present candidate clips, with the gold span from the
candidate's `clip_start` / `clip_end`. Spans longer than a window are not
usable. `extract` writes every window as an mp4 via ffmpeg (re-encoded;
keyframe cuts are not guaranteed at 1 fps); `embed` writes a
per-video-resumable `.npz` of window and query embeddings; `score` prints
the table. All state lives under `--out` (default `$SUBSTREAM_RUN_ROOT/retrieval_probe`).

### Retrieval backends

| backend | model | environment | notes |
|---|---|---|---|
| `qwen` | Qwen/Qwen3-VL-Embedding-8B | `full-stack` (needs transformers >= 4.57) | last non-pad-token pooling, right padding, L2-normalised; per-video pixel budget capped to 16 frames |
| `gve` | Alibaba-NLP/GVE-7B | `hf450` (its remote code needs transformers 4.50.x) | last-token pooling, left padding; flash-attention 2 required (the SDPA fallback materialises a whole-batch mask) |
| `dummy` | random vectors | any | plumbing tests |

The real backends need different environments (`docs/ENVIRONMENTS.md`) —
hence the staged `.npz` files.

## Composing a retrieval-augmented evaluation set

Two composers turn retrieval rankings into ordinary evaluation manifests:
per question, take the top-N windows, merge touching or overlapping spans
chronologically, cut each merged segment from the source video at 1 fps, and
join segments with a **2-second bright magenta delimiter frame labeled
"SEGMENT BREAK"**. Output: a standard `questions_dataset.json` (one composed
video per question) plus media, ready for `substream-eval`.

**`compose.py`** uses the probe's manifest and embeddings:

```bash
python -m substream.architectures.rag.compose \
    --backend qwen --topn 2 --domain both --out runs/rag_qwen_top2 \
    --full-questions [--syn-no-per-video 2] [--embed-missing] [--limit N] \
    [--probe-dir runs/retrieval_probe] [--question-set density_topup] [--density-band all]
```

`--full-questions` matters: the probe's queries are yes-only, and an all-yes
evaluation is gamed by answering yes; the flag pulls **all** questions (yes
and no) for the manifest's videos from the published dataset (natural: the
selected `--question-set` at L = 512; synthetic: the probe's yes questions
plus `--syn-no-per-video` absent candidates). The no-questions are where
retrieval hurts: it fetches the nearest-looking windows whether or not the
event happened. `--embed-missing` backfills query embeddings the probe never
computed (needs a GPU and the backend's environment), cached to a side
`.npz`.

**`pipeline.py`** scales the recipe to a whole bucket, with idempotent
stages:

```bash
python -m substream.architectures.rag.pipeline plan    --corpus nat64
python -m substream.architectures.rag.pipeline extract --corpus nat64 [--jobs 8]
python -m substream.architectures.rag.pipeline embed   --corpus nat64 [--batch 4]
python -m substream.architectures.rag.pipeline compose --corpus nat64 [--topn 3]
```

`--corpus` is `nat64`/`nat128`/`nat512` (natural, L = 64/128/512; natural
video has no L = 256 bucket) or `syn64` (synthetic, L = 64, `low` and
`max-entropy` bands). `plan` prints the work volume; `embed` uses the Qwen
backend and seeds window and query embeddings from the probe's `.npz` under
`--probe-dir` (default `$SUBSTREAM_RUN_ROOT/retrieval_probe`) when they
match; `compose` writes
`<out-root>/rag_qwen_top{N}_{corpus}/questions_dataset.json` with per-video
`_rag` provenance (backend, retrieved windows, merged segments, source video
as its Hub path). Windows and embeddings also live under `--out-root`
(default `$SUBSTREAM_RUN_ROOT/rag_corpus`). `--question-set`,
`--density-band`, `--repo` and `--revision` are as for `compose.py`.

### Data

All three scripts read the published dataset through
`substream.data.hf_loader.fetch` (`docs/DATASET.md`), the same download the
evaluation runner uses, so they see the same rows and media. `--repo`
(default `anonstreammem/substream-recollection`) and `--revision` (default:
latest) select the copy. Selection is by dataset field:

- **Natural** (`natural_video` config): `length_L` picks the stream length;
  `--question-set` picks `density_topup` (default; the mined questions
  behind the density analysis), `designed`, or `all`; `--density-band`
  restricts to the `low` or `high` codec-distortion band (default `all`).
  The text query target is the row's `query_class` (from the config's
  `questions.json`). Rows with `media_included: false` (provenance-only
  clips) are skipped, as in the runner.
- **Synthetic** (`synthetic_video` config): `length_L` picks the stream
  length; every candidate clip question is used, `present` giving the
  answer. Question ids are prefixed with the bucket directory and stream id
  because they recur across lengths and bands.

The published rows already omit the questions excluded from the density
analysis, so no exclusion list is needed.

### Evaluating a composed set

Always pass `--rag`, which prepends the disclosure preamble (verbatim in
`docs/PROTOCOL.md`); without it the model is misled about its input:

```bash
substream-eval --data runs/rag_qwen_top2/questions_dataset.json \
    --asset-root runs/rag_qwen_top2 --rag \
    --model qwen3_full --mode video --variant spatial --fps 1
```

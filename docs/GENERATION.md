# Regenerating the synthetic benchmark

This guide shows you how to regenerate the synthetic streams and their
rendered videos from scratch.

To reproduce the released synthetic sets in one command each, run the
`bucket_cli` orchestrator with the paired configs
`src/substream/generation/configs/bucket_spatial_uniform_eval_tokens.yaml`
(token streams) and `bucket_spatial_uniform_eval_lanes.yaml` (lane-control
streams):

```bash
python -m substream.generation.seq2vid.bucket_cli generate --config <buckets.yaml> --out-dir runs_seq
python -m substream.generation.seq2vid.bucket_cli render   --config <buckets.yaml> --sequences-root runs_seq --out-dir runs_render
python -m substream.generation.seq2vid.bucket_cli manifest --sequences-manifest runs_seq/bucket_generation_manifest.json
```

The pipeline underneath has two stages, in `src/substream/generation/`:

- **`vidgeom`** — a geometry-only renderer (no textures). A template YAML
  describes the scene, timing, vocabulary-to-glyph mapping, rules, and
  variants; all geometry is procedural in normalized 0..1 coordinates, so
  resolution and framerate changes are automatic.
- **`seq2vid`** — wraps rule-discovered token sequences into rendered videos
  with binary question metadata; you run its CLIs (`gen_cli`, `render_cli`,
  and the `bucket_cli` orchestrator).

Conveyor renders are letter-only: each token spawns one centered glyph on its
own frame; belts and lighting stay static. The spatial template
(`src/substream/generation/examples/template_conveyor_spatial.yaml`) pairs
the token stream with a dedicated lane-control sequence, so spatial
(belt-order) questions never change the rendered token order. Registered
scenes: `conveyor_v2`, `locker_room_v1_geom`, `sorting_hub_v1_geom`,
`shape_flash_v1` (see `SCENE_REGISTRY`).

## Questions JSON format

Each entry under `videos[].questions[]` is binary:

```json
{
  "question": "Did this sequence appear anywhere in the video?",
  "question_mode": "exists",
  "question_format": "binary_yes_no",
  "question_variant": "sequential",
  "answer": "yes",
  "candidate": {
    "sequence": ["4", "7", "10", "13"],
    "clip_path": "clips/video_1_q0_true_sequential.mp4",
    "clip_start": 9.04,
    "clip_end": 12.89,
    "present": true
  }
}
```

Half the questions are "yes" (a real slice), half "no" (a rerendered
distractor). `questions.json` also records causal empirical entropy (`entropy_overall.empirical_bits`, a
Lempel–Ziv estimator using only prefix context), the analytic bits from
generation (`entropy_overall.analytic_bits`), and per-question
`entropy_prefix` (the same metric at the question's position). Continuation
buckets add `prefix_clip_path` (and start/end) alongside the usual
`candidate`.

## Stage 1: generate sequences

```bash
python -m substream.generation.seq2vid.gen_cli \
  --out-dir runs_seq \
  --num-seqs 10 \
  --seq-lens 120 \
  --vocab-sizes 12 \
  --entropy-mins 0.15 --entropy-maxs 1.10 --ngram-max 6 --top-k 500 \
  --max-attempts 30 \
  --gen-workers 2 \
  --discover-len-mult 6 --max-rules 60 \
  --rule-mode probabilistic \
  --seed 123
```

Key flags (`--help` for all):

| flag | default | meaning |
|---|---|---|
| `--num-seqs` | 1 | total pool size; names auto-assigned (`SEQ_1`, `SEQ_2`, …) |
| `--seq-lens` / `--vocab-sizes` | 120 / 16 | comma-separated per-sequence values; a single value broadcasts |
| `--entropy-mins` / `--entropy-maxs` | 0.15 / 0.45 | per-sequence `[min, max)` bounds in bits/symbol |
| `--max-attempts` | 10 | attempts per sequence to hit the entropy bounds |
| `--discover-len-mult` | 4.0 | multiplier for the rule-discovery simulation length |
| `--max-rules` | 100 | maximum rules for discovery |
| `--rule-mode` | probabilistic | `probabilistic` or `deterministic` |
| `--disable-entropy-drop-guard` | off | relaxes only the per-iteration downward entropy guard (bounds and upward guard stay enforced); bucket configs expose it as `disable_entropy_drop_guard: true` (set for every low-entropy bucket) |
| `--gen-workers` | 1 | parallel generation workers |
| `--top-k` / `--ngram-max` | 500 / 6 | n-gram statistics stored per sequence |
| `--seed` | none | reproducibility |

Entropy bounds are checked analytically in bits/symbol using the rule
automaton; a no-rules uniform base reports `log2(vocab_size)` bits. Generation
fails fast if the bounds cannot be met within `--max-attempts`.

## Stage 2: render videos and questions

```bash
python -m substream.generation.seq2vid.render_cli \
  --template src/substream/generation/examples/template_conveyor_multi.yaml \
  --sequences-file runs_seq/sequences.json \
  --out-dir runs_render \
  --num-questions 8 \
  --num-videos 5 \
  --clip-options \
  --question-min-len 3 \
  --question-mode exists \
  --log-progress
```

Key flags:

| flag | default | meaning |
|---|---|---|
| `--template` | `examples/template_conveyor.yaml` | vidgeom template; declares its required sequence IDs in `sequences: [...]` |
| `--sequences-file` / `--sequences-root` | — / `runs_seq` | classic single-pool mode vs per-bucket mode |
| `--num-questions` / `--num-videos` | bucket defaults / no cap | questions per video; cap on videos rendered |
| `--clip-options` | off | render an mp4 clip per candidate ("yes" = matches the main video; "no" = rerendered distractor) |
| `--questions-only` | off | write `questions.json` only, no mp4 rendering (`--clip-options` ignored) |
| `--question-mode` | exists | `exists` ("did this appear?") or `continuation` ("does this follow the prefix?") |
| `--questions-at-end` | interspersed | force all questions to reference the full video |
| `--target-seq-lens` | none | comma-separated lengths; longer sequences are trimmed, shorter ones raise |
| `--assignment-seed` | none | shuffles the pool before assigning one entry per template sequence ID per video; each sequence used at most once |
| `--spatial-question-fraction` | 0.0 | fraction of questions rendered as spatial (lane-order) variants; spatial "no" candidates keep the token slice and sample lane traces that never co-occurred (sequential fallback after 100 failed attempts) |
| `--sequence-source seq_id=path` | none | dedicated pools per template sequence (e.g. token stream + lane-control stream); provenance recorded under `sequence_sources` |
| `--render-workers` | 1 | per-video process pool |
| `--fps` | template value | output frame rate |
| `--ffmpeg-crf` / `--ffmpeg-preset` / `--ffmpeg-codec` | 23 / veryfast / libx264 | compression (`libx265` is accepted) |
| `--question-min-len` | 3 | minimum candidate n-gram length |
| `--skip-validation` | off | skip `validation_manifest.json` (debugging only) |
| `--debug-frame-metadata` | off | dump per-frame token/letter/lane maps |

## EasyHuman

The EasyHuman sequences are shipped in the dataset's `easyhuman` config and
are not generated by this code. Their videos use the 3-belt template
(`template_conveyor_spatial.yaml`).

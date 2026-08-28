# Evaluation protocol

This guide shows you what a model sees during a run and how the calibration
(abstention-pricing) conditions work. Every prompt below is verbatim from the
source of truth, `src/substream/eval/processors/question_processor.py`.

## The task

Run an evaluation with `substream-eval`:

```bash
substream-eval --data natural_video --model qwen3_full --mode video --variant spatial
```

- `--data` accepts a Hub config name (`text`, `synthetic_video`,
  `natural_video`, `easyhuman`), a `<repo_id>:<config>` pair, or a local
  `manifest.json` path (with `--asset-root` for the media).
- `--mode text` shows the stream as tokens, `--mode video` as rendered video.
- `--variant sequential` keeps only order-based questions; `spatial` asks all.

Each item is a `(stream, probe, label)` tuple: a long input stream (token
sequence, rendered synthetic video, or natural clip), a short candidate (for
natural video, a question about an action), and whether it occurred inside
the stream. Answers are binary with an explicit abstain option in curly
braces: `{0}` / `{1}` / `{2}` (`{uncertain}` on natural video). A response
with no parseable answer extracts as `2` — see "Abstention versus failure to
answer".

## Exact prompts

Three base templates, built by
`QuestionProcessor.build_native_binary_question_text`.

**Natural video** (the question text is injected directly):

```
Watch this video closely, paying attention to the entire video. Did the person
open a fridge in this video?

Write {0} if yes, {1} if no or {uncertain} if you are unsure. Please put your
answer in curly braces, e.g, {0}.
```

**Synthetic text sequence** (`--mode text`):

```
You are given a main sequence of tokens and a candidate sequence. Your task:
Determine if the candidate sequence appears as a contiguous subsequence within
the main sequence, in the exact same order. The candidate must appear with all
its tokens consecutive and in order - no gaps, no extra tokens in between.
Answer {0} if the candidate appears as a contiguous subsequence, {1} if it does
not, or {2} if you are unsure. Please put your answer in curly braces, e.g, {2}.
```

**Synthetic video** (the conveyor-belt membership template):

```
In the main video and option video, letters moved down three conveyer belts. A
letter that appeared in any of the videos on a specific conveyer belt ALWAYS
stayed on the same conveyer belt and only moved down, not left or right. The
letters appeared sequentially. Only one was on the screen at a time. The letters
continued moving down the conveyer belt until they left the screen. Examine the
main video and the option video closely. Did this option video show events that
exactly appeared SOMEWHERE in the main video, down to the order they appeared
and which conveyer belt each letter was on? Write {0} if yes, {1} if no, or {2}
if unsure. Please put your answer in curly braces, e.g, {2}.
```

## Calibration arms: two-arm abstention pricing

The base prompts state no cost for a wrong answer, so answering weakly
dominates; the calibration conditions price wrong answers explicitly.

Pass `--arm A` or `--arm B` (this exports `CALIB_ARM` before any prompt is
built) to append **one sentence and nothing else** to the base prompt, so any
A/B difference is attributable to that sentence alone. With no arm set, the
prompt is byte-identical to the control condition; an unrecognised `CALIB_ARM`
value raises rather than silently emitting control data.

`{X}` below is whichever abstain token the base prompt used — `{uncertain}` on
natural video, `{2}` on both synthetic templates.

| arm | appended verbatim |
|---|---|
| **A** | ` Scoring: a correct answer earns +3 points, a wrong answer costs -1 point, and {X} earns 0 points. Answer only if you are more than 25% confident; otherwise answer {X}.` |
| **B** | ` Scoring: a correct answer earns +1 point, a wrong answer costs -4 points, and {X} earns 0 points. Answer only if you are more than 80% confident; otherwise answer {X}.` |

The payoffs are `(reward, penalty, threshold)` = A `(3, 1, 25)`, B `(1, 4, 80)`
(the `_CALIB_ARMS` table in `question_processor.py`); the implied confidence
threshold is tau = penalty / (reward + penalty) = 25% and 80%. The prompt
states both payoffs and threshold so a model cannot fail on arithmetic. The
thresholds straddle chance (50%): under arm A even a guessing model should
answer, under arm B it should abstain. Arm A observes the model's answers on
exactly the items arm B declines, so selectivity is read from the paired
runs — run the same cell twice:

```bash
substream-eval --data natural_video --model qwen3_full --arm A --name calibA_myrun
substream-eval --data natural_video --model qwen3_full --arm B --name calibB_myrun
```

One failure mode remains: low coverage caused by output degeneration, not
abstention, can fake a selectivity result — always read coverage alongside
response length.

### Band protocol for natural rows

When you split natural-video calibration results by information density, use
the **codec-distortion** bands of `docs/DATASET.md`: x264 re-encode at
32 kbit/frame, distortion measured in DINOv2 embedding space, one global
low/high cutoff at every length. Synthetic rows are never codec-banded — they
use their known entropy conditions (low / medium / max-entropy).

## RAG disclosure

Pass `--rag` (this exports `RAG_PREAMBLE=1`) to prepend this text to every
prompt. Use it whenever you evaluate retrieval-composed videos
(`docs/ARCHITECTURES.md`); without it the model is misled about its input.

```
NOTE: A retrieval system has pre-selected only the segments of the original
video most relevant to the question below. You are NOT seeing the complete
video. Distinct segments are separated by a bright magenta frame labeled
SEGMENT BREAK.
```

It is prepended, not appended, because the natural-prompt routing dispatches
on the question's first word.

## Frame geometry

Frames are decoded at 1 fps for natural video. Per-model frame sizes are set
by each model's sampler under `src/substream/samplers/`; per-model video
length caps are `max_video_bucket` in `src/substream/registry.py`.

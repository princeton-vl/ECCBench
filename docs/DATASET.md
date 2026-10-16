# The published dataset

This guide shows you how to load the published dataset and how to read its
schema and media layout.

The benchmark is published on the Hugging Face Hub as
[`anonstreammem/substream-recollection`](https://huggingface.co/datasets/anonstreammem/substream-recollection).
Each row is a `(stream, probe, label)` tuple: a long input stream, a short
candidate, and whether the candidate (for natural video, the referenced
action) occurred inside the stream.

Two terms recur below: a **bucket** fixes the stream length L; synthetic rows
carry an **entropy band** (`low`/`medium`/`max-entropy`, enforced analytically
at generation); natural rows carry a measured **density band** (`low`/`high`,
codec-distortion protocol below).

## Configs

| config | rows | content |
|---|---|---|
| `text` | 7,640 | synthetic substream questions, text modality, L = 8 … 4096 |
| `synthetic_video` | 6,065 | the same streams rendered as video |
| `natural_video` | 1,726 | 1,015 EPIC-Kitchens-100 and SoccerNet clips, L = 8, 16, 64, 128, 512, 1024, two question sets |
| `easyhuman` | 672 | 3-belt EasyHuman: 224 video rows (L = 256), 448 text rows (224 at L = 256, 224 at L = 1024); see `modality` |

## Three encodings

Every config ships the same rows three ways:

| file | format | intended consumer |
|---|---|---|
| `questions.parquet` | Parquet, one row per question | `datasets.load_dataset` |
| `questions.json` | NDJSON (one JSON object per line) | ad-hoc tooling |
| `manifest.json` | nested (videos, each with a `questions` list) | the evaluation runner |

Load the rows directly:

```python
from datasets import load_dataset
ds = load_dataset("anonstreammem/substream-recollection", "natural_video")["train"]
```

You do not need to download anything by hand for evaluation.
`eccbench-eval --data natural_video` first downloads the question files of
the config. It then downloads the media of the streams that you select with
`--buckets` and `--limit` (`eccbench.data.hf_loader`). The runner reads
`manifest.json`, but it keeps only the questions that are also rows of
`questions.json`.

## Media join rule

`video_path` and `clip_path` are relative to the config that **holds the
media**, given by the `media_config` column:

```python
from huggingface_hub import snapshot_download
root = snapshot_download("anonstreammem/substream-recollection", repo_type="dataset")
path = f"{root}/{row['media_config']}/{row['video_path']}"   # rows with a non-null video_path
```

`video_path` is `null` on `text` rows and `easyhuman` text-modality rows.

`media_included` has three values: `true` = media published; `false` =
NDA-gated and absent (provenance-only: the 161 SoccerNet rows); `null` = the
row has no media file. Filter with `media_included != False`, not a truthiness test, or
`null` rows drop too; the runner skips provenance-only rows itself.

## Keys and question sets

`answer` is the ground truth (`"yes"`/`"no"`). The row key is
`(question_id, length_L, entropy_band, question_variant)` — plus `modality` in
`easyhuman`. **`question_id` alone is not unique**: synthetic questions recur
across lengths and bands; ids are unique only within a question set.

`natural_video` holds two disjoint question sets over the same 1,015 clips:

| `question_set` | rows | what it asks |
|---|---|---|
| `designed` | 1,007 | the question the clip was built for, at every length |
| `density_topup` | 719 | mined questions about *other* event classes on 377 codec-banded clips (L = 64 … 1024), for the density analysis |

They share no `question_id`; no clip is asked the same question twice.

## Density bands (`natural_video`)

`density_band` is `low` / `high`: each clip is re-encoded with x264 at a fixed
budget of **32 kbit/frame** and banded by the resulting distortion (measured
in DINOv2 embedding space against the original), using **one global cutoff**
held constant at every stream length; the band is a property of the video,
uniform across its questions.

`density_band` is `null` on the 496 `designed` rows at L = 8 and 16 (no band
is assigned at these lengths) and on 117 unbanded rows at higher lengths:
613 `null` rows in total, all in `designed`.

Other columns:

| column | meaning |
|---|---|
| `event_start_frame`, `event_end_frame` | frame indices of the clip's designed event |
| `event_clip_start`, `event_clip_end` | same instants in seconds; clips are 1 fps, frame = `floor(seconds)` |
| `event_timing` | `measured` (EPIC: extent from narration) or `centered_by_construction` (SoccerNet: clip centred on the labelled instant, nominal 2 s window) |
| `clip_sha256` | sha256 of the published mp4; `null` where NDA-gated |
| `label_source` | `designed`, or `narration_margin` / `whole_video_absence` / `labels_v2_window` / `labels_v2_window_absence` for mined questions; `annotation_corrected` where the answer was checked against the source annotations and corrected |
| `in_density_comparison`, `in_calibration`, `in_rag_comparison` | whether the paper's density, calibration or retrieval comparison uses the question |
| `query_class` | the event class this question asks about |
| `question_time` | question position in clip seconds; always `length_L` |
| `source_provenance` | JSON: source video id, clip/event positions, slowdown factor, SoccerNet match metadata |

The event columns give the position of the **clip's** designed event. This
is the subject of the question only when `question_set == "designed"`. For
`density_topup` rows, compare `event_type` with `query_class` first.

## Licenses

Synthetic and EasyHuman content is CC BY 4.0. EPIC-Kitchens-100 clips are
CC BY-NC 4.0. SoccerNet rows are provenance-only under the SoccerNet NDA.
Each row has a `license` column, and the full texts are in `LICENSES/`. Cite
EPIC-KITCHENS-100 (Damen et al., IJCV 2022) and SoccerNet (Giancola et al.,
CVPR Workshops 2018) together with the benchmark.

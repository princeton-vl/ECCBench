# The published dataset

This guide shows you how to load the published dataset and how to read its
schema, media layout, and known caveats.

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
| `natural_video` | 1,813 | 1,028 EPIC-Kitchens-100 + SoccerNet clips, L = 8 … 1024, two question sets |
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

You do not need to download anything by hand for evaluation:
`substream-eval --data natural_video` downloads and resolves the dataset
itself (`substream.data.hf_loader.fetch`).

## Media join rule

`video_path` and `clip_path` are relative to the config that **holds the
media**, given by the `media_config` column:

```python
from huggingface_hub import snapshot_download
root = snapshot_download("anonstreammem/substream-recollection", repo_type="dataset")
path = f"{root}/{row['media_config']}/{row['video_path']}"   # rows with a non-null video_path
```

`video_path` is `null` on `text` rows and `easyhuman` text-modality rows.

`media_included` is tri-state: `true` = media published; `false` = NDA-gated
and absent (provenance-only — some SoccerNet-derived clips); `null` = no gated
media file. Filter with `media_included != False`, not a truthiness test, or
`null` rows drop too; the runner skips provenance-only rows itself.

## Keys and question sets

`answer` is the ground truth (`"yes"`/`"no"`). The row key is
`(question_id, length_L, entropy_band, question_variant)` — plus `modality` in
`easyhuman`. **`question_id` alone is not unique**: synthetic questions recur
across lengths and bands; ids are unique only within a question set.

`natural_video` holds two disjoint question sets over the same 1,028 clips:

| `question_set` | rows | what it asks |
|---|---|---|
| `designed` | 1,028 | the question the clip was built for, at every length L = 8 … 1024 |
| `density_topup` | 785 | mined questions about *other* event classes on the 403 codec-banded clips (L = 64 … 1024), for the density analysis |

They share no `question_id`; no clip is asked the same question twice.

## Density bands (`natural_video`)

`density_band` is `low` / `high`: each clip is re-encoded with x264 at a fixed
budget of **32 kbit/frame** and banded by the resulting distortion (measured
in DINOv2 embedding space against the original), using **one global cutoff**
held constant at every stream length; the band is a property of the video,
uniform across its questions.

`density_band` is `null` on the 504 `designed` rows at L = 8/16 (no band
assigned at L = 8/16) and on 121 unbanded clips at higher lengths —
625 null `designed` rows total.

Other columns:

| column | meaning |
|---|---|
| `event_start_frame`, `event_end_frame` | frame indices of the clip's designed event |
| `event_clip_start`, `event_clip_end` | same instants in seconds; clips are 1 fps, frame = `floor(seconds)` |
| `event_timing` | `measured` (EPIC: extent from narration) or `centered_by_construction` (SoccerNet: clip centred on the labelled instant, nominal 2 s window) |
| `clip_sha256` | sha256 of the published mp4; `null` where NDA-gated |
| `label_source` | `designed`, or `narration_margin` / `whole_video_absence` / `labels_v2_window` / `labels_v2_window_absence` for mined questions |
| `query_class` | the event class this question asks about |
| `question_time` | question position in clip seconds; always `length_L` |
| `source_provenance` | JSON: source video id, clip/event positions, slowdown factor, SoccerNet match metadata |

The event columns localise the **clip's** designed event, the question's
subject only when `question_set == "designed"`; for `density_topup` rows,
check `event_type` against `query_class` first.

## Licences

Synthetic and EasyHuman content is CC BY 4.0; EPIC-Kitchens-100-derived clips
CC BY-NC 4.0; SoccerNet rows provenance-only under the SoccerNet NDA.
Each row carries a `license` column; full texts ship in `LICENSES/`. Cite
EPIC-KITCHENS-100 (Damen et al., IJCV 2022) and SoccerNet (Giancola et al.,
CVPR Workshops 2018) alongside the benchmark.

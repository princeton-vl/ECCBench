# ECCBench: Evaluating the Memory of Vision-Language Models

**Paper:** *Good Memory has ECC: Evaluating the Memory of Vision-Language Models Beyond Accuracy*, Shmuel Berman and Jia Deng, Princeton University (arXiv link forthcoming).

ECCBench (published on the Hub as `substream-recollection`) is how we evaluate the memory of vision-language models beyond accuracy: **capacity** (how much incompressible content a model can retain), **compression** (whether structure in the input lets it remember more), and **calibration** (whether it abstains when it cannot answer reliably). The task is associative recall: a model watches a long stream — a video, or a plain token sequence — and is asked whether a short excerpt, a *substream*, appeared inside it. Streams have known ground truth and a known amount of structure, from highly compressible to nearly random, which is what makes the three properties separable. This repository has everything we use to evaluate models on the benchmark, regenerate the synthetic portion from scratch, and train the small backbones we use as reference memory systems.

## Getting Started

Install the package, then add the `api` extra, which any evaluation needs:

```bash
pip install -e .
pip install -e ".[api]"
```

Run a sanity check that needs no model weights or GPU. The dummy model cycles through canned answers, which exercises the data loading, prompting, and logging end to end:

```bash
substream-eval --data text --model dummy_eval --mode text --variant sequential \
    --buckets low/L8 --limit 2
```

The runner downloads the dataset automatically the first time you name a config. Every run writes a resumable state file and a per-question CSV under `--out` (default `runs/`); continue an interrupted run with `--resume`.

Evaluating real models requires more setup: the evaluated models need mutually incompatible library versions, so each group lives in its own virtual environment. See [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) to set these up.

## Documentation

- [docs/DATASET.md](docs/DATASET.md) — schema, media layout, and known caveats
- [docs/PROTOCOL.md](docs/PROTOCOL.md) — prompts, scoring, and the calibration arms
- [docs/GENERATION.md](docs/GENERATION.md) — regenerating the synthetic data
- [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md) — retrieval-augmented evaluation and M3-Agent
- [docs/BACKBONES.md](docs/BACKBONES.md) — training and evaluating the alternative-architecture memory backbones
- [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) — which model needs which environment, and every environment variable the package reads

## The dataset

We publish the dataset on the Hugging Face Hub at [`anonstreammem/substream-recollection`](https://huggingface.co/datasets/anonstreammem/substream-recollection).

| Config | Questions | Media | Contents |
|---|---|---|---|
| `text` | 7,640 | none | token sequences read as plain text |
| `synthetic_video` | 6,065 | 128 MB | the same sequences rendered as videos |
| `natural_video` | 1,813 | 5.2 GB | real video from EPIC-Kitchens-100; SoccerNet clips are listed by provenance only, their media is not redistributed |
| `easyhuman` | 672 | 10 MB | simple hand-designed patterns, a diagnostic floor |

Streams range from 8 to 4096 steps in the text config (video configs stop at 1024); each length is called a *bucket*. Synthetic streams come in three entropy bands, from highly predictable (`low`) through `medium` to fully random (`max-entropy`). See [docs/DATASET.md](docs/DATASET.md) for the full schema.

## Citation

If you use ECCBench in your work, please cite:

```bibtex
@article{berman2026eccbench,
  title  = {Good Memory has {ECC}: Evaluating the Memory of Vision-Language Models Beyond Accuracy},
  author = {Berman, Shmuel and Deng, Jia},
  journal = {arXiv preprint},
  year   = {2026}
}
```

## Getting Help

Please post a GitHub Issue on this repository. Include the exact command you ran, the full output logs, and your environment (OS, GPU(s), and which pip extra you installed).

## License

First-party code is MIT licensed; vendored components under `external/` keep their upstream licenses (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), and the dataset card documents per-row licensing, including the EPIC-Kitchens-100-derived clips (CC BY-NC 4.0) and the provenance-only SoccerNet rows.

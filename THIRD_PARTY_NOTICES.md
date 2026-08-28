# Third-Party Notices

This repository vendors several upstream projects under their original
licenses. The top-level `LICENSE` (MIT) covers only first-party code in
this repository; it does **not** relicense any vendored component below.
Each vendored file or directory remains under its upstream license, and
the corresponding upstream `LICENSE` is included in this repository at
the location indicated.

Where one upstream contributes several single-file modules, its `LICENSE`
lives once under `external_licenses/` at the repository root and is
cross-referenced from the relevant entries below rather than duplicated.

---

## Vendored directories

### `external/LongVILA/`

- **Upstream:** NVlabs/VILA (LongVILA) — https://github.com/NVlabs/VILA
- **License:** Apache License 2.0
- **In-tree LICENSE:** `external/LongVILA/LICENSE`

### `external/TimeChat/`

- **Upstream:** yaolinli/TimeChat-Online — https://github.com/yaolinli/TimeChat-Online
  (the Qwen2.5-VL fork with token dropping, `modeling_qwen2_5_vl_DTD.py`,
  used for the `wyccccc/TimeChatOnline-7B` checkpoint)
- **License:** BSD 3-Clause (Copyright 2025, LANCO @ Peking University)
- **In-tree LICENSE:** `external/TimeChat/LICENSE` (copy at
  `external_licenses/TimeChat-Online.LICENSE`)

### `external/TimeChat/qwen2_5_vl/`

- **Upstream:** Qwen2.5-VL reference modeling code from `huggingface/transformers`
  (https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl)
- **License:** Apache License 2.0 (Copyright 2018- The Hugging Face team)
- **In-tree LICENSE:** `external/TimeChat/qwen2_5_vl/LICENSE`

### `src/substream/backbones/backend/models/external/ttt_fast/ThunderKittens/`

- **Upstream:** HazyResearch/ThunderKittens — https://github.com/HazyResearch/ThunderKittens
- **License:** MIT License (Copyright 2024 HazyResearch)
- **In-tree LICENSE:** `src/substream/backbones/backend/models/external/ttt_fast/ThunderKittens/LICENSE`

### `src/substream/backbones/backend/models/external/ttt_fast/` (parent module)

- **Upstream provenance:** Derivative of `test-time-training/ttt-lm-pytorch`
  (MIT). The upstream `ttt.py` defines the TTT primitive that
  `modeling_ttt.py` extends. Additionally, `generation.py` carries a
  per-file header stating "Modified from
  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/generation.py"
  (Apache-2.0).
- **In-tree LICENSE:** `external_licenses/ttt-lm-pytorch.LICENSE` (MIT)
  and `external_licenses/mamba.LICENSE` (Apache 2.0) for the respective
  derivative portions.

### `src/substream/generation/`

- **Origin:** First-party. The `generation/vidgeom/` package and the
  `generation/seq2vid/` CLI suite are authored as part of this work and
  covered by the repository-level `LICENSE` (MIT).

---

## Single-file vendored modules under `src/substream/backbones/backend/models/external/`

Individual `.py` files copied or adapted from upstream projects. Each
upstream `LICENSE` is included once at the listed path in
`external_licenses/`.

| File | Upstream | License | In-tree LICENSE |
|---|---|---|---|
| `compressive_transformer.py` | lucidrains/compressive-transformer-pytorch — https://github.com/lucidrains/compressive-transformer-pytorch | MIT | `external_licenses/compressive-transformer-pytorch.LICENSE` |
| `deltaformer_parallel.py` | fla-org/flash-linear-attention — https://github.com/fla-org/flash-linear-attention (Copyright 2023-2025 Songlin Yang, Yu Zhang) | MIT | `external_licenses/flash-linear-attention.LICENSE` |
| `mamba2.py` | state-spaces/mamba — https://github.com/state-spaces/mamba (Copyright 2024 Tri Dao, Albert Gu) | Apache License 2.0 | `external_licenses/mamba.LICENSE` |
| `memory_mosaics_blocks.py` | facebookresearch/MemoryMosaics — https://github.com/facebookresearch/MemoryMosaics (Copyright Meta Platforms, Inc. and affiliates); per-file header states "derived from nanoGPT" (karpathy/nanoGPT — https://github.com/karpathy/nanoGPT) | Apache License 2.0 (nanoGPT base: MIT) | `external_licenses/MemoryMosaics.LICENSE`; `external_licenses/nanoGPT.LICENSE` |
| `stm.py` | thaihungle/SAM — https://github.com/thaihungle/SAM (Self-Attentive Associative Memory; Copyright 2020 Tony) | MIT | `external_licenses/SAM.LICENSE` |
| `ttt.py` | test-time-training/ttt-lm-pytorch — https://github.com/test-time-training/ttt-lm-pytorch | MIT | `external_licenses/ttt-lm-pytorch.LICENSE` |

---

## Top-level summary of in-tree license files

```
LICENSE                                                              # MIT, first-party
external/LongVILA/LICENSE                                            # Apache 2.0
external/TimeChat/LICENSE                                            # BSD 3-Clause
external/TimeChat/qwen2_5_vl/LICENSE                                 # Apache 2.0
external_licenses/
  ├── compressive-transformer-pytorch.LICENSE                        # MIT
  ├── flash-linear-attention.LICENSE                                 # MIT
  ├── mamba.LICENSE                                                  # Apache 2.0
  ├── MemoryMosaics.LICENSE                                          # Apache 2.0
  ├── nanoGPT.LICENSE                                                # MIT
  ├── SAM.LICENSE                                                    # MIT
  ├── TimeChat-Online.LICENSE                                        # BSD 3-Clause
  ├── ttt-lm-pytorch.LICENSE                                         # MIT
  └── xformers.LICENSE                                               # BSD 3-Clause
src/substream/backbones/backend/models/external/ttt_fast/
  └── ThunderKittens/LICENSE                                         # MIT
```

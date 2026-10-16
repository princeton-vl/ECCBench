"""Analytical per-response FLOPs, as defined in Appendix B of the paper.

A response costs

    F = F_vis + F_conn + F_pre(N) + F_dec(N, T_out)

where N is the number of tokens the language model reads and T_out the number
of tokens it generates. A matrix product of shapes (a, b) x (b, c) costs 2abc
FLOPs; elementwise operations are not counted.

Text inputs have no vision or connector cost. Video inputs read
``text_tokens`` prompt tokens plus the vision tokens of every frame.

Run ``python -m eccbench.flops RUN.csv --model ALIAS`` to add a ``flops``
column to a per-question log written by ``eccbench-eval``.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

HEAD_DIM = 128
VIDEO_PROMPT_TOKENS = 128
GLM_DENSE_FFN = 10944


@dataclass(frozen=True)
class ModelConstants:
    """Architecture constants from the paper's Table ``tab:flops_constants``.

    ``family`` selects the vision geometry and connector. ``vision_ffn`` is
    ``"G"`` (two matrices) or ``"SG"`` (three, gated). ``moe`` is
    ``(experts, active, shared_width)`` or ``None``.
    """

    family: str
    L_v: int
    d_v: int
    f_v: int
    vision_ffn: str
    patch: int
    packs_frames: bool
    L_t: int
    d_t: int
    n_q: int
    n_kv: int
    f_t: int
    vocab: int
    moe: tuple[int, int, int] | None = None
    lora_rank: int = 0


_QWEN25 = ModelConstants("qwen25", 32, 1280, 3420, "SG", 14, True, 28, 3584, 28, 4, 18944, 152064)
_QWEN3 = ModelConstants("qwen3", 27, 1152, 4304, "G", 16, True, 36, 4096, 32, 8, 12288, 151936)
_INTERNVL_8B = ModelConstants(
    "internvl", 24, 1024, 4096, "G", 14, False, 36, 4096, 32, 8, 12288, 151936
)
_INTERNVL_30B = ModelConstants(
    "internvl", 24, 1024, 4096, "G", 14, False, 48, 2048, 32, 4, 768, 151936, moe=(128, 8, 0)
)
_INTERNVL_38B = ModelConstants(
    "internvl", 45, 3200, 12800, "G", 14, False, 64, 5120, 64, 8, 25600, 151936
)

MODELS: dict[str, ModelConstants] = {
    "qwen_full": _QWEN25,
    "qwen3_full": _QWEN3,
    "qwen3_full_thinking": _QWEN3,
    "qwen3_omni": ModelConstants(
        "qwen3", 27, 1152, 4304, "G", 16, True, 48, 2048, 32, 4, 768, 152064, moe=(128, 8, 0)
    ),
    "glm45v": ModelConstants(
        "glm", 24, 1536, 4096, "SG", 14, True, 46, 4096, 96, 8, 1408, 151552, moe=(128, 8, 1408)
    ),
    "internvl-3-5": _INTERNVL_8B,
    "internvl-3-5-thinking": _INTERNVL_8B,
    "internvl-3-5-30b-a3b": _INTERNVL_30B,
    "internvl-3-5-30b-a3b-thinking": _INTERNVL_30B,
    "internvl-3-5-38b": _INTERNVL_38B,
    "internvl-3-5-38b-thinking": _INTERNVL_38B,
    "mimo-vl": ModelConstants(
        "qwen25", 32, 1280, 3456, "SG", 14, True, 36, 4096, 32, 8, 11008, 151680
    ),
    "minicpm-4-5": ModelConstants(
        "minicpm45", 27, 1152, 4304, "G", 14, False, 36, 4096, 32, 8, 12288, 151748
    ),
    "minicpm": ModelConstants(
        "minicpm26", 27, 1152, 4304, "G", 14, False, 28, 3584, 28, 4, 18944, 151666
    ),
    "longvila": ModelConstants(
        "longvila", 27, 1152, 4304, "G", 14, False, 28, 3584, 28, 4, 18944, 151651
    ),
    "phi_multimodal": ModelConstants(
        "phi4mm", 27, 1152, 4304, "G", 14, False, 32, 3072, 24, 8, 8192, 200064, lora_rank=256
    ),
}

_NATIVE_RESOLUTION = {"qwen25", "qwen3", "glm"}


class Flops(NamedTuple):
    vision: float
    connector: float
    prefill: float
    decode: float

    @property
    def total(self) -> float:
        return self.vision + self.connector + self.prefill + self.decode


def _get(model: str | ModelConstants) -> ModelConstants:
    return MODELS[model] if isinstance(model, str) else model


def _patch_grid(m: ModelConstants, width: int, height: int) -> tuple[int, int]:
    """Patches per side, each side rounded to a whole 2x2 merge unit."""
    unit = 2 * m.patch
    return max(1, round(width / unit)) * 2, max(1, round(height / unit)) * 2


def _minicpm45_packs(frames: int) -> list[int]:
    size = min(6, max(1, math.ceil(frames / 180)))
    full, rest = divmod(frames, size)
    return [size] * full + ([rest] if rest else [])


def _longvila_pool(frames: int) -> int:
    bucket = 1
    while bucket < 8 and bucket < frames / 256:
        bucket *= 2
    return 4 * bucket


def vision_tokens(model: str | ModelConstants, frames: int, width: int = 448,
                  height: int = 448) -> int:
    """Number of vision tokens V_LLM the connector passes to the language model."""
    m = _get(model)
    if frames <= 0:
        return 0
    if m.family in _NATIVE_RESOLUTION:
        nw, nh = _patch_grid(m, width, height)
        return nw * nh // 4 * math.ceil(frames / 2)
    if m.family == "internvl":
        return 256 * frames
    if m.family == "minicpm26":
        return 64 * frames
    if m.family == "minicpm45":
        return 64 * len(_minicpm45_packs(frames))
    if m.family == "longvila":
        return 257 * (frames // _longvila_pool(frames))
    if m.family == "phi4mm":
        return 545 * frames
    raise ValueError(f"unknown family {m.family!r}")


def vision_flops(model: str | ModelConstants, frames: int, width: int = 448,
                 height: int = 448) -> float:
    """Vision-encoder FLOPs for ``frames`` frames."""
    m = _get(model)
    if frames <= 0:
        return 0.0
    if m.family in _NATIVE_RESOLUTION:
        nw, nh = _patch_grid(m, width, height)
        passes, V, V_pre = math.ceil(frames / 2), nw * nh, nw * nh
        spans = [(4, V), (m.L_v - 4, 64)] if m.family == "qwen25" else [(m.L_v, V)]
    elif m.family == "internvl":
        passes, V, V_pre, spans = frames, 1025, 1024, [(m.L_v, 1025)]
    elif m.family == "phi4mm":
        passes, V, V_pre, spans = 2 * frames, 1024, 1024, [(m.L_v, 1024)]
    else:
        passes, V, V_pre, spans = frames, 1024, 1024, [(m.L_v, 1024)]

    d_v = m.d_v
    d_pat = 3 * m.patch**2 * (2 if m.packs_frames else 1)
    ffn = 6 if m.vision_ffn == "SG" else 4
    per_pass = 2 * V_pre * d_pat * d_v
    for blocks, S in spans:
        per_pass += blocks * (8 * V * d_v**2 + 4 * V * S * d_v + ffn * V * d_v * m.f_v)
    return float(passes * per_pass)


def connector_flops(model: str | ModelConstants, frames: int, width: int = 448,
                    height: int = 448) -> float:
    """FLOPs of the layers that map vision features into the language model."""
    m = _get(model)
    if frames <= 0:
        return 0.0
    d_v, d_t = m.d_v, m.d_t
    merged = 4 * d_v
    tokens = vision_tokens(m, frames, width, height)
    if m.family == "longvila":
        tokens = 256 * frames
    if m.family in ("qwen25", "longvila"):
        return float(tokens * (2 * merged**2 + 2 * merged * d_t))
    if m.family == "qwen3":
        return float(tokens * 4 * (2 * merged**2 + 2 * merged * d_t))
    if m.family == "glm":
        return float(tokens * (2 * merged * d_t + 2 * d_t**2 + 6 * d_t * GLM_DENSE_FFN))
    if m.family == "internvl":
        return float(tokens * (2 * merged * d_t + 2 * d_t**2))
    if m.family == "phi4mm":
        return float(tokens * (2 * d_v * d_t + 2 * d_t**2))

    def resampler(K: int) -> int:
        return 2 * K * d_v * d_t + 4 * K * d_t**2 + 256 * d_t * K + 384 * d_t**2

    if m.family == "minicpm26":
        return float(frames * resampler(1024))
    if m.family == "minicpm45":
        return float(sum(resampler(1024 * p) for p in _minicpm45_packs(frames)))
    raise ValueError(f"unknown family {m.family!r}")


def _ffn_per_token(m: ModelConstants) -> int:
    if m.moe is None:
        return 6 * m.d_t * m.f_t
    experts, active, shared = m.moe
    return 2 * m.d_t * experts + active * 6 * m.d_t * m.f_t + 6 * m.d_t * shared


def coeff_B(model: str | ModelConstants, vision: bool = True) -> int:
    """Per-token language-model FLOPs, attention scores excluded."""
    m = _get(model)
    d_q, d_kv = m.n_q * HEAD_DIM, m.n_kv * HEAD_DIM
    projections = 2 * m.d_t * (d_q + 2 * d_kv) + 2 * d_q * m.d_t
    if m.family == "glm":
        B = m.L_t * projections + 6 * m.d_t * GLM_DENSE_FFN + (m.L_t - 1) * _ffn_per_token(m)
    else:
        B = m.L_t * (projections + _ffn_per_token(m))
    if vision and m.lora_rank:
        r = m.lora_rank
        B += m.L_t * 2 * r * (
            (m.d_t + d_q + 2 * d_kv) + (d_q + m.d_t) + (m.d_t + 2 * m.f_t) + (m.f_t + m.d_t)
        )
    return B


def coeff_A(model: str | ModelConstants) -> int:
    """FLOPs of one query-key pair, summed over layers."""
    m = _get(model)
    return 4 * m.L_t * m.n_q * HEAD_DIM


def vocab_flops(model: str | ModelConstants) -> int:
    """FLOPs of one output projection to the vocabulary."""
    m = _get(model)
    return 2 * m.d_t * m.vocab


def prefill_flops(model: str | ModelConstants, n: int, vision: bool = True) -> float:
    """F_pre(N) = B N + A N (N + 1) / 2."""
    return float(coeff_B(model, vision) * n + coeff_A(model) * (n * (n + 1) // 2))


def decode_flops(model: str | ModelConstants, n: int, t_out: int, vision: bool = True) -> float:
    """FLOPs to generate ``t_out`` tokens after reading ``n`` tokens."""
    if t_out <= 0:
        return 0.0
    A, B = coeff_A(model), coeff_B(model, vision)
    return float(
        t_out * vocab_flops(model)
        + (t_out - 1) * B
        + A * ((t_out - 1) * n + t_out * (t_out - 1) // 2)
    )


def response_flops(
    model: str | ModelConstants,
    *,
    text_tokens: int,
    generated_tokens: int,
    frames: int = 0,
    width: int = 448,
    height: int = 448,
    candidate_frames: int = 0,
) -> Flops:
    """FLOPs of one response.

    ``text_tokens`` is the number of prompt text tokens; for text inputs it is
    the stream length. ``frames`` is the number of stream frames and
    ``candidate_frames`` the number of candidate-clip frames, both at
    ``width`` x ``height``.
    """
    m = _get(model)
    vision = frames > 0 or candidate_frames > 0
    vis = conn = 0.0
    n = text_tokens
    for count in (frames, candidate_frames):
        if count > 0:
            vis += vision_flops(m, count, width, height)
            conn += connector_flops(m, count, width, height)
            n += vision_tokens(m, count, width, height)
    return Flops(
        vision=vis,
        connector=conn,
        prefill=prefill_flops(m, n, vision),
        decode=decode_flops(m, n, generated_tokens, vision),
    )


_ANSWER_MARKER = re.compile(r"\{[012]\}")


def tokens_through_answer(response: str, token_count: int, answered: bool = False) -> int:
    """Estimate the tokens up to and including the last answer marker.

    The count is ``token_count`` scaled by the character position where the
    last ``{0}``, ``{1}`` or ``{2}`` ends. If there is no marker, an answered
    response counts half of ``token_count`` and an abstention counts all of it.
    """
    matches = list(_ANSWER_MARKER.finditer(response or ""))
    if not matches or not response:
        return max(1, math.ceil(token_count / 2)) if answered and token_count > 0 else token_count
    return max(1, math.ceil(token_count * matches[-1].end() / len(response)))


def _is_true(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _frames_from_bucket(bucket: str) -> int | None:
    match = re.search(r"L0*(\d+)", bucket or "")
    return int(match.group(1)) if match else None


def annotate_csv(
    src: Path,
    dst: Path,
    model: str,
    *,
    frames: int | None = None,
    width: int = 448,
    height: int = 448,
    prompt_tokens: int = VIDEO_PROMPT_TOKENS,
    candidate_frames: int = 5,
    through_answer: bool = True,
) -> int:
    """Copy a per-question CSV to ``dst`` with an added ``flops`` column."""
    with open(src, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "flops" not in fields:
        fields.append("flops")

    for row in rows:
        length = frames or _frames_from_bucket(row.get("bucket", ""))
        if length is None:
            raise ValueError(f"no stream length for row {row.get('question_id')}; pass --frames")
        t_out = int(float(row.get("response_token_count") or 0))
        if through_answer:
            answered = not _is_true(row.get("is_dont_know"))
            t_out = tokens_through_answer(row.get("response_full", ""), t_out, answered)
        if row.get("input_mode") == "video":
            clip = candidate_frames if row.get("candidate_clip_end") not in (None, "") else 0
            cost = response_flops(
                model, text_tokens=prompt_tokens, generated_tokens=t_out, frames=length,
                width=width, height=height, candidate_frames=clip,
            )
        else:
            cost = response_flops(model, text_tokens=length, generated_tokens=t_out)
        row["flops"] = f"{cost.total:.6e}"

    with open(dst, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eccbench.flops",
        description="Add an analytical FLOPs column to an eccbench-eval CSV log.",
    )
    parser.add_argument("csv", type=Path, help="per-question CSV written by eccbench-eval")
    parser.add_argument("--model", required=True, choices=sorted(MODELS), metavar="ALIAS",
                        help="registry alias, e.g. qwen_full")
    parser.add_argument("--out", type=Path, help="output CSV (default: <csv>.flops.csv)")
    parser.add_argument("--frames", type=int,
                        help="stream length L; read from the bucket column if omitted")
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--prompt-tokens", type=int, default=VIDEO_PROMPT_TOKENS,
                        help="prompt text tokens for video rows")
    parser.add_argument("--candidate-frames", type=int, default=5,
                        help="frames charged for a candidate clip on video rows")
    parser.add_argument("--all-tokens", action="store_true",
                        help="count every generated token, not only those up to the answer")
    args = parser.parse_args(argv)

    out = args.out or args.csv.with_suffix(".flops.csv")
    n = annotate_csv(
        args.csv, out, args.model, frames=args.frames, width=args.width, height=args.height,
        prompt_tokens=args.prompt_tokens, candidate_frames=args.candidate_frames,
        through_answer=not args.all_tokens,
    )
    print(f"wrote {n} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The FLOPs module reproduces the paper's Appendix B tables."""

import csv

import pytest

from eccbench import flops
from eccbench.registry import REGISTRY

# tab:prefill_decode: 64 frames (or text symbols), a three-token answer.
# (alias, text F_pre TF, text F_dec GF,
#  synthetic N, F_pre TF, F_dec GF, natural N, F_pre TF, F_dec GF)
PREFILL_DECODE = [
    ("qwen_full", 0.84, 29, 8320, 122, 36, 9696, 145, 37),
    ("qwen3_full", 0.89, 32, 6400, 101, 39, 7168, 115, 40),
    ("qwen3_omni", 0.35, 13, 6400, 51, 23, 7168, 59, 24),
    ("glm45v", 1.56, 53, 8320, 281, 90, 9696, 343, 96),
    ("internvl-3-5", 0.89, 32, 16512, 310, 51, 16512, 310, 51),
    ("internvl-3-5-30b-a3b", 0.35, 13, 16512, 197, 39, 16512, 197, 39),
    ("internvl-3-5-38b", 4.00, 130, 16512, 1316, 199, 16512, 1316, 199),
    ("mimo-vl", 0.82, 29, 8320, 127, 39, 9696, 151, 41),
    ("minicpm-4-5", 0.89, 32, 4224, 64, 36, 4224, 64, 36),
    ("minicpm", 0.84, 29, 4224, 59, 33, 4224, 59, 33),
    ("longvila", 0.84, 29, 4240, 59, 33, 4240, 59, 33),
    ("phi_multimodal", 0.41, 17, 35008, 492, 46, 35008, 492, 46),
]

# tab:flops_coeffs: B (MF/token), A (kF/pair), F_voc (MF).
COEFFS = {
    "qwen_full": (13051, 401, 1090),
    "qwen3_full": (13892, 590, 1245),
    "qwen3_omni": (5461, 786, 623),
    "glm45v": (24363, 2261, 1242),
    "internvl-3-5": (13892, 590, 1245),
    "internvl-3-5-30b-a3b": (5461, 786, 622),
    "internvl-3-5-38b": (62411, 2097, 1556),
    "mimo-vl": (12759, 590, 1243),
    "minicpm-4-5": (13892, 590, 1243),
    "minicpm": (13051, 401, 1087),
    "longvila": (13051, 401, 1087),
    "phi_multimodal": (7181, 393, 1229),
}


def test_every_alias_is_a_registry_model():
    assert set(flops.MODELS) <= set(REGISTRY)


@pytest.mark.parametrize("alias", sorted(COEFFS))
def test_coefficients(alias):
    B, A, voc = COEFFS[alias]
    assert round(flops.coeff_B(alias) / 1e6) == B
    assert round(flops.coeff_A(alias) / 1e3) == A
    assert round(flops.vocab_flops(alias) / 1e6) == voc


@pytest.mark.parametrize("row", PREFILL_DECODE, ids=[r[0] for r in PREFILL_DECODE])
def test_prefill_decode_at_64_frames(row):
    alias, t_pre, t_dec, s_n, s_pre, s_dec, n_n, n_pre, n_dec = row

    text = flops.response_flops(alias, text_tokens=64, generated_tokens=3)
    assert text.vision == text.connector == 0
    assert round(text.prefill / 1e12, 2) == t_pre
    assert round(text.decode / 1e9) == t_dec

    for (width, height), n, pre, dec in (((448, 448), s_n, s_pre, s_dec),
                                         ((640, 360), n_n, n_pre, n_dec)):
        assert 128 + flops.vision_tokens(alias, 64, width, height) == n
        video = flops.response_flops(alias, text_tokens=128, generated_tokens=3,
                                     frames=64, width=width, height=height)
        assert round(video.prefill / 1e12) == pre
        assert round(video.decode / 1e9) == dec


@pytest.mark.parametrize(
    "frames, size, candidate, tera",
    [(64, (640, 360), 0, 87), (128, (640, 360), 0, 213), (512, (640, 360), 0, 1786),
     (64, (448, 448), 5, 84)],
)
def test_full_video_cost_of_one_answer(frames, size, candidate, tera):
    """Qwen3-Omni full-video column of tab:rag (vision, connector, prefill, one token)."""
    cost = flops.response_flops("qwen3_omni", text_tokens=128, generated_tokens=1,
                                frames=frames, width=size[0], height=size[1],
                                candidate_frames=candidate)
    assert round(cost.total / 1e12) == tera


def test_tokens_through_answer():
    assert flops.tokens_through_answer("so {0} and more text", 20) < 20
    assert flops.tokens_through_answer("no marker here", 7) == 7


def test_annotate_csv(tmp_path):
    src = tmp_path / "run.csv"
    with open(src, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "question_id", "bucket", "input_mode", "response_full", "response_token_count"])
        writer.writeheader()
        writer.writerow({"question_id": "q1", "bucket": "low/L64", "input_mode": "sequence",
                         "response_full": "{0}", "response_token_count": "3"})
    dst = tmp_path / "out.csv"
    assert flops.annotate_csv(src, dst, "qwen_full", through_answer=False) == 1
    with open(dst, newline="") as handle:
        row = next(csv.DictReader(handle))
    expected = flops.response_flops("qwen_full", text_tokens=64, generated_tokens=3).total
    assert float(row["flops"]) == pytest.approx(expected, rel=1e-6)

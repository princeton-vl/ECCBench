"""Unit tests for the Qwen frame sampler path resolution and decoding."""

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "ones"):
    pytest.skip("real torch required (conftest stub active)", allow_module_level=True)
pytest.importorskip("cv2")
pytest.importorskip("qwen_vl_utils")

from eccbench.samplers.qwen_sampler import QwenFrameSampler  # noqa: E402


def test_resolve_video_path_prefers_largest_candidate(tmp_path):
    """When the canonical asset is missing, use the largest matching variant."""

    missing_path = tmp_path / "pattern_video.mp4"
    assert not missing_path.exists()

    candidate_small = tmp_path / "pattern_video_1fps_256.mp4"
    candidate_large = tmp_path / "pattern_video_2fps_384.mp4"
    candidate_small.write_bytes(b"a")
    candidate_large.write_bytes(b"ab")

    sampler = QwenFrameSampler()
    resolved = sampler._resolve_video_path(str(missing_path))

    assert resolved == str(candidate_large)


def test_sample_frames_delegates_to_decord_decoder(monkeypatch, tmp_path):
    """sample_frames resolves the path and decodes all frames via decord."""

    fake_video = tmp_path / "pattern_video_2fps_384.mp4"
    fake_video.write_text("")

    sampler = QwenFrameSampler()

    decoded_frames = torch.ones((3, 3, 4, 4), dtype=torch.uint8)
    decode_calls = []

    def fake_resolve(self, path):
        return str(fake_video)

    def fake_decode(self, video_path, max_frames, min_pixels, max_pixels):
        decode_calls.append(
            {
                "video_path": video_path,
                "max_frames": max_frames,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )
        return decoded_frames

    monkeypatch.setattr(QwenFrameSampler, "_resolve_video_path", fake_resolve)
    monkeypatch.setattr(
        QwenFrameSampler, "_decode_with_decord_all_frames", fake_decode
    )

    result = sampler.sample_frames(
        "/missing/pattern_video.mp4", fps=2, max_frames=4, max_pixels=99
    )

    assert torch.equal(result, decoded_frames)
    assert decode_calls == [
        {
            "video_path": str(fake_video),
            "max_frames": 4,
            "min_pixels": None,
            "max_pixels": 99,
        }
    ]

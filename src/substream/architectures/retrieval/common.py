"""Shared frame decoding for retrieval backends.

decode_window(path, start, end) -> list of PIL images at 1 fps, one per second
of [start, end). Videos are already 1 fps, so this is frames [start, end).
When start/end are None the whole clip is returned (used for query clips).
"""
from __future__ import annotations

import subprocess

import numpy as np
from PIL import Image

RES = 448   # decode size; backends may resize further


def decode_window(path, start=None, end=None, res: int = RES):
    sel = []
    if start is not None:
        sel += ["-ss", str(start)]
    if end is not None and start is not None:
        sel += ["-t", str(end - start)]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", *sel, "-i", str(path),
         "-vf", f"fps=1,scale={res}:{res}", "-pix_fmt", "rgb24",
         "-f", "rawvideo", "-"],
        capture_output=True, timeout=600, check=True).stdout
    fsz = res * res * 3
    n = len(raw) // fsz
    arr = np.frombuffer(raw[: n * fsz], np.uint8).reshape(n, res, res, 3)
    return [Image.fromarray(f) for f in arr]

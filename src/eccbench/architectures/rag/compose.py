#!/usr/bin/env python3
"""Compose a retrieval-augmented evaluation set from the retrieval probe outputs.

For each query, rank the video's windows by cosine similarity, merge the
touching or overlapping top-N spans, join them with a magenta SEGMENT BREAK
clip, and write ``questions_dataset.json``. With ``--full-questions`` the
questions come from the published dataset instead of the probe manifest.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from eccbench import config
from eccbench.architectures.rag import data

DEFAULT_PROBE_DIR = config.RUN_ROOT / "retrieval_probe"
DELIM_SEC = 2
VLEN = 512


def merge_spans(spans, gap=0.0):
    """Merge spans that overlap or are at most ``gap`` apart, in time order."""
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def probe_res(video: str):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video],
        capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    w, h = r.split(",")[:2]
    return int(w), int(h)


def compose(video: str, segments, dst: Path):
    """Cut each segment at 1 fps, join with magenta SEGMENT BREAK delimiters."""
    w, h = probe_res(video)
    tmp = dst.parent / f".{dst.stem}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    delim = tmp / "delim.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", f"color=c=magenta:size={w}x{h}:rate=1:duration={DELIM_SEC}",
         "-vf", ("drawtext=text='SEGMENT BREAK':fontcolor=black:"
                 f"fontsize={max(16, h // 8)}:x=(w-text_w)/2:y=(h-text_h)/2"),
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-y", str(delim)], capture_output=True, timeout=120, check=True)
    for i, (a, b) in enumerate(segments):
        seg = tmp / f"seg{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a),
             "-i", video, "-vf", "fps=1", "-an", "-c:v", "libx264",
             "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-y", str(seg)],
            capture_output=True, timeout=300, check=True)
        if i:
            parts.append(delim)
        parts.append(seg)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", "-y", str(dst)], capture_output=True, timeout=300, check=True)
    shutil.rmtree(tmp)
    n_frames = int(sum(b - a for a, b in segments) + DELIM_SEC * (len(segments) - 1))
    return n_frames


def full_question_queries(man, syn_no_per_video: int, seed: int = 0, *,
                          question_set: str = "density_topup", density_band: str = "all",
                          repo: str = data.hf_loader.DEFAULT_REPO,
                          revision: str | None = None):
    """Map each manifest video to all its questions, yes and no.

    Natural videos get every question of ``question_set`` at L = 512.
    Synthetic videos get the probe's yes questions plus ``syn_no_per_video``
    sampled absent candidates.
    """
    import random
    rng = random.Random(seed)
    nat = {it["video_key"]: it["queries"]
           for it in data.natural_items(VLEN, question_set=question_set,
                                        density_band=density_band,
                                        repo=repo, revision=revision)}
    syn = {}
    for it in data.synthetic_items(VLEN, repo=repo, revision=revision):
        no = [q for q in it["queries"] if not q["present"]]
        syn[it["video_key"]] = rng.sample(no, min(syn_no_per_video, len(no)))
    out = {}
    for e in man["entries"]:
        if e["domain"] == "natural":
            out[e["video_key"]] = nat.get(e["video_key"], [])
        else:
            yes = [dict(q, answer="yes") for q in e["queries"]]
            out[e["video_key"]] = yes + syn.get(e["video_key"], [])
    return out


class _EmbStore:
    """Embeddings from emb_<backend>.npz and emb_<backend>_extra.npz.

    With ``embed_missing``, queries without an embedding are embedded and saved.
    """

    def __init__(self, probe_dir: Path, backend, embed_missing):
        self.z = np.load(probe_dir / f"emb_{backend}.npz")
        self.extra_path = probe_dir / f"emb_{backend}_extra.npz"
        self.extra = dict(np.load(self.extra_path)) if self.extra_path.exists() else {}
        self.be = None
        if embed_missing:
            from eccbench.architectures.retrieval.qwen import Backend
            self.be = Backend()

    def get(self, key):
        if key in self.z.files:
            return self.z[key]
        return self.extra.get(key)

    def query(self, q):
        k = f'q::{q["question_id"]}'
        v = self.get(k)
        if v is not None:
            return v
        if self.be is None:
            return None
        v = (self.be.embed_text(q["text"], event=q.get("event")) if q["kind"] == "text"
             else self.be.embed_video_files([q["clip"]], batch=1)[0])
        self.extra[k] = v
        np.savez_compressed(self.extra_path, **self.extra)
        return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen", choices=["qwen"])
    ap.add_argument("--topn", type=int, default=2)
    ap.add_argument("--domain", choices=["natural", "synthetic", "both"], default="both")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap questions (0 = all)")
    ap.add_argument("--full-questions", action="store_true",
                    help="use all questions (yes and no) of the manifest's videos from the "
                         "published dataset, not only the probe's yes questions")
    ap.add_argument("--syn-no-per-video", type=int, default=2)
    ap.add_argument("--embed-missing", action="store_true",
                    help="embed queries absent from the npz (needs GPU + backend env); "
                         "saved to emb_<backend>_extra.npz for reuse")
    ap.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR,
                    help="retrieval probe state dir (default: %(default)s)")
    data.add_natural_filter_args(ap)
    data.add_dataset_args(ap)
    a = ap.parse_args()

    man = json.loads((a.probe_dir / "manifest.json").read_text())
    wins = man["windows"]
    store = _EmbStore(a.probe_dir, a.backend, a.embed_missing)
    fullq = (full_question_queries(man, a.syn_no_per_video,
                                   question_set=a.question_set, density_band=a.density_band,
                                   repo=a.repo, revision=a.revision)
             if a.full_questions else None)

    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(exist_ok=True)

    videos_json, n_done, n_skip = [], 0, 0
    frames_hist = []
    for e in man["entries"]:
        if a.domain != "both" and e["domain"] != a.domain:
            continue
        wkey = f"win::{e['video_key']}"
        W = store.get(wkey)
        queries = fullq[e["video_key"]] if fullq is not None else e["queries"]
        if W is None:
            n_skip += len(queries)
            continue
        W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
        for q in queries:
            if a.limit and n_done >= a.limit:
                break
            v = store.query(q)
            if v is None:
                n_skip += 1
                continue
            v = v / max(np.linalg.norm(v), 1e-9)
            top = np.argsort(-(W @ v))[: a.topn]
            segments = merge_spans([tuple(wins[int(i)]) for i in top])
            name = f'rag_{q["question_id"]}.mp4'.replace("/", "_")
            n_frames = compose(e["video"], segments, out / "videos" / name)
            frames_hist.append(n_frames)

            entry = dict(
                video_index=n_done + 1, variant=0,
                video_path=f"videos/{name}",
                bucket=f"rag_{a.backend}_top{a.topn}",
                duration_seconds=n_frames, actual_frames=n_frames,
                extracted_at_fps=1,
                _rag=dict(backend=a.backend, topn=a.topn,
                          source_video=e.get("hub_path", e["video"]),
                          retrieved_windows=[list(wins[int(i)]) for i in top],
                          segments=[list(s) for s in segments],
                          gold=q.get("gold")),
                question_count=1, questions=[])
            if q["kind"] == "text":
                entry["questions"].append(dict(
                    question=q["text"], question_id=f'rag_{q["question_id"]}',
                    question_time=float(n_frames), question_mode="exists",
                    question_format="binary_yes_no",
                    answer=q.get("answer", "yes"),
                    event_type_normalized=q.get("event"),
                    label_source="rag_probe", candidate={}))
            else:
                cname = Path(q["clip"]).name
                shutil.copy(q["clip"], out / "clips" / cname)
                entry["questions"].append(dict(
                    question="Did this option video appear in the main video?",
                    question_id=f'rag_{q["question_id"]}',
                    question_time=float(n_frames), question_mode="exists",
                    question_format="binary_yes_no",
                    answer=q.get("answer", "yes"),
                    candidate=dict(clip_path=f"clips/{cname}",
                                   present=q.get("answer", "yes") == "yes")))
            videos_json.append(entry)
            n_done += 1
        if a.limit and n_done >= a.limit:
            break

    ds = dict(dataset_name=f"rag_{a.backend}_top{a.topn}_{a.domain}",
              bucket_count=1, video_count=len(videos_json), videos=videos_json)
    (out / "questions_dataset.json").write_text(json.dumps(ds, indent=1))
    fh = sorted(frames_hist)
    print(f"composed {n_done} question-videos ({n_skip} skipped, embeddings missing)")
    if fh:
        print(f"frames per composed video: min={fh[0]} med={fh[len(fh)//2]} max={fh[-1]}")
    print(f"wrote {out}/questions_dataset.json")
    print(f"\nrun with (--rag discloses retrieval in the prompt):\n"
          f"  eccbench-eval --data {out}/questions_dataset.json --rag \\\n"
          f"    --asset-root {out} --mode video --variant spatial \\\n"
          f"    --model <model> --fps 1 --max-frames 64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

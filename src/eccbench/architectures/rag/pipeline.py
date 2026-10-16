#!/usr/bin/env python3
"""RAG pipeline for a whole bucket: embed windows, retrieve the top-N windows per
question, compose the retrieved clips and write questions_dataset.json.

Each stage (extract, embed, compose) skips work that is already done. ``nat64``,
``nat128`` and ``nat512`` select natural_video streams at L = 64, 128 and 512;
``syn64`` selects synthetic_video streams at L = 64 in the low and max-entropy
bands. All state and outputs live under ``--out-root``.
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
from eccbench.architectures.rag.compose import DELIM_SEC, compose, merge_spans

DEFAULT_OUT_ROOT = config.RUN_ROOT / "rag_corpus"
DEFAULT_PROBE_DIR = config.RUN_ROOT / "retrieval_probe"
WIN, STRIDE = 16, 4

# Natural video has no 256-frame bucket, hence the gap between 128 and 512.
CORPORA = {
    "nat64": dict(kind="natural", vlen=64),
    "nat128": dict(kind="natural", vlen=128),
    "nat512": dict(kind="natural", vlen=512),
    "syn64": dict(kind="synthetic", vlen=64, entropy_bands=("low", "max-entropy")),
}


def windows(vlen):
    return [(s, s + WIN) for s in range(0, vlen - WIN + 1, STRIDE)]


def corpus_items(name, opts):
    """Return (video_key, video_path, hub_path, queries) for every stream of the corpus."""
    c = CORPORA[name]
    if c["kind"] == "natural":
        src = data.natural_items(c["vlen"], question_set=opts.question_set,
                                 density_band=opts.density_band,
                                 repo=opts.repo, revision=opts.revision)
    else:
        src = data.synthetic_items(c["vlen"], entropy_bands=c["entropy_bands"],
                                   repo=opts.repo, revision=opts.revision)
    items = []
    for it in src:
        key = f'{name}_{it["video_key"]}'
        qs = [dict(q, question_id=q["question_id"] if q["kind"] == "text"
                   else f'{name}_{q["question_id"]}')
              for q in it["queries"]]
        if qs:
            items.append((key, it["video"], it["hub_path"], qs))
    return items


def stage_extract(name, opts, jobs=8):
    from concurrent.futures import ThreadPoolExecutor
    c = CORPORA[name]
    wins = windows(c["vlen"])
    tasks = []
    for key, vp, _, _ in corpus_items(name, opts):
        d = opts.out_root / "windows" / key
        d.mkdir(parents=True, exist_ok=True)
        for i, (a, b) in enumerate(wins):
            f = d / f"w{i:03d}.mp4"
            if not (f.exists() and f.stat().st_size > 0):
                tasks.append((vp, a, b, f))
    print(f"{name}: {len(tasks)} windows to extract")

    def one(t):
        src, a, b, dst = t
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a),
                        "-i", src, "-vf", "fps=1", "-an", "-c:v", "libx264",
                        "-preset", "ultrafast", "-crf", "23", "-y", str(dst)],
                       capture_output=True, timeout=300, check=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for n, _ in enumerate(pool.map(one, tasks), 1):
            if n % 500 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)
    print("extract done")


def stage_embed(name, opts, batch=4):
    from eccbench.architectures.retrieval.qwen import Backend
    c = CORPORA[name]
    wins = windows(c["vlen"])
    path = opts.out_root / f"emb_qwen_{name}.npz"
    out = dict(np.load(path)) if path.exists() else {}
    items = corpus_items(name, opts)
    probe = opts.probe_dir / "emb_qwen.npz"
    if not out and probe.exists():
        # The retrieval probe uses the same windows; reuse its embeddings.
        zp = np.load(probe)
        want_q = {q["question_id"] for _, _, _, qs in items for q in qs}
        for key, _, _, _ in items:
            pk = f'win::{key[len(name) + 1:]}'
            if pk in zp.files:
                out[f"win::{key}"] = zp[pk]
        for k in zp.files:
            if k.startswith("q::") and k[3:] in want_q:
                out[k] = zp[k]
        if out:
            print(f"seeded {len(out)} keys from the retrieval probe")
    be = Backend()
    for key, vp, _, qs in items:
        if f"win::{key}" not in out:
            files = [str(opts.out_root / "windows" / key / f"w{i:03d}.mp4")
                     for i in range(len(wins))]
            out[f"win::{key}"] = be.embed_video_files(files, batch=batch)
        for q in qs:
            qk = f'q::{q["question_id"]}'
            if qk in out:
                continue
            out[qk] = (be.embed_text(q["text"], event=q.get("event"))
                       if q["kind"] == "text"
                       else be.embed_video_files([q["clip"]], batch=1)[0])
        np.savez_compressed(path, **out)
        print(f"  embedded {key} (+{len(qs)} queries)", flush=True)
    print(f"wrote {path} ({len(out)} keys)")


def stage_compose(name, opts, topn):
    c = CORPORA[name]
    wins = windows(c["vlen"])
    z = np.load(opts.out_root / f"emb_qwen_{name}.npz")
    out = opts.out_root / f"rag_qwen_top{topn}_{name}"
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(exist_ok=True)
    videos_json, frames_hist, n_skip = [], [], 0
    for key, vp, hub_path, qs in corpus_items(name, opts):
        wkey = f"win::{key}"
        if wkey not in z.files:
            n_skip += len(qs)
            continue
        W = z[wkey]
        W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
        for q in qs:
            qk = f'q::{q["question_id"]}'
            if qk not in z.files:
                n_skip += 1
                continue
            v = z[qk] / max(np.linalg.norm(z[qk]), 1e-9)
            top = np.argsort(-(W @ v))[:topn]
            segs = merge_spans([tuple(wins[int(i)]) for i in top])
            vid = f'rag_{q["question_id"]}.mp4'.replace("/", "_")
            dst = out / "videos" / vid
            if not (dst.exists() and dst.stat().st_size > 0):
                compose(vp, segs, dst)
            n_frames = int(sum(b - a for a, b in segs) + DELIM_SEC * (len(segs) - 1))
            frames_hist.append(n_frames)
            entry = dict(video_index=len(videos_json) + 1, variant=0,
                         video_path=f"videos/{vid}",
                         bucket=f"rag_qwen_top{topn}_{name}",
                         duration_seconds=n_frames, actual_frames=n_frames,
                         extracted_at_fps=1,
                         _rag=dict(backend="qwen", topn=topn, corpus=name,
                                   source_video=hub_path,
                                   retrieved_windows=[list(wins[int(i)]) for i in top],
                                   segments=[list(s) for s in segs]),
                         question_count=1, questions=[])
            if q["kind"] == "text":
                entry["questions"].append(dict(
                    question=q["text"], question_id=f'rag_{q["question_id"]}',
                    question_time=float(n_frames), question_mode="exists",
                    question_format="binary_yes_no", answer=q["answer"],
                    event_type_normalized=q.get("event"),
                    label_source="rag", candidate={}))
            else:
                cname = Path(q["clip"]).name
                if not (out / "clips" / cname).exists():
                    shutil.copy(q["clip"], out / "clips" / cname)
                entry["questions"].append(dict(
                    question="Did this option video appear in the main video?",
                    question_id=f'rag_{q["question_id"]}',
                    question_time=float(n_frames), question_mode="exists",
                    question_format="binary_yes_no", answer=q["answer"],
                    candidate=dict(clip_path=f"clips/{cname}",
                                   present=q["answer"] == "yes")))
            videos_json.append(entry)
    ds = dict(dataset_name=f"rag_qwen_top{topn}_{name}", bucket_count=1,
              video_count=len(videos_json), videos=videos_json)
    (out / "questions_dataset.json").write_text(json.dumps(ds, indent=1))
    fh = sorted(frames_hist)
    ans = [e["questions"][0]["answer"] for e in videos_json]
    print(f"{name}: composed {len(videos_json)} question-videos ({n_skip} skipped), "
          f"yes={ans.count('yes')} no={ans.count('no')}")
    if fh:
        print(f"  frames/video: min={fh[0]} med={fh[len(fh)//2]} max={fh[-1]} "
              f"(original {c['vlen']})")
    print(f"  wrote {out}/questions_dataset.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["extract", "embed", "compose", "plan"])
    ap.add_argument("--corpus", required=True, choices=list(CORPORA))
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help="windows, embeddings and composed sets go here (default: %(default)s)")
    ap.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR,
                    help="retrieval probe state dir whose embeddings seed `embed` "
                         "(default: %(default)s)")
    data.add_natural_filter_args(ap)
    data.add_dataset_args(ap)
    a = ap.parse_args()
    if a.stage == "plan":
        items = corpus_items(a.corpus, a)
        nw = len(windows(CORPORA[a.corpus]["vlen"]))
        nq = sum(len(q) for _, _, _, q in items)
        ans = [q["answer"] for _, _, _, qs in items for q in qs]
        print(f"{a.corpus}: {len(items)} videos x {nw} windows = {len(items)*nw} "
              f"window embeds; {nq} queries (yes={ans.count('yes')} no={ans.count('no')})")
    elif a.stage == "extract":
        stage_extract(a.corpus, a, a.jobs)
    elif a.stage == "embed":
        stage_embed(a.corpus, a, a.batch)
    else:
        stage_compose(a.corpus, a, a.topn)

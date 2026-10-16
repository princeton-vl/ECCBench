#!/usr/bin/env python3
"""Retrieval probe: measure how often a video embedder ranks a window that contains
the queried event first, among the 16 s windows at stride 4 of an L = 512 video.

Natural queries are the designed yes-questions of natural_video, with the gold
span from ``event_clip_start`` and ``event_clip_end``. Synthetic queries are
present candidate clips of synthetic_video, with the gold span from the
candidate's ``clip_start`` and ``clip_end``. A window is correct if it fully
contains a gold span. All state lives under ``--out``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from eccbench import config
from eccbench.architectures.rag import data

DEFAULT_OUT = config.RUN_ROOT / "retrieval_probe"

WIN, STRIDE, VLEN = 16, 4, 512
N_VIDEOS, Q_PER_VIDEO = 32, 2


def windows():
    return [(s, s + WIN) for s in range(0, VLEN - WIN + 1, STRIDE)]


def build(out: Path, seed: int, repo: str, revision: str | None) -> None:
    rng = random.Random(seed)
    entries = []

    # Natural: text queries. Keep events short enough to fit in one window.
    per_video = []
    for it in data.natural_items(VLEN, question_set="designed", repo=repo, revision=revision):
        if it["event"] is None:
            continue
        a, b = max(0.0, it["event"][0]), min(float(VLEN), it["event"][1])
        if not 0 < b - a <= WIN:
            continue
        qs = [dict(q, gold=[(a, b)]) for q in it["queries"] if q["answer"] == "yes"]
        if qs:
            per_video.append((it, qs))

    rng.shuffle(per_video)
    per_video.sort(key=lambda t: -min(len(t[1]), Q_PER_VIDEO))
    n_full = sum(1 for _, qs in per_video if len(qs) >= Q_PER_VIDEO)
    for it, qs in per_video[:N_VIDEOS]:
        take = rng.sample(qs, min(Q_PER_VIDEO, len(qs)))
        entries.append(dict(domain="natural", video=it["video"], hub_path=it["hub_path"],
                            video_key=it["video_key"], queries=take))
    n_nat_q = sum(len(e["queries"]) for e in entries)
    print(f"natural: {len(entries)} videos ({n_full} had >= {Q_PER_VIDEO} usable queries), "
          f"{n_nat_q} queries")

    # Synthetic: candidate-clip queries.
    svids = data.synthetic_items(VLEN, repo=repo, revision=revision)
    rng.shuffle(svids)
    n_syn = 0
    for it in svids[:N_VIDEOS]:
        yes = [q for q in it["queries"]
               if q["present"] and (q["gold"][0][1] - q["gold"][0][0]) <= WIN]
        take = rng.sample(yes, min(Q_PER_VIDEO, len(yes)))
        n_syn += len(take)
        entries.append(dict(domain="synthetic", video=it["video"], hub_path=it["hub_path"],
                            video_key=it["video_key"], queries=take))
    print(f"synthetic: {min(N_VIDEOS, len(svids))} videos, {n_syn} queries")

    out.mkdir(parents=True, exist_ok=True)
    man = dict(win=WIN, stride=STRIDE, vlen=VLEN, windows=windows(), entries=entries)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {out}/manifest.json  "
          f"({len(entries)} videos, {sum(len(e['queries']) for e in entries)} queries, "
          f"{len(entries) * len(windows())} window embeddings per backend)")


def _win_dir(out: Path):
    d = out / "windows"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract(out: Path, jobs: int) -> None:
    """Write every window as an mp4, skipping files that exist.

    Windows are re-encoded because stream-copy cuts need not land on keyframes.
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    man = json.loads((out / "manifest.json").read_text())
    wins = man["windows"]
    tasks = []
    for e in man["entries"]:
        vd = _win_dir(out) / e["video_key"]
        vd.mkdir(exist_ok=True)
        for i, (a, b) in enumerate(wins):
            f = vd / f"w{i:03d}.mp4"
            if not (f.exists() and f.stat().st_size > 0):
                tasks.append((e["video"], a, b, f))
    print(f"windows to extract: {len(tasks)} (rest already on disk)")

    def one(t):
        src, a, b, dst = t
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a),
                        "-i", src, "-vf", "fps=1", "-an",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                        "-y", str(dst)], capture_output=True, timeout=300, check=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for n, _ in enumerate(pool.map(one, tasks), 1):
            if n % 500 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)
    print("extract done")


def embed(out: Path, backend_name: str, batch: int) -> None:
    man = json.loads((out / "manifest.json").read_text())
    if backend_name == "dummy":
        from eccbench.architectures.retrieval.dummy import Backend
    elif backend_name == "qwen":
        from eccbench.architectures.retrieval.qwen import Backend
    else:
        raise SystemExit(f"unknown backend {backend_name}")
    be = Backend()

    n_win = len(man["windows"])
    out_path = out / f"emb_{backend_name}.npz"
    z = dict(np.load(out_path)) if out_path.exists() else {}
    done0 = len(z)
    for e in man["entries"]:
        key = e["video_key"]
        if f"win::{key}" not in z:
            files = [str(_win_dir(out) / key / f"w{i:03d}.mp4") for i in range(n_win)]
            z[f"win::{key}"] = be.embed_video_files(files, batch=batch)
        for q in e["queries"]:
            qk = f'q::{q["question_id"]}'
            if qk in z:
                continue
            z[qk] = (be.embed_text(q["text"], event=q.get("event"))
                     if q["kind"] == "text"
                     else be.embed_video_files([q["clip"]], batch=1)[0])
        np.savez_compressed(out_path, **z)
        print(f"  embedded {key} ({len(e['queries'])} queries)", flush=True)
    print(f"wrote {out_path} ({len(z)} keys, {done0} pre-existing)")


def score(out: Path, backends: list[str], seed: int) -> None:
    man = json.loads((out / "manifest.json").read_text())
    wins = man["windows"]

    def correct_set(gold):
        return {i for i, (ws, we) in enumerate(wins)
                if any(ws <= a and b <= we for a, b in gold)}

    def points(rank):
        return 10.0 / (2 ** (rank - 1)) if rank <= 10 else 0.0

    rows = []
    for bk in backends:
        z = np.load(out / f"emb_{bk}.npz")
        for e in man["entries"]:
            W = z[f"win::{e['video_key']}"]
            W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
            for q in e["queries"]:
                v = z[f'q::{q["question_id"]}']
                v = v / max(np.linalg.norm(v), 1e-9)
                order = np.argsort(-(W @ v))
                cs = correct_set(q["gold"])
                rank = next((i + 1 for i, w in enumerate(order) if int(w) in cs), 999)
                rows.append(dict(backend=bk, domain=e["domain"],
                                 qid=q["question_id"], n_correct=len(cs),
                                 rank=rank, score=points(rank)))
    import pandas as pd
    df = pd.DataFrame(rows)

    # Random baseline: the same number of correct windows, in a shuffled ranking.
    rng = np.random.default_rng(seed)
    base = []
    one = df[df.backend == backends[0]]
    for _, r in one.iterrows():
        for _ in range(200):
            order = rng.permutation(len(wins))
            rank = next((i + 1 for i, w in enumerate(order) if w < r.n_correct), 999)
            base.append(dict(backend="RANDOM", domain=r.domain, rank=rank,
                             score=points(rank)))
    df = pd.concat([df, pd.DataFrame(base)], ignore_index=True)

    g = df.groupby(["backend", "domain"]).agg(
        n=("rank", "size"), mean_score=("score", "mean"),
        hit1=("rank", lambda r: (r == 1).mean()),
        hit3=("rank", lambda r: (r <= 3).mean()),
        hit10=("rank", lambda r: (r <= 10).mean()),
        med_rank=("rank", "median")).round(3)
    print(g.to_string())
    df[df.backend != "RANDOM"].to_csv(out / "scores.csv", index=False)
    print(f"\nwrote {out}/scores.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="probe state dir: manifest, windows, embeddings, scores "
                         "(default: %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=0)
    data.add_dataset_args(b)
    x = sub.add_parser("extract")
    x.add_argument("--jobs", type=int, default=8)
    e = sub.add_parser("embed")
    e.add_argument("--backend", required=True, choices=["qwen", "dummy"])
    e.add_argument("--batch", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("--backends", nargs="+", required=True)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.seed, a.repo, a.revision)
    elif a.cmd == "extract":
        extract(a.out, a.jobs)
    elif a.cmd == "embed":
        embed(a.out, a.backend, a.batch)
    else:
        score(a.out, a.backends, a.seed)

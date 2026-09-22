"""Stage-1 frozen sentence embeddings, cached by text hash. Runs in its OWN env, not the project .venv.

Environment (built once; never install torch into the uv-managed project .venv):

    uv venv --python 3.12 ~/venvs/emb
    uv pip install --python ~/venvs/emb/bin/python torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python ~/venvs/emb/bin/python sentence-transformers numpy

Run (after stage1_build.py wrote data/eval/pipeline/stage1/texts.jsonl):

    OMP_NUM_THREADS=6 ~/venvs/emb/bin/python scripts/eval/pipeline/stage1_embed.py

Model: `intfloat/multilingual-e5-small` (118M params, 384-d, 12 layers x 384 hidden),
FROZEN -- inference only, no fine-tuning. Chosen over `jhgan/ko-sroberta-multitask`
(768 hidden, ~3x the CPU cost per token) because this box is shared and CPU-only
and ~180k texts must be embedded; e5 was trained on large multilingual (incl.
Korean) weakly-supervised pairs and is a standard strong frozen feature
extractor. E5 requires an input prefix: "query: " for symmetric / non-retrieval
use (classification, regression features), which is what this is. Embeddings
are L2-normalised. Max 256 tokens (a title + a <=103-char Toss summary fits).

Cache (append-only, keyed by sha1 of the exact model text WITHOUT the prefix --
`stage1_text.text_hash`):
    data/eval/pipeline/stage1/emb/<model-slug>/hashes.txt   one hash per line
    data/eval/pipeline/stage1/emb/<model-slug>/emb_NNNN.npy  float32 [n, 384] chunks, in hash order
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL = "intfloat/multilingual-e5-small"
PREFIX = "query: "
TEXTS = REPO / "data/eval/pipeline/stage1/texts.jsonl"


def cache_dir(model: str = MODEL) -> Path:
    return REPO / "data/eval/pipeline/stage1/emb" / model.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--texts", default=str(TEXTS))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=20000, help="texts per saved .npy chunk")
    ap.add_argument("--limit", type=int, default=None, help="embed at most N new texts (timing)")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(min(args.threads, 6))
    out = cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    hfile = out / "hashes.txt"
    have = set(hfile.read_text().split()) if hfile.exists() else set()
    todo = []
    seen = set()
    with open(args.texts, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["hash"] not in have and r["hash"] not in seen:
                seen.add(r["hash"])
                todo.append((r["hash"], r["text"]))
    # longest-first batches waste less padding
    todo.sort(key=lambda x: len(x[1]))
    if args.limit:
        todo = todo[: args.limit]
    print(f"cached {len(have)}, to embed {len(todo)}", flush=True)
    if not todo:
        return
    model = SentenceTransformer(MODEL, device="cpu")
    model.max_seq_length = 256
    k = len(list(out.glob("emb_*.npy")))
    t0 = time.time()
    for s in range(0, len(todo), args.chunk):
        part = todo[s : s + args.chunk]
        emb = model.encode(
            [PREFIX + t for _, t in part],
            batch_size=args.batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        # write the chunk first, then its hashes: a crash leaves at worst an
        # orphan chunk, never hashes without vectors
        np.save(out / f"emb_{k:04d}.npy", emb)
        with hfile.open("a") as f:
            f.write("".join(h + "\n" for h, _ in part))
        k += 1
        done = s + len(part)
        rate = done / (time.time() - t0)
        print(
            f"{done}/{len(todo)}  {rate:.1f} texts/s  eta {(len(todo) - done) / rate / 60:.1f} min",
            flush=True,
        )
    (out / "meta.json").write_text(
        json.dumps({"model": MODEL, "prefix": PREFIX, "normalized": True, "max_seq_length": 256})
    )


def load_cache(model: str = MODEL) -> tuple[dict[str, int], np.ndarray]:
    """{hash: row}, matrix. Used by stage1_select (project .venv, numpy only)."""
    out = cache_dir(model)
    hfile = out / "hashes.txt"
    if not hfile.exists():
        raise FileNotFoundError(out)
    hashes = hfile.read_text().split()
    mats = [np.load(p) for p in sorted(out.glob("emb_*.npy"))]
    emb = np.concatenate(mats) if mats else np.zeros((0, 384), np.float32)
    if len(hashes) != len(emb):
        raise RuntimeError(
            f"embedding cache {out} inconsistent: {len(hashes)} hashes, {len(emb)} rows"
        )
    return {h: i for i, h in enumerate(hashes)}, emb


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    sys.exit(main())

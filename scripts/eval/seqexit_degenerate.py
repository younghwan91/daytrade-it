"""Is the learned exit policy anything more than "cut the first minute below entry"?

    .venv/bin/python scripts/eval/seqexit_degenerate.py \
        --cache data/eval/exitseq/nn_book_fresh.pkl \
        --windows data/eval/exitseq/FRESH_CONFIRM_WINDOW.json

Two neural arms reached the same inner-holdout score to two decimals, which a
continuous policy would not do by accident unless both collapsed onto the same
discrete rule. The candidate is the simplest rule the action space allows: act
at every legal step, i.e. cut at the first minute the position is below its
entry and never add. This scores that rule on every window the study reports,
beside hold-to-close, so the learned arms can be read against it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/eval")

from seqexit_core import LATTICES, NOTIONAL, Sim  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    args = ap.parse_args()

    with args.cache.open("rb") as f:
        book, cost, _ = pickle.load(f)
    sim = Sim(book, LATTICES["novol"], cost)
    win = json.loads(args.windows.read_text())["track_d_fresh"]
    days = book.day
    confirm = np.isin(days, win["confirm"])
    lo, hi = win["search"]
    search = np.array([lo <= d <= hi for d in days.tolist()])
    months = sorted({d[:7] for d in days[search].tolist()})

    cut_first = sim.cut_legal.copy()
    rules = {
        "hold": np.zeros(book.n_steps, dtype=bool),
        "cut_first_below_entry": cut_first,
    }
    windows = {"search-OOS (folds)": search & np.isin([d[:7] for d in days], months[1:])}
    windows[f"inner holdout {months[-1]}"] = search & np.array(
        [d[:7] == months[-1] for d in days.tolist()]
    )
    windows["confirm"] = confirm
    print(f"{'rule':24s} " + " ".join(f"{w:>24s}" for w in windows))
    for name, fires in rules.items():
        net, exit_at, _ = sim.run_steps(fires)
        cut = exit_at < book.last_idx
        cells = [
            f"{net[m].mean() / NOTIONAL * 1e4:+8.2f} (cut {cut[m].mean():.3f})"
            for m in windows.values()
        ]
        print(f"{name:24s} " + " ".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()

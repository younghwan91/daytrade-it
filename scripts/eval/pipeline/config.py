"""The pipeline's one shared configuration, and the guard that makes the freeze real.

Every stage imports `load()` instead of defining its own dates. A holdout that
lives in each script's constants gets redefined, drifts, and is eventually
read by accident; one that lives here and is checked at the point where labels
are loaded cannot be.

The guard is about LABELS, not rows. Stage 4 needs stage 1's score for a
holdout pair, and computing a score only reads decision-time features. What no
stage before 4 may do is look at what happened after `decide_at` on a holdout
day -- a return, an MFE, an exit outcome -- because that is the information the
holdout exists to keep unseen.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import cache
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "pipeline_config.json"


class HoldoutViolation(RuntimeError):
    """A stage before 4 tried to load an outcome on a frozen holdout day."""


@cache
def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def holdout_days() -> frozenset[str]:
    return frozenset(load()["frozen_holdout"]["days"])


def assert_no_holdout_labels(days: Iterable[str], stage: int) -> None:
    """Raise if a stage before 4 is about to read outcomes for a holdout day.

    Call it on the day set of every label/outcome file right after loading and
    before anything is computed from it.
    """
    if stage >= 4:
        return
    hit = sorted(set(days) & holdout_days())
    if hit:
        raise HoldoutViolation(
            f"stage {stage} loaded outcomes for {len(hit)} frozen holdout day(s), "
            f"first {hit[0]} -- drop them before use (see pipeline_config.json)"
        )


def drop_holdout(rows: list[dict], stage: int, day_key: str = "day") -> list[dict]:
    """The rows outside the holdout, with the guard applied to what remains."""
    held = holdout_days()
    kept = [r for r in rows if r[day_key] not in held]
    assert_no_holdout_labels((r[day_key] for r in kept), stage)
    return kept

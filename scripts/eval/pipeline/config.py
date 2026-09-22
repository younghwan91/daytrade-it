"""The pipeline's one shared configuration, and the guard that makes the freeze real.

Every stage imports `load()` instead of defining its own dates. A holdout that
lives in each script's constants gets redefined, drifts, and is eventually
read by accident; one that lives here and is checked at the point where labels
are loaded cannot be.

The holdout is a FORWARD rule, not a date list: every trading day on or after
`frozen_holdout.start`. The first holdout (a fixed 31-day list) was consumed by
an exit-only test before stage 4 existed; a window that is still accruing is
the only kind nobody can have scored yet. The consumed window is kept in
`consumed_windows`, with the decision on what stages 1-3 may do with it.

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


def holdout_start() -> str:
    return load()["frozen_holdout"]["start"]


def is_holdout(day: str) -> bool:
    """True for every day in the forward holdout (on or after its start)."""
    return day >= holdout_start()


def unlabelled_split(day: str) -> str:
    """The split of a row that is scored but carries no label.

    'holdout' on or after the forward start; otherwise 'post_train' -- scored by
    the final model from features only, a valid out-of-fold score for stages 2-3.
    """
    return "holdout" if is_holdout(day) else "post_train"


def consumed_ranges() -> list[tuple[str, str]]:
    return [tuple(w["range"]) for w in load().get("consumed_windows", [])]


def price_path_files() -> list[Path]:
    """The stage 2-3 price-path books, in date order. Raises if one is missing."""
    books = sorted(load()["price_paths"]["books"], key=lambda b: b["range"][0])
    files = [Path(b["file"]) for b in books]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"price-path book(s) missing: {missing} (see pipeline_config.json)")
    return files


def assert_no_holdout_labels(days: Iterable[str], stage: int) -> None:
    """Raise if a stage before 4 is about to read outcomes for a holdout day.

    Call it on the day set of every label/outcome file right after loading and
    before anything is computed from it.
    """
    if stage >= 4:
        return
    hit = sorted(d for d in set(days) if is_holdout(d))
    if hit:
        raise HoldoutViolation(
            f"stage {stage} loaded outcomes for {len(hit)} frozen holdout day(s), "
            f"first {hit[0]} -- the forward holdout starts {holdout_start()} "
            f"(see pipeline_config.json)"
        )


def drop_holdout(rows: list[dict], stage: int, day_key: str = "day") -> list[dict]:
    """The rows outside the holdout, with the guard applied to what remains."""
    kept = [r for r in rows if not is_holdout(r[day_key])]
    assert_no_holdout_labels((r[day_key] for r in kept), stage)
    return kept


def stage4_status(eligible_days: Iterable[str]) -> dict:
    """Whether stage 4 may run, given the eligible forward days that exist.

    Eligibility (news collected for the full universe, bar coverage) needs data
    checks, so the caller supplies the days; this only applies the rule.
    """
    h = load()["frozen_holdout"]
    days = sorted(d for d in set(eligible_days) if is_holdout(d))
    return {
        "eligible": len(days),
        "target": h["target_trading_days"],
        "ready": len(days) >= h["target_trading_days"],
        "first": days[0] if days else None,
        "last": days[-1] if days else None,
    }

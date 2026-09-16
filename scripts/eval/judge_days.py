"""Turn a dollar top-up into **whole judged days**, most recent first.

    uv run python scripts/eval/judge_days.py --budget-usd 10 \\
        --summary data/eval/universe_pairs_n300_summary.csv \\
        --pairs data/eval/universe_pairs_n300_prefiltered.jsonl \\
        --state-dir data/eval/judge_days_state

Why this exists: the first judging wave spent its whole budget on a
**stratified sample over `month,sector,session`**, which spread 97,721
judgments evenly over 125 days and finished none of them. Nothing downstream
(`build_universe_events` -> `replay_baseline` -> `optimize_policy` ->
`bandit_*`) can use a 30%-judged day, so that money bought zero usable days.
`judge_batch --days` fixes the aim; this script decides *which* days, enforces
the cap and proves each day actually finished.

Contract:

* `--budget-usd X` is a **hard cap**. Days are taken most-recent-first while the
  cumulative *estimate* fits; each day is then handed at most what is left, so
  `judge_batch`'s own `BudgetExceeded` guard is the second line of defence.
  The run aborts (`BudgetOverrun`) rather than starting a day it cannot pay for.
* It **refuses to start** unless one tiny Anthropic probe succeeds. An exhausted
  credit balance is the failure mode that actually happens here, and it returns
  a 400 that `judge_batch` would otherwise burn its retry budget on.
* Every day is **verified**: the day is complete only when >= `--min-judged`
  (default 0.99) of its prefiltered pairs have a valid cache row. The first day
  that cannot be finished stops the run -- a half-finished day is exactly the
  outcome this script exists to prevent, and continuing would spend the rest of
  the budget producing more of them.

Determinism and idempotence: each day gets its own `judge_batch` state file
(`--state-dir/<day>.json`), so a re-run resumes that day instead of re-sampling
it, and days already at >= the threshold cost nothing and are reported as
complete without a call.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_batch import CACHE, DEFAULT_MODEL, cache_row_valid  # noqa: E402

#: Columns the summary CSV must carry (written by `pairs_day_summary.py`).
COLUMNS = ("day", "pairs_total", "pairs_kept", "already_judged", "missing", "est_cost_usd")

#: A day counts as usable downstream at this judged fraction.
MIN_JUDGED = 0.99

#: Head-room over a day's estimate, so a slightly-off token estimate does not
#: abort the day -- still clamped to the remaining budget.
DAY_MARGIN = 1.10


class ProbeFailed(RuntimeError):
    """The Anthropic account cannot pay: do not start."""


class BudgetOverrun(RuntimeError):
    """A day reported back more spend than the cap allows: stop immediately."""


@dataclass(frozen=True)
class DayRow:
    """One row of the per-day summary."""

    day: str
    pairs_total: int
    pairs_kept: int
    already_judged: int
    missing: int
    est_cost_usd: float


# --------------------------------------------------------------------------
# Pure logic
# --------------------------------------------------------------------------


def load_summary(path: Path) -> list[DayRow]:
    """Read the per-day summary CSV, **most recent day first**."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for column in COLUMNS:
            if column not in fields:
                raise ValueError(f"{path}: missing column {column!r} (have {fields})")
        rows = [
            DayRow(
                day=r["day"],
                pairs_total=int(r["pairs_total"]),
                pairs_kept=int(r["pairs_kept"]),
                already_judged=int(r["already_judged"]),
                missing=int(r["missing"]),
                est_cost_usd=float(r["est_cost_usd"]),
            )
            for r in reader
        ]
    return sorted(rows, key=lambda r: r.day, reverse=True)


def pick_days(rows: Iterable[DayRow], budget_usd: float) -> tuple[list[DayRow], float]:
    """The most recent days whose cumulative estimate fits inside ``budget_usd``.

    The walk **stops** at the first day that does not fit rather than skipping
    it, so the selection is a contiguous most-recent window -- which is what
    LODO and the replay want. A day that is already finished (``missing == 0``)
    is free, is always taken, and consumes no budget.
    """
    if budget_usd < 0:
        raise ValueError(f"budget must be >= 0, got {budget_usd}")
    picked: list[DayRow] = []
    total = 0.0
    for r in rows:
        if r.missing == 0:
            picked.append(r)
            continue
        if total + r.est_cost_usd > budget_usd + 1e-9:
            break
        picked.append(r)
        total += r.est_cost_usd
    return picked, total


def format_table(results: list[dict]) -> str:
    """The completed-days table printed at the end of a run."""
    head = f"{'day':<12}{'judged':>9}{'kept':>9}{'frac':>8}{'spent$':>9}  status"
    lines = [head, "-" * len(head)]
    for r in results:
        status = "complete" if r["complete"] else (r.get("error") or "INCOMPLETE")
        lines.append(
            f"{r['day']:<12}{r['judged']:>9,}{r['kept']:>9,}"
            f"{r['judged_frac']:>8.3f}{r['spent_usd']:>9.2f}  {status}"
        )
    complete = sum(1 for r in results if r["complete"])
    spent = sum(r["spent_usd"] for r in results)
    lines += [
        "-" * len(head),
        f"{complete} complete / {len(results)} attempted, spent ${spent:.2f}",
    ]
    return "\n".join(lines)


def run_days(
    rows: Iterable[DayRow],
    *,
    budget_usd: float,
    judge: Callable[[str, float], dict],
    verify: Callable[[str], tuple[int, int]],
    probe: Callable[[], None],
    min_judged: float = MIN_JUDGED,
    log: Callable[[str], None] = lambda _: None,
) -> list[dict]:
    """Judge the affordable days to completion, newest first, under a hard cap.

    ``judge(day, budget_usd)`` runs one day and returns a `judge_batch` summary
    (its ``estimate.usd`` is the spend charged against the cap).
    ``verify(day)`` returns ``(judged, kept)`` for that day.
    ``probe()`` raises :class:`ProbeFailed` when the account cannot pay.
    """
    picked, planned = pick_days(rows, budget_usd)
    if not picked:
        log(f"nothing affordable inside ${budget_usd:.2f}")
        return []
    log(f"{len(picked)} day(s) planned, estimate ${planned:.2f} of ${budget_usd:.2f}")

    probe()  # never before this point: a failed probe must judge nothing

    results: list[dict] = []
    spent = 0.0
    for r in picked:
        error: str | None = None
        day_spend = 0.0
        if r.missing:
            remaining = budget_usd - spent
            day_budget = min(r.est_cost_usd * DAY_MARGIN, remaining)
            if day_budget <= 0:
                log(f"{r.day}: budget exhausted -- stopping")
                break
            try:
                summary = judge(r.day, day_budget)
            except Exception as exc:  # a subprocess failure is a result, not a crash
                error = str(exc)
                summary = {}
            day_spend = float(summary.get("estimate", {}).get("usd", 0.0))
            spent += day_spend
            if spent > budget_usd + 1e-9:
                results.append(_result(r.day, 0, r.pairs_kept, day_spend, False, error))
                raise BudgetOverrun(
                    f"{r.day} pushed spend to ${spent:.2f} over the ${budget_usd:.2f} cap"
                )
        judged, kept = verify(r.day)
        frac = judged / kept if kept else 1.0
        complete = error is None and frac >= min_judged
        results.append(_result(r.day, judged, kept, day_spend, complete, error))
        log(f"{r.day}: {judged}/{kept} = {frac:.3f} judged, ${day_spend:.2f}")
        if not complete:
            log(f"{r.day}: did not reach {min_judged:.0%} -- stopping")
            break
    return results


def _result(
    day: str, judged: int, kept: int, spent: float, complete: bool, error: str | None
) -> dict:
    return {
        "day": day,
        "judged": judged,
        "kept": kept,
        "judged_frac": judged / kept if kept else 1.0,
        "spent_usd": spent,
        "complete": complete,
        "error": error,
    }


# --------------------------------------------------------------------------
# Real layers (never imported by the tests)
# --------------------------------------------------------------------------


def probe_balance(model: str = DEFAULT_MODEL) -> None:
    """One tiny Messages call (``max_tokens=1``) to prove the account can pay.

    Costs a fraction of a cent. Raises :class:`ProbeFailed` on anything that is
    not a 200, which is exactly what an exhausted credit balance produces
    (400 ``invalid_request_error``: "Your credit balance is too low").
    """
    import httpx  # noqa: PLC0415
    from judge_batch import _api_key  # noqa: PLC0415

    key = _api_key()
    if not key:
        raise ProbeFailed("no ANTHROPIC_API_KEY")
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise ProbeFailed(f"probe request failed: {exc}") from exc
    if resp.status_code != 200:
        raise ProbeFailed(f"probe HTTP {resp.status_code}: {resp.text[:300]}")


def load_pairs_by_day(pairs: Path) -> dict[str, list[dict]]:
    """The prefiltered pairs file, grouped by `day` (the judging denominator)."""
    by_day: dict[str, list[dict]] = {}
    with pairs.open() as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                by_day.setdefault(str(p["day"]), []).append(p)
    return by_day


def cache_keys(cache: Path) -> dict[str, dict]:
    if not cache.exists():
        return {}
    rows: dict[str, dict] = {}
    with cache.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r
    return rows


def make_verifier(
    by_day: dict[str, list[dict]], cache: Path, model: str
) -> Callable[[str], tuple[int, int]]:
    """`(judged, kept)` for a day, re-reading the cache each time (it grows)."""

    def verify(day: str) -> tuple[int, int]:
        pairs = by_day.get(day, [])
        rows = cache_keys(cache)
        judged = sum(1 for p in pairs if cache_row_valid(rows.get(p["key"], {}), p, model))
        return judged, len(pairs)

    return verify


def make_judge(args: argparse.Namespace) -> Callable[[str, float], dict]:
    """Run one day through `judge_batch --days <day>` as a subprocess."""
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "judge_batch.py"

    def judge(day: str, budget_usd: float) -> dict:
        cmd = [
            sys.executable,
            str(script),
            "--pairs",
            str(args.pairs),
            "--days",
            day,
            "--sample",
            str(args.sample),
            "--budget-usd",
            f"{budget_usd:.4f}",
            "--model",
            args.model,
            "--cache",
            str(args.cache),
            "--state",
            str(state_dir / f"{day}.json"),
            "--chunk",
            str(args.chunk),
            "--chars-per-token",
            str(args.chars_per_token),
            "--est-output-tokens",
            str(args.est_output_tokens),
        ]
        if args.max_submit is not None:
            cmd += ["--max-submit", str(args.max_submit)]
        print(f"$ {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"judge_batch failed on {day} (exit {proc.returncode})")
        try:  # the summary is the last JSON object on stdout
            start = proc.stdout.rindex("{\n")
            return dict(json.loads(proc.stdout[start:]))
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"judge_batch on {day}: unreadable summary ({exc})") from exc

    return judge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--budget-usd", type=float, required=True, help="hard cap for the whole run")
    ap.add_argument("--summary", type=Path, required=True, help="per-day summary CSV")
    ap.add_argument("--pairs", type=Path, required=True, help="prefiltered pairs jsonl")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--state-dir", type=Path, default=Path("data/eval/judge_days_state"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--min-judged", type=float, default=MIN_JUDGED)
    ap.add_argument(
        "--sample", type=int, default=10**9, help="cap on pairs per day (all by default)"
    )
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--chars-per-token", type=float, default=1.321)
    ap.add_argument("--est-output-tokens", type=int, default=68)
    ap.add_argument("--max-submit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the result table as JSON")
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="print the days the budget buys and exit -- no probe, no spend",
    )
    args = ap.parse_args()

    rows = load_summary(args.summary)
    if args.plan_only:
        picked, total = pick_days(rows, args.budget_usd)
        for r in picked:
            print(f"{r.day}  missing {r.missing:>6,}  ${r.est_cost_usd:.3f}")
        print(f"{len(picked)} day(s), estimate ${total:.2f} of ${args.budget_usd:.2f}")
        return

    by_day = load_pairs_by_day(args.pairs)
    results = run_days(
        rows,
        budget_usd=args.budget_usd,
        judge=make_judge(args),
        verify=make_verifier(by_day, args.cache, args.model),
        probe=lambda: probe_balance(args.model),
        min_judged=args.min_judged,
        log=lambda m: print(m, flush=True),
    )
    print(format_table(results))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    if not results or not all(r["complete"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

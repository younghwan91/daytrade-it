"""Monthly, time-ordered folds for the entry bandit — the spec's readiness check.

    uv run python scripts/eval/bandit_monthly.py --events data/eval/n300_t30/events \\
        --dataset data/eval/n300_t30/bandit/dataset.jsonl --out data/eval/n300_t30/bandit/monthly

Spec `docs/superpowers/specs/2026-09-16-news-universe-a-plan-design.md`
("밴딧 준비 완료 판정") asks for **시간순 폴드(월 단위)** — monthly folds in time
order. `bandit_lodo.py` answers a different question: its 40 leave-one-day-out
folds share 39 of 40 training days, so they are one model forty times over and
their agreement is close to a tautology (A6 Addendum H). This script is the
measurement the spec actually asks for.

Fold m (every month but the first): train on every trainable row from a day
**strictly before** month m, replay each of month m's days with that model.
Nothing at or after the evaluated day is ever used to train it.

Reports, per the spec's first two criteria:

1. ``lambda_stable`` — the chosen λ across folds spans at most one step of
   ``bandit_lodo.LAMS``.
2. ``signs_consistent`` — every main feature (score, session, pre-article move)
   keeps one sign across folds.

Criterion 3 is `bayes_judge` over the pooled out-of-sample days. No Anthropic
call; the DB is read only.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import bandit_lodo as bl
import optimize_policy as op

from daytrade_it.application.trading.policy import NewsPolicy

OUT = Path("data/eval/bandit/monthly")
WORK = Path("data/eval/bandit_monthly_work")
#: The spec names these as the features whose sign must hold: score, session,
#: and the move before the article.
MAIN_FEATURES = ("score", "premarket", "pre_entry_move")

__all__ = ["MAIN_FEATURES", "folds_by_month", "lambda_stable", "run", "sign_table"]


def folds_by_month(days: list[dt.date]) -> list[tuple[str, list[dt.date]]]:
    """(month, its days) in time order, dropping the first month — it has no past."""
    by_month: dict[str, list[dt.date]] = collections.defaultdict(list)
    for d in sorted(days):
        by_month[f"{d.year:04d}-{d.month:02d}"].append(d)
    months = sorted(by_month)
    return [(m, by_month[m]) for m in months[1:]]


def lambda_stable(lams: list[float]) -> bool:
    """True when the chosen λ values span at most one step of `bandit_lodo.LAMS`."""
    if not lams:
        return False
    idx = [bl.LAMS.index(x) for x in lams]
    return max(idx) - min(idx) <= 1


def sign_table(models: dict[str, dict]) -> dict[str, dict[str, Any]]:
    """Per feature: the sign each fold's weight takes, and whether they agree."""
    names = next(iter(models.values()))["feature_names"]
    table: dict[str, dict[str, Any]] = {}
    for i, nm in enumerate(names):
        signs = {
            m: ("+" if mod["w"][i] > 0 else "-" if mod["w"][i] < 0 else "0")
            for m, mod in models.items()
        }
        distinct = set(signs.values())
        table[nm] = {
            "by_fold": signs,
            "consistent": len(distinct) == 1,
            "modal_share": max(collections.Counter(signs.values()).values()) / len(signs),
        }
    return table


def run(
    rows: list[dict],
    days: list[dt.date],
    base: NewsPolicy,
    out: Path,
    make_env,
) -> dict:
    from bayes_judge import judge  # noqa: PLC0415

    folds = folds_by_month(days)
    if not folds:
        raise SystemExit(
            f"need >= 2 distinct months, got {sorted({d.strftime('%Y-%m') for d in days})}"
        )
    train_rows = list(bl.trainable(rows))
    (out / "folds").mkdir(parents=True, exist_ok=True)

    saved, op._ENV = op._ENV, make_env()
    try:
        records, models = [], {}
        for month, month_days in folds:
            cutoff = min(month_days)
            past = [r for r in train_rows if dt.date.fromisoformat(r["day"]) < cutoff]
            if len({r["day"] for r in past}) < 2:
                print(f"fold {month}: skipped, {len(past)} row(s) before {cutoff}", file=sys.stderr)
                continue
            model_path = out / "folds" / f"{month}.model.json"
            started = time.monotonic()
            hp = bl.train(past, model_path)
            models[month] = json.loads(model_path.read_text())
            policy = bl.bandit_policy(base, model_path)
            candidate = [op._jsonable(op._ENV.evaluate(d, policy)) for d in month_days]
            baseline = [op._jsonable(op._ENV.evaluate(d, base)) for d in month_days]
            records.append(
                {
                    "month": month,
                    "train_days": sorted({r["day"] for r in past}),
                    "n_train_rows": len(past),
                    "eval_days": [str(d) for d in month_days],
                    "hyperparams": hp,
                    "candidate": candidate,
                    "baseline": baseline,
                }
            )
            print(
                f"fold {month}: train {len(past)} rows / "
                f"{len({r['day'] for r in past})}d lam={hp['lam']} "
                f"min_prob={hp['min_prob']} min_edge={hp['min_edge']} "
                f"cand={sum(float(x['net']) for x in candidate):,.0f} "
                f"base={sum(float(x['net']) for x in baseline):,.0f} "
                f"({time.monotonic() - started:.0f}s)",
                file=sys.stderr,
            )
    finally:
        op._ENV = saved

    if not records:
        raise SystemExit("no fold had >= 2 training days")
    oof = {
        "candidate": [x for r in records for x in r["candidate"]],
        "baseline": [x for r in records for x in r["baseline"]],
    }
    lams = [r["hyperparams"]["lam"] for r in records]
    signs = sign_table(models)
    verdict = judge({r["day"]: r for r in oof["candidate"]}, {r["day"]: r for r in oof["baseline"]})
    summary = {
        "n_folds": len(records),
        "months": [r["month"] for r in records],
        "lambdas": {r["month"]: r["hyperparams"]["lam"] for r in records},
        "lambda_stable": lambda_stable(lams),
        "min_probs": {r["month"]: r["hyperparams"]["min_prob"] for r in records},
        "min_edges": {r["month"]: r["hyperparams"]["min_edge"] for r in records},
        "signs": signs,
        "main_features": MAIN_FEATURES,
        "signs_consistent": all(signs[f]["consistent"] for f in MAIN_FEATURES if f in signs),
        "judge": verdict,
        "criteria": {
            "1_lambda_stable": lambda_stable(lams),
            "2_signs_consistent": all(signs[f]["consistent"] for f in MAIN_FEATURES if f in signs),
            "3_adopt": verdict["adopt"],
        },
    }
    op._dump(out / "folds.json", records)
    op._dump(out / "oof.json", oof)
    op._dump(out / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    from bandit_dataset import OUT as DATASET  # noqa: PLC0415
    from bandit_dataset import load_rows  # noqa: PLC0415
    from replay_common import CACHE, load_day  # noqa: PLC0415
    from replay_parity import DART, EVENTS, _export_quant_airflow_env  # noqa: PLC0415

    from daytrade_it.application.trading.policy import load_policy  # noqa: PLC0415
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (  # noqa: PLC0415
        QuantAirflowDBClient,
    )
    from daytrade_it.shared.runtime_host import require_backtest_host  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--events", type=Path, default=None)
    ap.add_argument("--base-policy", type=Path, help="default: live defaults")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dart", type=Path, default=DART)
    ap.add_argument("--work-root", type=Path, default=WORK)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    args = ap.parse_args(argv)

    require_backtest_host()
    if args.base_policy is not None and not args.base_policy.exists():
        ap.error(f"--base-policy {args.base_policy} does not exist")
    base = load_policy(args.base_policy) if args.base_policy else NewsPolicy()
    rows = load_rows(args.dataset)
    events_dir = args.events or EVENTS
    days = [dt.date.fromisoformat(p.stem) for p in sorted(events_dir.glob("2026-*.jsonl"))]
    extra = sorted({r["day"] for r in rows} - {str(d) for d in days})
    if extra:
        ap.error(f"dataset days without an events file: {extra}")
    _export_quant_airflow_env()
    stocks = QuantAirflowDBClient().get_stock_master()
    for d in days:
        load_day(d, args.cache_dir, events_dir)
    started = time.monotonic()
    summary = run(
        rows,
        days,
        base,
        out=args.out,
        make_env=partial(
            op._replay_env, days, stocks, args.dart, args.work_root, args.cache_dir, events_dir
        ),
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(f"wall {time.monotonic() - started:.0f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

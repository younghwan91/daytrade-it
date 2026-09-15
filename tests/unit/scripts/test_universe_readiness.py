"""universe_readiness (F7): per-day minute-bar + C1 news readiness for A6 -- pure, no DB."""

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from build_universe_events import load_c1, load_coverage  # noqa: E402
from universe_readiness import day_readiness, readiness_report  # noqa: E402

from daytrade_it.infrastructure.brokers.base import KST  # noqa: E402

D0 = dt.date(2026, 9, 11)
D1 = dt.date(2026, 9, 14)
D2 = dt.date(2026, 9, 15)


def _c1(root: Path, code: str, created: list[str], status: str = "done", since: str = "2026-09-01"):
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": f"{code}-{i}", "title": "t", "createdAt": c, "code": code}
        for i, c in enumerate(created)
    ]
    (root / f"{code}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / f"{code}.done").write_text(json.dumps({"status": status, "since": since}))


def test_load_coverage_matches_load_c1(tmp_path: Path) -> None:
    _c1(tmp_path, "000001", ["2026-09-14T17:00:00", "2026-09-15T09:00:00"], status="page_cap")
    _c1(tmp_path, "000002", ["2026-09-10T09:00:00"])
    assert load_coverage(tmp_path) == load_c1(tmp_path)[1]
    assert load_coverage(tmp_path)["000001"]["start"] == dt.datetime(2026, 9, 14, 17, tzinfo=KST)


def test_day_readiness_ready_partial_not(tmp_path: Path) -> None:
    _c1(tmp_path, "000001", ["2026-09-10T09:00:00"])
    _c1(tmp_path, "000002", ["2026-09-14T15:00:00"], status="page_cap")  # covered from D2 only
    coverage = load_coverage(tmp_path)
    universe = ["000001", "000002"]
    full = {(d, c): 380 for d in (D1, D2) for c in [*universe, "069500"]}

    ready = day_readiness(D2, D1, universe, full, coverage)
    assert ready["status"] == "ready"

    partial = day_readiness(D1, D0, universe, full, coverage)  # 000002 news not covered on D1
    assert partial["status"] == "partial"
    assert partial["news_uncovered"] == ["000002"] and partial["news_truncated"] == ["000002"]

    few_bars = {**full, (D2, "000002"): 10}
    r = day_readiness(D2, D1, universe, few_bars, coverage)
    assert r["status"] == "partial" and r["bars_missing"] == ["000002"]

    no_etf = {k: v for k, v in full.items() if k != (D2, "069500")}
    assert day_readiness(D2, D1, universe, no_etf, coverage)["status"] == "not_ready"

    missing_done = day_readiness(D2, D1, [*universe, "000003"], full, coverage)
    assert missing_done["status"] == "not_ready" and missing_done["news_missing"] == ["000003"]


def test_readiness_report_counts_and_lists_ready_days(tmp_path: Path) -> None:
    _c1(tmp_path, "000001", ["2026-09-10T09:00:00"])
    coverage = load_coverage(tmp_path)
    universes = {D1: ["000001"], D2: ["000001"]}
    prev = {D1: D0, D2: D1}
    bars = {(D1, "000001"): 380, (D1, "069500"): 380, (D2, "000001"): 380}
    report = readiness_report(universes, prev, bars, coverage)
    assert report["ready_days"] == ["2026-09-14"]
    assert report["counts"] == {"ready": 1, "partial": 0, "not_ready": 1}
    assert [d["day"] for d in report["days"]] == ["2026-09-14", "2026-09-15"]

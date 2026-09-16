"""pairs_day_summary: the per-day judging ledger. Files only -- no DB, no API."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from judge_batch import DEFAULT_MODEL, pair_input_hash  # noqa: E402
from pairs_day_summary import (  # noqa: E402
    count_by_day,
    group_by_day,
    load_cache,
    summarize,
)


def pair(day: str, key: str) -> dict:
    return {
        "key": key,
        "day": day,
        "code": "005930",
        "name": "삼성전자",
        "market": "KOSPI",
        "sector": "반도체",
        "title": f"{key} 제목",
        "body": f"{key} 본문",
        "category": "",
        "session": "regular",
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return path


class TestSummarize:
    def test_orders_days_most_recent_first(self, tmp_path: Path) -> None:
        kept = {d: [pair(d, f"{d}|a")] for d in ("2026-09-01", "2026-09-15", "2026-09-10")}
        rows = summarize({}, kept, {}, DEFAULT_MODEL, 1.321, 68)
        assert [r["day"] for r in rows] == ["2026-09-15", "2026-09-10", "2026-09-01"]

    def test_missing_is_kept_minus_valid_cache_rows(self) -> None:
        pairs = [pair("2026-09-15", f"k{i}") for i in range(3)]
        cache = {
            "k0": {"key": "k0", "input_hash": pair_input_hash(pairs[0], DEFAULT_MODEL)},
            "k1": {"key": "k1", "input_hash": "stale"},
        }
        (row,) = summarize(
            {"2026-09-15": 5}, {"2026-09-15": pairs}, cache, DEFAULT_MODEL, 1.321, 68
        )
        assert row["pairs_total"] == 5
        assert row["pairs_kept"] == 3
        assert row["already_judged"] == 1
        assert row["missing"] == 2

    def test_a_fully_judged_day_costs_nothing(self) -> None:
        pairs = [pair("2026-09-15", "k0")]
        cache = {"k0": {"key": "k0", "input_hash": pair_input_hash(pairs[0], DEFAULT_MODEL)}}
        (row,) = summarize({}, {"2026-09-15": pairs}, cache, DEFAULT_MODEL, 1.321, 68)
        assert row["missing"] == 0
        assert row["est_cost_usd"] == 0.0

    def test_cost_scales_with_the_number_of_missing_pairs(self) -> None:
        one = summarize({}, {"d": [pair("d", "k0")]}, {}, DEFAULT_MODEL, 1.321, 68)[0]
        two = summarize(
            {}, {"d": [pair("d", "k0"), pair("d", "k1")]}, {}, DEFAULT_MODEL, 1.321, 68
        )[0]
        assert two["est_cost_usd"] > one["est_cost_usd"] > 0

    def test_pairs_total_falls_back_to_kept_when_unfiltered_counts_are_absent(self) -> None:
        (row,) = summarize({}, {"d": [pair("d", "k0")]}, {}, DEFAULT_MODEL, 1.321, 68)
        assert row["pairs_total"] == 1


class TestReaders:
    def test_count_and_group_by_day(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "p.jsonl",
            [pair("2026-09-15", "a"), pair("2026-09-15", "b"), pair("2026-09-14", "c")],
        )
        assert count_by_day(path) == {"2026-09-15": 2, "2026-09-14": 1}
        assert sorted(group_by_day(path)) == ["2026-09-14", "2026-09-15"]
        assert len(group_by_day(path)["2026-09-15"]) == 2

    def test_missing_cache_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert load_cache(tmp_path / "nope.jsonl") == {}

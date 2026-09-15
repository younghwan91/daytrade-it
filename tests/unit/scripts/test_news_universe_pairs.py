"""news_universe_pairs: group identical newsIds across per-code files into (live id, ticker)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

import backfill_toss_company_news as bf  # noqa: E402
import news_universe_pairs as nup  # noqa: E402


def _row(news_id, code, created="2026-09-15T10:00:00"):
    return {"id": news_id, "code": code, "title": f"t-{news_id}", "createdAt": created}


def test_group_related_collects_codes_per_news_id():
    rows = [_row("n1", "005930"), _row("n1", "000660"), _row("n2", "005930"), _row("n1", "005930")]
    grouped = nup.group_related(rows)
    assert grouped["n1"]["codes"] == {"005930", "000660"}
    assert grouped["n2"]["codes"] == {"005930"}
    assert grouped["n1"]["title"] == "t-n1"


def test_build_pairs_one_row_per_ticker_with_live_id_and_related_set():
    pairs = nup.build_pairs(nup.group_related([_row("n1", "005930"), _row("n1", "000660")]))
    assert [p["ticker"] for p in pairs] == ["000660", "005930"]
    assert all(p["article_id"] == bf.live_article_id("n1") for p in pairs)
    assert pairs[0]["related"] == ["000660", "005930"]
    assert pairs[0]["news_id"] == "n1" and pairs[0]["published_at"] == "2026-09-15T10:00:00"


def test_iter_rows_reads_only_code_jsonl_files(tmp_path: Path):
    (tmp_path / "005930.jsonl").write_text(json.dumps(_row("n1", "005930")) + "\n\n")
    (tmp_path / "000660.jsonl").write_text(json.dumps(_row("n1", "000660")) + "\n")
    (tmp_path / "000660.partial.jsonl").write_text(json.dumps(_row("zz", "000660")) + "\n")
    (tmp_path / "pairs.jsonl").write_text("{}\n")
    ids = sorted(r["id"] + r["code"] for r in nup.iter_rows(tmp_path))
    assert ids == ["n1000660", "n1005930"]


def test_compare_sets_counts_recall_and_extras():
    got = {"a": {"1", "2"}, "b": {"3"}}
    ref = {"a": {"1"}, "b": {"3", "4"}, "c": {"9"}}
    s = nup.compare_sets(got, ref)
    assert s["articles_both"] == 2
    assert s["ref_pairs"] == 3 and s["ref_pairs_found"] == 2
    assert s["extra_pairs"] == 1
    assert s["exact_articles"] == 0

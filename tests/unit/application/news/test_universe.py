"""Pure library for the news universe pool: selection, sweep visit times,
live article key, related codes (A2, docs/superpowers/plans/2026-09-16-news-universe-a-plan.md C2).
"""

from __future__ import annotations

import datetime as dt

import pytest
from krx_quant_core.market.session import KST

from daytrade_it.application.news.universe import (
    SWEEP_INTERVAL,
    UNIVERSE_SIZE,
    SweepCursor,
    SweepRequest,
    judgment_input_hash,
    live_article_id,
    related_codes,
    select_universe,
    simulate_sweep,
)


class TestSelectUniverse:
    def test_sorts_by_value_desc_then_code_asc_on_ties(self) -> None:
        values = {"005930": 100.0, "000660": 100.0, "035420": 200.0}
        assert select_universe(values) == ["035420", "000660", "005930"]

    def test_caps_at_size(self) -> None:
        values = {f"{i:06d}": float(1000 - i) for i in range(10)}
        result = select_universe(values, size=3)
        assert result == ["000000", "000001", "000002"]

    def test_ignores_non_positive_values(self) -> None:
        values = {"005930": 0.0, "000660": -5.0, "035420": 1.0}
        assert select_universe(values) == ["035420"]

    def test_ignores_non_six_digit_codes(self) -> None:
        values = {"00593": 100.0, "0059300": 100.0, "AB1234": 100.0, "005930": 50.0}
        assert select_universe(values) == ["005930"]

    def test_default_size_is_universe_size_constant(self) -> None:
        values = {f"{i:06d}": float(2000 - i) for i in range(UNIVERSE_SIZE + 50)}
        assert len(select_universe(values)) == UNIVERSE_SIZE

    def test_empty_input(self) -> None:
        assert select_universe({}) == []


DAY = dt.date(2026, 9, 16)
STALE = dt.datetime(2026, 9, 15, 15, 20, tzinfo=KST)


def t(h: int, m: int, s: int = 0) -> dt.datetime:
    return dt.datetime.combine(DAY, dt.time(h, m, s), tzinfo=KST)


def _reqs(trace: list[SweepRequest]) -> list[tuple]:
    return [(r.kind, r.code, r.page, r.news_id) for r in trace]


class TestSweepCursor:
    def test_visits_codes_in_order_and_detail_follows_its_page(self) -> None:
        cur = SweepCursor(["A", "B"], STALE, page_size=3)
        r = cur.next_request()
        assert (r.kind, r.code, r.page) == ("page", "A", 1)
        assert cur.page_result(r, [("n1", t(8, 0)), ("n0", STALE - dt.timedelta(seconds=1))]) == [
            "n1"
        ]
        d = cur.next_request()
        assert (d.kind, d.code, d.news_id) == ("detail", "A", "n1")
        assert cur.next_request().code == "B"

    def test_seen_via_any_code_costs_nothing_and_stale_or_undated_skipped(self) -> None:
        cur = SweepCursor(["A", "B"], STALE, page_size=3)
        cur.page_result(cur.next_request(), [("n1", t(8, 0))])
        cur.next_request()  # detail n1
        r = cur.next_request()
        assert cur.page_result(r, [("n1", t(8, 0)), ("n2", None), (None, t(8, 0))]) == []
        assert cur.next_request() == SweepRequest("page", "A", 1)

    def test_pages_on_while_the_page_is_full_of_new_items(self) -> None:
        cur = SweepCursor(["A"], STALE, page_size=2)
        r = cur.next_request()
        cur.page_result(r, [("n4", t(8, 0)), ("n3", t(8, 0))])
        assert [cur.next_request().news_id for _ in range(2)] == ["n4", "n3"]
        r2 = cur.next_request()
        assert (r2.kind, r2.code, r2.page) == ("page", "A", 2)
        cur.page_result(r2, [("n2", t(8, 0)), ("n1", STALE - dt.timedelta(1))])  # one stale: stop
        assert cur.next_request().news_id == "n2"
        assert cur.next_request() == SweepRequest("page", "A", 1)

    def test_page_result_is_required_before_the_next_request(self) -> None:
        cur = SweepCursor(["A"], STALE)
        cur.next_request()
        with pytest.raises(RuntimeError):
            cur.next_request()


class TestSimulateSweep:
    def test_detail_costs_an_interval_and_discovery_is_detail_completion(self) -> None:
        arrivals = {"A": [("n1", t(7, 0))], "B": [("n2", t(7, 30)), ("n1", t(7, 0))]}
        trace: list[SweepRequest] = []
        found = simulate_sweep(["A", "B"], DAY, arrivals, STALE, end=dt.time(8, 0, 6), trace=trace)
        assert _reqs(trace) == [
            ("page", "A", 1, None),  # 08:00:00
            ("detail", "A", 1, "n1"),  # 08:00:01 -> found 08:00:02
            ("page", "B", 1, None),  # 08:00:02 (n1 already seen: free)
            ("detail", "B", 1, "n2"),  # 08:00:03 -> found 08:00:04
            ("page", "A", 1, None),  # 08:00:04
            ("page", "B", 1, None),  # 08:00:05
        ]
        assert found == {"n1": t(8, 0, 2), "n2": t(8, 0, 4)}

    def test_article_not_visible_before_created(self) -> None:
        arrivals = {"A": [("late", t(8, 0, 1))]}
        found = simulate_sweep(["A"], DAY, arrivals, STALE, end=dt.time(8, 0, 5))
        # 08:00:00 page: not yet published; 08:00:01 page sees it, detail 08:00:02 -> 08:00:03
        assert found == {"late": t(8, 0, 3)}

    def test_paging_through_a_full_backlog(self) -> None:
        arrivals = {"A": [(f"n{i}", t(7, 0, i)) for i in range(5)]}
        trace: list[SweepRequest] = []
        found = simulate_sweep(
            ["A"], DAY, arrivals, STALE, end=dt.time(8, 1), page_size=2, trace=trace
        )
        kinds = [(r.kind, r.page, r.news_id) for r in trace[:9]]
        assert kinds == [
            ("page", 1, None),
            ("detail", 1, "n4"),
            ("detail", 1, "n3"),
            ("page", 2, None),
            ("detail", 2, "n2"),
            ("detail", 2, "n1"),
            ("page", 3, None),
            ("detail", 3, "n0"),
            ("page", 1, None),  # page 3 was not full: back to the cycle
        ]
        assert found["n0"] == t(8, 0, 8)

    def test_stops_at_end_and_excludes_stale(self) -> None:
        arrivals = {"A": [("old", STALE - dt.timedelta(seconds=1)), ("n", t(15, 19, 59))]}
        found = simulate_sweep(["A"], DAY, arrivals, STALE, start=dt.time(15, 19, 58))
        # 15:19:58 page (not yet), 15:19:59 page sees n; its detail would start at 15:20 -> no
        assert found == {}

    def test_empty_universe(self) -> None:
        assert simulate_sweep([], DAY, {}, STALE) == {}

    def test_default_interval_is_one_second(self) -> None:
        assert dt.timedelta(seconds=1) == SWEEP_INTERVAL


class TestLiveArticleId:
    def test_matches_operational_key_formula(self) -> None:
        # Real archived pair: data/toss_detail/2026-09-15/006f4150-....json has
        # news_id "newspim_20260915000210"; the filename (the live daemon's own
        # uuid5 key for the article) is that same string.
        assert live_article_id("newspim_20260915000210") == "006f4150-d62b-5770-a154-fb84bd7b23a5"

    def test_matches_second_real_archived_pair(self) -> None:
        # Title-matched pair: data/toss_detail/2026-09-15/<id>.json for
        # news_id "seokyung_2KHIC2PH6Z" (title "기아 'PV7', 獨 하노버서 세계 첫
        # 공개") is filed under this uuid, which also equals the article_id
        # the live judgment archive recorded for that same article on
        # data/news_judgments/2026-09-15.jsonl.
        assert live_article_id("seokyung_2KHIC2PH6Z") == "595eead7-3889-57cd-bdc5-fc299f660c70"

    def test_deterministic(self) -> None:
        assert live_article_id("abc_123") == live_article_id("abc_123")

    def test_different_ids_differ(self) -> None:
        assert live_article_id("abc_123") != live_article_id("abc_124")


class TestRelatedCodes:
    def test_sorts_codes_per_id(self) -> None:
        rows = {"a1": {"005930", "000660"}, "a2": {"035420"}}
        assert related_codes(rows) == {
            "a1": ("000660", "005930"),
            "a2": ("035420",),
        }

    def test_empty_set_yields_empty_tuple(self) -> None:
        assert related_codes({"a1": set()}) == {"a1": ()}

    def test_empty_input(self) -> None:
        assert related_codes({}) == {}


def test_judgment_input_hash_covers_model_system_and_user() -> None:
    import hashlib

    h = judgment_input_hash("m", "sys", "user")
    assert len(h) == 64
    assert h == judgment_input_hash("m", "sys", "user")
    assert (
        len(
            {
                h,
                judgment_input_hash("m2", "sys", "user"),
                judgment_input_hash("m", "sys2", "user"),
                judgment_input_hash("m", "sys", "user2"),
            }
        )
        == 4
    )
    # unambiguous framing: moving text across the boundary changes the hash
    assert judgment_input_hash("m", "ab", "c") != judgment_input_hash("m", "a", "bc")
    assert h != hashlib.sha256(b"msysuser").hexdigest()

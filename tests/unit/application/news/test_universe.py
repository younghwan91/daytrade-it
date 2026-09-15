"""Pure library for the news universe pool: selection, sweep visit times,
live article key, related codes (A2, docs/superpowers/plans/2026-09-16-news-universe-a-plan.md C2).
"""

from __future__ import annotations

import datetime as dt

from krx_quant_core.market.session import KST

from daytrade_it.application.news.universe import (
    SWEEP_INTERVAL,
    UNIVERSE_SIZE,
    judgment_input_hash,
    live_article_id,
    next_visit,
    related_codes,
    select_universe,
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


class TestNextVisit:
    UNIVERSE = ["000001", "000002", "000003"]
    DAY = dt.date(2026, 9, 16)

    def test_first_visit_of_the_day_for_first_code(self) -> None:
        after = dt.datetime(2026, 9, 16, 8, 0, tzinfo=KST)
        result = next_visit("000001", self.UNIVERSE, self.DAY, after)
        assert result == dt.datetime(2026, 9, 16, 8, 0, 0, tzinfo=KST)

    def test_first_visit_offset_by_index(self) -> None:
        after = dt.datetime(2026, 9, 16, 8, 0, tzinfo=KST)
        result = next_visit("000003", self.UNIVERSE, self.DAY, after)
        assert result == dt.datetime(2026, 9, 16, 8, 0, 2, tzinfo=KST)

    def test_after_within_current_cycle_returns_next_cycle(self) -> None:
        # after 08:00:01 (already past 000001's 08:00:00 slot this cycle),
        # 000001's next visit is the next cycle: 08:00:03.
        after = dt.datetime(2026, 9, 16, 8, 0, 1, tzinfo=KST)
        result = next_visit("000001", self.UNIVERSE, self.DAY, after)
        assert result == dt.datetime(2026, 9, 16, 8, 0, 3, tzinfo=KST)

    def test_after_exactly_on_slot_returns_that_slot(self) -> None:
        after = dt.datetime(2026, 9, 16, 8, 0, 2, tzinfo=KST)
        result = next_visit("000003", self.UNIVERSE, self.DAY, after)
        assert result == dt.datetime(2026, 9, 16, 8, 0, 2, tzinfo=KST)

    def test_many_cycles_later(self) -> None:
        # 08:50:00.001 is just past the k=999 slot for code index 1
        # (08:00:01 + 999*3s = 08:50:00.0), so the next slot is k=1000:
        # 08:00:01 + 1000*3s = 08:50:01.
        after = (
            dt.datetime(2026, 9, 16, 8, 0, 0, tzinfo=KST)
            + 1000 * 3 * dt.timedelta(seconds=1)
            + dt.timedelta(milliseconds=1)
        )
        result = next_visit("000002", self.UNIVERSE, self.DAY, after)
        assert result == dt.datetime(2026, 9, 16, 8, 0, 1, tzinfo=KST) + 1000 * 3 * dt.timedelta(
            seconds=1
        )

    def test_accepts_naive_after_as_kst(self) -> None:
        after_naive = dt.datetime(2026, 9, 16, 8, 0, 0)
        result = next_visit("000001", self.UNIVERSE, self.DAY, after_naive)
        assert result == dt.datetime(2026, 9, 16, 8, 0, 0, tzinfo=KST)

    def test_after_before_start_clamps_to_start_slot(self) -> None:
        after = dt.datetime(2026, 9, 16, 0, 0, 0, tzinfo=KST)
        result = next_visit("000002", self.UNIVERSE, self.DAY, after)
        assert result == dt.datetime(2026, 9, 16, 8, 0, 1, tzinfo=KST)

    def test_none_when_code_not_in_universe(self) -> None:
        after = dt.datetime(2026, 9, 16, 8, 0, 0, tzinfo=KST)
        assert next_visit("999999", self.UNIVERSE, self.DAY, after) is None

    def test_none_when_visit_would_be_at_or_after_1520(self) -> None:
        after = dt.datetime(2026, 9, 16, 15, 19, 59, tzinfo=KST)
        result = next_visit("000001", self.UNIVERSE, self.DAY, after)
        # next slot for 000001 at/after 15:19:59 lands at/after 15:20:00.
        assert result is None

    def test_returns_last_visit_just_before_1520(self) -> None:
        # Build a universe sized so a slot lands exactly one interval before
        # the 15:20 cutoff, using a 1-code universe (visits every interval).
        after = dt.datetime(2026, 9, 16, 15, 19, 59, tzinfo=KST)
        result = next_visit("000001", ["000001"], self.DAY, after)
        assert result == dt.datetime(2026, 9, 16, 15, 19, 59, tzinfo=KST)

    def test_default_interval_is_one_second(self) -> None:
        assert dt.timedelta(seconds=1) == SWEEP_INTERVAL

    def test_custom_start_and_interval(self) -> None:
        after = dt.datetime(2026, 9, 16, 9, 0, 0, tzinfo=KST)
        result = next_visit(
            "000002",
            self.UNIVERSE,
            self.DAY,
            after,
            start=dt.time(9, 0),
            interval=dt.timedelta(minutes=1),
        )
        assert result == dt.datetime(2026, 9, 16, 9, 1, 0, tzinfo=KST)


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

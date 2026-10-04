"""판정이 계속 실패하면 신규 진입을 멈춘다.

09-16: 크레딧이 떨어져 09:07~14:58 동안 86 건이 실패했는데도 데몬은 조용히
계속 돌았다. 그날 판정 기록은 20 건(09:00~09:32)뿐이고 나머지 여섯 시간은
근거 없이 매매했다. 아무도 장이 끝날 때까지 몰랐다.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from uuid import uuid4

import pytest
import scripts.news_signal_poller as poller


class _Boom:
    """항상 실패하는 판정기."""

    def __init__(self) -> None:
        self.calls = 0

    async def score_article(self, **_: object) -> object:
        self.calls += 1
        raise RuntimeError("credit balance is too low")


class _Judgments:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def seen(self, key: str) -> bool:
        return key in self.rows

    def append(self, key: str, record: dict) -> None:
        self.rows[key] = record


@pytest.fixture
def isolated_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "live_STOP"
    monkeypatch.setattr(poller, "KILL_SWITCH_PATH", path)
    monkeypatch.setattr(poller, "_scoring_failures", 0)
    return path


def _article() -> object:
    from daytrade_it.domain.entities.news import NewsArticle

    return NewsArticle(
        id=uuid4(),
        title="테스트 기사",
        summary="요약",
        url="https://example.invalid/1",
        published_at=dt.datetime(2026, 9, 16, 9, 0, tzinfo=poller.KST),
        source="test",
        tickers=[],
        category=None,
    )


class TestScoringFailureGuard:
    def test_limit_is_reachable_and_small_enough_to_catch_a_dead_session(self) -> None:
        """한 번의 일시 오류로 하루를 날리지 않되, 장 초반에 잡힐 만큼 작다."""
        assert 1 < poller.SCORING_FAILURE_LIMIT <= 20

    def test_switch_absent_before_the_limit(self, isolated_switch: Path) -> None:
        poller._scoring_failures = poller.SCORING_FAILURE_LIMIT - 1
        assert not isolated_switch.exists()

    def test_kill_switch_written_at_the_limit(self, isolated_switch: Path) -> None:
        """한계에 도달하면 스위치 파일이 생기고, 원인이 그 안에 남는다."""
        isolated_switch.parent.mkdir(parents=True, exist_ok=True)
        isolated_switch.write_text(
            f"{dt.datetime.now(poller.KST).isoformat()} 판정 연속 실패 "
            f"{poller.SCORING_FAILURE_LIMIT} 회로 신규 진입을 자동 중단한다.\n"
            "마지막 오류: RuntimeError: credit balance is too low\n"
        )
        assert poller.kill_switch_active()
        body = isolated_switch.read_text()
        assert "판정 연속 실패" in body
        assert "credit balance" in body

    def test_kill_switch_only_blocks_entries(self) -> None:
        """스위치는 신규 진입만 막는다 -- 청산은 계속 돌아야 한다.

        폴러의 루프에서 `trader.manage()` 가 스위치 검사보다 **앞에** 있다.
        순서가 뒤집히면 보유 종목이 하루를 넘긴다.
        """
        src = Path(poller.__file__).read_text()
        manage = src.index("trader.manage()")
        check = src.index("if stop_now:")
        assert manage < check

    def test_stop_switch_is_mode_aware(self) -> None:
        """종이매매는 `paper_STOP` 만 본다 -- 실매매가 멈춰 있어도 판정은 쌓인다.

        신뢰할 수 있는 아카이브 날짜가 이틀뿐이라 모든 판단이 거기서 막힌다.
        멈춘 날을 데이터 없는 날로 버리지 않기 위한 구분.
        """
        src = Path(poller.__file__).read_text()
        gate = src[src.index("stop_now = (") : src.index("if stop_now:")]
        assert "kill_switch_active()" in gate
        assert "PAPER_KILL_SWITCH_PATH.exists()" in gate
        assert "paper_seen is None" in gate

    def test_counter_resets_on_success(self, isolated_switch: Path) -> None:
        """연속 실패만 센다 -- 한 건 성공하면 0 으로 돌아간다."""
        poller._scoring_failures = 5
        globals_reset = 'globals()["_scoring_failures"] = 0'
        assert globals_reset in Path(poller.__file__).read_text()


class TestTickPinRotation:
    """핀은 시간으로 자른다. 매일 새로 쓰고, TTL 이 지나면 실제로 빠진다.

    09-17 에 70 개를 덧붙였더니 09-18 에는 그 70 개가 구독 자리를 잡고 있어
    새 뉴스 종목이 못 들어갔다 -- 판정 이후 호가 커버리지 31% -> 23%.
    넣는 코드만 있고 빼는 코드가 없으면 기능이 스스로를 막는다.
    """

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(poller, "TICK_PIN_PATH", tmp_path / "pins.txt")
        monkeypatch.setattr(poller, "_pinned_codes", {})
        monkeypatch.setattr(poller, "_pin_day", None)

    @staticmethod
    def _kept() -> list[str]:
        return [
            line
            for line in poller.TICK_PIN_PATH.read_text().splitlines()
            if not line.startswith("#")
        ]

    def test_expired_pin_is_dropped(self) -> None:
        """TTL 이 지나면 새 판정이 없어도 자리를 내놓는다."""
        poller._pin_for_tick_collection("005930")
        assert self._kept() == ["005930"]
        poller._pinned_codes["005930"] = dt.datetime.now(poller.KST) - dt.timedelta(minutes=1)
        poller._sweep_tick_pins()
        assert self._kept() == []

    def test_held_lasts_until_the_close(self) -> None:
        """보유 종목은 종가까지 -- 청산 판단에 쓰는 유일한 자리다."""
        poller._pin_for_tick_collection("000660", held=True)
        assert poller._pinned_codes["000660"].time() == poller.TICK_PIN_HELD_UNTIL

    def test_candidate_repin_never_shortens_a_held_pin(self) -> None:
        """후보로 다시 들어와도 보유분 만료가 당겨지면 안 된다."""
        poller._pin_for_tick_collection("000660", held=True)
        poller._pin_for_tick_collection("000660")
        assert poller._pinned_codes["000660"].time() == poller.TICK_PIN_HELD_UNTIL

    def test_new_day_clears(self) -> None:
        poller._pinned_codes["005930"] = dt.datetime.now(poller.KST) + dt.timedelta(hours=1)
        poller._pin_day = dt.date(2020, 1, 1)
        poller._pin_for_tick_collection("000660")
        assert self._kept() == ["000660"]  # 어제 종목은 오늘 자리를 막지 않는다

    def test_ttl_matches_the_measured_concurrent_demand(self) -> None:
        """실측: 30분 13~14 종목, 60분 19~24, 120분 33~39. 상한 100 을 혼자 안 먹는다."""
        assert dt.timedelta(minutes=15) <= poller.TICK_PIN_TTL <= dt.timedelta(minutes=120)
        assert 0 < poller.TICK_PIN_MAX <= 50


class TestPreMarketPinCohort:
    """장전 판정 종목은 매수 여부와 무관하게 핀한다 -- 표본이 판정 결과에 조건부가 되지 않게."""

    def test_cutoff_is_just_after_the_open(self) -> None:
        assert dt.time(9, 0) <= poller.PRE_MARKET_PIN_UNTIL <= dt.time(9, 10)

    def test_cohort_fits_under_the_cap(self) -> None:
        """실측 장전 판정 12~23 종목/일 + 장중 매수 핀이 상한 30 안에 든다."""
        assert poller.TICK_PIN_MAX >= 23

    def test_pin_happens_before_the_buy_only_return(self) -> None:
        """핀 호출이 `if not sentiment.is_bullish: return` 보다 앞에 있어야 비매수 판정도 핀된다."""
        src = Path(poller.__file__).read_text()
        pin = src.index("PRE_MARKET_PIN_UNTIL:\n        _pin_for_tick_collection(ticker.value)")
        bullish_return = src.index("if not sentiment.is_bullish:\n        return")
        assert pin < bullish_return

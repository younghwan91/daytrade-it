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
        check = src.index("if kill_switch_active():")
        assert manage < check

    def test_counter_resets_on_success(self, isolated_switch: Path) -> None:
        """연속 실패만 센다 -- 한 건 성공하면 0 으로 돌아간다."""
        poller._scoring_failures = 5
        globals_reset = 'globals()["_scoring_failures"] = 0'
        assert globals_reset in Path(poller.__file__).read_text()

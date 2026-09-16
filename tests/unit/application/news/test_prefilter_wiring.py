"""The pre-LLM filter must be off by default and identical in live and backtest.

Covers the four call sites: `judge_pair` (live daemon + sweep shadow),
`build_universe_events.day_pairs` (backtest pools/pairs files) and
`judge_batch.BatchJudge.run` (batch judging).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "eval"))

from build_universe_events import day_pairs, load_articles  # noqa: E402

from daytrade_it.application.news import judging  # noqa: E402
from daytrade_it.application.news.judging import JudgmentLog, judge_pair  # noqa: E402
from daytrade_it.domain.value_objects import Ticker  # noqa: E402
from daytrade_it.infrastructure.brokers.base import KST  # noqa: E402
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo  # noqa: E402

ROUNDUP_TITLE = "[표] 코스피 기관·외국인 순매수 상위종목"
REAL_TITLE = "삼성 3조 수주"


@dataclass
class _Article:
    id: uuid.UUID
    title: str
    summary: str | None = "요약"
    content: str | None = "본문"
    category: str | None = "stock"
    published_at: dt.datetime = dt.datetime(2026, 9, 15, 8, 0, tzinfo=KST)


@dataclass
class _Features:
    subject: str = "mention"

    def score(self) -> float:
        return 0.0

    def to_sentiment(self) -> Any:
        class _S:
            score = 0.0

            class label:  # noqa: N801
                value = "neutral"

        return _S()


class _Claude:
    def __init__(self) -> None:
        self.calls = 0

    async def score_article(self, **_: Any) -> _Features:
        self.calls += 1
        return _Features()


STOCK = StockInfo(code="005930", name="삼성전자", market="거래소", sector="반도체")


async def _run(title: str, tmp_path: Path, prefilter: bool | None) -> tuple[_Claude, list[dict]]:
    judging.prefilter_skips = 0
    claude = _Claude()
    log = JudgmentLog(tmp_path, dt.date(2026, 9, 15))
    await judge_pair(
        _Article(id=uuid.uuid4(), title=title),
        Ticker(value="005930"),
        STOCK,
        claude,  # type: ignore[arg-type]
        log,
        None,
        prefilter=prefilter,
    )
    rows = (
        [json.loads(line) for line in log.path.read_text().splitlines()]
        if log.path.exists()
        else []
    )
    return claude, rows


class TestJudgePair:
    @pytest.mark.asyncio
    async def test_off_by_default_judges_a_roundup(self, tmp_path: Path) -> None:
        claude, rows = await _run(ROUNDUP_TITLE, tmp_path, None)
        assert claude.calls == 1
        assert rows[0]["v2"] is not None
        assert "prefilter_skipped" not in rows[0]

    @pytest.mark.asyncio
    async def test_on_skips_a_roundup_without_calling_the_model(self, tmp_path: Path) -> None:
        claude, rows = await _run(ROUNDUP_TITLE, tmp_path, True)
        assert claude.calls == 0
        assert judging.prefilter_skips == 1

    @pytest.mark.asyncio
    async def test_on_archives_the_skip_so_backtests_see_the_same_universe(
        self, tmp_path: Path
    ) -> None:
        _, rows = await _run(ROUNDUP_TITLE, tmp_path, True)
        assert len(rows) == 1
        assert rows[0]["prefilter_skipped"] is True
        assert rows[0]["prefilter_tag"] == "표"
        assert rows[0]["v2"] is None
        assert rows[0]["ticker"] == "005930"

    @pytest.mark.asyncio
    async def test_on_still_judges_real_news(self, tmp_path: Path) -> None:
        claude, rows = await _run(REAL_TITLE, tmp_path, True)
        assert claude.calls == 1
        assert rows[0]["v2"] is not None
        assert judging.prefilter_skips == 0

    @pytest.mark.asyncio
    async def test_env_switch_is_honoured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER", "1")
        claude, _ = await _run(ROUNDUP_TITLE, tmp_path, None)
        assert claude.calls == 0


def _c1(root: Path, code: str, title: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{code}.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "id": "n1",
                    "title": title,
                    "summary": "요약",
                    "createdAt": "2026-09-15T07:30:00",
                    "source": "x",
                    "newsType": "NEWS",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    (root / f"{code}.done").write_text(
        json.dumps({"status": "ok", "rows": 1, "since": "2026-09-01"})
    )


class TestDayPairs:
    @pytest.mark.parametrize("title", [ROUNDUP_TITLE, REAL_TITLE])
    def test_off_by_default_keeps_every_pair(self, tmp_path: Path, title: str) -> None:
        root = tmp_path / "c1"
        _c1(root, "005930", title)
        pairs = day_pairs(
            load_articles(root), dt.date(2026, 9, 15), dt.date(2026, 9, 14), ["005930"]
        )
        assert len(pairs) == 1

    def test_on_drops_the_roundup_and_counts_it(self, tmp_path: Path) -> None:
        root = tmp_path / "c1"
        _c1(root, "005930", ROUNDUP_TITLE)
        stats: dict = {}
        pairs = day_pairs(
            load_articles(root),
            dt.date(2026, 9, 15),
            dt.date(2026, 9, 14),
            ["005930"],
            None,
            stats,
            None,
            True,
        )
        assert pairs == []
        assert stats["prefiltered"] == 1

    def test_on_keeps_real_news(self, tmp_path: Path) -> None:
        root = tmp_path / "c1"
        _c1(root, "005930", REAL_TITLE)
        stats: dict = {}
        pairs = day_pairs(
            load_articles(root),
            dt.date(2026, 9, 15),
            dt.date(2026, 9, 14),
            ["005930"],
            None,
            stats,
            None,
            True,
        )
        assert len(pairs) == 1
        assert stats["prefiltered"] == 0

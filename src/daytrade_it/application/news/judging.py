"""The live daemon's per-pair judging step, shared by the live poller and the sweep shadow.

Extracted verbatim from `scripts/news_signal_poller.py` (`JudgmentLog`,
`session_close_kst`, and the scoring half of `process_pair`) so the universe
sweep shadow runner (`scripts/news_sweep_shadow.py`, plan C5) judges and
archives exactly like the live daemon without importing the daemon module --
which pulls in `AutoTrader` and the order path.

Nothing here places orders, persists signals, or touches a broker.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from krx_quant_core.market.session import KST

from daytrade_it.shared.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from daytrade_it.domain.entities.news import NewsArticle
    from daytrade_it.domain.value_objects import Ticker
    from daytrade_it.infrastructure.external.anthropic_client import (
        AnthropicClient,
        ArticleFeatures,
    )
    from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

logger = get_logger(__name__)

SESSION_CLOSE = dt.time(15, 20)  # signals expire here; day trades are flat by the close


def session_close_kst(day: dt.date) -> dt.datetime:
    return dt.datetime.combine(day, SESSION_CLOSE, tzinfo=KST)


class JudgmentLog:
    """Every (article, ticker) judgment, one JSONL per day.

    Kept for all pairs -- not just the ones that became signals -- so a
    later forward evaluation of v2 vs shadow v4 isn't biased toward what the
    live rule selected. Also dedups re-judging after a restart.
    """

    def __init__(self, directory: Path, day: dt.date) -> None:
        self._path = directory / f"{day.isoformat()}.jsonl"
        self._keys: set[str] = set()
        if self._path.exists():
            for line in self._path.open():
                self._keys.add(json.loads(line)["key"])

    @property
    def path(self) -> Path:
        return self._path

    def seen(self, key: str) -> bool:
        return key in self._keys

    def append(self, key: str, record: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(json.dumps({"key": key, **record}, ensure_ascii=False, default=str) + "\n")
        self._keys.add(key)


def pair_key(article: NewsArticle, ticker: Ticker) -> str:
    return f"{article.id}|{ticker.value}"


async def judge_pair(
    article: NewsArticle,
    ticker: Ticker,
    stock: StockInfo,
    claude: AnthropicClient,
    judgments: JudgmentLog,
    full_body_text: str | None,
) -> ArticleFeatures | None:
    """Score one (article, company) pair with v2 (+ shadow v4) and archive it.

    Returns the v2 features, or None if scoring failed (logged, never raised).
    """
    try:
        features = await claude.score_article(
            code=ticker.value,
            name=stock.name,
            market=stock.market,
            sector=stock.sector,
            title=article.title,
            body=article.summary or article.content,
            category=article.category,
        )
    except Exception:
        logger.exception(
            "news_signal_poller: scoring failed", article_id=str(article.id), ticker=ticker.value
        )
        return None

    sentiment = features.to_sentiment()
    logger.info(
        "news_signal_poller: judged",
        ticker=ticker.value,
        name=stock.name,
        subject=features.subject,
        score=round(sentiment.score, 3),
        label=sentiment.label.value,
        title=article.title[:80],
    )

    # Shadow v4 (full body): recorded for a forward comparison, never used
    # for signals. See anthropic_client._USER_TEMPLATE_FULL_BODY.
    shadow: ArticleFeatures | None = None
    if full_body_text:
        try:
            shadow = await claude.score_article(
                code=ticker.value,
                name=stock.name,
                market=stock.market,
                sector=stock.sector,
                title=article.title,
                body=full_body_text,
                category=article.category,
                full_body=True,
            )
        except Exception:
            logger.exception("news_signal_poller: shadow v4 scoring failed", ticker=ticker.value)

    judgments.append(
        pair_key(article, ticker),
        judgment_record(article, ticker, stock, features, shadow, full_body_text),
    )
    return features


def judgment_record(
    article: NewsArticle,
    ticker: Ticker,
    stock: StockInfo,
    features: ArticleFeatures,
    shadow: ArticleFeatures | None,
    full_body_text: str | None,
    judged_at: dt.datetime | None = None,
) -> dict:
    """One archive row (without its "key"), in the live `data/news_judgments` format."""
    return {
        "article_id": str(article.id),
        "ticker": ticker.value,
        "name": stock.name,
        "title": article.title,
        "category": article.category,
        "published_at": article.published_at.isoformat(),
        "judged_at": (judged_at or dt.datetime.now(KST)).isoformat(),
        "full_body_chars": len(full_body_text or ""),
        "v2": {**asdict(features), "score": features.score()},
        "v4_shadow": {**asdict(shadow), "score": shadow.score()} if shadow else None,
    }

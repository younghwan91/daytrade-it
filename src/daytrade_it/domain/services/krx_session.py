"""KRX 정규장 경계와 직전 정규장 종가 — 운영(`entry_filter`)·리플레이·연구 스크립트의 한 정문.

2026-09-14 부터 KRX 애프터마켓(16:00~20:00 연속매매, ETF·ETN 제외) 체결이 키움 일봉 → ``daily_bars``·
``daily_bars_adjusted``·``supply_demand.close`` 의 종가·고가·저가·거래량과 ``minute_bars`` 16:00~19:59 봉에 섞였다
(시가만 정상). 다음 날 가격제한 기준가 = 전일 **정규장** 종가 = 분봉 :data:`REGULAR_OPEN` <= ts < :data:`REGULAR_END`
마지막 봉 종가다. ``minute_bars.ts`` 는 시간대 없는 KST 다 — ``at time zone`` 을 걸지 않는다.

`scripts/eval/krx_session.py`(연구용 pandas 정문)는 이 모듈의 상수·함수를 다시 내보낸다. 운영 코드가 ``scripts`` 를
import 하지 않도록 정본은 여기 둔다.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

from krx_quant_core.market.session import REGULAR_OPEN

__all__ = ["AFTERMARKET_FROM", "REGULAR_END", "REGULAR_OPEN", "last_regular_closes"]

#: 일봉에 KRX 애프터마켓 체결이 섞이기 시작한 첫 거래일.
AFTERMARKET_FROM = dt.date(2026, 9, 14)
#: 정규장 + 종가 단일가 봉: 보통 15:30, 변동성 완화장치로 연장되면 15:32
#: (09-01..09-15: 15:30 봉 2,895, 15:32 봉 22, 15:31/15:33/15:34 없음). 그 뒤 봉은 시간외·애프터마켓이다.
REGULAR_END = dt.time(15, 33)


def last_regular_closes[P](
    rows: Iterable[tuple[str, dt.datetime, P]], day: dt.date
) -> dict[str, tuple[P, dt.date]]:
    """``(code, ts, close)`` 분봉 행에서 종목별로 `day` 이전 마지막 정규장 봉의 (종가, 날짜). ts 는 KST naive."""
    best: dict[str, tuple[dt.datetime, P]] = {}
    for code, ts, close in rows:
        if ts.date() >= day or not (REGULAR_OPEN <= ts.time() < REGULAR_END):
            continue
        if code not in best or ts > best[code][0]:
            best[code] = (ts, close)
    return {c: (cl, ts.date()) for c, (ts, cl) in best.items()}

"""전진(forward) 승격 게이트 -- "좋아 보인다"를 못 쓰게 만드는 7개 조건.

scalp-it `docs/PROMOTION_CHECKLIST.md` A절을 daytrade-it 장부에 그대로 적용한다.
전진 = 정책이 고정된 뒤 쌓인 거래일(실매매 장부 `data/auto_trader/`, 종이 장부
`data/auto_trader_paper/`). 입력은 순수 자료구조라 DB 없이 단위검증한다.

    G1 거래당 순수익 평균 > 0
    G2 날짜 클러스터 부트스트랩 95% CI 하한 > 0
    G3 상위 1% 거래 제거 후 평균 > 0
    G4 일평균 음수인 날 ≤ 1/3
    G5 Deflated Sharpe 확률 > 0.95 (누적 시행 N 반영)
    G6 무작위 진입 대조 대비 p < 0.05 (날짜 단위 짝 검정)
    G7 전진 거래일 ≥ 10 · 거래 ≥ 30

2026-10-04 기준 운영 정책은 G2·G3·G4 에서 떨어진다(13일·63거래: 평균 +27.8bp,
중앙값 −45.8, 상위 1건 제외 −20.5, CI [−62, +149], 음수일 7/13).
"""

from __future__ import annotations

import datetime as dt
import math
import random
import statistics as st
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

#: 측정된 왕복 비용(bp). 수수료·거세·스프레드 포함 -- `no-edge-news-daytrading` 메모 참조.
ROUND_TRIP_TOLL_BP = 36.7

ExitPrice = Callable[[str, dt.datetime, float], float]


@dataclass(frozen=True)
class Trade:
    day: dt.date
    ticker: str
    entry: float
    exit: float
    net_bp: float


def ledger_trades(
    events: Iterable[Mapping], *, exit_price: ExitPrice, toll_bp: float
) -> list[Trade]:
    """장부 이벤트 → 거래. 진입가 = 체결된 지정가(종이 브로커는 지정가에 체결),
    청산가 = ``exit_price(ticker, exit_ts, exit_limit)`` (보통 그 시각 봉 종가)."""
    events = list(events)
    submitted = {e["order_id"]: e for e in events if e.get("event") == "entry_submitted"}
    filled = {e["order_id"] for e in events if e.get("event") == "entry_resolved"}
    open_pos: dict[str, float] = {}
    for oid in filled:
        e = submitted.get(oid)
        if e:
            open_pos[e["ticker"]] = float(e["limit"])
    out: list[Trade] = []
    for e in events:
        if e.get("event") != "exit_submitted" or e["ticker"] not in open_pos:
            continue
        entry = open_pos.pop(e["ticker"])
        ts = dt.datetime.fromisoformat(e["ts"]).replace(tzinfo=None)
        fallback = float(e.get("limit") or entry)
        exit_ = float(exit_price(e["ticker"], ts, fallback))
        out.append(Trade(ts.date(), e["ticker"], entry, exit_, (exit_ / entry - 1) * 1e4 - toll_bp))
    return out


def _by_day(trades: Iterable[Trade]) -> dict[dt.date, list[float]]:
    d: dict[dt.date, list[float]] = {}
    for t in trades:
        d.setdefault(t.day, []).append(t.net_bp)
    return d


def gate_report(
    trades: list[Trade],
    control_by_day: Mapping[dt.date, float],
    *,
    n_trials: int,
    seed: int = 0,
    n_boot: int = 2000,
    min_days: int = 10,
    min_trades: int = 30,
) -> dict:
    """7개 게이트를 판정한다. ``control_by_day`` 는 그날 무작위 진입(같은 매수 풀에서
    무작위 5종목) 거래당 평균 bp. ``n_trials`` 는 이 정책에 이르기까지 시도한 후보 수."""
    rng = random.Random(seed)
    per = [t.net_bp for t in trades]
    days = sorted(_by_day(trades))
    byday = _by_day(trades)
    gates: dict[str, dict] = {}

    mean = st.mean(per) if per else math.nan
    gates["G1_mean_gt0"] = {"value": mean, "ok": bool(per) and mean > 0}

    if len(days) >= 2:
        boots = []
        for _ in range(n_boot):
            s = rng.choices(days, k=len(days))
            boots.append(st.mean([x for d in s for x in byday[d]]))
        boots.sort()
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    else:
        lo = hi = math.nan
    gates["G2_daycluster_ci_lo_gt0"] = {"value": lo, "hi": hi, "ok": lo > 0}

    srt = sorted(per)
    k = max(1, int(len(per) * 0.01))
    ex = st.mean(srt[:-k]) if len(srt) > k else math.nan
    gates["G3_ex_top1pct_gt0"] = {"value": ex, "ok": ex > 0}

    neg = sum(st.mean(byday[d]) < 0 for d in days)
    gates["G4_neg_days_le_third"] = {
        "value": neg,
        "days": len(days),
        "ok": bool(days) and neg * 3 <= len(days),
    }

    dsr = (
        deflated_sharpe_from_sample(per, n_trials)
        if len(per) >= 2
        else {"prob_deflated_sharpe": math.nan}
    )
    p_dsr = dsr["prob_deflated_sharpe"]
    gates["G5_deflated_sharpe_gt95"] = {
        "value": p_dsr,
        "ok": not math.isnan(p_dsr) and p_dsr > 0.95,
    }

    diffs = [st.mean(byday[d]) - control_by_day[d] for d in days if d in control_by_day]
    if len(diffs) >= 2:
        # 날짜 단위 부호 뒤집기 순열검정 (단측: 정책 > 대조)
        obs = st.mean(diffs)
        ge = 0
        n_perm = 4000
        for _ in range(n_perm):
            if st.mean([x if rng.random() < 0.5 else -x for x in diffs]) >= obs:
                ge += 1
        p = (ge + 1) / (n_perm + 1)
    else:
        obs, p = math.nan, math.nan
    gates["G6_vs_random_p_lt05"] = {
        "value": p,
        "mean_diff_bp": obs,
        "ok": not math.isnan(p) and p < 0.05,
    }

    gates["G7_sample"] = {
        "days": len(days),
        "trades": len(per),
        "ok": len(days) >= min_days and len(per) >= min_trades,
    }

    return {
        "days": [d.isoformat() for d in days],
        "n_trades": len(per),
        "median_bp": st.median(per) if per else math.nan,
        "per_day_sum_bp": {d.isoformat(): sum(byday[d]) for d in days},
        "gates": gates,
        "promote": all(g["ok"] for g in gates.values()),
    }

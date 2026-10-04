"""Forward promotion gate: the 7 statistical conditions from scalp-it's checklist."""

import datetime as dt
import random

from daytrade_it.application.research.forward_gate import (
    Trade,
    gate_report,
    ledger_trades,
)

D1 = dt.date(2026, 9, 21)
D2 = dt.date(2026, 9, 22)


def _ev(ts, event, **kw):
    return {"ts": f"{ts}+09:00", "event": event, **kw}


def test_ledger_trades_pairs_filled_entries_with_exits_and_nets_the_toll():
    events = [
        _ev("2026-09-21T09:00:01", "entry_submitted", ticker="A", order_id="1", limit="1000"),
        _ev("2026-09-21T09:00:02", "entry_submitted", ticker="B", order_id="2", limit="500"),
        _ev("2026-09-21T09:05:00", "entry_resolved", ticker="A", order_id="1", reason="filled"),
        _ev("2026-09-21T09:05:00", "entry_cancelled", ticker="B", order_id="2", reason="ttl"),
        _ev("2026-09-21T15:00:00", "exit_submitted", ticker="A", order_id="3", limit="990"),
    ]
    trades = ledger_trades(events, exit_price=lambda _t, _ts, _fb: 1010.0, toll_bp=36.7)
    assert len(trades) == 1
    t = trades[0]
    assert (t.day, t.ticker, t.entry, t.exit) == (D1, "A", 1000.0, 1010.0)
    assert round(t.net_bp, 1) == round(100 - 36.7, 1)


def test_ledger_trades_falls_back_to_exit_limit_when_no_bar():
    events = [
        _ev("2026-09-21T09:00:01", "entry_submitted", ticker="A", order_id="1", limit="1000"),
        _ev("2026-09-21T09:05:00", "entry_resolved", ticker="A", order_id="1", reason="filled"),
        _ev("2026-09-21T15:00:00", "exit_submitted", ticker="A", order_id="3", limit="990"),
    ]
    trades = ledger_trades(events, exit_price=lambda _t, _ts, fb: fb, toll_bp=0)
    assert trades[0].exit == 990.0


def test_ledger_trades_prefers_recorded_fill_prices_over_limits_and_bars():
    events = [
        _ev("2026-09-21T09:00:01", "entry_submitted", ticker="A", order_id="1", limit="1000"),
        _ev(
            "2026-09-21T09:00:30",
            "fill",
            ticker="A",
            order_id="1",
            side="BUY",
            quantity=1,
            avg_price="995",
            role="entry",
        ),
        _ev("2026-09-21T09:00:30", "entry_resolved", ticker="A", order_id="1", reason="filled"),
        _ev("2026-09-21T15:00:00", "exit_submitted", ticker="A", order_id="3", limit="990"),
        _ev("2026-09-21T15:01:00", "exit_submitted", ticker="A", order_id="4", limit="985"),
        _ev(
            "2026-09-21T15:01:10",
            "fill",
            ticker="A",
            order_id="4",
            side="SELL",
            quantity=1,
            avg_price="1005",
            role="exit",
        ),
    ]
    trades = ledger_trades(events, exit_price=lambda _t, _ts, _fb: 1010.0, toll_bp=0)
    assert [(t.day, t.ticker, t.entry, t.exit) for t in trades] == [(D1, "A", 995.0, 1005.0)]
    assert round(trades[0].net_bp, 2) == round((1005 / 995 - 1) * 1e4, 2)


def test_ledger_trades_uses_fill_for_one_leg_and_falls_back_for_the_other():
    events = [
        _ev("2026-09-21T09:00:01", "entry_submitted", ticker="A", order_id="1", limit="1000"),
        _ev("2026-09-21T09:05:00", "entry_resolved", ticker="A", order_id="1", reason="ttl"),
        _ev("2026-09-21T15:00:00", "exit_submitted", ticker="A", order_id="3", limit="990"),
        _ev(
            "2026-09-21T15:00:05",
            "fill",
            ticker="A",
            order_id="3",
            side="SELL",
            quantity=1,
            avg_price="992",
            role="exit",
        ),
    ]
    trades = ledger_trades(events, exit_price=lambda _t, _ts, _fb: 1010.0, toll_bp=0)
    assert (trades[0].entry, trades[0].exit) == (1000.0, 992.0)


def _trades(day_values):
    out = []
    for day, vals in day_values.items():
        out += [Trade(day, f"T{i}", 100.0, 100.0, v) for i, v in enumerate(vals)]
    return out


def test_gate_report_passes_a_clean_edge_and_fails_an_outlier_edge():
    rng = random.Random(0)
    days = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(12)]
    good = _trades({d: [rng.gauss(60, 20) for _ in range(5)] for d in days})
    control = dict.fromkeys(days, -50.0)
    rep = gate_report(good, control, n_trials=50, seed=0)
    assert all(g["ok"] for g in rep["gates"].values()), rep
    assert rep["promote"] is True

    bad = _trades({d: [-30.0] * 5 for d in days})
    bad[0] = Trade(days[0], "X", 100.0, 100.0, 3000.0)  # one lottery ticket carries the mean
    rep = gate_report(bad, control, n_trials=50, seed=0)
    assert rep["gates"]["G1_mean_gt0"]["ok"] is True
    assert rep["gates"]["G3_ex_top1pct_gt0"]["ok"] is False
    assert rep["promote"] is False


def test_gate_report_refuses_thin_samples():
    rep = gate_report(
        _trades({D1: [10.0] * 3, D2: [10.0] * 3}), {D1: 0.0, D2: 0.0}, n_trials=1, seed=0
    )
    assert rep["gates"]["G7_sample"]["ok"] is False
    assert rep["promote"] is False


def test_drop_days_without_market_removes_holiday_paper_fills():
    from daytrade_it.application.research.forward_gate import drop_days_without_market

    trades = _trades({D1: [10.0, 20.0], D2: [5.0]})
    kept, dropped = drop_days_without_market(trades, lambda _code, day: day == D1)
    assert dropped == [D2]
    assert {t.day for t in kept} == {D1}

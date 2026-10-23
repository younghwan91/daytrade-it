"""autoresearch 평가기 가드 — 위반을 주입해 실제로 막히는지 본다(통과만 보는 테스트는 규칙이 죽어도 초록)."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval" / "autoresearch"))

import prepare as P  # noqa: E402

M = P.M


def toy(n: int = 4, nd: int = 3, price: float = 10000.0) -> dict:
    """평탄한 가격(모든 봉 체결), 비용 0.004, 상·하한가 ±30%."""
    shape = (nd, n, M)
    C = np.full(shape, price, np.float32)
    D = {
        "O": C.copy(),
        "H": C * 1.001,
        "L": C * 0.999,
        "C": C.copy(),
        "Vol": np.full(shape, 100.0, np.float32),
    }
    for k in ("na", "nf", "nx", "nm"):
        D[k] = np.zeros(shape, np.uint8)
    for k in P.STATIC_KEYS:
        D[k] = np.zeros((nd, n), np.float32)
    D["uni"] = np.ones((nd, n), bool)
    D["prev_close"][:] = price
    D["lu"] = np.full((nd, n), price * 1.3, np.float32)
    D["ld"] = np.full((nd, n), price * 0.7, np.float32)
    D["next_px"] = np.full((nd, n), price, np.float32)
    D["cost_o"] = np.full((nd, n), 0.004, np.float32)
    D["cost_l"] = np.full((nd, n), 0.004, np.float32)
    D["days"] = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(nd)]
    D["codes"] = [[f"{i:06d}" for i in range(n)] for _ in range(nd)]
    D["half"] = 0
    P._derive(D)
    return D


def sig_at(D: dict, d: int, n: int, m: int, score: float = 1.0) -> dict:
    cand = np.zeros(D["O"].shape, bool)
    cand[d, n, m] = True
    return {"cand": cand, "score": np.full(D["O"].shape, score)}


def rederive(D: dict) -> dict:
    P._derive(D)
    return D


CFG = {"max_pos": 5, "max_entries": 10, "stop": 0.03, "exit_kind": "close"}


# ---------------------------------------------------------------- 체결(M3·M4·M5)


def test_signal_at_m_fills_at_next_bar_open_not_same_bar():
    D = toy()
    D["O"][0, 1, 11] = 10100.0
    D["C"][0, 1, 10] = 9000.0  # 신호 봉 종가로 체결되면 이 값이 나온다
    t = P.simulate(rederive(D), sig_at(D, 0, 1, 10), CFG)
    assert len(t) == 1 and t["entry"].iloc[0] == 11
    assert t["net"].iloc[0] == pytest.approx(
        10000 / 10100 - 1 - 0.004, rel=1e-5
    )  # 15:00 종가 10000 에 정리


def test_no_fill_on_bar_without_trades():
    D = toy()
    D["Vol"][0, 1, 11] = 0
    D["O"][0, 1, 11] = np.nan
    t = P.simulate(rederive(D), sig_at(D, 0, 1, 10), CFG)
    assert len(t) == 0  # 묵은 가격으로 사지 않는다


def test_no_buy_at_limit_up_open():
    D = toy()
    D["O"][0, 2, 6] = D["lu"][0, 2]
    t = P.simulate(rederive(D), sig_at(D, 0, 2, 5), CFG)
    assert len(t) == 0


def test_locked_limit_down_defers_stop_then_next_day_open():
    D = toy()
    ld = float(D["ld"][0, 0])
    for k in range(50, M):
        D["O"][0, 0, k] = D["H"][0, 0, k] = D["L"][0, 0, k] = D["C"][0, 0, k] = ld  # 하한가 잠김
    D["next_px"][0, 0] = ld * 0.95
    t = P.simulate(rederive(D), sig_at(D, 0, 0, 10), CFG)
    assert t["why"].iloc[0] == "overnight"
    assert t["net"].iloc[0] == pytest.approx(
        ld * 0.95 / 10000 - 1 - 0.004, rel=1e-4
    )  # 비용 포함(M4)


def test_stop_gap_fills_at_open_and_intrabar_one_tick_worse():
    D = toy()
    D["O"][0, 0, 40] = D["H"][0, 0, 40] = D["L"][0, 0, 40] = D["C"][0, 0, 40] = 9000.0  # 갭 하락
    t = P.simulate(rederive(D), sig_at(D, 0, 0, 10), CFG)
    assert t["why"].iloc[0] == "gap_stop" and t["exit"].iloc[0] == 40
    D = toy()
    D["L"][0, 0, 40] = 9600.0  # 손절 9700 을 봉 안에서 깸
    t = P.simulate(rederive(D), sig_at(D, 0, 0, 10), CFG)
    assert t["why"].iloc[0] == "stop"
    assert t["net"].iloc[0] == pytest.approx((9700 - 10) / 10000 - 1 - 0.004, rel=1e-5)


def test_alpha_subtracts_same_window_universe_move():
    D = toy(n=4)
    D["O"][0, :, 300:] = D["H"][0, :, 300:] = D["L"][0, :, 300:] = D["C"][0, :, 300:] = (
        10500.0  # 모두 +5%
    )
    t = P.simulate(rederive(D), sig_at(D, 0, 0, 10), CFG)
    assert t["net"].iloc[0] == pytest.approx(0.05 - 0.004, rel=1e-4)
    assert t["alpha"].iloc[0] == pytest.approx(
        -0.004, abs=1e-5
    )  # 시장이 같이 오른 몫은 점수가 아니다


def test_one_entry_per_code_per_day_and_slots():
    D = toy(n=4)
    cand = np.zeros(D["O"].shape, bool)
    cand[0, :, 10:20] = True
    t = P.simulate(D, {"cand": cand, "score": np.ones(D["O"].shape)}, {**CFG, "max_pos": 2})
    assert len(t) == 2 and t["row"].is_unique


def test_condition_exit_judged_at_close_filled_next_open():
    D = toy()
    D["C"][0, 0, 30] = 9900.0  # VWAP 아래로 마감
    D["O"][0, 0, 31] = 9950.0
    t = P.simulate(rederive(D), sig_at(D, 0, 0, 10), {**CFG, "exit_kind": "vwap", "exit_g": 0.001})
    assert t["why"].iloc[0] == "cond" and t["exit"].iloc[0] == 31
    assert t["net"].iloc[0] == pytest.approx(9950 / 10000 - 1 - 0.004, rel=1e-5)


# ---------------------------------------------------------------- 청산 메뉴(M2)


def test_clamp_cfg_cuts_narrow_stop_and_drops_unknown_exits():
    c = P.clamp_cfg(
        {
            "stop": 0.003,
            "max_pos": 99,
            "take_profit": 0.02,
            "trail": 0.01,
            "time_stop": 30,
            "breakeven": 0.01,
            "exit_w": 1,
        }
    )
    assert c["stop"] == P.STOP_FLOOR and c["max_pos"] == 10 and c["exit_w"] == 5
    assert not {"take_profit", "trail", "time_stop", "breakeven"} & set(c)


def test_clamp_cfg_rejects_exit_outside_menu():
    with pytest.raises(ValueError):
        P.clamp_cfg({"exit_kind": "trail"})


def test_struct_stop_never_narrower_than_floor():
    D = toy()
    D["L"][0, 0, :11] = 9990.0  # 진입 전 저가가 진입가 바로 아래
    D["L"][0, 0, 40] = 9900.0  # −1% — 바닥(1.5%)보다 좁은 구조선이면 여기서 털린다
    t = P.simulate(
        rederive(D), sig_at(D, 0, 0, 10), {**CFG, "exit_kind": "struct", "struct_b": 0.002}
    )
    assert t["why"].iloc[0] == "flat"


def test_uni_guard_cannot_be_bypassed_by_cand():
    D = toy()
    D["uni"][:] = False
    t = P.simulate(D, sig_at(D, 0, 0, 10), CFG)
    assert len(t) == 0


def test_no_entry_after_last_signal_bar():
    D = toy()
    t = P.simulate(D, sig_at(D, 0, 0, P.LAST_SIGNAL + 1), CFG)
    assert len(t) == 0


# ---------------------------------------------------------------- train.py 검사(C1·C3·M7)


@pytest.mark.parametrize(
    "src",
    [
        "import os\n",
        "import pickle\n",
        "from prepare import SEED\n",
        "x = open('a')\n",
        "y = getattr(1, 'real')\n",
        "import numpy as np\nnp.load('x.npy')\n",
        "import pandas as pd\npd.read_pickle('x')\n",
        "a = (1).__class__\n",
        "BAD = ['005930']\n",
        "Y = 2026\n",
        "S = '2026-03'\n",
        "def f(V):\n    return V['days'].year\n",
        "def f(V):\n    V['C'].setflags(write=True)\n",
        "print(1)\n",
        "def f(a):\n    return a.base\n",
    ],
)
def test_lint_rejects_escape_and_hindsight(src):
    assert P.lint_train(src)


def test_lint_accepts_baseline():
    assert P.lint_train(P.TRAIN.read_text(encoding="utf-8")) == []


def test_view_is_read_only_copy_without_dates_or_codes():
    D = toy()
    V = P.view(D)
    assert not {"days", "codes", "lu", "ld", "next_px", "cost_o", "cost_l"} & set(V)
    with pytest.raises(ValueError):
        V["C"][0, 0, 0] = 1.0
    assert V["C"].base is None or not np.shares_memory(V["C"], D["PC"])


def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "s.py"
    p.write_text(
        "import numpy as np\nimport pandas as pd\n"
        + body
        + "def get_strategy():\n    return {'params': {}, 'signals': signals, 'cfg': lambda x: {}}\n"
    )
    return p


def test_causality_catches_intraday_and_cross_day_lookahead(tmp_path):
    D = toy(n=3, nd=12)
    rng = np.random.default_rng(0)
    D["C"] = (10000 * np.cumprod(1 + rng.normal(0, 0.002, D["C"].shape), axis=2)).astype(np.float32)
    D["O"], D["H"], D["L"] = D["C"].copy(), D["C"] * 1.001, D["C"] * 0.999
    P._derive(D)
    res = {"folds": [{"params": {}}]}
    peek = _write(
        tmp_path,
        "def signals(V, x):\n    f = np.roll(V['C'], -1, axis=2)\n"
        "    return {'cand': f > V['C'], 'score': f}\n",
    )
    assert not P.causality(D, res, peek)[0]
    dayhi = _write(
        tmp_path,
        "def signals(V, x):\n    h = np.nanmax(V['H'], axis=2, keepdims=True)\n"
        "    return {'cand': V['C'] >= 0.995 * h, 'score': V['C'] * 0}\n",
    )
    assert not P.causality(D, res, dayhi)[0]  # 그날 고가(장 마감 뒤에 알 수 있다)
    nxt = _write(
        tmp_path,
        "def signals(V, x):\n    t = np.roll(V['C'][:, :, -1:], -1, axis=0)\n"
        "    return {'cand': V['C'] < t, 'score': V['C'] * 0}\n",
    )
    assert not P.causality(D, res, nxt)[0]  # 다음 날 가격
    honest = _write(
        tmp_path,
        "def signals(V, x):\n    p = np.concatenate([V['C'][:, :, :1], V['C'][:, :, :-1]], axis=2)\n"
        "    return {'cand': V['C'] > p, 'score': V['C'] - p}\n",
    )
    assert P.causality(D, res, honest)[0]


# ---------------------------------------------------------------- 선택·keep·원장(M1·M8)


def test_plateau_pick_prefers_plateau_over_spike():
    rng = np.random.default_rng(0)
    U = rng.random((60, 2))
    y = np.where(U[:, 0] > 0.6, 10.0, -10.0)
    U = np.vstack([U, [[0.05, 0.05]]])
    y = np.append(y, 100.0)
    assert U[P.plateau_pick(U, y), 0] > 0.6


def test_half_split_is_stable_and_balanced():
    codes = [f"{i:06d}" for i in range(4000)]
    h = np.array([P.half_of(c) for c in codes])
    assert 0.45 < h.mean() < 0.55
    assert [P.half_of(c) for c in codes[:50]] == list(h[:50])


def test_folds_are_monthly_and_stop_before_forward_window():
    f = P.folds()
    assert len(f) == 11 and f[-1][2] == dt.date(2026, 10, 13) and dt.date(2026, 10, 13) > P.END
    assert all(a < b < c for a, b, c in f) and f[0][0] == dt.date(2025, 8, 1)


def test_summarize_voids_thin_trading():
    import pandas as pd

    folds = [{"fold": i, "days": 20} for i in range(11)]
    trades = [
        pd.DataFrame(
            {
                "day": [0] * 3,
                "row": range(3),
                "entry": 1,
                "exit": 2,
                "net": 0.01,
                "alpha": 0.01,
                "cost": 0.004,
                "why": "flat",
                "fold": i,
            }
        )
        for i in range(11)
    ]
    s = P.summarize({"folds": folds, "trades": trades})
    assert s["void"] and s["score"] == P.VOID_SCORE


def test_beats_requires_paired_fold_improvement(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BEST", tmp_path / "BEST.json")
    (tmp_path / "BEST.json").write_text(
        json.dumps({"score": 0.0, "top1": 0.0, "fold_alpha": [0.0] * 11})
    )
    one_big = {
        "void": False,
        "score": 5.0,
        "_fold_alpha": [60.0] + [-0.1] * 10,
    }  # 한 폴드만 크게(잡음 겨냥)
    assert not P.beats(one_big, 0.0)[0]
    broad = {"void": False, "score": 3.0, "_fold_alpha": [3.0] * 11}
    assert P.beats(broad, 0.0)[0]
    assert not P.beats(broad, -5.0)[0]  # 상위 1% 를 빼면 무너지는 개선


def test_ledger_is_double_written_and_survives_one_copy_loss(tmp_path, monkeypatch):
    dirs = (tmp_path / "a", tmp_path / "b")
    monkeypatch.setattr(P, "LEDGER_DIRS", dirs)
    for s in ("x1", "x2", "x1"):
        P._append("RUNS.jsonl", {"train_sha": s})
    (dirs[0] / "RUNS.jsonl").unlink()
    assert P.n_trials() == 2


def test_final_refuses_after_three_lockbox_openings(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "LEDGER_DIRS", (tmp_path / "a", tmp_path / "b"))
    monkeypatch.setattr(P, "integrity", lambda **_: None)
    for _ in range(P.LOCKBOX_MAX):
        P._append("LOCKBOX.jsonl", {"pass": False})
    with pytest.raises(SystemExit, match="잠금"):
        P.final(1)

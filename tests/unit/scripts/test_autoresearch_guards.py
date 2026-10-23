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
        # 평가기 리뷰 M-4(a) 우회 예시와 그 이웃
        "import dataclasses\nm = dataclasses.inspect.importlib.import_module('subprocess')\n",
        "import pandas as pd\nh = pd.io.common.get_handle('x', 'w')\n",
        "import numpy as np\nx = np._core\n",
        "import numpy as np\nx = np.random.mtrand\n",
        "from numpy import random as r\nx = r.mtrand\n",
        "import collections\nx = collections.abc\n",
        "from pandas import io\n",
        "import numpy.random\n",
        "from numpy import *\n",
        "import typing\n",
        # 2차 리뷰 N3: 생성기 프레임 → builtins, 모듈을 값으로, 파일 쓰기, 동적 dunder 조립
        "def g():\n    yield 1\nb = g().gi_frame.f_builtins\n",
        "import numpy as np\ndef f():\n    x = np\n    return x.core\n",
        "import numpy as np\ndef f(h):\n    return h(np)\n",
        "def f(a):\n    a.dump('p')\n",
        "import pandas as pd\ndef f():\n    pd.DataFrame({}).to_string(buf='p')\n",
        "k = '__imp' + 'ort__'\n",
    ],
)
def test_lint_rejects_escape_and_hindsight(src):
    assert P.lint_train(src)


def test_lint_accepts_baseline_and_ordinary_numpy_pandas():
    assert P.lint_train(P.TRAIN.read_text(encoding="utf-8")) == []
    ok = (
        "import numpy as np\nimport pandas as pd\nfrom numpy import random as r\n"
        "a = np.random.default_rng(0)\nb = np.linalg.norm([1.0])\nc = r.default_rng(1)\n"
        "d = pd.DataFrame({'x': [1.0]}).rolling(1).mean().to_numpy()\n"
    )
    assert P.lint_train(ok) == []


def test_module_cannot_be_held_as_value_even_allowed_submodule():
    # 3차 리뷰: r = np.random → r.bit_generator.re.enum.bltns.exec(...) 로 builtins 에 닿았다
    esc = (
        "import numpy as np\nr = np.random\nb = r.bit_generator.re.enum.bltns\n"
        "b.exec('import json', {})\n"
    )
    bad = P.lint_train(esc)
    assert any("값으로" in b for b in bad) and any(".exec" in b for b in bad)
    assert P.lint_train("import pandas as pd\nk = pd.compat\n")
    assert P.lint_train("import numpy as np\nx = np.random.default_rng(0).random(3)\n") == []


def test_lockbox_markets_frozen_at_build():
    import pandas as pd

    now = pd.DataFrame(
        {"code": ["000001"], "market": ["거래소"]}
    )  # 000002 는 그사이 폐지, 000001 은 이전상장
    fixed = {"market": {"000001": "코스닥", "000002": "코스닥"}}
    assert P._markets(now, fixed) == {"000001": "코스닥", "000002": "코스닥"}
    assert P._markets(now, None) == {"000001": "거래소"}


def test_stray_ignored_pyc_only_when_source_exists(tmp_path):
    (tmp_path / "scripts/eval").mkdir(parents=True)
    (tmp_path / "scripts/eval/costs.py").write_text("")
    out = (
        "!! scripts/eval/__pycache__/costs.cpython-313.pyc\n"
        "!! scripts/eval/__pycache__/numpy.cpython-313.pyc\n"
        "!! scripts/eval/autoresearch/numpy.pyc\n"
        "!! scripts/eval/autoresearch/results.tsv\n"
    )
    assert P._stray(out, tmp_path) == [
        "!! scripts/eval/__pycache__/numpy.cpython-313.pyc",
        "!! scripts/eval/autoresearch/numpy.pyc",
    ]


def test_day_hashes_cover_engine_inputs():
    D = toy()
    h0 = P.day_hashes(D)
    D["cost_l"][1, 2] += 0.001
    D["nm"][2, 0, 5] = 1
    h1 = P.day_hashes(D)
    assert h0[0] == h1[0] and h0[1] != h1[1] and h0[2] != h1[2]


def test_stray_files_outside_agent_file_are_reported():
    out = "?? scripts/eval/krx_quant_core/__init__.py\n M scripts/eval/autoresearch/train.py\n"
    assert P._stray(out) == ["?? scripts/eval/krx_quant_core/__init__.py"]
    assert P._stray(" M scripts/eval/autoresearch/train.py\n") == []


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


def _rand_toy(nd: int = 20) -> dict:
    D = toy(n=3, nd=nd)
    rng = np.random.default_rng(0)
    D["C"] = (10000 * np.cumprod(1 + rng.normal(0, 0.002, D["C"].shape), axis=2)).astype(np.float32)
    D["O"], D["H"], D["L"] = D["C"].copy(), D["C"] * 1.001, D["C"] * 0.999
    D["na"][:] = (rng.random(D["na"].shape) < 0.01).astype(np.uint8)  # 드문 기사
    D["pre_all"][:] = (rng.random(D["pre_all"].shape) < 0.5).astype(np.float32)  # 장전 기사
    D["vol20"][:] = rng.random(D["vol20"].shape).astype(np.float32)
    P._derive(D)
    return D


RES = {"folds": [{"params": {}}]}


@pytest.mark.parametrize(
    "body",
    [
        # 한 봉 앞 엿보기(모든 봉 후보)
        "def signals(V, x):\n    f = np.roll(V['C'], -1, axis=2)\n    return {'cand': f > V['C'], 'score': f}\n",
        # 그날 고가(장 마감 뒤에 안다)
        "def signals(V, x):\n    h = np.nanmax(V['H'], axis=2, keepdims=True)\n"
        "    return {'cand': V['C'] >= 0.995 * h, 'score': V['C'] * 0}\n",
        # 다음 날 가격
        "def signals(V, x):\n    t = np.roll(V['C'][:, :, -1:], -1, axis=0)\n"
        "    return {'cand': V['C'] < t, 'score': V['C'] * 0}\n",
        # 리뷰 C-1: 드문 후보 — 기사 봉에서 다음 봉이 양봉이면(한 봉 앞)
        "def signals(V, x):\n    nxt = np.roll(V['C'], -1, axis=2)\n"
        "    return {'cand': (V['na'] > 0) & (nxt > V['C']), 'score': V['C'] * 0}\n",
        # 리뷰 C3: 달력(날짜 인덱스)으로 구간을 끄기
        "def signals(V, x):\n    c = V['na'] > 0\n    c = c.copy()\n    c[:5] = False\n"
        "    return {'cand': c, 'score': V['C'] * 0}\n",
        # 검사 중인지 알아채기: 마지막 봉 NaN 여부(v2 는 다른 날 실제 값으로 바꿔 모양·값이 같다) + 미래 엿보기
        "def signals(V, x):\n    f = np.roll(V['C'], -2, axis=2)\n    peek = ~np.isnan(V['C'][:, :, -1:])\n"
        "    return {'cand': (V['na'] > 0) & (f > V['C']) & peek, 'score': f}\n",
        # 2차 리뷰 N1: 장전 경로(봉 0) — 09:05 종가를 보고 09:01 시가에 산다
        "def signals(V, x):\n    c = np.zeros(V['C'].shape, bool)\n"
        "    c[:, :, 0] = (V['pre_all'] > 0) & (V['C'][:, :, 5] > V['C'][:, :, 0])\n"
        "    return {'cand': c, 'score': V['C'] * 0}\n",
        # 2차 리뷰 N2: 순서와 무관한 날짜 횡단 계산(전 기간 분위수)
        "def signals(V, x):\n    thr = np.nanquantile(V['vol20'], 0.5)\n"
        "    c = (V['na'] > 0) & (V['vol20'] < thr)[:, :, None]\n"
        "    return {'cand': c, 'score': V['C'] * 0}\n",
    ],
)
def test_causality_catches_lookahead(tmp_path, body):
    assert not P.causality(_rand_toy(), RES, _write(tmp_path, body))[0]


def test_causality_passes_honest_sparse_strategy(tmp_path):
    body = (
        "def signals(V, x):\n    p = np.concatenate([V['C'][:, :, :1], V['C'][:, :, :-1]], axis=2)\n"
        "    return {'cand': (V['na'] > 0) & (V['C'] > p), 'score': V['C'] - p + V['mkt_prev1'][:, None, None]}\n"
    )
    assert P.causality(_rand_toy(), RES, _write(tmp_path, body)) == (True, "ok")


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


def test_beats_requires_paired_bootstrap_not_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BEST", tmp_path / "BEST.json")
    rng = np.random.default_rng(1)
    base = {i: float(v) for i, v in enumerate(rng.normal(-40, 60, 200))}
    (tmp_path / "BEST.json").write_text(
        json.dumps({"score": -50.0, "top1": -55.0, "daily_alpha": base})
    )
    noise = {k: v + float(rng.normal(0, 60)) for k, v in base.items()}  # 잡음만 바뀐 변형
    noise_mean = np.mean([noise[k] - base[k] for k in base])
    s_noise = {"void": False, "score": -50.0 + max(3.0, noise_mean), "_daily_alpha": noise}
    ok, why = P.beats(s_noise, -55.0)
    assert not ok or P.paired_boot_p(noise, base)[0] < P.KEEP_P, why
    better = {k: v + 15.0 + float(rng.normal(0, 5)) for k, v in base.items()}  # 날마다 고르게 +15bp
    assert P.beats({"void": False, "score": -35.0, "_daily_alpha": better}, -45.0)[0]
    assert not P.beats({"void": False, "score": -35.0, "_daily_alpha": better}, -60.0)[
        0
    ]  # 상위 1% 빼면 무너짐
    assert not P.beats({"void": False, "score": -49.0, "_daily_alpha": better}, -45.0)[
        0
    ]  # Δscore < 2bp


def test_paired_bootstrap_rejects_pure_noise_most_of_the_time():
    rng = np.random.default_rng(7)
    base = {i: float(v) for i, v in enumerate(rng.normal(0, 80, 220))}
    hits = sum(
        P.paired_boot_p({k: v + float(rng.normal(0, 80)) for k, v in base.items()}, base, seed=r)[0]
        < P.KEEP_P
        for r in range(40)
    )
    assert hits <= 6  # 무작위면 ~5% (이항 40회 95% 상한 근처)


def test_peer_benchmark_uses_same_characteristic_bucket():
    D = toy(n=12)
    D["kosdaq"][0, 6:] = 1.0  # 행 6~11 코스닥
    D["tv_prev"][0] = np.arange(12, 0, -1)
    P._derive(D)
    D["O"][0, 6:, 300:] = D["H"][0, 6:, 300:] = D["L"][0, 6:, 300:] = D["C"][0, 6:, 300:] = (
        11000.0  # 코스닥만 +10%
    )
    P._derive(D)
    t = P.simulate(D, sig_at(D, 0, 7, 10), CFG)  # 코스닥 종목 하나
    assert t["net"].iloc[0] == pytest.approx(0.10 - 0.004, rel=1e-3)
    assert abs(t["alpha"].iloc[0]) < 0.01  # 코스닥 기울기는 알파가 아니다(전체 평균이면 +5%)


def test_rank_null_degenerate_is_excluded_not_failed():
    assert P.beats_null(-30.0, [-30.0] * 100) is None
    assert P.beats_null(5.0, list(np.linspace(-20, 4, 100))) is True
    assert P.beats_null(0.0, list(np.linspace(-20, 4, 100))) is False


def test_factor_intercept_t_recovers_planted_alpha():
    rng = np.random.default_rng(3)
    F = rng.normal(0, 0.01, (400, 4))
    y = 0.002 + F @ np.array([1.0, 0.5, -0.3, 0.2]) + rng.normal(0, 0.003, 400)
    a, t = P.nw_intercept_t(y, F)
    assert a == pytest.approx(0.002, abs=5e-4) and t > 5


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

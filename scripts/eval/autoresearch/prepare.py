"""autoresearch 고정 평가기 — 진입점이자 심판. 에이전트는 이 파일을 고치지 않는다(program.md · 사전등록).

karpathy/autoresearch 의 prepare.py 자리 + dietmarwo/autoresearch-trading 의 "구조는 LLM, 숫자는 최적화기" 분리를
swing-it 평가기 v2(리뷰 C1~C4·M1~M8 반영)의 가드 그대로 이 레포의 장중 매매에 옮겼다.

대상: 토스 종목 뉴스(전일 거래대금 상위 300, LLM 판정 없음)가 보인 뒤 분봉에서 1주 진입 → 메뉴 청산 → 15:00 정리.
``train.py`` 의 ``get_strategy()`` 가 후보·순위·파라미터 범위·엔진 설정(청산은 메뉴)을 내면, 파라미터 점 48개
(라틴 하이퍼큐브)를 전 기간에 한 번씩 돌리고(포지션은 날마다 0 으로 돌아가 날짜끼리 독립이다), 폴드마다 학습 3개월의
점수로 고원(이웃 5점 중앙값 최대)을 골라 시험 1개월을 그 점의 거래로 잰다.

가드(사전등록 docs/research/frozen/2026-10-24-autoresearch-prereg.json):
- C1 진입점은 이 파일. train.py 는 AST 허용 목록(파일·OS·동적 실행·더블언더스코어·종목코드·연도 리터럴 금지)을 통과해야
  로드된다. 태그 뒤 평가기·의존 파일 변경 거부 + MANIFEST.json sha256(코드·캐시) 대조.
- C2 미래정보: 평가마다 (날짜, 분) 절단 4곳에서 그 시점까지만 남긴 V 로 새 모듈이 만든 신호 == 전체 신호 앞부분. 다르면 void.
- C3 V 에는 날짜·종목코드가 없다. 행 = 그날 유니버스 절반을 전일 거래대금 순으로 늘어놓은 것(날마다 다른 종목).
- C4 점수 = 거래당 **비용 차감 후 선정 알파**(순수익 − 같은 진입·청산 시각 사이 유니버스 동일가중 수익). 장중 하락 흐름·
  시각 선택은 점수에 0 이다. 실제 순수익(흐름 포함)은 보고 열·졸업 바 1. 종목 절반 잠금은 최대 3회(코드로 강제).
- M1 시행 원장·잠금 횟수는 레포 밖 고정 경로 두 곳에 덧붙인다(git reset 이 못 지운다). DSR N = 고유 train sha 수.
- M2 청산은 메뉴뿐: 15:00 정리 · 하드 손절(≥ 1.5%) · VWAP 이탈 · 분봉 이동평균 이탈 · 진입 전 당일 저가(구조선, 거리 ≥ 1.5%).
  익절·추적·본전 이동·보유 시간 손절은 메뉴에 없다.
- M3 체결은 거래가 있는 분봉에서만(묵은 가격 체결 없음). 상한가 시가 매수 없음, 하한가 잠김 매도 미룸.
- M4 모든 보유분은 그날 비용을 내고 청산된다(15:00 이후 체결이 없으면 다음 날 시가).
- M5 분봉 m 의 신호(m 의 종가까지 정보) → m+1 시가 진입. 조건 청산도 m 종가 판정 → 다음 체결 봉 시가.
- M6 졸업 널 = 순위 무작위 · 후보 무작위 솎기 · 파라미터 무작위 각 100회 95 백분위 + 일 순수익의 시장 회귀 절편 NW t > 2.
- M7 V 의 배열은 읽기 전용 복사본이다(제자리 수정은 오류).
- M8 keep 은 평가기가 쌍 비교로 판정한다. run.log 에는 요약만 나온다(폴드별 값은 레포 밖 캐시에만).

사용(작업 트리 루트):
  uv run python scripts/eval/autoresearch/prepare.py --build      # 탐색 절반 캐시(한 번, KR_QUANT_DB) → MANIFEST.json
  uv run python scripts/eval/autoresearch/prepare.py > run.log 2>&1   # 실험 한 번
  uv run python scripts/eval/autoresearch/prepare.py --accept     # 직전 실행이 beats_best 면 BEST 갱신
  uv run python scripts/eval/autoresearch/prepare.py --final      # 졸업 심사 — 메인 세션만, 잠금 최대 3회
"""

# ruff: noqa: E741 — O·H·L·C 는 분봉 배열 이름 그대로 쓴다
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "eval"), str(ROOT / "src")]

TRAIN = HERE / "train.py"
CACHE = Path(os.path.expanduser("~/.cache/daytrade-it/autoresearch"))
LEDGER_DIRS = (
    Path(os.path.expanduser("~/.cache/daytrade-it/autoresearch_ledger")),
    Path(os.path.expanduser("~/.local/state/daytrade-it/autoresearch_ledger")),
)
BEST, LAST = CACHE / "BEST.json", CACHE / "last_run.json"
MANIFEST = HERE / "MANIFEST.json"
PREREG_TAG = "daytrade-autoresearch-prereg-1"
GUARDED_FILES = (
    "scripts/eval/autoresearch/prepare.py",
    "scripts/eval/costs.py",
    "scripts/eval/krx_session.py",
    "src/daytrade_it/domain/services/krx_session.py",
    "uv.lock",
    "docs/research/frozen/2026-10-24-autoresearch-prereg.json",
)

START, END = (
    dt.date(2025, 8, 1),
    dt.date(2026, 10, 12),
)  # 10-13~ 는 전진 게이트(regular_close) 몫 — 읽지 않는다
TEST_STARTS = [dt.date(2025, m, 1) for m in (11, 12)] + [dt.date(2026, m, 1) for m in range(1, 10)]
TRAIN_MONTHS = 3
UNIVERSE = 300
LAG_MIN = 5  # 기사 createdAt + 5분에 보인다고 본다(1단계 +4분보다 보수적)
M = 380  # 09:00..15:19 분봉
FLAT = 360  # 15:00 봉 종가 정리
LAST_SIGNAL = 329  # 신호 ≤ 14:29 → 진입 ≤ 14:30
OPEN_WINDOW = 5  # 진입 봉 < 09:05 면 장초 스프레드
STOP_FLOOR = 0.015
LOCKBOX_MAX = 3
SEED = 20261024
N_EVAL = 48
K_PLATEAU = 5
MIN_TRAIN_TRADES = 60
MIN_FOLD_TRADES = 30
MAX_THIN_FOLDS = 2
MIN_OOS_TRADES = 600
DISPERSION = 0.5
KEEP_DSCORE, KEEP_FOLDS, KEEP_TOP_SLACK = 2.0, 7, 2.0  # bp, 폴드 수(11 중), bp
VOID_SCORE = -999.0
N_NULL = 100
CFG_BOUNDS = {
    "max_pos": (1, 10),
    "max_entries": (1, 30),
    "stop": (STOP_FLOOR, 0.08),
    "exit_g": (0.0, 0.01),
    "exit_w": (5, 60),
    "struct_b": (0.002, 0.02),
}
CFG_INT = {"max_pos", "max_entries", "exit_w"}
CFG_CHOICES = {"exit_kind": ("close", "vwap", "ma", "struct")}
CFG_DEFAULT = {
    "max_pos": 5,
    "max_entries": 10,
    "stop": 0.03,
    "exit_kind": "close",
    "exit_g": 0.0,
    "exit_w": 20,
    "struct_b": 0.005,
}
BAR_KEYS = ("O", "H", "L", "C", "Vol", "vwap", "na", "nf", "nx", "nm")
STATIC_KEYS = (
    "uni",
    "prev_close",
    "ret1",
    "ret5",
    "ret20",
    "prev_range",
    "tv_prev",
    "tv_rank",
    "kosdaq",
    "pre_all",
    "pre_feat",
    "pre_move",
    "hist5",
)
FEAT_RE = re.compile(r"특징주")
EXCL_RE = re.compile(r"단독")
MOVE_RE = re.compile(
    r"급등|급락|상한가|하한가|신고가|신저가|강세|약세|껑충|폭등|폭락|들썩|치솟|곤두박질|↑|↓|%\s*(상승|하락|올라|내려)"
)
_G: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - _G.get('t0', time.time()):6.0f}s] {msg}", file=sys.stderr, flush=True)


def half_of(code: str) -> int:
    """종목 절반 잠금 — 코드 sha256 첫 바이트 짝홀. 0 = 탐색, 1 = 잠금."""
    return hashlib.sha256(str(code).encode()).digest()[0] & 1


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def folds() -> list[tuple[dt.date, dt.date, dt.date]]:
    """(학습 시작, 시험 시작, 시험 끝 미포함). 마지막 폴드는 09-01..10-12."""
    out = []
    for i, ts in enumerate(TEST_STARTS):
        te = TEST_STARTS[i + 1] if i + 1 < len(TEST_STARTS) else END + dt.timedelta(days=1)
        y, m = ts.year, ts.month - TRAIN_MONTHS
        while m < 1:
            y, m = y - 1, m + 12
        out.append((dt.date(y, m, 1), ts, te))
    return out


# ---------------------------------------------------------------- train.py 검사(C1·C3)

ALLOWED_IMPORTS = {
    "numpy",
    "pandas",
    "math",
    "functools",
    "itertools",
    "collections",
    "dataclasses",
    "typing",
    "__future__",
}
BANNED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
    "memoryview",
    "help",
    "exit",
    "quit",
    "print",
}
BANNED_ATTR = re.compile(
    r"^(load|loads|save|savez|savez_compressed|savetxt|loadtxt|genfromtxt|fromfile|tofile|memmap|fromregex|"
    r"DataSource|ctypeslib|ctypes|f2py|testing|lib|os|sys|system|eval|query|setflags|flags|base|"
    r"read_\w+|to_(pickle|csv|parquet|sql|json|excel|hdf|feather|clipboard|html|latex|markdown|xml|stata|orc)|"
    r"year|month|day|dayofweek|weekday|quarter|date|dates|days|codes|code)$"
)
YEAR_LIKE = range(1990, 2031)


def lint_train(src: str) -> list[str]:
    """train.py 허용 목록 — 위반 목록(비면 통과). 완전한 샌드박스는 아니다: 우회 시도는 program.md 위반 = 결과 무효."""
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    bad.append(f"line {node.lineno}: import {m}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            bad.append(f"line {node.lineno}: 이름 {node.id}")
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or BANNED_ATTR.match(node.attr)
        ):
            bad.append(f"line {node.lineno}: 속성 .{node.attr}")
        elif isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, str) and (re.search(r"\d{6}", v) or re.search(r"(19|20)\d{2}", v)):
                bad.append(f"line {node.lineno}: 종목코드·날짜 같은 문자열 {v!r}")
            if isinstance(v, int) and not isinstance(v, bool) and v in YEAR_LIKE:
                bad.append(f"line {node.lineno}: 연도 같은 정수 {v}")
    return bad


def load_strategy(path: Path = TRAIN) -> dict:
    """train.py 를 **새 모듈 인스턴스**로 로드(캐시 공유 없음). 허용 목록 위반이면 거부."""
    src = Path(path).read_text(encoding="utf-8")
    bad = lint_train(src)
    if bad:
        raise SystemExit("train.py 허용 목록 위반 — 평가 거부:\n  " + "\n  ".join(bad))
    spec = importlib.util.spec_from_file_location(f"ar_train_{time.time_ns()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    S = mod.get_strategy()
    for k in ("params", "signals", "cfg"):
        if k not in S:
            raise SystemExit(f"get_strategy() 에 {k!r} 가 없다")
    S["sha"] = hashlib.sha256(src.encode()).hexdigest()[:16]
    return S


# ---------------------------------------------------------------- 캐시(--build, 탐색 절반만 디스크)


def _dsn() -> str:
    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        for line in (ROOT.parent / "quant-airflow" / ".env").read_text().splitlines():
            if line.startswith("KR_QUANT_DB="):
                dsn = line.split("=", 1)[1].strip().strip("\"'")
    if not dsn:
        raise SystemExit("KR_QUANT_DB 가 없다")
    return dsn


def _archive_dir() -> Path:
    for p in (
        ROOT / "data" / "toss_universe_42m",
        ROOT.parent / "daytrade-it" / "data" / "toss_universe_42m",
    ):
        if p.is_dir():
            return p
    raise SystemExit("data/toss_universe_42m 이 없다")


def load_articles(codes: set[str], lo: dt.datetime, hi: dt.datetime) -> pd.DataFrame:
    """토스 종목 뉴스 아카이브 → (code, tau = createdAt + LAG_MIN, feat, excl, move). (id, code) 중복 제거."""
    rows = []
    arch = _archive_dir()
    for code in sorted(codes):
        f = arch / f"{code}.jsonl"
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    a = json.loads(line)
                    t = dt.datetime.fromisoformat(a["createdAt"][:19])
                except (ValueError, KeyError, TypeError):
                    continue
                if lo <= t <= hi:
                    title = a.get("title") or ""
                    rows.append(
                        (
                            a.get("id"),
                            code,
                            t,
                            bool(FEAT_RE.search(title)),
                            bool(EXCL_RE.search(title)),
                            bool(MOVE_RE.search(title)),
                        )
                    )
    df = pd.DataFrame(rows, columns=["id", "code", "t", "feat", "excl", "move"]).drop_duplicates(
        ["id", "code"]
    )
    df["tau"] = df["t"] + pd.Timedelta(minutes=LAG_MIN)
    return df.drop(columns=["id", "t"]).sort_values("tau").reset_index(drop=True)


def build_half(con, half: int) -> dict:  # noqa: ANN001, C901
    """한 절반의 캐시. 날짜·종목코드는 ``days``·``codes`` 에만 두고 V 로 넘기지 않는다."""
    from costs import MEASURED_SPREAD
    from krx_quant_core.costs.model import round_trip_cost
    from krx_quant_core.market.codes import Market
    from krx_quant_core.market.limits import limit_down_price, limit_up_price
    from krx_quant_core.market.ticks import tick_size_int
    from krx_session import regular_daily

    mk = dict(
        pd.read_sql(
            "select code, market from stocks where market in ('거래소','코스닥')", con
        ).values
    )
    daily = regular_daily(con, START - dt.timedelta(days=60), END)
    daily = daily[daily["code"].isin(mk)].sort_values(["code", "date"])
    cal = sorted(daily.loc[daily["code"] == "005930", "date"])  # 거래일 달력
    g = daily.groupby("code")
    for k, col in (("p1", "close"), ("p2", "close"), ("p6", "close"), ("p21", "close")):
        daily[k] = g[col].shift(int(k[1:]))
    daily["prev_tv"] = g["trade_value"].shift(1)
    daily["prev_rng"] = (g["high"].shift(1) - g["low"].shift(1)) / daily["p1"]
    daily["next_open"] = g["open"].shift(-1)
    daily["next_date"] = g["date"].shift(-1)
    dk = daily.set_index(["date", "code"])

    cur = con.cursor()
    sess = [d for d in cal if START <= d <= END]
    uni_by_day: dict[dt.date, list[str]] = {}
    for d in sess:
        x = daily[(daily["date"] == d) & np.isfinite(daily["prev_tv"])]
        top = (
            x.sort_values(["prev_tv", "code"], ascending=[False, True])["code"]
            .head(UNIVERSE)
            .tolist()
        )
        uni_by_day[d] = [c for c in top if half_of(c) == half]
    codes_all = set().union(*uni_by_day.values())
    art = load_articles(
        codes_all,
        dt.datetime.combine(cal[0], dt.time(0)),
        dt.datetime.combine(END, dt.time(15, 20)),
    )
    art_by_code = dict(tuple(art.groupby("code")))
    log(f"half {half}: 기사 {len(art):,}건, 종목 {len(codes_all)}")

    nmax = max(len(v) for v in uni_by_day.values())
    Ds = len(sess)
    f32 = np.float32
    bars = {k: np.full((Ds, nmax, M), np.nan, f32) for k in ("O", "H", "L", "C")}
    bars["Vol"] = np.zeros((Ds, nmax, M), f32)
    news = {k: np.zeros((Ds, nmax, M), np.uint8) for k in ("na", "nf", "nx", "nm")}
    st = {k: np.full((Ds, nmax), np.nan, f32) for k in STATIC_KEYS if k != "uni"}
    uni = np.zeros((Ds, nmax), bool)
    lu, ld, nxt = (np.full((Ds, nmax), np.nan, f32) for _ in range(3))
    cost_o, cost_l = (np.full((Ds, nmax), np.nan, f32) for _ in range(2))
    codes_by_day, keep_days, diag = [], [], {"days_skipped": [], "news_cov": [], "bar_cov": []}
    ci = {d: i for i, d in enumerate(cal)}
    di = 0
    for d in sess:
        codes = uni_by_day[d]
        lo9 = dt.datetime.combine(d, dt.time(9, 0))
        cur.execute(
            "select code, ts, open, high, low, close, volume from minute_bars where code = any(%s) "
            "and ts >= %s and ts < %s",
            (codes, lo9, lo9 + dt.timedelta(minutes=M)),
        )
        rows = cur.fetchall()
        have = {r[0] for r in rows}
        if len(have) < 0.5 * len(codes):
            diag["days_skipped"].append((str(d), len(have), len(codes)))
            continue
        row_of = {c: i for i, c in enumerate(codes)}
        for code, ts, o, h, low, c, v in rows:
            m = ts.hour * 60 + ts.minute - 540
            if 0 <= m < M and v and v > 0:
                i = row_of[code]
                (
                    bars["O"][di, i, m],
                    bars["H"][di, i, m],
                    bars["L"][di, i, m],
                    bars["C"][di, i, m],
                ) = o, h, low, c
                bars["Vol"][di, i, m] = v
        p = cal[ci[d] - 1]
        p5 = cal[ci[d] - 6] if ci[d] >= 6 else cal[0]
        pre_lo, pre_hi = pd.Timestamp(dt.datetime.combine(p, dt.time(15, 0))), pd.Timestamp(lo9)
        h_lo = pd.Timestamp(dt.datetime.combine(p5, dt.time(15, 0)))
        n_cov = 0
        for code, i in row_of.items():
            r = dk.loc[(d, code)]
            pc = float(r["p1"])
            if not (pc > 0) or code not in have:
                continue
            uni[di, i] = True
            st["prev_close"][di, i] = pc
            st["ret1"][di, i] = pc / r["p2"] - 1
            st["ret5"][di, i] = pc / r["p6"] - 1
            st["ret20"][di, i] = pc / r["p21"] - 1
            st["prev_range"][di, i] = r["prev_rng"]
            st["tv_prev"][di, i] = math.log1p(max(float(r["prev_tv"]), 0.0))
            st["tv_rank"][di, i] = i / max(len(codes) - 1, 1)
            st["kosdaq"][di, i] = 1.0 if mk[code] == "코스닥" else 0.0
            lu[di, i], ld[di, i] = limit_up_price(pc), limit_down_price(pc)
            nd = r["next_date"]
            nxt[di, i] = r["next_open"] if isinstance(nd, dt.date) and nd <= END else np.nan
            tick = tick_size_int(pc)
            for at_open, arr in ((True, cost_o), (False, cost_l)):
                sp = max(MEASURED_SPREAD[(mk[code], at_open)], tick / pc)
                arr[di, i] = round_trip_cost(
                    d,
                    Market.KOSDAQ if mk[code] == "코스닥" else Market.KOSPI,
                    slippage_one_way=sp / 2,
                )
            a = art_by_code.get(code)
            if a is None:
                st["pre_all"][di, i] = st["pre_feat"][di, i] = st["pre_move"][di, i] = st["hist5"][
                    di, i
                ] = 0
                continue
            tau = a["tau"]
            pre = a[(tau > pre_lo) & (tau <= pre_hi)]
            st["pre_all"][di, i], st["pre_feat"][di, i], st["pre_move"][di, i] = (
                len(pre),
                pre["feat"].sum(),
                pre["move"].sum(),
            )
            st["hist5"][di, i] = int(((tau > h_lo) & (tau <= pre_lo)).sum())
            ins = a[(tau > pre_hi) & (tau <= pre_hi + pd.Timedelta(minutes=M))]
            if len(pre) or len(ins):
                n_cov += 1
            if len(ins):
                mm = np.clip(
                    np.ceil((ins["tau"] - pre_hi).dt.total_seconds().to_numpy() / 60).astype(int)
                    - 1,
                    0,
                    M - 1,
                )
                for key, sel in (("na", None), ("nf", "feat"), ("nx", "excl"), ("nm", "move")):
                    w = np.ones(len(mm)) if sel is None else ins[sel].to_numpy(float)
                    news[key][di, i] = np.minimum(
                        np.bincount(mm, weights=w, minlength=M), 255
                    ).astype(np.uint8)
        diag["news_cov"].append((str(d), n_cov / max(uni[di].sum(), 1)))
        diag["bar_cov"].append((str(d), int(uni[di].sum()), len(codes)))
        codes_by_day.append(codes)
        keep_days.append(d)
        di += 1
        if di % 20 == 0:
            log(f"half {half}: {di} 일")
    sl = slice(0, di)
    D = {k: v[sl] for k, v in {**bars, **news}.items()}
    D.update({k: v[sl] for k, v in st.items()})
    D.update(
        uni=uni[sl], lu=lu[sl], ld=ld[sl], next_px=nxt[sl], cost_o=cost_o[sl], cost_l=cost_l[sl]
    )
    D.update(days=keep_days, codes=codes_by_day, half=half, diag=diag)
    _derive(D)
    return D


def _derive(D: dict) -> None:
    """원 분봉(빈 봉 NaN)에서 파생: has·이월 종가(PC)·봉 시가 시점 가격(PO)·VWAP·시장 경로."""
    O, C, V = D["O"], D["C"], D["Vol"]
    has = (V > 0) & np.isfinite(O)
    pc0 = np.where(np.isfinite(D["prev_close"]), D["prev_close"], np.nan)[:, :, None]
    Cf = pd.DataFrame(C.reshape(-1, M).T).ffill().to_numpy().T.reshape(C.shape)
    Cf = np.where(np.isfinite(Cf), Cf, pc0).astype(np.float32)
    prevc = np.concatenate([np.broadcast_to(pc0, C.shape[:2] + (1,)), Cf[:, :, :-1]], axis=2)
    D["has"] = has
    D["PC"] = Cf  # 봉 m 종가 시점의 마지막 거래가
    D["PO"] = np.where(has, O, prevc).astype(np.float32)  # 봉 m 시가 시점의 가격
    tp = np.where(has, C, 0.0).astype(np.float64) * V
    cv = np.cumsum(V, axis=2, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        D["vwap"] = np.where(cv > 0, np.cumsum(tp, axis=2) / cv, np.nan).astype(np.float32)
        rel = D["PC"] / D["PO"][:, :, :1]
    u = D["uni"][:, :, None] & np.isfinite(rel)
    with np.errstate(invalid="ignore"):
        D["mkt"] = (np.where(u, rel, 0).sum(1) / np.maximum(u.sum(1), 1)).astype(np.float32)


def build() -> None:
    import psycopg2

    CACHE.mkdir(parents=True, exist_ok=True)
    con = psycopg2.connect(_dsn())
    con.set_session(readonly=True)
    D = build_half(con, 0)  # 탐색 절반만 디스크에(C1)
    path = CACHE / "search_s0.pkl"
    with open(path, "wb") as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
    cov = np.array([c for _, c in D["diag"]["news_cov"]])
    log(
        f"탐색 절반 {len(D['days'])}일 × 최대 {D['O'].shape[1]}행, 건너뛴 날 {D['diag']['days_skipped']}"
    )
    log(f"뉴스 커버리지(행 중 기사 있는 비율) 일 중앙 {np.median(cov):.3f}, 최소 {cov.min():.3f}")
    write_manifest()


def write_manifest() -> None:
    """의존 파일·캐시 sha256 → MANIFEST.json. 태그 뒤에는 MANIFEST 자체가 바뀌면 평가 거부(integrity)."""
    man = {p: sha_file(ROOT / p) for p in GUARDED_FILES if (ROOT / p).exists()}
    missing = [p for p in GUARDED_FILES if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"의존 파일이 없다: {missing}")
    man["cache/search_s0.pkl"] = sha_file(CACHE / "search_s0.pkl")
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
    log(f"MANIFEST.json 기록 — {len(man)} 항목")


def load_search() -> dict:
    with open(CACHE / "search_s0.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- 무결성(C1·M1)


def _git(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)


def tagged() -> bool:
    return _git("rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG}").returncode == 0


def integrity(check_cache: bool = True) -> None:
    if not tagged():
        print(f"경고: 태그 {PREREG_TAG} 없음 — 사전등록 전 시험 실행", file=sys.stderr)
        return
    guarded = (*GUARDED_FILES, "scripts/eval/autoresearch/MANIFEST.json")
    changed = _git("log", "--format=%h %s", f"{PREREG_TAG}..HEAD", "--", *guarded).stdout.strip()
    dirty = _git("status", "--porcelain", "--", *guarded).stdout.strip()
    if changed or dirty:
        raise SystemExit(
            f"평가기·의존 파일이 사전등록({PREREG_TAG}) 뒤 바뀌었다 — 평가 거부.\n{changed}\n{dirty}"
        )
    man = json.loads(MANIFEST.read_text())
    for p, h in man.items():
        if p.startswith("cache/"):
            if check_cache and sha_file(CACHE / p[6:]) != h:
                raise SystemExit(f"캐시 {p} sha 불일치 — 평가 거부")
        elif sha_file(ROOT / p) != h:
            raise SystemExit(f"{p} sha 불일치 — 평가 거부")


def _append(name: str, rec: dict) -> None:
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    for d in LEDGER_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        with open(d / name, "a", encoding="utf-8") as f:
            f.write(line)


def _read_ledger(name: str) -> list:
    """두 사본 중 긴 쪽(한쪽이 지워져도 남는다)."""
    out = []
    for d in LEDGER_DIRS:
        p = d / name
        if p.exists():
            out.append(
                [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            )
    return max(out, key=len) if out else []


def n_trials() -> int:
    return len({r["train_sha"] for r in _read_ledger("RUNS.jsonl")})


# ---------------------------------------------------------------- V 와 청산 메뉴


def view(D: dict, upto: tuple[int, int] | None = None) -> dict:
    """train.py 에 넘기는 묶음 — 읽기 전용 복사본, 날짜·종목코드 없음(C3·M7).

    가격 O·H·L·C 는 빈 봉을 직전 거래가로 채운 값, Vol·뉴스 수는 빈 봉 0. ``upto=(d, m)`` 이면 날짜 d 까지만 남기고
    d 의 봉 m+1 이후를 NaN·0 으로 지운다(미래정보 검사)."""
    nd = len(D["days"]) if upto is None else upto[0] + 1
    has = D["has"][:nd]
    fill = D["PC"][:nd]
    V = {
        "O": np.where(has, D["O"][:nd], fill),
        "H": np.where(has, D["H"][:nd], fill),
        "L": np.where(has, D["L"][:nd], fill),
        "C": fill.copy(),
        "Vol": D["Vol"][:nd].copy(),
        "vwap": D["vwap"][:nd].copy(),
        "mkt": D["mkt"][:nd].copy(),
    }
    for k in ("na", "nf", "nx", "nm"):
        V[k] = D[k][:nd].copy()
    for k in STATIC_KEYS:
        V[k] = np.array(D[k][:nd], copy=True)
    if upto is not None:
        d, m = upto
        for k in ("O", "H", "L", "C", "vwap"):
            V[k][d, :, m + 1 :] = np.nan
        for k in ("Vol", "na", "nf", "nx", "nm"):
            V[k][d, :, m + 1 :] = 0
        V["mkt"][d, m + 1 :] = np.nan
    for v in V.values():
        v.setflags(write=False)
    return V


def clamp_cfg(cfg: dict) -> dict:
    """엔진 설정 — 메뉴 밖 키는 버리고(시간 손절·익절·추적 없음), 숫자는 범위로 자른다(M2)."""
    out = dict(CFG_DEFAULT)
    for k, (lo, hi) in CFG_BOUNDS.items():
        if k in cfg:
            v = float(np.clip(cfg[k], lo, hi))
            out[k] = int(round(v)) if k in CFG_INT else v
    for k, ch in CFG_CHOICES.items():
        if k in cfg:
            if cfg[k] not in ch:
                raise ValueError(f"cfg[{k!r}]={cfg[k]!r} 는 허용 밖: {ch}")
            out[k] = cfg[k]
    return out


def _sma(C: np.ndarray, w: int) -> np.ndarray:
    cs = np.cumsum(np.nan_to_num(C, nan=0.0), axis=-1, dtype=np.float64)
    out = np.full(C.shape, np.nan)
    out[..., w - 1 :] = (
        cs[..., w - 1 :] - np.concatenate([np.zeros(C.shape[:-1] + (1,)), cs[..., :-w]], axis=-1)
    ) / w
    return out


def _exit_cond(D: dict, d: int, c: dict) -> np.ndarray | None:
    """봉 종가에 판정하는 조건 청산(N×M bool) — 다음 체결 봉 시가에 나간다(M5)."""
    if c["exit_kind"] == "vwap":
        with np.errstate(invalid="ignore"):
            return D["PC"][d] < D["vwap"][d] * (1 - c["exit_g"])
    if c["exit_kind"] == "ma":
        with np.errstate(invalid="ignore"):
            return D["PC"][d] < _sma(D["PC"][d], c["exit_w"])
    return None


def _locked_down(D: dict, d: int, n: int, k: int) -> bool:
    ld = D["ld"][d, n]
    return bool(np.isfinite(ld) and D["H"][d, n, k] == D["L"][d, n, k] and D["L"][d, n, k] <= ld)


def _exit(
    D: dict, d: int, n: int, e: int, c: dict, cond: np.ndarray | None
) -> tuple[int, str, float, str]:
    """(청산 봉, 시각 'o'|'c', 가격, 사유). 같은 봉 안 순서: 손절(저가) → 조건(종가) → 다음 체결 봉 시가."""
    from krx_quant_core.market.ticks import tick_size_int

    O, L, C, has = D["O"][d, n], D["L"][d, n], D["C"][d, n], D["has"][d, n]
    E = float(O[e])
    stop = E * (1 - c["stop"])
    if c["exit_kind"] == "struct":
        low_before = np.nanmin(np.where(has[:e], L[:e], np.nan)) if has[:e].any() else np.nan
        if np.isfinite(low_before):
            stop = max(min(low_before * (1 - c["struct_b"]), E * (1 - STOP_FLOOR)), stop)
    k = e
    while k < M:
        if has[k] and not _locked_down(D, d, n, k):
            if k > e and O[k] <= stop:
                return k, "o", float(O[k]), "gap_stop"
            if L[k] <= stop:
                px = max(float(L[k]), stop - tick_size_int(stop))
                return k, "c", px, "stop"
        if k >= FLAT:
            if has[k] and not _locked_down(D, d, n, k):
                return k, ("c" if k == FLAT else "o"), float(C[k] if k == FLAT else O[k]), "flat"
        elif cond is not None and cond[n, k]:
            for k2 in range(k + 1, M):
                if has[k2] and not _locked_down(D, d, n, k2):
                    if k2 > FLAT:
                        return k2, "o", float(O[k2]), "flat"
                    return k2, "o", float(O[k2]), "cond"
            break
        k += 1
    nx = D["next_px"][d, n]
    if np.isfinite(nx):
        return M - 1, "c", float(nx), "overnight"
    last = float(D["PC"][d, n, M - 1])
    return (
        M - 1,
        "c",
        last * (1 - D["cost_l"][d, n]),
        "overnight_nopx",
    )  # 다음 날 가격이 없으면 한 번 더 비용


def simulate_day(D: dict, d: int, cand: np.ndarray, score: np.ndarray, c: dict) -> list[tuple]:
    """하루 — 신호 봉 m → m+1 시가 진입(M5), 빈 슬롯에 score 높은 순, 종목당 하루 1회."""
    uni, has, O, lu = D["uni"][d], D["has"][d], D["O"][d], D["lu"][d]
    cm = np.asarray(cand, bool) & uni[:, None]
    cm[:, LAST_SIGNAL + 1 :] = False
    if not cm.any():
        return []
    sc = np.where(np.isfinite(score), score, -np.inf)
    cond = _exit_cond(D, d, c)
    entered = np.zeros(len(uni), bool)
    free_at: list[int] = []
    out = []
    for m in np.flatnonzero(cm.any(0)):
        e = m + 1
        nopen = sum(f > e for f in free_at)
        room = min(c["max_pos"] - nopen, c["max_entries"] - len(out))
        if room <= 0:
            if len(out) >= c["max_entries"]:
                break
            continue
        with np.errstate(invalid="ignore"):
            ok = cm[:, m] & ~entered & has[:, e] & (O[:, e] < lu)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        idx = idx[np.lexsort((idx, -sc[idx, m]))][:room]
        for n in idx:
            k, when, px, why = _exit(D, d, n, e, c, cond)
            E = float(O[n, e])
            cost = float(D["cost_o"][d, n] if e < OPEN_WINDOW else D["cost_l"][d, n])
            pa = D["PO"][d, :, e]
            pb = D["PO"][d, :, k] if when == "o" else D["PC"][d, :, k]
            u = uni & np.isfinite(pa) & np.isfinite(pb) & (pa > 0)
            bench = float(np.mean(pb[u] / pa[u]) - 1) if u.any() else 0.0
            net = px / E - 1 - cost
            out.append((d, int(n), int(e), int(k), net, net - bench, cost, why))
            entered[n] = True
            free_at.append(k if when == "o" else k + 1)
    return out


TRADE_COLS = ["day", "row", "entry", "exit", "net", "alpha", "cost", "why"]


def simulate(D: dict, sig: dict, cfg: dict, days: np.ndarray | None = None) -> pd.DataFrame:
    cand, score = sig["cand"], sig["score"]
    if np.shape(cand) != D["O"].shape or np.shape(score) != D["O"].shape:
        raise ValueError(f"cand·score 는 (날짜, 행, 분) = {D['O'].shape} 모양이어야 한다")
    c = clamp_cfg(cfg)
    rows = []
    for d in range(len(D["days"])) if days is None else days:
        rows += simulate_day(D, int(d), cand[d], np.asarray(score[d], float), c)
    return pd.DataFrame(rows, columns=TRADE_COLS)


# ---------------------------------------------------------------- 워크포워드


def sample_params(spec: dict, n: int, seed: int) -> tuple[list[dict], np.ndarray]:
    names = list(spec)
    if not names:
        return [{}], np.zeros((1, 0))
    rng = np.random.default_rng(seed)  # 라틴 하이퍼큐브
    U = (np.argsort(rng.random((len(names), n)), axis=1).T + rng.random((n, len(names)))) / n
    pts = []
    for u in U:
        x = {}
        for k, name in enumerate(names):
            s = spec[name]
            if s[0] == "choice":
                x[name] = s[1][min(int(u[k] * len(s[1])), len(s[1]) - 1)]
            else:
                v = s[1] + u[k] * (s[2] - s[1])
                x[name] = int(round(v)) if s[0] == "int" else float(v)
        pts.append(x)
    return pts, U


def plateau_pick(U: np.ndarray, y: np.ndarray, k: int = K_PLATEAU) -> int:
    """이웃 k 개(자기 포함) **중앙값**이 가장 높은 점 — 고원."""
    if U.shape[1] == 0 or len(y) <= k:
        return int(np.argmax(y))
    dd = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    nb = np.argsort(dd, axis=1)[:, :k]
    return int(np.argmax(np.median(y[nb], axis=1)))


def _point(args: tuple) -> tuple[int, pd.DataFrame | None]:
    j, x, days = args
    D, S, V = _G["D"], _G["S"], _G["V"]
    try:
        return j, simulate(D, S["signals"](V, x), S["cfg"](x), days)
    except Exception as e:  # noqa: BLE001 — 한 점 실패는 그 점만 버린다
        log(f"점 #{j} 실패: {e!r}")
        return j, None


def _pool_map(tasks: list, workers: int) -> list:
    if workers <= 1:
        return [_point(t) for t in tasks]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_point, tasks, chunksize=1)


def day_index(D: dict, lo: dt.date, hi: dt.date) -> np.ndarray:
    return np.array([i for i, d in enumerate(D["days"]) if lo <= d < hi], int)


def walk_forward(D: dict, workers: int, train_path: Path = TRAIN) -> dict:
    """새 모듈로 train.py 로드 → 점 48개를 전 기간에 한 번씩 → 폴드마다 학습 3개월 고원 선택 → 시험 1개월."""
    S = load_strategy(train_path)
    V = view(D)
    _G.update(D=D, S=S, V=V)
    pts, U = sample_params(S["params"], N_EVAL, SEED)
    S["signals"](V, pts[0])  # 부모에서 train.py 캐시를 데운다(fork 공유)
    out = dict(_pool_map([(j, x, None) for j, x in enumerate(pts)], workers))
    res = {"folds": [], "trades": [], "pts": pts}
    for fi, (tr0, ts, te) in enumerate(folds()):
        tri, tei = day_index(D, tr0, ts), day_index(D, ts, te)
        y = np.full(len(pts), -1e4)
        for j in range(len(pts)):
            t = out.get(j)
            if t is None:
                continue
            s = t[t["day"].isin(tri)]
            if len(s) >= MIN_TRAIN_TRADES:
                y[j] = s["alpha"].mean() * 1e4
        j = plateau_pick(U, y)
        t = out.get(j)
        test = t[t["day"].isin(tei)] if t is not None else pd.DataFrame(columns=TRADE_COLS)
        res["folds"].append(
            {
                "fold": fi,
                "test": [str(ts), str(te)],
                "params": pts[j],
                "point": j,
                "train_bp": float(y[j]),
                "alpha_bp": float(test["alpha"].mean() * 1e4) if len(test) else float("nan"),
                "net_bp": float(test["net"].mean() * 1e4) if len(test) else float("nan"),
                "n": int(len(test)),
                "days": int(len(tei)),
            }
        )
        res["trades"].append(test.assign(fold=fi))
    return res


def summarize(res: dict, top_frac: float = 0.0) -> dict:
    tr = pd.concat(res["trades"]) if res["trades"] else pd.DataFrame(columns=TRADE_COLS + ["fold"])
    tr = tr.reset_index(drop=True)
    if top_frac and len(tr):
        k = max(5, int(round(top_frac * len(tr))))
        tr = tr.drop(tr["alpha"].astype(float).nlargest(k).index)
    nf = len(res["folds"])
    al = np.array(
        [
            tr.loc[tr["fold"] == f, "alpha"].mean() * 1e4 if (tr["fold"] == f).any() else np.nan
            for f in range(nf)
        ]
    )
    n = np.array([int((tr["fold"] == f).sum()) for f in range(nf)])
    ndays = sum(f["days"] for f in res["folds"])
    void = bool(
        (n < MIN_FOLD_TRADES).sum() > MAX_THIN_FOLDS
        or n.sum() < MIN_OOS_TRADES
        or not np.isfinite(al).all()
    )
    win, loss = tr.loc[tr["net"] > 0, "net"], tr.loc[tr["net"] <= 0, "net"]
    return {
        "score": float(al.mean() - DISPERSION * al.std(ddof=1)) if not void else VOID_SCORE,
        "void": void,
        "fold_alpha_mean": float(np.nanmean(al)) if np.isfinite(al).any() else float("nan"),
        "folds_pos": int((al > 0).sum()),
        "oos_trades": int(n.sum()),
        "trades_per_day": float(n.sum() / max(ndays, 1)),
        "thin_folds": int((n < MIN_FOLD_TRADES).sum()),
        "alpha_bp": float(tr["alpha"].mean() * 1e4) if len(tr) else float("nan"),
        "net_bp": float(tr["net"].mean() * 1e4) if len(tr) else float("nan"),
        "cost_bp": float(tr["cost"].mean() * 1e4) if len(tr) else float("nan"),
        "payoff_ratio": float(win.mean() / -loss.mean())
        if len(win) and len(loss) and loss.mean() < 0
        else float("nan"),
        "win_rate": float((tr["net"] > 0).mean()) if len(tr) else float("nan"),
        "_fold_alpha": al.tolist(),
    }


def causality(D: dict, res: dict, train_path: Path = TRAIN) -> tuple[bool, str]:
    """미래정보 검사(C2): (날짜, 분) 절단 4곳에서 그 시점까지만 남긴 V 로 새 모듈이 만든 신호 == 전체 신호 앞부분."""
    x = res["folds"][-1]["params"]
    full = load_strategy(train_path)["signals"](view(D), x)
    nd = len(D["days"])
    for frac, m in ((0.35, 45), (0.6, 200), (0.8, 10), (0.97, 320)):
        d = int(nd * frac)
        part = load_strategy(train_path)["signals"](view(D, upto=(d, m)), x)
        a, b = np.asarray(full["cand"], bool), np.asarray(part["cand"], bool)
        if b.shape != (d + 1,) + a.shape[1:]:
            return False, f"cut ({d},{m}): 모양 {b.shape}"
        pre_a = np.concatenate([a[:d].reshape(-1), a[d, :, : m + 1].reshape(-1)])
        pre_b = np.concatenate([b[:d].reshape(-1), b[d, :, : m + 1].reshape(-1)])
        if (pre_a != pre_b).any():
            return False, f"cut ({d},{m}): cand 불일치 {(pre_a != pre_b).sum()}"
        sa = np.concatenate(
            [
                np.asarray(full["score"], float)[:d].reshape(-1),
                np.asarray(full["score"], float)[d, :, : m + 1].reshape(-1),
            ]
        )
        sb = np.concatenate(
            [
                np.asarray(part["score"], float)[:d].reshape(-1),
                np.asarray(part["score"], float)[d, :, : m + 1].reshape(-1),
            ]
        )
        if not np.allclose(sa[pre_a], sb[pre_a], rtol=1e-6, atol=1e-9, equal_nan=True):
            return False, f"cut ({d},{m}): score 불일치"
    return True, "ok"


def beats(s: dict, top: float) -> tuple[bool, str]:
    """keep 판정(M8): Δscore ≥ 2bp ∧ 시험 폴드 ≥ 7/11 개선(쌍 비교) ∧ 상위 1% 제거 점수 악화 ≤ 2bp."""
    if s["void"]:
        return False, "void"
    if not BEST.exists():
        return True, "첫 유효 실행"
    b = json.loads(BEST.read_text())
    better = sum(f > g for f, g in zip(s["_fold_alpha"], b["fold_alpha"], strict=True))
    ok = (
        s["score"] >= b["score"] + KEEP_DSCORE
        and better >= KEEP_FOLDS
        and top >= b["top1"] - KEEP_TOP_SLACK
    )
    return (
        bool(ok),
        f"dscore {s['score'] - b['score']:+.2f}bp · better_folds {better}/{len(b['fold_alpha'])} · dtop1 {top - b['top1']:+.2f}bp",
    )


SHOWN = (
    "score",
    "void",
    "beats_best",
    "beats_detail",
    "lookahead_check",
    "folds_pos",
    "oos_trades",
    "trades_per_day",
    "thin_folds",
    "top1pct_removed_score",
    "alpha_bp",
    "net_bp",
    "cost_bp",
    "payoff_ratio",
    "win_rate",
    "n_trials",
    "train_sha",
    "elapsed_s",
)


def evaluate(workers: int, smoke: bool = False) -> dict:
    integrity()
    _G["t0"] = time.time()
    D = load_search()
    S = load_strategy()
    res = walk_forward(D, workers)
    s = summarize(res)
    ok, why = causality(D, res)
    if not ok:
        s.update(void=True, score=VOID_SCORE)
    s["lookahead_check"] = why
    s["top1pct_removed_score"] = (
        summarize(res, top_frac=0.01)["score"] if not s["void"] else float("nan")
    )
    if smoke:  # 사전등록 전 배선 확인 — 점수·수익은 출력하지 않는다
        print("---")
        for k in ("void", "lookahead_check", "oos_trades", "trades_per_day", "thin_folds"):
            print(f"{k + ':':22s} {s[k]}")
        print("fold_trades:          ", [f["n"] for f in res["folds"]])
        return s
    s["beats_best"], s["beats_detail"] = beats(s, s["top1pct_removed_score"])
    seen = {r["train_sha"] for r in _read_ledger("RUNS.jsonl")}
    s["n_trials"] = len(seen | {S["sha"]})
    s["train_sha"] = S["sha"]
    s["elapsed_s"] = round(time.time() - _G["t0"], 1)
    _append(
        "RUNS.jsonl",
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "train_sha": S["sha"],
            "score": s["score"],
            "void": s["void"],
            "beats_best": s["beats_best"],
            "lookahead": why,
            "tagged": tagged(),
        },
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    LAST.write_text(
        json.dumps({"summary": s, "folds": res["folds"]}, ensure_ascii=False, indent=1, default=str)
    )
    print("---")
    for k in SHOWN:  # 폴드별 값은 숨긴다(M8)
        v = s[k]
        print(f"{k + ':':24s} {v:.4f}" if isinstance(v, float) else f"{k + ':':24s} {v}")
    return s


def accept() -> None:
    integrity(check_cache=False)
    last = json.loads(LAST.read_text())
    cur = load_strategy()["sha"]
    sm = last["summary"]
    if sm["train_sha"] != cur or not sm["beats_best"]:
        raise SystemExit("직전 실행이 현재 train.py 의 beats_best=True 가 아니다 — 갱신 거부")
    BEST.write_text(
        json.dumps(
            {
                "score": sm["score"],
                "top1": sm["top1pct_removed_score"],
                "train_sha": cur,
                "fold_alpha": sm["_fold_alpha"],
            },
            indent=1,
        )
    )
    _append(
        "RUNS.jsonl",
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "train_sha": cur,
            "accept": True,
            "score": sm["score"],
        },
    )
    print(f"BEST 갱신: score {sm['score']:.2f}bp")


# ---------------------------------------------------------------- 졸업 심사(--final, 메인 세션만)


def nw_intercept_t(y: np.ndarray, x: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """y = a + b·x 의 절편과 Newey-West t."""
    X = np.column_stack([np.ones(len(y)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    Sg = (X * e[:, None]).T @ (X * e[:, None])
    for lag in range(1, lags + 1):
        G = (X[lag:] * e[lag:, None]).T @ (X[:-lag] * e[:-lag, None])
        Sg += (1 - lag / (lags + 1)) * (G + G.T)
    Vc = XtX_inv @ Sg @ XtX_inv
    return float(b[0]), float(b[0] / np.sqrt(Vc[0, 0]))


def _null_point(args: tuple) -> float:
    """널 한 번: 폴드마다 그 폴드 파라미터(또는 무작위)로 시험 날짜만, 신호를 kind 대로 흐트린다."""
    kind, rep = args
    D, S, V, res = _G["D"], _G["S"], _G["V"], _G["res"]
    rng = np.random.default_rng(SEED + 7919 * rep + {"rank": 1, "thin": 2, "params": 3}[kind])
    trades = []
    for f in res["folds"]:
        x = f["params"]
        if kind == "params":
            x = sample_params(S["params"], 1, int(rng.integers(1 << 30)))[0][0]
        sig = S["signals"](V, x)
        tei = day_index(D, dt.date.fromisoformat(f["test"][0]), dt.date.fromisoformat(f["test"][1]))
        cand, score = np.asarray(sig["cand"], bool), np.asarray(sig["score"], float)
        if kind in ("rank", "thin"):
            score = rng.random(score.shape)
        if kind == "thin":
            base = D["uni"][:, :, None] & D["has"]
            base[:, :, LAST_SIGNAL + 1 :] = False
            new = np.zeros_like(cand)
            for d in tei:
                k = int((cand[d] & base[d]).sum())
                pool = np.flatnonzero(base[d].reshape(-1))
                if k and len(pool):
                    new[d].reshape(-1)[rng.choice(pool, min(k, len(pool)), replace=False)] = True
            cand = new
        trades.append(
            simulate(D, {"cand": cand, "score": score}, S["cfg"](x), tei).assign(fold=f["fold"])
        )
    return summarize({"folds": res["folds"], "trades": trades})["score"]


def final(workers: int) -> None:
    import psycopg2
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

    integrity()
    used = len(_read_ledger("LOCKBOX.jsonl"))
    if used >= LOCKBOX_MAX:
        raise SystemExit(f"잠금 절반은 이미 {used}회 열렸다(최대 {LOCKBOX_MAX}) — 사전등록")
    if n_trials() == 0:
        raise SystemExit("시행 원장이 비었다 — 루프를 돈 뒤에 실행해야 시행 수 N 이 맞다")
    _G["t0"] = time.time()
    S = load_strategy()
    D = load_search()
    search = walk_forward(D, workers)
    s0 = summarize(search)
    top_s = summarize(search, top_frac=0.01)["score"]
    _G.update(D=D, S=S, V=view(D), res=search)
    with mp.get_context("fork").Pool(workers) as pool:
        nulls = {
            k: pool.map(_null_point, [(k, r) for r in range(N_NULL)], chunksize=1)
            for k in ("rank", "thin", "params")
        }
    log("널 완료")
    tr = pd.concat(search["trades"])
    daily = tr.groupby("day").agg(net=("net", "mean"), alpha=("alpha", "mean"))
    mret = np.array([float(D["mkt"][d, FLAT] / D["mkt"][d, 1] - 1) for d in daily.index])
    a_t = nw_intercept_t(daily["net"].to_numpy(), mret)
    N = n_trials()
    dsr = deflated_sharpe_from_sample(daily["alpha"].to_numpy(), N)
    con = psycopg2.connect(_dsn())
    con.set_session(readonly=True)
    DL = build_half(con, 1)  # 잠금 절반 — 메모리에서만
    lock = walk_forward(DL, workers)
    s1 = summarize(lock)
    bars = {
        "b1_search_positive": s0["score"] > 0 and s0["net_bp"] > 0,
        "b2_beats_nulls_p95": all(
            s0["score"] > float(np.quantile(v, 0.95)) for v in nulls.values()
        ),
        "b3_top1pct_removed_positive": top_s > 0,
        "b4_lockbox": s1["score"] > 0 and s1["folds_pos"] >= 7 and not s1["void"],
        "b5_dsr_ge_095": float(dsr.get("prob_deflated_sharpe") or 0) >= 0.95,
        "b6_trades": not s0["void"],
        "b7_factor_t": a_t[1] > 2.0,
    }
    out = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_sha": S["sha"],
        "n_trials": N,
        "search": s0,
        "search_top1pct_removed": top_s,
        "nulls_p95": {k: float(np.quantile(v, 0.95)) for k, v in nulls.items()},
        "nulls_median": {k: float(np.median(v)) for k, v in nulls.items()},
        "lockbox": s1,
        "factor_daily_net": {"alpha": a_t[0], "t_nw": a_t[1]},
        "dsr_daily_alpha": dsr,
        "bars": bars,
        "pass": all(bars.values()),
        "folds_search": search["folds"],
        "folds_lockbox": lock["folds"],
    }
    _append("LOCKBOX.jsonl", out)  # 잠금 사용의 증인(덧붙이기, 두 경로)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument(
        "--manifest", action="store_true", help="캐시는 두고 MANIFEST.json 만 다시 쓴다(태그 전)"
    )
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="사전등록 전 배선 확인(점수 출력 없음)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    _G["t0"] = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.build:
        build()
    elif a.manifest:
        if tagged():
            raise SystemExit(
                f"태그 {PREREG_TAG} 뒤에는 MANIFEST 를 다시 쓰지 않는다 — 정정 사전등록·새 태그로"
            )
        write_manifest()
    elif a.final:
        final(a.workers)
    elif a.accept:
        accept()
    else:
        evaluate(a.workers, smoke=a.smoke)


if __name__ == "__main__":
    main()

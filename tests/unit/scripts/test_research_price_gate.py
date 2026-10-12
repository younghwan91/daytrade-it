"""연구 코드가 상한가·추격 판정에 일봉 종가를 직접 넣지 못하게 막는다.

2026-09-14 부터 ``daily_bars``·``daily_bars_adjusted`` 의 종가·고가에 KRX 애프터마켓(16:00~20:00)이 섞였다.
가격제한 기준가는 전일 **정규장** 종가인데, ``rr_entry_filter``·``volume_entry`` 는 ``limit_up_price(daily_bars 전일
종가)`` 를 썼다(2단계를 돌렸다면 오염). 정문은 ``scripts/eval/krx_session.py`` (``regular_daily``).

규칙: ``scripts/eval``·``scripts/research`` 의 파일이 가격제한·추격 식별자(``limit_up``·``limit_down``·``chase``)를 쓰면서
일봉을 직접 읽으면(SQL ``FROM daily_bars[_adjusted]`` 또는 ``get_adjusted_daily_bars``) 위반이다. 예외는
:data:`ALLOWED` 에 이유와 함께 적힌 파일뿐이다. 주석·독스트링은 보지 않는다(AST).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCOPES = (REPO / "scripts" / "eval", REPO / "scripts" / "research")
GATE = "krx_session"

#: 파일명 → 예외 이유. 새 연구 파일은 여기 넣지 말고 krx_session 을 쓴다.
ALLOWED = {
    "news_multiday_drift.py": (
        "기각된 연구(2026-10-05). limit_up 플래그를 일봉 원 종가 비율로 만든다 — 2026-09-14 이후 날짜는 "
        "오염이라 다시 돌리지 않는다(파일 머리 경고)."
    ),
    "replay_parity.py": (
        "실매매 리플레이. no-chase 는 R17(라이브가 애프터마켓 포함 일봉 종가를 쓴다)을 그대로 흉내 내고, "
        "가격제한 기준가는 krx_session.last_regular_closes(R18)로 따로 만든다. daily_bars_adjusted 는 "
        "vintage 플래그에만 쓴다."
    ),
}

_LIMIT = re.compile(r"limit_?(up|down)|chase", re.IGNORECASE)
_DAILY_SQL = re.compile(r"\bfrom\s+daily_bars(_adjusted)?\b", re.IGNORECASE)
_DAILY_CALL = "get_adjusted_daily_bars"


def _identifiers(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.alias):
            out.add(node.asname or node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.arg | ast.keyword) and node.arg:
            out.add(node.arg)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.isidentifier()
        ):
            # 열·딕셔너리 키 이름("limit_up")만 — SQL 같은 긴 문자열은 아래에서 따로 본다.
            out.add(node.value)
    return out


def _direct_daily_reads(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _DAILY_SQL.search(node.value)
        ) or (isinstance(node, ast.Attribute) and node.attr == _DAILY_CALL):
            lines.append(node.lineno)
    return sorted(set(lines))


def violations(src: str) -> list[str]:
    """가격제한·추격 식별자 + 일봉 직접 읽기가 한 파일에 있으면 그 읽기 줄마다 한 건."""
    tree = ast.parse(src)
    limits = sorted(i for i in _identifiers(tree) if _LIMIT.search(i))
    if not limits:
        return []
    return [
        f"{ln}행: 일봉 직접 읽기 + 가격제한·추격({', '.join(limits[:3])}) — {GATE}.regular_daily 로 읽는다"
        for ln in _direct_daily_reads(tree)
    ]


def _research_files() -> list[Path]:
    return sorted(p for d in SCOPES for p in d.glob("*.py") if p.stem != GATE)


def test_research_scripts_do_not_feed_daily_closes_into_limit_or_chase() -> None:
    bad = {
        p.name: v
        for p in _research_files()
        if p.name not in ALLOWED and (v := violations(p.read_text()))
    }
    assert not bad, bad


def test_allowlist_is_live() -> None:
    """예외 목록의 파일이 사라졌거나 더는 규칙에 걸리지 않으면 목록에서 뺀다(죽은 예외 금지)."""
    names = {p.name: p for p in _research_files()}
    for name, why in ALLOWED.items():
        assert why.strip()
        assert name in names, f"{name} 가 없다 — ALLOWED 에서 뺀다"
        assert violations(names[name].read_text()), f"{name} 는 이제 규칙에 안 걸린다 — 뺀다"


# -- 위반 주입: 규칙이 죽으면 이 테스트들이 빨개진다 -----------------------------------------


ORIGINAL_RR = """
from krx_quant_core.market.limits import limit_up_price
import pandas as pd

def load_daily(con, first, last):
    d = pd.read_sql(
        "select code, date, high, close, trade_value from daily_bars where date >= %s and date <= %s",
        con, params=(first, last),
    )
    d["prev_close"] = d.groupby("code")["close"].shift(1)
    return d

def simulate(prev_close):
    return limit_up_price(prev_close)
"""


def test_flags_the_original_rr_entry_filter_pattern() -> None:
    v = violations(ORIGINAL_RR)
    assert len(v) == 1 and v[0].startswith("7행")


def test_flags_adjusted_table_and_db_client_reads() -> None:
    adjusted = 'from x import limit_down_price\nQ = "SELECT close FROM daily_bars_adjusted"\n'
    client = "def no_chase(db, t):\n    return db.get_adjusted_daily_bars(t)[-1].close\n"
    assert violations(adjusted)
    assert violations(client)


def test_flags_ratio_flag_without_core_function() -> None:
    """core 함수 없이 LIMIT_UP 비율 상수로 판정해도 걸린다(news_multiday_drift 꼴)."""
    src = 'LIMIT_UP = 0.295\nQ = "select code, date, close from daily_bars"\n'
    assert violations(src)


def test_gate_reads_pass() -> None:
    src = (
        "from krx_quant_core.market.limits import limit_up_price\n"
        "from krx_session import regular_daily\n"
        "def f(con, a, b):\n"
        "    d = regular_daily(con, a, b)\n"
        "    return limit_up_price(d['close'].iloc[-1])\n"
    )
    assert violations(src) == []


def test_comments_and_unrelated_daily_reads_pass() -> None:
    """주석의 상한가 언급, 가격제한과 무관한 일봉 읽기는 이 규칙의 대상이 아니다."""
    src = (
        "# limit_up_price(prev_close) 를 쓰지 않는다\n"
        'Q = "select date from daily_bars_adjusted where code = %s"\n'
    )
    assert violations(src) == []


@pytest.mark.parametrize("name", ["rr_entry_filter.py", "volume_entry.py"])
def test_fixed_scripts_use_the_gate(name: str) -> None:
    src = (REPO / "scripts" / "eval" / name).read_text()
    assert "from krx_session import regular_daily" in src
    assert violations(src) == []

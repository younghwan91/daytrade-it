"""뉴스 사전 필터 — LLM 판정 전에 규칙으로 거른다. 평가와 운영이 같은 규칙을 써야 한다.

패턴은 Toss 종목별 과거 기사(2025-08~2026-09, 매체별 제목 템플릿 빈도)에서 뽑았다.

- robot: 자동 생성 기사. 대부분 이미 일어난 가격·수급을 옮긴다(뉴스핌 로보뉴스
  "[장중수급포착]", 매경 "AI 기자 MK시그널", 연합 "애프터마켓서 N%대 급등").
  뉴스핌 "[리포트 브리핑]"도 로보뉴스다. 증권사 리포트라는 새 정보를 담지만 사용자
  지시(2026-09-15 "로봇 기사는 당연히 빼야지")로 뺀다.
- listing: 시황·공시 목록·표·일정·헤드라인 모음. 특정 회사의 재료가 아니다.
- not_about_company: 제목에 회사명이 없다. Toss 종목 페이지에는 시장 전체 기사도
  섞이고 목록 API의 `stockCodes`는 비어 있어서, 제목이 사실상 유일한 관련성 신호다.

`reports_price_move`는 거르지 않고 표시만 한다(사람이 쓴 "[특징주]" 등). 평가에서 따로 본다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ROBOT_MARKERS = ("로보뉴스", "씽크풀", "AI 기자", "자동생성", "알고리즘에 의해")
_ROBOT_TITLE = re.compile(
    r"^\s*\[(장중수급포착|리포트 브리핑|MK 골든크로스|MK시그널)"
    r"|애프터마켓서 \d+%대 (급등|급락)"
)
_LISTING_TITLE = re.compile(
    r"^\s*\[(\d{4})?(마감체크|개장시황|마감시황|테마시황|시황레이더|표|오늘의 주요공시|오늘의 IR|오늘의 증시일정"
    r"|오늘증시|아침밥|모닝 리포트|마켓뷰|급등락주 짚어보기|베스트&워스트|게시판|사설|기자수첩"
    r"|미리보는 이데일리 신문|연합뉴스 이 시각 헤드라인|전국 주요 신문)"
    r"|주요공시\]|순매수도 상위종목|시세표|오늘의 메모\[|기업 공시 ?\[|오프라인 헤드라인|오늘의 인사"
)
_PRICE_MOVE_TITLE = re.compile(r"특징주|급등|급락|강세|약세|신고가|상한가|하한가|주가 [+-]?\d")
_CORP_MARKS = re.compile(r"\s|\(주\)|㈜|주식회사")


@dataclass(frozen=True)
class PrefilterDecision:
    keep: bool
    reason: str
    reports_price_move: bool = False


def _norm(text: str) -> str:
    return _CORP_MARKS.sub("", text or "")


def prefilter(*, title: str, summary: str, source: str, company_name: str) -> PrefilterDecision:  # noqa: ARG001
    title = title or ""
    head = f"{title} {(summary or '')[:200]}"
    if _ROBOT_TITLE.search(title) or any(m in head for m in _ROBOT_MARKERS):
        return PrefilterDecision(False, "robot")
    if _LISTING_TITLE.search(title):
        return PrefilterDecision(False, "listing")
    name = _norm(company_name)
    if not name or name not in _norm(title):
        return PrefilterDecision(False, "not_about_company")
    return PrefilterDecision(True, "ok", bool(_PRICE_MOVE_TITLE.search(title)))

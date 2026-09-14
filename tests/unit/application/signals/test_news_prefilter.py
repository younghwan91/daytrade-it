"""news_prefilter — 실제 Toss 과거 기사 제목·요약에서 뽑은 예시로 고정한다."""

import pytest

from daytrade_it.application.signals.news_prefilter import prefilter


@pytest.mark.parametrize(
    ("title", "summary", "source", "reason"),
    [
        (
            "[장중수급포착] 예시반도체, 외국인/기관 동시 순매수… 주가 +3.13%",
            "[서울=뉴스핌] 로보뉴스 = 14일 10시 15분 현재 예시반도체는",
            "newspim",
            "robot",
        ),
        (
            "[리포트 브리핑]CJ대한통운, 'NDR 후기: 방향성에 집중' 목표가 110,000원 - 신한투자증권",
            " = 신한투자증권에서 20일 CJ대한통운에 대해",
            "newspim",
            "robot",
        ),
        (
            "[MK 골든크로스 돌파종목 : 유한양행 & 현대건설]",
            "핫 이슈 종목을 분석해드리는 AI 기자 ‘MK시그널’ 입니다.",
            "maekyung",
            "robot",
        ),
        (
            "대신증권, 애프터마켓서 10%대 급등",
            "(서울=연합뉴스) 코스피 상장사 대신증권",
            "yna",
            "robot",
        ),
        (
            "[0508마감체크] 코스피, 또 사상 최고치",
            "▶마감체크 ■ 코스피지수",
            "infostockdaily",
            "listing",
        ),
        ("[표] 거래소 기관 순매수도 상위종목(11일)", "", "yna", "listing"),
        ("[코스피·코스닥 전 거래일(10일) 주요공시]", "", "yna", "listing"),
        ("[마감시황] 코스피 6600선 마감", "", "edaily", "listing"),
    ],
)
def test_robot_and_listing_articles_are_dropped(
    title: str, summary: str, source: str, reason: str
) -> None:
    d = prefilter(title=title, summary=summary, source=source, company_name="대신증권")
    assert (d.keep, d.reason) == (False, reason)


def test_company_name_must_be_in_title() -> None:
    d = prefilter(
        title="상법 개정 후 첫 주총 시즌… 기업 '정관 변경·자사주 소각' 대응",
        summary="",
        source="infostockdaily",
        company_name="예시반도체",
    )
    assert (d.keep, d.reason) == (False, "not_about_company")


def test_company_name_match_ignores_spaces_and_corp_marks() -> None:
    d = prefilter(
        title="HD현대 마린솔루션, 발전엔진 생산능력 확대",
        summary="",
        source="hankyung",
        company_name="HD현대마린솔루션(주)",
    )
    assert d.keep


def test_human_article_about_company_is_kept_with_price_report_flag() -> None:
    d = prefilter(
        title="[특징주] 삼성중공업, LNG운반선 6척 수주에 강세",
        summary="",
        source="newspim",
        company_name="삼성중공업",
    )
    assert (d.keep, d.reason, d.reports_price_move) == (True, "ok", True)


def test_plain_catalyst_article_is_kept() -> None:
    d = prefilter(
        title='"예시반도체, 목표가↑…DDR5 외주물량 증가세"-한투',
        summary="",
        source="yna",
        company_name="예시반도체",
    )
    assert (d.keep, d.reports_price_move) == (True, False)

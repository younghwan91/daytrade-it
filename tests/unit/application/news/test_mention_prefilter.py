"""The deterministic pre-LLM filter: which (article, ticker) pairs are worth judging.

Measured on 97,721 judged pairs (see
`.superpowers/sdd/2026-09-16-bayes-stages/prefilter-report.md`): 13.34% of pairs
skipped, 0.024% BUY (score > 0.15) recall loss.
"""

from __future__ import annotations

import pytest

from daytrade_it.application.news.mention_prefilter import (
    ROUNDUP_TAGS,
    prefilter_enabled,
    roundup_tag,
    should_judge,
)

REAL = ("기아", "000270", "기아 ‘PV7’, 獨 하노버서 세계 첫 공개", "기아가 상용 PBV PV7을 공개했다.")


class TestRoundupTag:
    @pytest.mark.parametrize(
        "title",
        [
            "[표] 코스피 기관·외국인·개인 순매수도 상위종목",
            "[개장시황] 엔비디아 호실적 영향 코스피, 2%대 상승 출발",
            "[마감시황] 코스피, 삼성전자 주주환원 기대에 6900선 회복",
            "[오늘의 IR] 삼성전자 등",
            "[티타임] 어쩌고",
            "[食전食후] 어쩌고",
            "[오늘의 증시일정] 9월 16일",
            "[연합뉴스 이 시각 헤드라인] 오전 9시",
            "[AI MY 증시전망] 9월 16일",
            "[장중시황] 코스피 보합",
            "[ETF 시황] 어쩌고",
            "[주간수급리포트] 어쩌고",
            "[주간증시전망] 어쩌고",
            "[오늘증시] 어쩌고",
        ],
    )
    def test_matches_known_roundup_tags(self, title: str) -> None:
        assert roundup_tag(title) is not None

    @pytest.mark.parametrize(
        "title",
        ["[0519마감체크]", "[0810개장체크] 어쩌고", "[0703주요일정] 어쩌고"],
    )
    def test_matches_dated_roundup_tags(self, title: str) -> None:
        assert roundup_tag(title) is not None

    @pytest.mark.parametrize(
        "title",
        [
            # editorial tags that carry real BUYs (15-48% BUY rate in ground truth)
            "[단독]\"전세계인 다 먹은 꼴\"…'매운맛 뚝심' 불닭, 100억개 신화 썼다",
            "[특징주] 삼성전기, 10%대 급등에 신고가 경신",
            "[리포트 브리핑] 어쩌고",
            "[모닝 리포트] 어쩌고",
            "[속보] 어쩌고",
            # 4.8% BUY rate -- deliberately NOT filtered
            "[장중수급포착] 한화생명, 외국인/기관 동시 순매수… 주가 +8.15%",
            # no bracket at all
            "HD현대, 1분기 실적 '역대 최대'…영업익 2조8348억원",
            "",
        ],
    )
    def test_does_not_match_tradeable_titles(self, title: str) -> None:
        assert roundup_tag(title) is None

    def test_returns_the_matched_tag_for_logging(self) -> None:
        assert roundup_tag("[표] 코스피 순매수 상위") == "표"

    def test_tolerates_leading_whitespace_and_full_width_brackets(self) -> None:
        assert roundup_tag("  [표] 어쩌고") == "표"
        assert roundup_tag("【표】 어쩌고") == "표"

    def test_tolerates_none(self) -> None:
        assert roundup_tag(None) is None

    def test_tag_set_is_non_empty_and_frozen(self) -> None:
        assert "표" in ROUNDUP_TAGS
        assert isinstance(ROUNDUP_TAGS, frozenset)


class TestShouldJudge:
    def test_off_by_default_judges_everything(self) -> None:
        assert should_judge(*REAL) is True
        assert should_judge("삼성전자", "005930", "[표] 코스피 순매수 상위", "") is True

    def test_enabled_skips_a_roundup_table(self) -> None:
        assert (
            should_judge("삼성전자", "005930", "[표] 코스피 순매수 상위", "", enabled=True) is False
        )

    def test_enabled_still_judges_real_news(self) -> None:
        assert should_judge(*REAL, enabled=True) is True

    def test_enabled_does_not_depend_on_body(self) -> None:
        """Live sees the full body, backtests see a 100-char stub -- the filter
        must read the title only so both worlds skip exactly the same pairs."""
        title = "HD현대, 1분기 실적 '역대 최대'…영업익 2조8348억원"
        assert should_judge("HD현대일렉트릭", "267260", title, "", enabled=True) is True
        assert should_judge("HD현대일렉트릭", "267260", title, "x" * 5000, enabled=True) is True

    def test_enabled_does_not_depend_on_name_or_code(self) -> None:
        title = "[표] 코스피 순매수 상위"
        assert should_judge("", "", title, "", enabled=True) is False
        assert should_judge("아무거나", "999999", title, "", enabled=True) is False

    def test_handles_missing_title(self) -> None:
        assert should_judge("기아", "000270", None, None, enabled=True) is True


class TestPrefilterEnabled:
    def test_default_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAYTRADE_NEWS_PREFILTER", raising=False)
        assert prefilter_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_env_turns_it_on(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER", value)
        assert prefilter_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_env_keeps_it_off(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER", value)
        assert prefilter_enabled() is False

    def test_explicit_override_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER", "1")
        assert prefilter_enabled(False) is False
        monkeypatch.delenv("DAYTRADE_NEWS_PREFILTER", raising=False)
        assert prefilter_enabled(True) is True

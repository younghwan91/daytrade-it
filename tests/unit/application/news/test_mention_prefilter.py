"""The deterministic pre-LLM filter: which (article, ticker) pairs are worth judging.

Measured on 97,721 judged pairs (see
`.superpowers/sdd/2026-09-16-bayes-stages/prefilter-report.md`): 13.34% of pairs
skipped, 0.024% BUY (score > 0.15) recall loss.
"""

from __future__ import annotations

import pytest

from daytrade_it.application.news.mention_prefilter import (
    FANOUT_MAX,
    INDEX_FANOUT_MIN,
    ROUNDUP_TAGS,
    fanout_max,
    index_fanout_min,
    prefilter_enabled,
    roundup_tag,
    should_judge,
    skip_reason,
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


class TestFanout:
    """Second filter: articles Toss tags with many companies are roundups.

    Measured on the same 97,721 judged pairs (report §2a): fan-out > 15
    combined with the roundup tags skips 22.94% at 0.183% BUY recall loss.
    """

    def test_unknown_fanout_never_drops(self) -> None:
        """A caller that cannot count related codes must behave as before."""
        assert should_judge(*REAL, None, enabled=True) is True
        assert should_judge("삼성전자", "005930", "진짜 뉴스", "", None, enabled=True) is True

    def test_small_fanout_is_judged(self) -> None:
        assert should_judge("삼성전자", "005930", "진짜 뉴스", "", 1, enabled=True) is True
        assert should_judge("삼성전자", "005930", "진짜 뉴스", "", FANOUT_MAX, enabled=True) is True

    def test_large_fanout_is_skipped(self) -> None:
        assert (
            should_judge("삼성전자", "005930", "진짜 뉴스", "", FANOUT_MAX + 1, enabled=True)
            is False
        )
        assert should_judge("삼성전자", "005930", "진짜 뉴스", "", 99, enabled=True) is False

    def test_off_by_default_keeps_a_huge_fanout(self) -> None:
        assert should_judge("삼성전자", "005930", "진짜 뉴스", "", 999) is True

    def test_threshold_override_wins(self) -> None:
        judged = should_judge("삼성전자", "005930", "진짜 뉴스", "", 20, enabled=True, max_codes=25)
        assert judged is True
        assert (
            should_judge("삼성전자", "005930", "진짜 뉴스", "", 6, enabled=True, max_codes=5)
            is False
        )

    def test_zero_threshold_disables_the_fanout_rule(self) -> None:
        assert (
            should_judge("삼성전자", "005930", "진짜 뉴스", "", 999, enabled=True, max_codes=0)
            is True
        )

    def test_a_real_dropped_buy_from_the_measurement(self) -> None:
        """`[오늘의 주요공시] SK하이닉스·HLB·…` -- 28 codes, a disclosure roundup."""
        title = "[오늘의 주요공시] SK하이닉스·HLB·진에어·포스코홀딩스·네이버·대우건설 등"
        assert should_judge("SK하이닉스", "000660", title, "", 28, enabled=True) is False


class TestSkipReason:
    def test_none_when_off(self) -> None:
        assert skip_reason("[표] 어쩌고", 99, enabled=False) is None

    def test_none_for_real_news(self) -> None:
        assert skip_reason("삼성 3조 수주", 2, enabled=True) is None

    def test_roundup_reason_names_the_tag(self) -> None:
        assert skip_reason("[표] 어쩌고", 1, enabled=True) == "roundup:표"

    def test_fanout_reason_names_the_count(self) -> None:
        assert skip_reason("진짜 뉴스", 40, enabled=True) == "fanout:40"

    def test_roundup_wins_when_both_apply(self) -> None:
        assert skip_reason("[표] 어쩌고", 40, enabled=True) == "roundup:표"

    def test_should_judge_agrees_with_skip_reason(self) -> None:
        for title in ("[표] 어쩌고", "진짜 뉴스"):
            for n in (None, 1, 15, 16, 99):
                assert should_judge("이름", "000000", title, "", n, enabled=True) is (
                    skip_reason(title, n, enabled=True) is None
                )


class TestFanoutMax:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAYTRADE_NEWS_PREFILTER_MAX_CODES", raising=False)
        assert fanout_max() == FANOUT_MAX

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER_MAX_CODES", "7")
        assert fanout_max() == 7

    def test_bad_env_falls_back_to_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER_MAX_CODES", "nonsense")
        assert fanout_max() == FANOUT_MAX

    def test_explicit_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER_MAX_CODES", "7")
        assert fanout_max(30) == 30


class TestIndexCommentary:
    """Rule 3: an index name in the title of a widely-tagged article.

    `코스피`/`코스닥` in the title of an article Toss tags to >= 5 companies is
    market-index commentary, not company news. Measured marginally over the
    roundup + fan-out rules on 97,721 judged pairs: +5.0% of pairs skipped for
    +0.12% BUY recall loss.
    """

    def test_index_token_with_wide_fanout_is_skipped(self) -> None:
        title = "코스피, 7999 찍고 7643 마감…외국인 5.7조 순매도"
        assert should_judge("삼성전자", "005930", title, "", 9, enabled=True) is False

    def test_kosdaq_counts_too(self) -> None:
        title = "코스닥 900선 회복…개인 순매수"
        assert should_judge("에코프로", "086520", title, "", 7, enabled=True) is False

    def test_index_token_on_a_narrow_article_is_judged(self) -> None:
        """Company news that merely names the index must survive."""
        title = "한미반도체, 코스피 시총 10위 진입"
        assert should_judge("한미반도체", "042700", title, "", 2, enabled=True) is True

    def test_wide_fanout_without_an_index_token_falls_to_the_fanout_rule(self) -> None:
        assert should_judge("삼성전자", "005930", "삼성전자 3조 수주", "", 6, enabled=True) is True

    def test_unknown_fanout_never_fires(self) -> None:
        title = "코스피, 7999 찍고 7643 마감"
        assert should_judge("삼성전자", "005930", title, "", None, enabled=True) is True

    def test_reason_names_the_token_and_the_count(self) -> None:
        assert skip_reason("코스피 급락", 8, enabled=True) == "index:코스피:8"

    def test_fanout_rule_wins_when_both_apply(self) -> None:
        assert skip_reason("코스피 급락", 40, enabled=True) == "fanout:40"

    def test_off_by_default_switch(self) -> None:
        assert should_judge("삼성전자", "005930", "코스피 급락", "", 9, enabled=False) is True


class TestIndexFanoutMin:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAYTRADE_NEWS_PREFILTER_INDEX_MIN_CODES", raising=False)
        assert index_fanout_min() == INDEX_FANOUT_MIN

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER_INDEX_MIN_CODES", "3")
        assert index_fanout_min() == 3

    def test_bad_env_falls_back_to_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER_INDEX_MIN_CODES", "nonsense")
        assert index_fanout_min() == INDEX_FANOUT_MIN

    def test_explicit_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAYTRADE_NEWS_PREFILTER_INDEX_MIN_CODES", "3")
        assert index_fanout_min(99) == 99

    def test_zero_disables_the_index_rule(self) -> None:
        assert (
            should_judge(
                "삼성전자", "005930", "코스피 급락", "", 9, enabled=True, index_min_codes=0
            )
            is True
        )


class TestAdoptedThresholds:
    """The measured constants. A change here changes what live judging spends."""

    def test_fanout_max_is_eleven(self) -> None:
        assert FANOUT_MAX == 11

    def test_index_fanout_min_is_five(self) -> None:
        assert INDEX_FANOUT_MIN == 5

    def test_twelve_codes_is_dropped_and_eleven_is_kept(self) -> None:
        assert should_judge(*REAL, 12, enabled=True) is False
        assert should_judge(*REAL, 11, enabled=True) is True

    def test_index_boundary_is_five_codes(self) -> None:
        """코스피 in the title: 4 codes is judged, 5 codes is skipped."""
        title = "코스피, 7999 찍고 7643 마감…외국인 5.7조 순매도"
        assert should_judge("삼성전자", "005930", title, "", 4, enabled=True) is True
        assert should_judge("삼성전자", "005930", title, "", 5, enabled=True) is False

    def test_index_token_alone_never_drops_without_related_codes(self) -> None:
        """A title-only match with no related-code count must stay judged.

        Live callers that cannot count the article's companies (`related_count`
        is None) have to behave exactly as if the rule did not exist.
        """
        for title in ("코스피 급락", "코스닥 900선 회복", "코스피·코스닥 동반 하락"):
            assert should_judge("삼성전자", "005930", title, "", None, enabled=True) is True
            assert skip_reason(title, None, enabled=True) is None

    def test_every_rule_is_reachable_through_should_judge(self) -> None:
        """One function, three rules -- callers never re-implement any of them."""
        cases = {
            "[표] 코스피 거래대금 상위": (1, "roundup:표"),
            "기아 PV7 공개": (12, "fanout:12"),
            "코스피, 7999 찍고 7643 마감": (5, "index:코스피:5"),
        }
        for title, (n, expected) in cases.items():
            assert skip_reason(title, n, enabled=True) == expected
            assert should_judge("기아", "000270", title, "", n, enabled=True) is False

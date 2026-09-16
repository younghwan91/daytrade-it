"""주춤 익절·본절 손절 규칙 -- 사용자 지시 2026-09-16.

"저항에 부딪치거나 주춤거리면 그냥 익절해, 오르면 피라미딩이고."
"익절이 될 걸 손실로 바꾸진 않아 절대."
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from daytrade_it.application.trading.auto_trader import breakeven_stop_hit, stall_exit_hit
from daytrade_it.infrastructure.brokers.base import MinuteBar

NOW = dt.datetime(2026, 9, 16, 10, 30)
#: 봉 목록이 09:00 부터 시작하므로, 기본 진입은 그 직전으로 둔다.
OPEN_ENTRY = dt.datetime(2026, 9, 16, 8, 59)


def bars(closes: list[int], highs: list[int] | None = None) -> list[MinuteBar]:
    """09:00 부터 1분 간격. highs 를 안 주면 close 를 고점으로 쓴다."""
    highs = highs or closes
    return [
        MinuteBar(
            ts=dt.datetime(2026, 9, 16, 9, 0) + dt.timedelta(minutes=i),
            open=Decimal(c),
            high=Decimal(h),
            low=Decimal(c),
            close=Decimal(c),
            volume=100,
        )
        for i, (c, h) in enumerate(zip(closes, highs, strict=True))
    ]


class TestStallExit:
    def test_rising_does_not_fire(self) -> None:
        """오르는 동안에는 고점이 갱신되므로 걸리지 않는다 -- 그때는 피라미딩."""
        hit, gain = stall_exit_hit(
            bars(list(range(100, 120))), Decimal(100), NOW, Decimal("0.02"), 5, OPEN_ENTRY
        )
        assert hit is False
        assert gain is not None and gain > Decimal("0.02")

    def test_stalls_in_profit_fires(self) -> None:
        """이익 구간에서 5분간 고점을 못 넘으면 익절."""
        closes = list(range(100, 115)) + [114, 113, 114, 113, 114]
        hit, gain = stall_exit_hit(bars(closes), Decimal(100), NOW, Decimal("0.02"), 5, OPEN_ENTRY)
        assert hit is True
        assert gain is not None and gain > 0

    def test_stall_below_min_gain_does_not_fire(self) -> None:
        """손실·미미한 이익 구간에서는 켜지지 않는다 -- 그쪽은 손절의 몫."""
        closes = [100] * 20
        hit, _ = stall_exit_hit(bars(closes), Decimal(100), NOW, Decimal("0.02"), 5, OPEN_ENTRY)
        assert hit is False

    def test_not_enough_bars(self) -> None:
        hit, _ = stall_exit_hit(
            bars([100, 101, 102]), Decimal(100), NOW, Decimal("0.02"), 5, OPEN_ENTRY
        )
        assert hit is False

    def test_disabled_by_zero_minutes(self) -> None:
        hit, _ = stall_exit_hit(
            bars(list(range(100, 120))), Decimal(100), NOW, Decimal("0.02"), 0, OPEN_ENTRY
        )
        assert hit is False


class TestBreakevenStop:
    def test_arms_then_fires_on_return_to_entry(self) -> None:
        """한 번 +2% 갔다가 본전선으로 되돌아오면 판다."""
        closes = list(range(100, 106)) + list(range(105, 99, -1))
        hit, peak_gain, close = breakeven_stop_hit(
            bars(closes), Decimal(100), NOW, Decimal("0.02"), Decimal("0.003"), OPEN_ENTRY
        )
        assert hit is True
        assert peak_gain is not None and peak_gain >= Decimal("0.02")
        assert close == Decimal(100)

    def test_does_not_arm_below_gain(self) -> None:
        """이익이 난 적 없으면 켜지지 않는다 -- 진입가 아래로는 절대 안 자른다."""
        closes = [100, 100, 99, 98, 97]
        hit, peak_gain, _ = breakeven_stop_hit(
            bars(closes), Decimal(100), NOW, Decimal("0.02"), Decimal("0.003"), OPEN_ENTRY
        )
        assert hit is False
        assert peak_gain is not None and peak_gain < Decimal("0.02")

    def test_armed_but_still_up_does_not_fire(self) -> None:
        """올라 있는 동안에는 안 판다."""
        closes = list(range(100, 110))
        hit, _, _ = breakeven_stop_hit(
            bars(closes), Decimal(100), NOW, Decimal("0.02"), Decimal("0.003"), OPEN_ENTRY
        )
        assert hit is False

    def test_buffer_covers_costs(self) -> None:
        """본전선은 수수료·세금만큼 위다: 진입가 정확히 복귀 전에 나간다."""
        closes = list(range(1000, 1060, 10)) + [1003]
        hit, _, close = breakeven_stop_hit(
            bars(closes), Decimal(1000), NOW, Decimal("0.02"), Decimal("0.003"), OPEN_ENTRY
        )
        assert hit is True
        assert close == Decimal(1003)

    def test_peak_uses_high_not_close(self) -> None:
        """고점은 봉의 high 로 센다 -- 장중에 찍고 내려온 것도 이익으로 본다."""
        closes = [100, 100, 100, 100]
        highs = [100, 105, 100, 100]
        hit, peak_gain, _ = breakeven_stop_hit(
            bars(closes, highs), Decimal(100), NOW, Decimal("0.02"), Decimal("0.003"), OPEN_ENTRY
        )
        assert hit is True
        assert peak_gain == Decimal("0.05")


class TestPeakIsMeasuredAfterEntry:
    """진입 **전** 고점을 자기 고점으로 세면 안 된다.

    오전에 10% 오른 종목을 오후에 산 경우, 그 10% 는 우리 이익이 아니다.
    이 구분이 없으면 본절 손절은 한 번도 오른 적 없는 포지션을 자르고,
    주춤 익절은 "이익 나자마자 매도"로 퇴화한다. rl-builder 가 잡아냈다.
    """

    #: 09:00~09:10 에 100 -> 110, 그 뒤 13:00 까지 평평.
    @staticmethod
    def morning_spike() -> list[MinuteBar]:
        out = []
        for i in range(11):
            out.append(
                MinuteBar(
                    ts=dt.datetime(2026, 9, 16, 9, 0) + dt.timedelta(minutes=i),
                    open=Decimal(100 + i),
                    high=Decimal(100 + i),
                    low=Decimal(100),
                    close=Decimal(100 + i),
                    volume=100,
                )
            )
        for i in range(30):
            out.append(
                MinuteBar(
                    ts=dt.datetime(2026, 9, 16, 13, 0) + dt.timedelta(minutes=i),
                    open=Decimal(100),
                    high=Decimal(100),
                    low=Decimal(100),
                    close=Decimal(100),
                    volume=100,
                )
            )
        return out

    AFTERNOON = dt.datetime(2026, 9, 16, 13, 0)
    LATER = dt.datetime(2026, 9, 16, 13, 20)

    def test_breakeven_does_not_fire_on_a_position_that_never_gained(self) -> None:
        hit, peak_gain, _ = breakeven_stop_hit(
            self.morning_spike(),
            Decimal(100),
            self.LATER,
            Decimal("0.02"),
            Decimal("0.003"),
            self.AFTERNOON,
        )
        assert hit is False
        assert peak_gain == Decimal(0)

    def test_stall_does_not_fire_just_because_the_day_high_is_unreachable(self) -> None:
        hit, _ = stall_exit_hit(
            self.morning_spike(),
            Decimal(100),
            self.LATER,
            Decimal("0.005"),
            5,
            self.AFTERNOON,
        )
        # 진입 이후 구간은 평평해서 이익이 0 이다. 오전 고점(+10%)을 물었다면
        # "고점을 못 넘었다"로 판정해 곧바로 매도가 나갔을 자리다.
        assert hit is False

    def test_unknown_entry_time_is_treated_as_not_armed(self) -> None:
        """진입 시각을 모르면 봉이 하나도 안 잡혀 발동하지 않는다."""
        hit, _, _ = breakeven_stop_hit(
            self.morning_spike(),
            Decimal(100),
            self.LATER,
            Decimal("0.02"),
            Decimal("0.003"),
            dt.datetime(2026, 9, 16, 15, 0),
        )
        assert hit is False

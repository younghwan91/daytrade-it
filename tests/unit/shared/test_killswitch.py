"""정지 파일 판정이 코어 KillSwitch 로 넘어가도 비고정 규약은 그대로다."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import scripts.news_signal_poller as poller
from krx_quant_core.risk.killswitch import KILL_MANUAL, KillSwitch, KillSwitchConfig

from daytrade_it.application.trading import auto_trader
from daytrade_it.interfaces.mcp import tools
from daytrade_it.shared.killswitch import stop_file_active


def test_present_is_active_absent_is_not(tmp_path: Path) -> None:
    stop = tmp_path / "live_STOP"
    assert not stop_file_active(stop)
    stop.write_text("stop")
    assert stop_file_active(stop)


def test_not_sticky_removing_the_file_resumes(tmp_path: Path) -> None:
    """daytrade-it 규약: 파일이 있는 동안만 막고, 지우면 즉시 재개한다."""
    stop = tmp_path / "live_STOP"
    stop.write_text("stop")
    assert stop_file_active(stop)
    stop.unlink()
    assert not stop_file_active(stop)


def test_reason_is_core_manual_stop(tmp_path: Path) -> None:
    """내용 없는 빈 파일도 코어 ``manual_stop`` 사유로 잡힌다(크기·내용 무관)."""
    stop = tmp_path / "live_STOP"
    stop.touch()
    switch = KillSwitch(KillSwitchConfig(stop_file=stop, sticky_stop_file=False))
    assert switch.check(dt.date(2026, 10, 5)) == KILL_MANUAL
    assert not switch.killed  # 비고정 -- 래치되지 않는다


@pytest.mark.parametrize(
    ("module", "attr", "check"),
    [
        (tools, "_KILL_SWITCH_PATH", lambda: tools._kill_switch_active()),
        (tools, "_KILL_SWITCH_PATH", lambda: auto_trader._kill_switch_on()),
        (poller, "KILL_SWITCH_PATH", lambda: poller.kill_switch_active()),
    ],
    ids=["mcp_execute_trade", "auto_trader", "news_poller"],
)
def test_call_sites_read_their_module_global_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: object, attr: str, check: object
) -> None:
    """세 호출부 모두 임포트 때가 아니라 판정 때의 전역 경로를 본다."""
    first = tmp_path / "a_STOP"
    second = tmp_path / "b_STOP"
    monkeypatch.setattr(module, attr, first)
    assert not check()
    first.write_text("stop")
    assert check()
    first.unlink()
    assert not check()
    # 전역을 바꿔 끼우면(종이매매 폴러·테스트) 새 경로를 본다.
    monkeypatch.setattr(module, attr, second)
    assert not check()
    second.write_text("stop")
    assert check()


def test_paper_swap_moves_order_path_to_paper_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """폴러의 종이매매 전환(``tools._KILL_SWITCH_PATH = PAPER_KILL_SWITCH_PATH``)이
    주문 경로·AutoTrader 판정을 ``paper_STOP`` 으로 옮기고 ``live_STOP`` 은 무시한다."""
    live = tmp_path / "live_STOP"
    paper = tmp_path / "paper_STOP"
    monkeypatch.setattr(poller, "KILL_SWITCH_PATH", live)
    monkeypatch.setattr(poller, "PAPER_KILL_SWITCH_PATH", paper)
    monkeypatch.setattr(tools, "_KILL_SWITCH_PATH", poller.PAPER_KILL_SWITCH_PATH)

    live.write_text("stop")
    assert poller.kill_switch_active()  # 실매매 판정은 여전히 live_STOP
    assert not tools._kill_switch_active()
    assert not auto_trader._kill_switch_on()

    paper.write_text("stop")
    assert tools._kill_switch_active()
    assert auto_trader._kill_switch_on()

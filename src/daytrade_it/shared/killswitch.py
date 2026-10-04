"""정지 파일 판정 -- 코어 ``KillSwitch`` 로 위임한다.

daytrade-it 의 킬 스위치는 ``data/live_STOP``(종이매매는 ``data/paper_STOP``)이
**있는 동안만** 신규 주문을 거부하고, 지우면 즉시 재개되는 비고정 규약이다. 예전엔
세 곳(MCP ``execute_trade``·``AutoTrader``·뉴스 폴러)이 각자 ``Path.exists()`` 를
불렀다. 이제 같은 판정을 코어 :class:`krx_quant_core.risk.killswitch.KillSwitch` 에
``sticky_stop_file=False`` 로 맡긴다 -- 동작은 그대로이고, 뒤에 손실한도 같은
규칙을 붙일 때 세 곳이 한 판정기를 보게 된다.

경로는 **호출 시점에** 넘겨 받는다. 폴러가 종이매매 모드에서 모듈 전역
``_KILL_SWITCH_PATH`` 를 바꿔 끼우고 테스트도 monkeypatch 하므로, 임포트 때
경로를 잡아 두면 안 된다. 코어 설정은 frozen 이라 판정마다 새로 만든다 --
파일 stat 한 번이 전부라 비용은 없다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from krx_quant_core.risk.killswitch import KillSwitch, KillSwitchConfig

__all__ = ["stop_file_active"]

_KST = ZoneInfo("Asia/Seoul")


def stop_file_active(stop_file: str | Path) -> bool:
    """``stop_file`` 이 지금 존재하면 True(수동 정지). 비고정 -- 지우면 바로 False."""
    switch = KillSwitch(KillSwitchConfig(stop_file=stop_file, sticky_stop_file=False))
    return switch.check(datetime.now(_KST).date()) is not None

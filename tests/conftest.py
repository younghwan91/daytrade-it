"""
Pytest Configuration and Fixtures
"""

import pytest


@pytest.fixture
def sample_ticker() -> str:
    """Sample ticker symbol for testing."""
    return "AAPL"


@pytest.fixture
def sample_news_content() -> dict:
    """Sample news article content."""
    return {
        "title": "Apple Reports Record Quarterly Revenue",
        "content": "Apple Inc. announced record quarterly revenue today...",
        "source": "Reuters",
        "published_at": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def sample_price_data() -> dict:
    """Sample price data."""
    return {
        "open": 185.50,
        "high": 188.25,
        "low": 184.75,
        "close": 187.50,
        "volume": 45000000,
    }


@pytest.fixture(autouse=True)
def _isolate_kill_switch(tmp_path_factory: pytest.TempPathFactory) -> object:
    """실제 `data/live_STOP` 이 테스트 결과를 바꾸지 못하게 한다.

    킬 스위치 경로는 작업 디렉터리 상대 경로라, 운영자가 매매를 멈춰 두면
    주문 경로 테스트 11 개가 "kill switch active" 로 무더기로 깨졌다. 스위치가
    켜져 있느냐는 운영 상태이지 테스트 입력이 아니다. 스위치 동작 자체를
    확인하는 테스트는 각자 경로를 monkeypatch 한다.
    """
    import scripts.news_signal_poller as poller

    from daytrade_it.interfaces.mcp import tools

    path = tmp_path_factory.mktemp("killswitch") / "live_STOP"
    old_tools, old_poller = tools._KILL_SWITCH_PATH, poller.KILL_SWITCH_PATH
    tools._KILL_SWITCH_PATH = path
    poller.KILL_SWITCH_PATH = path
    yield path
    tools._KILL_SWITCH_PATH, poller.KILL_SWITCH_PATH = old_tools, old_poller

import importlib.util
import sys
from pathlib import Path

_path = Path(__file__).resolve().parents[3] / "scripts" / "eval" / "exit_samples.py"
_spec = importlib.util.spec_from_file_location("exit_samples", _path)
es = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = es
_spec.loader.exec_module(es)


def test_pre_market_and_after_close_enter_at_open() -> None:
    assert es.entry_minute("pre_market", "2026-09-02T07:10:00", "2026-09-02") == "2026-09-02T09:00"
    assert es.entry_minute("after_close", "2026-09-01T17:00:00", "2026-09-02") == "2026-09-02T09:00"


def test_intraday_enters_minute_after_publish_plus_60s() -> None:
    assert es.entry_minute("intraday", "2026-09-01T11:30:46", "2026-09-01") == "2026-09-01T11:32"
    assert es.entry_minute("intraday", "2026-09-01T11:30:00", "2026-09-01") == "2026-09-01T11:31"


def test_split() -> None:
    assert es.split_of("2026-09-08") == "dev"
    assert es.split_of("2026-09-09") == "test"
    assert es.split_of("2026-09-14") == "forward"

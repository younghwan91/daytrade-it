"""JudgmentLog: every judged pair recorded once per day, restart-safe."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from news_signal_poller import JudgmentLog  # noqa: E402


def test_judgment_log_appends_and_dedups_across_restart(tmp_path: Path) -> None:
    day = dt.date(2026, 9, 14)
    log = JudgmentLog(tmp_path, day)
    assert not log.seen("a|005930")
    log.append("a|005930", {"v2": {"score": 0.3}, "v4_shadow": None})
    assert log.seen("a|005930")

    restarted = JudgmentLog(tmp_path, day)
    assert restarted.seen("a|005930")
    assert not restarted.seen("a|000660")
    assert len((tmp_path / "2026-09-14.jsonl").read_text().splitlines()) == 1

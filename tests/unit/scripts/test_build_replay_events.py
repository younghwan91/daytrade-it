"""parse_poll_line: daemon-log poll-line parsing for replay event building."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from build_replay_events import parse_poll_line  # noqa: E402

from daytrade_it.infrastructure.brokers.base import KST


def test_parse_poll_line_extracts_kst_timestamp_from_poll_complete_line() -> None:
    line = '{"event": "Toss news poll complete", "timestamp": "2026-09-14T00:00:35.123456Z"}'
    ts = parse_poll_line(line)
    assert ts is not None
    assert ts.tzinfo is not None
    assert ts.astimezone(KST).isoformat() == "2026-09-14T09:00:35.123456+09:00"


def test_parse_poll_line_ignores_non_poll_json_lines() -> None:
    line = '{"event": "Toss article judged", "timestamp": "2026-09-14T00:00:35Z"}'
    assert parse_poll_line(line) is None


def test_parse_poll_line_ignores_non_json_lines() -> None:
    assert parse_poll_line("not json at all\n") is None


def test_parse_poll_line_ignores_json_missing_timestamp() -> None:
    line = '{"event": "Toss news poll complete"}'
    assert parse_poll_line(line) is None

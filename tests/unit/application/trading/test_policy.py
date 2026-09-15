"""NewsPolicy: one JSON file shared by the live poller and replay."""

import datetime as dt
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from daytrade_it.application.signals.news_entry import SelectionConfig
from daytrade_it.application.trading.auto_trader import TradingLimits
from daytrade_it.application.trading.policy import NewsPolicy, dump_policy, load_policy


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_policy(tmp_path / "nope.json") == NewsPolicy()


def test_round_trip_defaults_and_non_defaults(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    dump_policy(NewsPolicy(), path)
    assert load_policy(path) == NewsPolicy()

    policy = NewsPolicy(
        limits=TradingLimits(
            stop_loss_pct=Decimal("0.015"),
            stop_time=dt.timedelta(minutes=45),
            entry_start=dt.time(9, 5),
            entry_vwap_reclaim=True,
            entry_limit_ticks=2,
            vwap_exit_gap=None,
        ),
        selection=SelectionConfig(
            min_score=0.4,
            subjects=("main",),
            exclude_material=("수주",),
            min_pre_move=Decimal("-0.02"),
            max_pre_move=Decimal("0.05"),
            sessions=("premarket",),
            rank_key="score",
        ),
    )
    dump_policy(policy, path)
    raw = json.loads(path.read_text())
    assert raw["limits"]["stop_loss_pct"] == "0.015"
    assert raw["limits"]["stop_time"] == 2700
    assert raw["limits"]["entry_start"] == "09:05:00"
    assert raw["selection"]["subjects"] == ["main"]
    assert load_policy(path) == policy


def test_partial_file_fills_defaults(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"selection": {"min_score": 0.3}}))
    assert load_policy(path) == NewsPolicy(selection=SelectionConfig(min_score=0.3))


@pytest.mark.parametrize(
    "raw",
    [{"bogus": 1}, {"limits": {"bogus": 1}}, {"selection": {"bogus": 1}}],
)
def test_unknown_keys_raise(tmp_path: Path, raw: dict) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_policy(path)


def test_invalid_rank_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"selection": {"rank_key": "random"}}))
    with pytest.raises(ValueError):
        load_policy(path)


def test_poller_policy_parts_default_when_file_absent(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
    import news_signal_poller

    assert Path("config/news_policy.json") == news_signal_poller.POLICY_PATH
    monkeypatch.chdir(tmp_path)  # no config/ here
    policy = load_policy(news_signal_poller.POLICY_PATH)
    limits, selection, rank_key = news_signal_poller._policy_parts(policy)
    assert limits == TradingLimits()
    assert selection == SelectionConfig()
    assert rank_key == "surprise"


@pytest.mark.parametrize(
    "raw",
    [
        '{"selection": {"min_score": NaN}}',
        '{"selection": {"min_score": Infinity}}',
        '{"selection": {"max_pre_move": "NaN"}}',
        '{"limits": {"stop_loss_pct": "-Infinity"}}',
        '{"limits": {"stop_loss_pct": NaN}}',
        '{"limits": {"stop_time": Infinity}}',
    ],
)
def test_non_finite_numbers_raise(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "p.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_policy(path)


def test_unknown_subject_raises(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"selection": {"subjects": ["main", "headline"]}}))
    with pytest.raises(ValueError):
        load_policy(path)
    path.write_text(json.dumps({"selection": {"subjects": ["mention"]}}))
    assert load_policy(path).selection.subjects == ("mention",)

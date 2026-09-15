"""NewsPolicy: trading limits + stock selection, one JSON file.

The live poller (`scripts/news_signal_poller.py`, `config/news_policy.json`)
and the replay engine read the same file, so an optimized policy runs live
exactly as it was replayed. A missing file means the defaults, which are
the live behavior; a partial file overrides only the keys it names.

JSON encoding: Decimal as a string, timedelta as seconds, dt.time as
"HH:MM:SS", tuples as lists, None as null. Unknown keys raise ValueError.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import types
import typing
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from daytrade_it.application.signals.news_entry import SESSIONS, SelectionConfig
from daytrade_it.application.trading.auto_trader import RANK_KEYS, TradingLimits
from daytrade_it.infrastructure.external.anthropic_client import SUBJECT_LEVELS


@dataclass(frozen=True)
class NewsPolicy:
    limits: TradingLimits = field(default_factory=TradingLimits)
    selection: SelectionConfig = field(default_factory=SelectionConfig)


def _encode(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dt.timedelta):
        seconds = value.total_seconds()
        return int(seconds) if seconds == int(seconds) else seconds
    if isinstance(value, dt.time):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def _decode(tp: Any, value: Any, name: str) -> Any:
    if value is None:
        if type(None) in typing.get_args(tp):
            return None
        raise ValueError(f"{name}: null is not allowed")
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        (inner,) = [a for a in typing.get_args(tp) if a is not type(None)]
        return _decode(inner, value, name)
    try:
        if tp is Decimal:
            if isinstance(value, bool) or not isinstance(value, str | int | float):
                raise TypeError
            number = Decimal(str(value))
            if not number.is_finite():
                raise ValueError
            return number
        if tp is dt.timedelta:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError
            if not math.isfinite(value):
                raise ValueError
            return dt.timedelta(seconds=value)
        if tp is dt.time:
            return dt.time.fromisoformat(value)
        if tp is bool:
            if not isinstance(value, bool):
                raise TypeError
            return value
        if tp is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError
            return value
        if tp is float:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError
            if not math.isfinite(value):
                raise ValueError
            return float(value)
        if tp is str:
            if not isinstance(value, str):
                raise TypeError
            return value
        if origin is tuple:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise TypeError
            return tuple(value)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"{name}: invalid value {value!r}") from exc
    raise ValueError(f"{name}: unsupported field type {tp!r}")


def _from_dict(cls: type, raw: Any, section: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"{section}: expected an object")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"{section}: unknown keys {sorted(unknown)}")
    return cls(**{k: _decode(hints[k], v, f"{section}.{k}") for k, v in raw.items()})


def _validate(policy: NewsPolicy) -> None:
    sel = policy.selection
    if sel.rank_key not in RANK_KEYS:
        raise ValueError(f"selection.rank_key: {sel.rank_key!r} not in {RANK_KEYS}")
    bad = set(sel.sessions) - set(SESSIONS)
    if bad:
        raise ValueError(f"selection.sessions: unknown {sorted(bad)}")
    bad = set(sel.subjects) - set(SUBJECT_LEVELS)
    if bad:
        raise ValueError(f"selection.subjects: unknown {sorted(bad)}")


def load_policy(path: Path) -> NewsPolicy:
    """The policy in `path`; defaults when the file doesn't exist."""
    if not path.exists():
        return NewsPolicy()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object")
    unknown = set(raw) - {"limits", "selection"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    policy = NewsPolicy(
        limits=_from_dict(TradingLimits, raw.get("limits", {}), "limits"),
        selection=_from_dict(SelectionConfig, raw.get("selection", {}), "selection"),
    )
    _validate(policy)
    return policy


def policy_to_dict(policy: NewsPolicy) -> dict[str, dict[str, Any]]:
    return {
        section: {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        for section, obj in (("limits", policy.limits), ("selection", policy.selection))
    }


def dump_policy(policy: NewsPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy_to_dict(policy), ensure_ascii=False, indent=2) + "\n")

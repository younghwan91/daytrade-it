"""Decision-time attributes, and the null-differenced return they explain.

Everything here is known before the entry minute. Nothing reads the body text,
nothing reads a later data vintage, nothing costs an API call.

The dependent variable is deliberately NOT the raw return. `alpha` is

    alpha = ret_close - null(day, acting minute, sigma x value cell)

so the day, the minute and the volatility/liquidity bucket are differenced out
before any feature is looked at. Three of tonight's retracted findings were
one of those three leaking in -- `after_10am` was position shrinkage, the
dashboard's forward maximum was 2x pre-entry volatility, and the pool's whole
mean was the index. Conditioning on `alpha` cannot reproduce them.

Cost cancels in `alpha` (both legs are one round trip), so a positive alpha of
X bp is X bp of real selection; to be tradeable on its own it still has to
clear the 21.5 bp round trip on the RAW return, which `net` carries.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterator
from pathlib import Path

COST = 0.00215
#: leading "[...]" of a title -- 급등락주 / N마감체크 / 표 / 특징주 ...
TAG = re.compile(r"^\s*\[([^\]]{1,16})\]")
#: "[이데일리 ...]" / "[서울=뉴스핌] ..." -- the wire that filed the story
WIRE = re.compile(r"\[(?:[^\]\[=]{1,12}=)?([가-힣A-Za-z][가-힣A-Za-z0-9 ]{1,10}?)\s*(?:[^\]]*)?\]")


def _bucket(value: float | None, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    if value is None:
        return "na"
    for edge, label in zip(edges, labels, strict=False):
        if value < edge:
            return label
    return labels[-1]


def wire_of(summary: str | None) -> str:
    """The filing wire, from the dateline the Korean agencies all prepend."""
    if not summary:
        return "na"
    m = WIRE.search(summary[:60])
    if not m:
        return "na"
    name = m.group(1).strip()
    return name if 1 < len(name) <= 10 else "na"


def title_tag(title: str | None) -> str:
    m = TAG.match(title or "")
    return m.group(1) if m else "none"


def features(master: dict, label: dict, strat: dict | None) -> dict[str, object]:
    v2 = master.get("v2") or {}
    day = dt.date.fromisoformat(label["day"])
    fan = master.get("related_n")
    return {
        "market": master.get("market") or "na",
        "sector": master.get("sector") or "na",
        "session": master.get("session") or "na",
        "weekday": day.strftime("%a"),
        "tag": title_tag(master.get("title")),
        "wire": wire_of(master.get("summary")),
        "fanout": _bucket(
            fan, (2, 3, 6, 11, 21, 41), ("1", "2", "3-5", "6-10", "11-20", "21-40", "41+")
        ),
        "pre_move": _bucket(
            label.get("pre_move"),
            (-0.03, -0.01, -0.003, 0.003, 0.01, 0.03),
            ("<-3%", "-3..-1%", "-1..-0.3%", "flat", "0.3..1%", "1..3%", ">3%"),
        ),
        "sigma_q": f"s{strat['sig_q']}" if strat else "na",
        "value_q": f"v{strat['val_q']}" if strat else "na",
        "act_min": _bucket(
            label.get("act_min"),
            (1, 30, 60, 120, 240),
            ("open", "9:01-9:29", "9:30-9:59", "10-11", "11-13", "13+"),
        ),
        "score": _bucket(
            v2.get("score"), (0.001, 0.15, 0.3, 0.5), ("0", "0-.15", ".15-.3", ".3-.5", ".5+")
        ),
        "subject": v2.get("subject") or "na",
        "material": v2.get("material_type") or "na",
        "persistence": v2.get("persistence") or "na",
        "freshness": v2.get("freshness") or "na",
        "scale": v2.get("scale_vs_size") or "na",
        "reports_move": str(v2.get("reports_price_move")),
        "sent_dir": str(v2.get("sentiment_direction")),
        "sent_str": str(v2.get("sentiment_strength")),
        "surprise": str(v2.get("surprise")),
        "specificity": str(v2.get("specificity")),
    }


#: free and deterministic first, then what the prompt already asks
AXES = (
    "market",
    "sector",
    "session",
    "weekday",
    "tag",
    "wire",
    "fanout",
    "pre_move",
    "sigma_q",
    "value_q",
    "act_min",
    "score",
    "subject",
    "material",
    "persistence",
    "freshness",
    "scale",
    "reports_move",
    "sent_dir",
    "sent_str",
    "surprise",
    "specificity",
)


def load(
    labels: Path, master: Path, null: Path, strata: Path | None, min_cell: int = 5
) -> Iterator[dict]:
    """Join the four files into one row per pair, with `alpha` attached."""
    nulls: dict[tuple[str, int], dict] = {}
    with null.open() as f:
        for line in f:
            r = json.loads(line)
            nulls[(r["day"], r["act_min"])] = r

    strat: dict[tuple[str, str], dict] = {}
    if strata:
        with strata.open() as f:
            for line in f:
                s = json.loads(line)
                strat[(s["day"], s["code"])] = s

    lab: dict[str, dict] = {}
    with labels.open() as f:
        for line in f:
            r = json.loads(line)
            lab[r["key"]] = r

    with master.open() as f:
        for line in f:
            m = json.loads(line)
            label = lab.get(m["key"])
            if label is None:
                continue
            n = nulls.get((label["day"], label["act_min"]))
            if n is None:
                continue
            code = m.get("code") or m["key"].split("|")[1]
            s = strat.get((label["day"], code))
            base = n["null_mean"]
            if strat:
                cell = (n.get("cells") or {}).get(f"{s['sig_q']}|{s['val_q']}") if s else None
                if not cell or cell["n"] < min_cell:
                    continue
                base = cell["mean"]
            ret = label["ret_close"]
            yield {
                "key": m["key"],
                "day": label["day"],
                "month": label["day"][:7],
                "code": code,
                "ret": ret,
                "net": ret - COST,
                "null": base,
                "alpha": ret - base,
                "mkt": n.get("mkt_ret"),
                "pre_move_raw": label.get("pre_move"),
                "act_min": label["act_min"],
                "score_raw": v2.get("score") if (v2 := m.get("v2") or {}) else None,
                "sigma": (s or {}).get("sigma"),
                "f": features(m, label, s),
            }

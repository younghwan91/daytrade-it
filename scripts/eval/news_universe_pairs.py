"""Build (live article_id, ticker) pairs from the per-code Toss universe files.

    uv run python scripts/eval/news_universe_pairs.py [--in-dir data/toss_universe] [--out data/toss_universe_pairs.jsonl]

The per-company endpoint's `stockCodes` is always null, so an article's related tickers are
recovered by grouping: the set of codes whose list contains the same newsId. The live daemon
takes related tickers from the detail endpoint's `companyCodes`; validation of the two lives in
`validate_news_universe.py`. `article_id` is the live daemon's uuid5 key, so pairs join directly
against data/news_judgments.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from backfill_toss_company_news import live_article_id

CODE_FILE = re.compile(r"^\d{6}\.jsonl$")


def iter_rows(in_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(in_dir.iterdir()):
        if CODE_FILE.match(path.name):
            with path.open() as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)


def group_related(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for r in rows:
        g = grouped.setdefault(
            r["id"], {"codes": set(), "title": r.get("title"), "createdAt": r.get("createdAt")}
        )
        g["codes"].add(r["code"])
    return grouped


def build_pairs(grouped: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for news_id, g in grouped.items():
        related = sorted(g["codes"])
        aid = live_article_id(news_id)
        for ticker in related:
            pairs.append(
                {
                    "article_id": aid,
                    "news_id": news_id,
                    "ticker": ticker,
                    "related": related,
                    "published_at": g["createdAt"],
                    "title": g["title"],
                }
            )
    return pairs


def compare_sets(got: dict[str, set[str]], ref: dict[str, set[str]]) -> dict[str, int]:
    """Pair-level agreement on articles present in both mappings (keys = article ids)."""
    both = got.keys() & ref.keys()
    return {
        "articles_both": len(both),
        "ref_pairs": sum(len(ref[k]) for k in both),
        "ref_pairs_found": sum(len(ref[k] & got[k]) for k in both),
        "extra_pairs": sum(len(got[k] - ref[k]) for k in both),
        "exact_articles": sum(got[k] == ref[k] for k in both),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("data/toss_universe"))
    ap.add_argument("--out", type=Path, default=Path("data/toss_universe_pairs.jsonl"))
    a = ap.parse_args()
    pairs = build_pairs(group_related(iter_rows(a.in_dir)))
    tmp = a.out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.rename(a.out)
    print(f"{len(pairs)} pairs over {len({p['news_id'] for p in pairs})} articles -> {a.out}")


if __name__ == "__main__":
    main()

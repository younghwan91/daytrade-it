"""Fill the 13,370 master rows whose pairs file predates `related_n`.

Fan-out is the number of universe codes Toss tagged on one article, which is
recoverable from any pairs file by counting the rows that share an article id
(the key's prefix). The pre-`related_n` a6 file carries the same article
grouping, so the count is exact for every article it holds in full.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

D = Path("data/eval")
MASTER = D / "prompt_v3/master.jsonl"


def main() -> None:
    # article -> related_n, from whichever file states it
    stated: dict[str, int] = {}
    # article -> rows seen, as the fallback count
    seen: Counter[str] = Counter()
    for name in ("universe_pairs_n300.jsonl", "universe_pairs_a6.jsonl"):
        with (D / name).open() as f:
            for line in f:
                r = json.loads(line)
                art = r["key"].split("|")[0]
                if r.get("related_n") is not None:
                    stated[art] = r["related_n"]
                seen[art] += 1
    print(f"articles: {len(seen)}, with a stated related_n: {len(stated)}")

    rows = [json.loads(line) for line in MASTER.open()]
    filled = exact = 0
    for r in rows:
        if r.get("related_n") is not None:
            continue
        art = r["key"].split("|")[0]
        if art in stated:
            r["related_n"] = stated[art]
            exact += 1
        else:
            # a6-only article: the a6 file holds every pair of it, so the row
            # count is the fan-out.
            r["related_n"] = seen[art]
            r["related_n_source"] = "counted"
            filled += 1
    print(f"filled: {exact} from a stated value, {filled} by counting rows")

    with MASTER.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    missing = sum(1 for r in rows if r.get("related_n") is None)
    print(f"still missing: {missing}")


if __name__ == "__main__":
    main()

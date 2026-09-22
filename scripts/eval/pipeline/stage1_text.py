"""Stage-1 text: exactly the title + summary the live judge receives, and nothing else.

Stdlib only -- imported both by the project `.venv` (stage1_build / stage1_select)
and by the separate embedding env (stage1_embed), so the two sides cannot drift
on what "the text" of a pair is.

Live parity (src/daytrade_it/application/news/judging.py -> anthropic_client.build_prompt):
    title    = article.title.strip()
    body     = (article.summary or article.content).strip()   -- the dashboard summary
    category = article.category                                 -- None on every judged pair
The archive (`data/toss_universe_42m/<code>.jsonl`) carries `title` and `summary`
but no body, so a pair whose summary is empty gets title only (live would have
sent the content; 30 of 135,946 judged pairs). Category is null on all 135,946
judged pairs, so it carries no information and is not used. The stock name and
code that the live prompt also contains are NOT part of the model text: they are
ticker identity, the confound this stage has to avoid learning.

    python scripts/eval/pipeline/stage1_text.py parity     # archive vs live-judged text
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / "data/toss_universe_42m"
JUDGED_MASTER = REPO / "data/eval/prompt_v3/master.jsonl"
MASK = "[종목]"


def live_fields(title: str | None, summary: str | None) -> tuple[str, str]:
    """(title, body) exactly as build_prompt formats them (it .strip()s both)."""
    return (title or "").strip(), (summary or "").strip()


def model_text(title: str | None, summary: str | None) -> str:
    t, s = live_fields(title, summary)
    return f"{t}\n{s}" if s else t


def text_hash(text: str) -> str:
    """Cache key for one exact text (model prefixes are added by the embedder, not here)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---- stock-name masking (ablation) ----------------------------------------
_PREF = re.compile(r"(\d?우[A-C]?|\(전환\))$")


def name_variants(name: str) -> list[str]:
    """The stock's own name and its obvious spellings, longest first.

    `삼성전자우` -> 삼성전자우, 삼성전자; `SK 하이닉스` -> SK 하이닉스, SK하이닉스;
    `(주)` / `㈜` prefixes and suffixes are covered by masking the bare name.
    Names shorter than 2 characters are not masked (would hit ordinary words).
    """
    out = set()
    base = name.strip()
    for v in (base, base.replace(" ", "")):
        if len(v) >= 2:
            out.add(v)
        stem = _PREF.sub("", v)
        if len(stem) >= 2 and stem != v:
            out.add(stem)
    return sorted(out, key=len, reverse=True)


def mask_text(text: str, name: str | None, code: str) -> str:
    out = text.replace(code, MASK)
    if name:
        for v in name_variants(name):
            out = re.sub(re.escape(v), MASK, out, flags=re.IGNORECASE)
    return out


# ---- archive access ---------------------------------------------------------
def archive_rows(code: str, days: set[str] | None = None) -> Iterable[dict]:
    path = ARCHIVE / f"{code}.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if days is not None and (r.get("createdAt") or "")[:10] not in days:
                continue
            yield r


def archive_texts(wanted: dict[str, set[str]]) -> dict[tuple[str, str], tuple[str, str]]:
    """{(news_id, code): (title, summary)} for `wanted` = {code: {day, ...}}."""
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for code, days in wanted.items():
        for r in archive_rows(code, days):
            out.setdefault((r["id"], code), (r.get("title") or "", r.get("summary") or ""))
    return out


def parity() -> dict:
    """Fraction of live-judged pairs whose archive title/summary match exactly."""
    sys.path.insert(0, str(REPO / "src"))
    from daytrade_it.application.news.universe import live_article_id

    judged = [json.loads(line) for line in JUDGED_MASTER.open(encoding="utf-8")]
    wanted: dict[str, set[str]] = {}
    for r in judged:
        wanted.setdefault(r["code"], set()).add(r["day"])
        # overnight (pre_market) pairs were published the evening before
        wanted[r["code"]].add(r["published_at"][:10])
    by_key: dict[str, tuple[str, str]] = {}
    for code, days in wanted.items():
        for r in archive_rows(code, days):
            by_key.setdefault(
                f"{live_article_id(r['id'])}|{code}", (r.get("title") or "", r.get("summary") or "")
            )
    n = found = t_eq = s_eq = both = t_strip = s_strip = 0
    s_prefix = 0
    examples = []
    for r in judged:
        n += 1
        a = by_key.get(r["key"])
        if a is None:
            continue
        found += 1
        lt, ls = r.get("title") or "", r.get("summary") or ""
        at, as_ = a
        t_eq += at == lt
        s_eq += as_ == ls
        both += at == lt and as_ == ls
        t_strip += at.strip() == lt.strip()
        s_strip += as_.strip() == ls.strip()
        if as_ != ls:
            s_prefix += bool(ls) and (as_.startswith(ls[:40]) or ls.startswith(as_[:40]))
            if len(examples) < 8:
                examples.append({"key": r["key"], "live": ls[:120], "archive": as_[:120]})
        elif at != lt and len(examples) < 8:
            examples.append({"key": r["key"], "live_title": lt, "archive_title": at})
    return {
        "judged_pairs": n,
        "found_in_archive": found,
        "title_exact": t_eq / found if found else None,
        "summary_exact": s_eq / found if found else None,
        "both_exact": both / found if found else None,
        "title_exact_after_strip": t_strip / found if found else None,
        "summary_exact_after_strip": s_strip / found if found else None,
        "summary_mismatch_sharing_40char_prefix": s_prefix,
        "mismatch_examples": examples,
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["parity"]:
        print(json.dumps(parity(), indent=1, ensure_ascii=False))
    else:
        sys.exit(__doc__)

"""`refilter_pairs` re-applies the adopted prefilter to a frozen pairs file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from refilter_pairs import refilter  # noqa: E402


def _line(**kw: object) -> str:
    row = {"name": "기아", "code": "000270", "title": "기아 PV7 공개", "body": "", "related_n": 2}
    row.update(kw)
    return json.dumps(row, ensure_ascii=False) + "\n"


class TestRefilter:
    def test_keeps_real_company_news(self) -> None:
        kept, reasons = refilter([_line()])
        assert len(kept) == 1
        assert reasons == {}

    def test_drops_a_roundup_and_names_the_rule(self) -> None:
        kept, reasons = refilter([_line(title="[표] 코스피 상위 거래대금")])
        assert kept == []
        assert reasons["roundup"] == 1

    def test_drops_a_wide_fanout(self) -> None:
        kept, reasons = refilter([_line(related_n=30)])
        assert kept == []
        assert reasons["fanout"] == 1

    def test_drops_index_commentary(self) -> None:
        kept, reasons = refilter([_line(title="코스피, 7999 찍고 7643 마감", related_n=8)])
        assert kept == []
        assert reasons["index"] == 1

    def test_preserves_input_order_and_bytes(self) -> None:
        good_a, bad, good_b = _line(code="000270"), _line(related_n=30), _line(code="005930")
        kept, _ = refilter([good_a, bad, good_b])
        assert kept == [good_a, good_b]

    def test_missing_related_n_is_kept(self) -> None:
        kept, _ = refilter([_line(related_n=None)])
        assert len(kept) == 1

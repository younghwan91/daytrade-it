"""Recent DART disclosures for the entry risk gate -- re-exported from krx-quant-core.

The reader over scalp-it's ``data/dart.db`` moved verbatim to
:mod:`krx_quant_core.risk.dart_db` (same query, staleness limit, lookback
and fail-closed ``DisclosureDataUnavailable``); this module keeps the import
path the poller, MCP tools and ``scripts/preopen_check.sh`` use.

Fail-closed by design: missing, unreadable or stale data raises
`DisclosureDataUnavailable`; the caller refuses the *entry*. Exits never
consult this module.
"""

from __future__ import annotations

from krx_quant_core.risk.dart_db import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_STALENESS,
    DartDisclosureDB,
    DisclosureDataUnavailable,
)

__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_MAX_STALENESS",
    "DartDisclosureDB",
    "DisclosureDataUnavailable",
]

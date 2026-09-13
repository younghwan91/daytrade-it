"""Risk-gate domain service -- re-exported from krx-quant-core.

Sentiment/disclosure as a RISK GATE (entry-blocker; hard disclosures halt or
force exit), never a signal generator. The keyword taxonomy, trading-halt
reason judgement, decision order and reason strings moved verbatim to
:mod:`krx_quant_core.risk.disclosure`, which also carries the full design
rationale and research grounding. This module keeps the import path.

The only change from the original: ``RiskGate.evaluate`` takes any
``SentimentLike`` (``score``, ``confidence``, ``is_bearish``) instead of this
package's ``Sentiment`` value object, which satisfies it structurally.
"""

from __future__ import annotations

from krx_quant_core.risk.disclosure import (
    DisclosureSeverity,
    GateAction,
    RiskGate,
    RiskGateConfig,
    RiskGateDecision,
    SentimentLike,
    classify_disclosure_severity,
)

__all__ = [
    "DisclosureSeverity",
    "GateAction",
    "RiskGate",
    "RiskGateConfig",
    "RiskGateDecision",
    "SentimentLike",
    "classify_disclosure_severity",
]

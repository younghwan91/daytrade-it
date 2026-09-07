"""Task 3 spike: kiwoom-client mock-trading (is_mock=True) end-to-end check.

Goal: confirm kiwoom-client's mock-trading flow actually works and record the
real request/response field names for account/quote/order calls, before
Phase 3 designs KiwoomBroker (gpt-quant-v2's BrokerInterface implementation)
against assumed shapes.

Credential discovery (in order):
  1. KIWOOM_APP_KEY / KIWOOM_APP_SECRET / KIWOOM_ACCOUNT_NO env vars
  2. A non-example .env file in ~/Documents/git/kiwoom-client or this repo

If no credentials are found, this script does NOT fabricate a live call. It
exits with a clear "SKIPPED — no credentials" message. See
docs/spikes/003-kiwoom-mock-trading.md for how the API shapes were instead
documented by reading kiwoom-client's source directly (allowed by the task
brief, step 3) when this was the actual outcome on 2026-09-07.

Usage:
    python scripts/spikes/kiwoom_mock_spike.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

KIWOOM_CLIENT_REPO = Path.home() / "Documents" / "git" / "kiwoom-client"
STK = "005930"  # Samsung Electronics -- real KRX ticker used per the brief


def _load_dotenv_if_present(path: Path) -> dict[str, str]:
    """Minimal .env parser (no python-dotenv dependency), mirroring the
    pattern kiwoom-client's own tests/integration_api_smoke.py uses."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def discover_credentials() -> tuple[str | None, str | None, str | None]:
    """Look for KIWOOM_APP_KEY / KIWOOM_APP_SECRET / KIWOOM_ACCOUNT_NO in the
    environment, then in a .env file in kiwoom-client's repo or this repo."""
    app_key = os.environ.get("KIWOOM_APP_KEY")
    app_secret = os.environ.get("KIWOOM_APP_SECRET")
    account_no = os.environ.get("KIWOOM_ACCOUNT_NO")

    if not app_key or not app_secret:
        for env_path in (
            KIWOOM_CLIENT_REPO / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ):
            values = _load_dotenv_if_present(env_path)
            app_key = app_key or values.get("KIWOOM_APP_KEY")
            app_secret = app_secret or values.get("KIWOOM_APP_SECRET")
            account_no = account_no or values.get("KIWOOM_ACCOUNT_NO")

    # Treat placeholder values from .env.example-style files as "not set".
    if app_key in (None, "your_app_key"):
        app_key = None
    if app_secret in (None, "your_app_secret"):
        app_secret = None
    if account_no in (None, "your_account_number"):
        account_no = None

    return app_key, app_secret, account_no


def run_live_spike(app_key: str, app_secret: str, account_no: str | None) -> dict:
    """Only reached when real mock-mode credentials are available.

    Connects with is_mock=True, fetches account info, fetches a quote for
    005930, and places + immediately cancels one mock limit order. Returns a
    dict of the exact response shapes (field names) observed, for
    docs/spikes/003-kiwoom-mock-trading.md.
    """
    from kiwoom_client import KiwoomAPI
    from kiwoom_client.base import KiwoomAPIError

    findings: dict = {"live_call_made": True, "steps": []}
    account_no = account_no or "1234567890"

    api = KiwoomAPI(app_key=app_key, app_secret=app_secret, is_mock=True)
    try:
        login_resp = api.login()
        findings["steps"].append(
            {"step": "login", "response_keys": sorted(login_resp.keys())}
        )

        # Account: 예수금상세현황요청 (kt00001)
        deposit_resp = api.account.deposit_detail(qry_tp="3")
        findings["steps"].append(
            {
                "step": "account.deposit_detail (kt00001)",
                "response_keys": sorted(deposit_resp.keys()),
            }
        )

        # Quote: 주식호가요청 (ka10004)
        quote_resp = api.market.stock_quote(stk_cd=STK)
        findings["steps"].append(
            {
                "step": "market.stock_quote (ka10004)",
                "response_keys": sorted(quote_resp.keys()),
            }
        )

        # Order: place a mock limit buy far below market (unlikely to fill
        # instantly), then cancel it immediately.
        buy_resp = api.order.buy_order(
            dmst_stex_tp="01",
            stk_cd=STK,
            ord_qty="1",
            trde_tp="00",
            ord_uv="1000",
        )
        findings["steps"].append(
            {"step": "order.buy_order (kt10000)", "response": buy_resp}
        )

        ord_no = buy_resp.get("ord_no")
        if ord_no:
            cancel_resp = api.order.cancel_order(
                org_ord_no=ord_no,
                stk_cd=STK,
                ord_qty="1",
            )
            findings["steps"].append(
                {"step": "order.cancel_order (kt10003)", "response": cancel_resp}
            )
        else:
            findings["steps"].append(
                {
                    "step": "order.cancel_order (kt10003)",
                    "skipped": "buy_order response carried no ord_no",
                }
            )

        api.logout()
    except KiwoomAPIError as exc:
        findings["error"] = {"code": exc.code, "message": exc.message}
    finally:
        api.close()

    return findings


def main() -> int:
    app_key, app_secret, account_no = discover_credentials()

    if not app_key or not app_secret:
        print(
            "SKIPPED: no KIWOOM_APP_KEY / KIWOOM_APP_SECRET found in the "
            "environment or a non-example .env file "
            f"(checked {KIWOOM_CLIENT_REPO / '.env'} and this repo's .env).\n"
            "Per the task brief (Global Constraints: no live broker "
            "credentials in Phase 0), this script does NOT prompt for "
            "credentials. See docs/spikes/003-kiwoom-mock-trading.md for the "
            "API shapes documented instead by reading kiwoom-client's source."
        )
        return 0

    findings = run_live_spike(app_key, app_secret, account_no)
    print(json.dumps(findings, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

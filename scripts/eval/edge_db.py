"""kr_quant connection, READ-ONLY, credentials from quant-airflow/.env.

No secret is written to the repo; `.env` stays the single source. Every
consumer here opens a read-only session -- this is the shared production DB.
"""

from __future__ import annotations

import os
import urllib.parse as _u
from pathlib import Path

ENV = Path.home() / "git" / "quant-airflow" / ".env"


def dsn() -> str:
    if os.environ.get("KR_QUANT_DB"):
        return os.environ["KR_QUANT_DB"]
    e = dict(
        line.strip().split("=", 1)
        for line in ENV.read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    pw = _u.quote(e["TIMESCALE_PASSWORD"], safe="")
    return (
        f"postgresql://{e['TIMESCALE_USER']}:{pw}"
        f"@{e['TIMESCALE_HOST']}:{e['TIMESCALE_PORT']}/{e['TIMESCALE_DB']}"
    )


def connect():
    import psycopg2

    conn = psycopg2.connect(dsn(), connect_timeout=8)
    conn.set_session(readonly=True, autocommit=True)
    return conn

"""Unit-test defaults shared across packages."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _fresh_empty_dart_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point execute_trade's disclosure gate at a fresh, empty dart.db.

    Live, the gate reads scalp-it's data/dart.db and refuses entries when it
    is missing or stale. Tests that aren't about the gate need it present
    and clean; the gate's own tests in test_tools.py build their own file.
    """
    from daytrade_it.interfaces.mcp import tools  # noqa: PLC0415

    path = tmp_path / "autouse_dart.db"  # own name: gate tests build tmp_path/"dart.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE disclosures (rcept_no TEXT PRIMARY KEY, rcept_dt TEXT NOT NULL, "
        "code TEXT NOT NULL, corp_name TEXT, corp_cls TEXT, report_nm TEXT, "
        "material_type TEXT, is_correction INTEGER)"
    )
    con.commit()
    con.close()
    monkeypatch.setattr(tools, "_DART_DB_PATH", path)
    return path

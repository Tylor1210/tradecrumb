"""
SQLite history of regime snapshots (data/market_regime.db), one row per as_of
date. Lets the weekly review and dashboard show how fear/greed and pressure
moved, and lets trades be correlated with the regime they entered in.

Called by: cli.py (`regime`, `weekly-cycle`), dashboard/server.py.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path("data/market_regime.db")

_SQL = """
CREATE TABLE IF NOT EXISTS regime_snapshots (
    as_of        TEXT PRIMARY KEY,
    fear_greed   REAL,
    pressure     REAL NOT NULL,
    label        TEXT,
    payload_json TEXT NOT NULL,
    saved_at     TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


class RegimeStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_SQL)

    def save(self, regime) -> None:
        d = regime.to_dict()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO regime_snapshots (as_of, fear_greed, pressure, label, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (d["as_of"], d["fear_greed"], d["pressure"], d["pressure_label"], json.dumps(d)),
            )

    def latest(self) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT payload_json FROM regime_snapshots ORDER BY as_of DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    def history(self, limit: int = 60) -> list:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT as_of, fear_greed, pressure, label FROM regime_snapshots ORDER BY as_of DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"as_of": r[0], "fear_greed": r[1], "pressure": r[2], "label": r[3]} for r in reversed(rows)]

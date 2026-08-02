"""
SQLite-backed tracker for trade calls made by outside traders (Discord, X,
Instagram, etc.) — e.g. aristotle_investments. Entries are added manually
(no scraping/bot integration — see README_TRADER_TRACKING.md for why) and
scored against real price outcomes the same way paper trades are, so you can
compare "signal-engine's own picks" against "calls from people you follow."

Stored in data/trader_calls.db (gitignored).
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/trader_calls.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trader_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id         TEXT UNIQUE NOT NULL,
    trader_name     TEXT NOT NULL,
    platform        TEXT NOT NULL,
    handle          TEXT,
    ticker          TEXT NOT NULL,
    call_type       TEXT NOT NULL DEFAULT 'BUY',
    price_at_call   REAL,
    target_price    REAL,
    stop_price      REAL,
    timeframe       TEXT DEFAULT 'unspecified',
    notes           TEXT,
    source_url      TEXT,
    called_at       TEXT NOT NULL,
    status          TEXT DEFAULT 'OPEN',
    exit_price      REAL,
    outcome_pct     REAL,
    closed_at       TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


class TraderCallTracker:
    """Persistent log of third-party trader calls, scored like paper trades."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def add_call(
        self,
        trader_name: str,
        platform: str,
        ticker: str,
        called_at: datetime,
        call_type: str = "BUY",
        handle: Optional[str] = None,
        price_at_call: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timeframe: str = "unspecified",
        notes: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> str:
        call_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trader_calls (
                    call_id, trader_name, platform, handle, ticker, call_type,
                    price_at_call, target_price, stop_price, timeframe,
                    notes, source_url, called_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """,
                (
                    call_id, trader_name, platform, handle, ticker.upper(), call_type.upper(),
                    price_at_call, target_price, stop_price, timeframe,
                    notes, source_url, called_at.isoformat(),
                ),
            )
            conn.commit()
        logger.info("Trader call logged: %s %s %s @ %s", trader_name, call_type, ticker, price_at_call)
        return call_id

    def close_call(
        self,
        call_id: str,
        exit_price: float,
        closed_at: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> Optional[float]:
        closed_at = closed_at or datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT price_at_call, call_type FROM trader_calls WHERE call_id = ?", (call_id,)
            ).fetchone()
            if row is None:
                return None
            entry_price, call_type = row
            outcome_pct = None
            if entry_price:
                raw = (exit_price - entry_price) / entry_price * 100
                outcome_pct = round(raw if call_type != "SELL" else -raw, 4)

            fields = ["exit_price = ?", "outcome_pct = ?", "status = 'CLOSED'", "closed_at = ?"]
            params: list = [exit_price, outcome_pct, closed_at.isoformat()]
            if notes is not None:
                fields.append("notes = ?")
                params.append(notes)
            params.append(call_id)

            conn.execute(f"UPDATE trader_calls SET {', '.join(fields)} WHERE call_id = ?", params)
            conn.commit()
        return outcome_pct

    def delete_call(self, call_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM trader_calls WHERE call_id = ?", (call_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_calls(
        self,
        trader_name: Optional[str] = None,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        query = "SELECT * FROM trader_calls WHERE 1=1"
        params: list = []
        if trader_name:
            query += " AND trader_name = ?"
            params.append(trader_name)
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        if status:
            query += " AND status = ?"
            params.append(status.upper())
        query += " ORDER BY called_at DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_trader_summary(self) -> list[dict]:
        """Per-trader win rate / call counts, closed calls only for outcome stats."""
        calls = self.list_calls()
        by_trader: dict = {}
        for c in calls:
            by_trader.setdefault(c["trader_name"], []).append(c)

        summary = []
        for name, trader_calls in by_trader.items():
            closed = [c for c in trader_calls if c["status"] == "CLOSED" and c["outcome_pct"] is not None]
            open_count = sum(1 for c in trader_calls if c["status"] == "OPEN")
            wins = [c for c in closed if c["outcome_pct"] > 0]
            summary.append({
                "trader_name": name,
                "platform": trader_calls[0]["platform"],
                "total_calls": len(trader_calls),
                "open_calls": open_count,
                "closed_calls": len(closed),
                "win_rate_pct": round(len(wins) / len(closed) * 100, 2) if closed else None,
                "avg_outcome_pct": round(sum(c["outcome_pct"] for c in closed) / len(closed), 2) if closed else None,
            })
        summary.sort(key=lambda s: (s["win_rate_pct"] is None, -(s["win_rate_pct"] or 0)))
        return summary

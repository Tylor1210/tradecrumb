"""
SQLite-backed paper trade logger.

Records every paper trade's entry, exit, and outcome for post-analysis.
Stored in data/paper_trades.db (gitignored).

Schema:
  trades table:
    id                  INTEGER PRIMARY KEY
    trade_id            TEXT UNIQUE (Alpaca order ID or generated)
    ticker              TEXT
    signal_date         TEXT (ISO)
    entry_date          TEXT (ISO)
    entry_price         REAL
    exit_date           TEXT (ISO, nullable)
    exit_price          REAL (nullable)
    qty                 REAL
    notional            REAL
    outcome_pct         REAL (nullable — set on close)
    status              TEXT ('OPEN' | 'CLOSED')
    recommendation      TEXT ('BUY' | 'SHORT')
    composite_score     REAL
    hold_period         TEXT ('intraday' | 'swing' | 'monthly')
    signal_breakdown_json TEXT (full SignalReport breakdown as JSON)
    notes               TEXT
    created_at          TEXT
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/paper_trades.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id              TEXT UNIQUE NOT NULL,
    ticker                TEXT NOT NULL,
    signal_date           TEXT NOT NULL,
    entry_date            TEXT,
    entry_price           REAL,
    exit_date             TEXT,
    exit_price            REAL,
    qty                   REAL,
    notional              REAL,
    outcome_pct           REAL,
    status                TEXT DEFAULT 'OPEN',
    recommendation        TEXT,
    composite_score       REAL,
    hold_period           TEXT,
    signal_breakdown_json TEXT,
    notes                 TEXT,
    created_at            TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


class TradeTracker:
    """
    Persistent paper trade log using SQLite.

    Thread-safe for single-process use (SQLite default).
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
        logger.debug("Trade tracker DB ready: %s", self.db_path)

    # ── Entry ──────────────────────────────────────────────────────────────

    def log_entry(
        self,
        ticker: str,
        signal_date: datetime,
        entry_date: datetime,
        entry_price: float,
        qty: float,
        notional: float,
        recommendation: str,
        composite_score: float,
        signal_breakdown: dict,
        hold_period: str = "swing",
        trade_id: Optional[str] = None,
    ) -> str:
        """
        Record a new paper trade entry.

        Returns the trade_id string.
        """
        if trade_id is None:
            trade_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trades (
                    trade_id, ticker, signal_date, entry_date, entry_price,
                    qty, notional, recommendation, composite_score, hold_period,
                    signal_breakdown_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """,
                (
                    trade_id,
                    ticker,
                    signal_date.isoformat(),
                    entry_date.isoformat(),
                    entry_price,
                    qty,
                    notional,
                    recommendation,
                    composite_score,
                    hold_period,
                    json.dumps(signal_breakdown),
                ),
            )
            conn.commit()

        logger.info("Trade entry logged: %s %s @ $%.2f", ticker, trade_id[:8], entry_price)
        return trade_id

    # ── Exit ───────────────────────────────────────────────────────────────

    def log_exit(
        self,
        trade_id: str,
        exit_date: datetime,
        exit_price: float,
        notes: str = "",
    ) -> Optional[float]:
        """
        Record a trade exit and compute outcome_pct.

        Returns outcome_pct (positive = profit, negative = loss, None if not found).
        Win definition: outcome_pct > 0 (even +0.01% = win).
        For SHORT trades, profit is inverted — price falling is a win.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT entry_price, recommendation FROM trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()

            if row is None:
                logger.warning("Trade ID not found for exit: %s", trade_id)
                return None

            entry_price, recommendation = float(row[0]), row[1]
            raw_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
            outcome_pct = -raw_pct if recommendation == "SHORT" else raw_pct

            conn.execute(
                """
                UPDATE trades
                SET exit_date = ?, exit_price = ?, outcome_pct = ?,
                    status = 'CLOSED', notes = ?
                WHERE trade_id = ?
                """,
                (
                    exit_date.isoformat(),
                    exit_price,
                    round(outcome_pct, 4),
                    notes,
                    trade_id,
                ),
            )
            conn.commit()

        result = "WIN" if outcome_pct > 0 else "LOSS"
        logger.info(
            "Trade exit logged: %s %s → %.2f%% (%s)",
            trade_id[:8], result, outcome_pct, result,
        )
        return outcome_pct

    # ── Queries ─────────────────────────────────────────────────────────────

    def get_open_trades(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_date DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_closed_trades(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY exit_date DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_trade(self, trade_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_performance_summary(self) -> dict:
        """
        Compute win rate and performance stats from closed trades.

        Win rate = profitable trades / total trades (even +0.01% = win).
        """
        closed = self.get_closed_trades()
        if not closed:
            return {
                "message": "No closed trades yet",
                "total_closed_trades": 0,
            }

        outcomes = [t["outcome_pct"] for t in closed if t.get("outcome_pct") is not None]
        if not outcomes:
            return {"message": "No outcomes recorded", "total_closed_trades": len(closed)}

        profitable = [o for o in outcomes if o > 0]  # +0.01% = win
        losing = [o for o in outcomes if o <= 0]

        return {
            "total_closed_trades": len(closed),
            "profitable_trades": len(profitable),
            "losing_trades": len(losing),
            "win_rate_pct": round(len(profitable) / len(outcomes) * 100, 2),
            "avg_outcome_pct": round(sum(outcomes) / len(outcomes), 2),
            "median_outcome_pct": round(sorted(outcomes)[len(outcomes) // 2], 2),
            "best_trade_pct": round(max(outcomes), 2),
            "worst_trade_pct": round(min(outcomes), 2),
            "cumulative_return_pct": round(sum(outcomes), 2),
            "meets_70pct_target": (len(profitable) / len(outcomes) * 100) >= 70.0,
        }

    def get_ticker_performance(self, ticker: str) -> dict:
        """Performance breakdown for a specific ticker."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE ticker = ? AND status = 'CLOSED'",
                (ticker,),
            ).fetchall()

        trades = [dict(r) for r in rows]
        outcomes = [t["outcome_pct"] for t in trades if t.get("outcome_pct") is not None]

        if not outcomes:
            return {"ticker": ticker, "closed_trades": 0}

        profitable = [o for o in outcomes if o > 0]
        return {
            "ticker": ticker,
            "closed_trades": len(trades),
            "win_rate_pct": round(len(profitable) / len(outcomes) * 100, 2),
            "avg_outcome_pct": round(sum(outcomes) / len(outcomes), 2),
            "total_return_pct": round(sum(outcomes), 2),
        }

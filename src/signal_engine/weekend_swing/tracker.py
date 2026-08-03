"""
Weekend Swing Signals — persistent storage (Part 6).

SQLite-backed, following the same conventions as paper_trading/tracker.py
and trader_tracking/tracker.py: a status-progression lifecycle table rather
than separate tables per state, since a signal moves through the same
states in order (TRIGGERED -> SUBMITTED -> CLOSED, or TRIGGERED -> EXPIRED).

Stored in data/weekend_swing.db (gitignored). Scan results are a separate,
replaced-per-week snapshot table (not a lifecycle) since they're
informational, not something that gets submitted/closed.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/weekend_swing.db")

_CREATE_SCAN_RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS weekend_scan_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start         TEXT NOT NULL,
    ticker             TEXT NOT NULL,
    strategy           TEXT NOT NULL,
    pivot_high         REAL,
    pivot_low          REAL,
    current_price      REAL,
    distance_pct       REAL,
    earnings_excluded  INTEGER DEFAULT 0,
    data_quality       TEXT,
    scanned_at         TEXT DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_SIGNALS_SQL = """
CREATE TABLE IF NOT EXISTS weekend_signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id        TEXT UNIQUE NOT NULL,
    week_start       TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    strategy         TEXT NOT NULL,
    gap_type         TEXT,
    direction        TEXT NOT NULL,
    trigger_price    REAL NOT NULL,
    stop_price       REAL NOT NULL,
    triggered_at     TEXT NOT NULL,
    breakdown_json   TEXT,
    status           TEXT DEFAULT 'TRIGGERED',
    qty              REAL,
    notional         REAL,
    broker_trade_id  TEXT,
    entry_price      REAL,
    submitted_at     TEXT,
    exit_date        TEXT,
    exit_price       REAL,
    exit_reason      TEXT,
    outcome_pct      REAL,
    closed_at        TEXT,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


class WeekendSwingTracker:
    """Persistent log for Weekend Swing Signals: scan snapshots + the triggered->submitted->closed lifecycle."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_SCAN_RESULTS_SQL)
            conn.execute(_CREATE_SIGNALS_SQL)
            conn.commit()

    # ── Scan results (replaced per week) ────────────────────────────────────

    def save_scan_results(self, week_start: str, candidates: list) -> None:
        """Replaces this week's scan snapshot with the given candidates."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM weekend_scan_results WHERE week_start = ?", (week_start,))
            for c in candidates:
                conn.execute(
                    """
                    INSERT INTO weekend_scan_results (
                        week_start, ticker, strategy, pivot_high, pivot_low,
                        current_price, distance_pct, earnings_excluded, data_quality
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        week_start, c["ticker"], c["strategy"], c["pivot_high"], c["pivot_low"],
                        c["current_price"], c["distance_pct"], int(c.get("earnings_excluded", False)),
                        c.get("data_quality", ""),
                    ),
                )
            conn.commit()
        logger.info("Saved %d scan results for week %s", len(candidates), week_start)

    def get_scan_results(self, week_start: Optional[str] = None) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if week_start is None:
                row = conn.execute("SELECT MAX(week_start) FROM weekend_scan_results").fetchone()
                week_start = row[0] if row else None
                if week_start is None:
                    return []
            rows = conn.execute(
                "SELECT * FROM weekend_scan_results WHERE week_start = ? ORDER BY distance_pct ASC",
                (week_start,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Signal lifecycle: TRIGGERED -> SUBMITTED -> CLOSED / EXPIRED ───────

    def log_triggered_signal(self, signal, week_start: str) -> str:
        """
        Records a newly-fired signal as TRIGGERED. Idempotent per
        (week_start, ticker, strategy, gap_type): re-checking the same
        candidate on a later poll won't create a duplicate row.
        """
        gap_type = signal.breakdown.get("gap_type")
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                """
                SELECT signal_id FROM weekend_signals
                WHERE week_start = ? AND ticker = ? AND strategy = ? AND IFNULL(gap_type, '') = IFNULL(?, '')
                  AND status != 'EXPIRED'
                """,
                (week_start, signal.ticker, signal.strategy, gap_type),
            ).fetchone()
            if existing:
                return existing[0]

            signal_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO weekend_signals (
                    signal_id, week_start, ticker, strategy, gap_type, direction,
                    trigger_price, stop_price, triggered_at, breakdown_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'TRIGGERED')
                """,
                (
                    signal_id, week_start, signal.ticker, signal.strategy, gap_type, signal.direction,
                    signal.entry_price, signal.stop_price, signal.date.isoformat()
                    if hasattr(signal.date, "isoformat") else str(signal.date),
                    json.dumps(signal.breakdown, default=str),
                ),
            )
            conn.commit()
        logger.info("Triggered signal logged: %s %s %s", signal.ticker, signal.strategy, signal_id[:8])
        return signal_id

    def get_signals(self, status: Optional[str] = None) -> list:
        query = "SELECT * FROM weekend_signals"
        params: list = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY triggered_at DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_signal(self, signal_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM weekend_signals WHERE signal_id = ?", (signal_id,)).fetchone()
        return dict(row) if row else None

    def submit_signal(
        self, signal_id: str, qty: float, notional: float, broker_trade_id: str, entry_price: float
    ) -> bool:
        """TRIGGERED -> SUBMITTED. Becomes an 'active position' on the page."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE weekend_signals
                SET status = 'SUBMITTED', qty = ?, notional = ?, broker_trade_id = ?,
                    entry_price = ?, submitted_at = ?
                WHERE signal_id = ? AND status = 'TRIGGERED'
                """,
                (qty, notional, broker_trade_id, entry_price, datetime.now().isoformat(), signal_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def close_signal(
        self, signal_id: str, exit_price: float, exit_reason: str, exit_date: Optional[datetime] = None
    ) -> Optional[float]:
        """SUBMITTED -> CLOSED. Direction-aware outcome_pct (short profits when price falls)."""
        exit_date = exit_date or datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT entry_price, direction FROM weekend_signals WHERE signal_id = ?", (signal_id,)
            ).fetchone()
            if row is None or row[0] is None:
                logger.warning("Cannot close signal %s — not found or never submitted", signal_id)
                return None
            entry_price, direction = float(row[0]), row[1]
            raw_pct = (exit_price - entry_price) / entry_price * 100 if entry_price else 0.0
            outcome_pct = -raw_pct if direction == "short" else raw_pct

            conn.execute(
                """
                UPDATE weekend_signals
                SET status = 'CLOSED', exit_date = ?, exit_price = ?, exit_reason = ?,
                    outcome_pct = ?, closed_at = ?
                WHERE signal_id = ?
                """,
                (exit_date.isoformat(), exit_price, exit_reason, round(outcome_pct, 4),
                 datetime.now().isoformat(), signal_id),
            )
            conn.commit()
        return outcome_pct

    def expire_signal(self, signal_id: str) -> bool:
        """TRIGGERED -> EXPIRED, for signals never acted on (e.g. week ended)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE weekend_signals SET status = 'EXPIRED' WHERE signal_id = ? AND status = 'TRIGGERED'",
                (signal_id,),
            )
            conn.commit()
            return cur.rowcount > 0

    # ── History / summary ───────────────────────────────────────────────────

    def get_history(
        self,
        strategy: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> list:
        query = "SELECT * FROM weekend_signals WHERE status = 'CLOSED'"
        params: list = []
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if date_from:
            query += " AND closed_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND closed_at <= ?"
            params.append(date_to)
        query += " ORDER BY closed_at DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_summary_stats(
        self,
        strategy: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> dict:
        """
        Win rate / avg win / avg loss / total P&L, combined and broken down
        by strategy. When `strategy` is given, "combined" reflects only that
        strategy (matching whatever the caller filtered the trade list to).
        """
        closed = self.get_history(strategy=strategy, date_from=date_from, date_to=date_to)

        def _summarize(trades: list) -> dict:
            outcomes = [t["outcome_pct"] for t in trades if t.get("outcome_pct") is not None]
            if not outcomes:
                return {"total_trades": len(trades), "win_rate_pct": None, "avg_win_pct": None,
                        "avg_loss_pct": None, "total_pl_pct": 0.0}
            wins = [o for o in outcomes if o > 0]
            losses = [o for o in outcomes if o <= 0]
            return {
                "total_trades": len(outcomes),
                "win_rate_pct": round(len(wins) / len(outcomes) * 100, 2),
                "avg_win_pct": round(sum(wins) / len(wins), 2) if wins else 0.0,
                "avg_loss_pct": round(sum(losses) / len(losses), 2) if losses else 0.0,
                "total_pl_pct": round(sum(outcomes), 2),
            }

        by_strategy = {}
        for strat in {"prior_week_breakout", "overnight_range_breakout", "gap_strength"}:
            strat_trades = [t for t in closed if t["strategy"] == strat]
            if strat_trades:
                by_strategy[strat] = _summarize(strat_trades)

        return {"combined": _summarize(closed), "by_strategy": by_strategy}

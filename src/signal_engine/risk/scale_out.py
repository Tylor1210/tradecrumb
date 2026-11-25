"""
Partial profit taking: the "sell half at +20%, let the rest run" rule.

This is the upside mirror of `loss_cap.py`. In plain terms:

  * Under +20% (the trigger) nothing changes. The strategy's own exits run
    the trade.
  * The first close at +20% or better:
      - If the move was FAST (see below), nothing is sold. The full position
        is kept and managed with a trailing stop.
      - Otherwise, if the setup still points to continuation, sell 50% and
        keep the other 50% as a runner.
      - If the setup does NOT point to continuation on that bar, close the
        whole position at the gain (no runner worth leaving).
  * The runner (or the full fast-move position) then closes on the first of:
      - a close more than `trail_pct` below its best close (trailing stop),
      - a close back at or below breakeven (`runner_floor_pct`, 0 = entry),
        so a winner never turns into a loser,
      - the setup turning (runner only; a fast-move position rides the trail),
      - `runner_max_bars` bars after the trigger, if set.

What counts as a FAST move (either one is enough):
  * the trigger was reached within `fast_move_bars` bars of entry
    (default 2, so the first or second close after entry), or
  * the bar that crossed the trigger moved more than `fast_move_atr_mult`
    times the ticker's ATR in a single bar (default 3x ATR14). Needs an ATR
    from the caller; without one only the bar-count test applies.

All checks run on bar closes, same as the loss cap. A gap can carry the
runner through its trailing stop or breakeven floor between two closes, so a
runner's realized exit can be worse than the level itself.

"Setup still points to continuation" is decided by the caller, because each
harness has different signals available (see `backtest/engine.py` and
`weekend_swing/backtest.py`). This module only takes a yes/no per bar.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

EXIT_PARTIAL = "scale_out_partial"            # sold sell_fraction at the trigger, runner kept
EXIT_NO_RUNNER = "scale_out_full_no_runner"   # hit the trigger but no continuation: closed it all
EXIT_TRAIL = "scale_out_trailing_stop"        # runner / fast position hit its trailing stop
EXIT_BREAKEVEN = "scale_out_breakeven"        # runner / fast position came back to its floor
EXIT_SETUP_TURNED = "scale_out_setup_turned"  # runner closed because the setup turned
EXIT_RUNNER_TIMEOUT = "scale_out_runner_timeout"  # runner hit runner_max_bars

PHASE_WATCH = "watch"    # below the trigger so far
PHASE_RUNNER = "runner"  # partial taken, runner open
PHASE_FAST = "fast"      # fast move, full position on a trailing stop
PHASE_DONE = "done"


@dataclass
class ScaleOutAction:
    """What to do on this bar. sell_fraction is a fraction of what is still held (1.0 = close it all)."""

    sell_fraction: float
    reason: str

    @property
    def closes_position(self) -> bool:
        return self.sell_fraction >= 1.0


@dataclass
class ScaleOutRule:
    """Thresholds are positive percentages of gain from entry (20.0 = 20% up)."""

    enabled: bool = True
    trigger_gain_pct: float = 20.0
    sell_fraction: float = 0.5
    # Fast move: no forced sale, full position rides the trailing stop instead.
    fast_move_bars: int = 2           # trigger reached within this many bars of entry (0 = off)
    fast_move_atr_mult: float = 3.0   # or the crossing bar moved more than this x ATR (0 = off)
    # Runner management.
    trail_pct: float = 8.0            # close when price is this % below its best close since entry
    runner_floor_pct: float = 0.0     # never let it come back below this gain (0.0 = breakeven)
    runner_max_bars: int = 20         # max bars a runner stays open after the trigger (0 = no limit)
    # Daily engine only: the flat take_profit_pct (10% by default) would always
    # close the trade before +20%, so the rule could never run. When True the
    # flat take profit is switched off while this rule is on.
    replaces_take_profit: bool = True
    # Daily engine only: the runner also ignores hold_period_days, since the
    # point is to let it run. runner_max_bars still caps it.
    runner_ignores_time_exit: bool = True

    def __post_init__(self) -> None:
        if self.trigger_gain_pct <= 0:
            raise ValueError("trigger_gain_pct must be > 0")
        if not (0 < self.sell_fraction <= 1):
            raise ValueError("sell_fraction must be in (0, 1]")
        if not (0 < self.trail_pct < 100):
            raise ValueError("trail_pct must be between 0 and 100")
        if self.runner_floor_pct >= self.trigger_gain_pct:
            raise ValueError("runner_floor_pct must be below trigger_gain_pct")
        if self.fast_move_bars < 0 or self.fast_move_atr_mult < 0 or self.runner_max_bars < 0:
            raise ValueError("fast_move_bars, fast_move_atr_mult and runner_max_bars must be >= 0")

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["ScaleOutRule"]:
        """Build from a yaml `scale_out:` block. Returns None when missing or disabled."""
        if not cfg or not cfg.get("enabled", False):
            return None
        return cls.from_config_values(cfg)

    @classmethod
    def from_config_values(cls, cfg: Optional[dict]) -> "ScaleOutRule":
        """Build from a yaml block even if `enabled` is false (used by the compare flags)."""
        cfg = cfg or {}
        d = cls()
        return cls(
            enabled=True,
            trigger_gain_pct=float(cfg.get("trigger_gain_pct", d.trigger_gain_pct)),
            sell_fraction=float(cfg.get("sell_fraction", d.sell_fraction)),
            fast_move_bars=int(cfg.get("fast_move_bars", d.fast_move_bars)),
            fast_move_atr_mult=float(cfg.get("fast_move_atr_mult", d.fast_move_atr_mult)),
            trail_pct=float(cfg.get("trail_pct", d.trail_pct)),
            runner_floor_pct=float(cfg.get("runner_floor_pct", d.runner_floor_pct)),
            runner_max_bars=int(cfg.get("runner_max_bars", d.runner_max_bars)),
            replaces_take_profit=bool(cfg.get("replaces_take_profit", d.replaces_take_profit)),
            runner_ignores_time_exit=bool(cfg.get("runner_ignores_time_exit", d.runner_ignores_time_exit)),
        )

    def new_trade(self) -> "ScaleOutState":
        return ScaleOutState(self)

    def describe(self) -> str:
        fast = []
        if self.fast_move_bars:
            fast.append(f"within {self.fast_move_bars} bars")
        if self.fast_move_atr_mult:
            fast.append(f"a single bar over {self.fast_move_atr_mult:g}x ATR")
        return (
            f"sell {self.sell_fraction:.0%} at +{self.trigger_gain_pct:g}%, runner on a "
            f"{self.trail_pct:g}% trailing stop with a floor at +{self.runner_floor_pct:g}%; "
            f"fast move ({' or '.join(fast) or 'off'}) keeps the full position"
        )


class ScaleOutState:
    """Per-trade state for one open position. Call `update()` once per bar close."""

    def __init__(self, rule: ScaleOutRule) -> None:
        self.rule = rule
        self.phase = PHASE_WATCH
        self.bars = 0
        self.bars_since_trigger = 0
        self.max_gain_pct = float("-inf")
        self.reached_trigger = False
        self.fast_move = False
        self.partial_taken = False

    @property
    def managing(self) -> bool:
        """True once the trigger was hit and this rule owns the exit (runner or fast move)."""
        return self.phase in (PHASE_RUNNER, PHASE_FAST)

    def update(
        self, gain_pct: float, setup_alive: bool, bar_move_atr: Optional[float] = None
    ) -> Optional[ScaleOutAction]:
        """
        gain_pct: current gain from entry as a positive number (22.0 = 22% up).
                  Negative means the trade is losing.
        setup_alive: True if the setup still points to continuation on this bar.
        bar_move_atr: this bar's move in the trade's direction, in ATRs
                      (close minus previous close, over ATR). Optional.

        Returns a ScaleOutAction when something should be sold on this bar.
        """
        rule = self.rule
        if self.phase == PHASE_DONE:
            return None

        self.bars += 1
        self.max_gain_pct = max(self.max_gain_pct, gain_pct)

        if self.phase == PHASE_WATCH:
            if gain_pct < rule.trigger_gain_pct:
                return None
            self.reached_trigger = True
            fast_by_bars = rule.fast_move_bars > 0 and self.bars <= rule.fast_move_bars
            fast_by_atr = (
                rule.fast_move_atr_mult > 0
                and bar_move_atr is not None
                and bar_move_atr >= rule.fast_move_atr_mult
            )
            if fast_by_bars or fast_by_atr:
                self.fast_move = True
                self.phase = PHASE_FAST
                return None
            if not setup_alive or rule.sell_fraction >= 1.0:
                self.phase = PHASE_DONE
                return ScaleOutAction(1.0, EXIT_NO_RUNNER)
            self.phase = PHASE_RUNNER
            self.partial_taken = True
            return ScaleOutAction(rule.sell_fraction, EXIT_PARTIAL)

        # Runner or fast-move position.
        self.bars_since_trigger += 1
        peak_price = 1 + self.max_gain_pct / 100
        trail_gain = (peak_price * (1 - rule.trail_pct / 100) - 1) * 100

        reason = None
        if gain_pct <= rule.runner_floor_pct:
            reason = EXIT_BREAKEVEN
        elif gain_pct <= trail_gain:
            reason = EXIT_TRAIL
        elif self.phase == PHASE_RUNNER and not setup_alive:
            reason = EXIT_SETUP_TURNED
        elif rule.runner_max_bars and self.bars_since_trigger >= rule.runner_max_bars:
            reason = EXIT_RUNNER_TIMEOUT

        if reason is None:
            return None
        self.phase = PHASE_DONE
        return ScaleOutAction(1.0, reason)


def gain_pct_from_entry(direction: str, entry_price: float, price: float) -> float:
    """Gain from entry as a positive percentage (negative when losing)."""
    raw = (price - entry_price) / entry_price * 100
    return raw if direction == "long" else -raw


def atr_series(df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    """Simple-average ATR from high/low/close. None if those columns are missing."""
    if not {"high", "low", "close"}.issubset(df.columns):
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def bar_move_in_atr(
    direction: str, close: float, prev_close: Optional[float], atr: Optional[float]
) -> Optional[float]:
    """This bar's move in the trade's direction, measured in ATRs. None when not computable."""
    if prev_close is None or atr is None or pd.isna(prev_close) or pd.isna(atr) or atr <= 0:
        return None
    move = close - prev_close
    if direction != "long":
        move = -move
    return move / atr


@dataclass
class ScaleOutStats:
    """How often trades got to the trigger, and what the rule did with them."""

    trades: int = 0
    reached_trigger: int = 0
    scaled_out: int = 0
    fast_move_held: int = 0
    closed_no_runner: int = 0
    # Best close-to-entry gain of each trade (only trades that had at least one close after entry).
    max_gain_pcts: list = field(default_factory=list)

    def add(self, state: Optional["ScaleOutState"]) -> None:
        self.trades += 1
        if state is not None and state.bars:
            self.max_gain_pcts.append(state.max_gain_pct)
        if state is None or not state.reached_trigger:
            return
        self.reached_trigger += 1
        if state.fast_move:
            self.fast_move_held += 1
        elif state.partial_taken:
            self.scaled_out += 1
        else:
            self.closed_no_runner += 1

    def merge(self, other: "ScaleOutStats") -> "ScaleOutStats":
        return ScaleOutStats(
            trades=self.trades + other.trades,
            reached_trigger=self.reached_trigger + other.reached_trigger,
            scaled_out=self.scaled_out + other.scaled_out,
            fast_move_held=self.fast_move_held + other.fast_move_held,
            closed_no_runner=self.closed_no_runner + other.closed_no_runner,
            max_gain_pcts=self.max_gain_pcts + other.max_gain_pcts,
        )

    def reached_at_least(self, gain_pct: float) -> int:
        """How many trades closed at or above this gain at least once while open."""
        return sum(1 for g in self.max_gain_pcts if g >= gain_pct)

    @property
    def reached_pct(self) -> float:
        return round(self.reached_trigger / self.trades * 100, 1) if self.trades else 0.0

    def to_dict(self) -> dict:
        return {
            "trades": self.trades,
            "reached_trigger": self.reached_trigger,
            "reached_pct": self.reached_pct,
            "scaled_out": self.scaled_out,
            "fast_move_held": self.fast_move_held,
            "closed_no_runner": self.closed_no_runner,
        }

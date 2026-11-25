"""
Per-trade loss cap: the "10% review / 20% get out" rule.

The rule, in plain terms:

  * Under 10% down — nothing changes. Whatever stop/target/time exit the
    strategy already has keeps running the trade.
  * 10% down or worse ("review zone") — the trade does NOT auto-close. It
    stays open only while the setup still shows a chance of gain. The first
    bar where the setup looks weak, it closes.
  * 20% down or worse ("closing zone") — the trade starts working toward an
    exit. It closes on the first weak-setup bar, or after `exit_within_bars`
    more bars at the latest, whichever comes first. Once a trade reaches the
    closing zone it stays there even if price bounces, so a dead-cat bounce
    can be used as the exit instead of resetting the clock.
  * 25% down or worse (backstop) — close immediately, no questions asked.

All checks run on bar closes. A gap can carry a trade past a threshold
between two closes, so the realized loss can be worse than the threshold
itself; backtest reports show the worst single trade so that stays visible.

What "setup still shows a chance of gain" means is decided by the caller,
because each harness has different signals available (see
`backtest/engine.py` and `weekend_swing/backtest.py`). This module only
takes a yes/no per bar.
"""

from dataclasses import dataclass
from typing import Optional

EXIT_REVIEW = "loss_cap_review"        # 10%+ down and the setup went weak
EXIT_WEAK = "loss_cap_weak_signal"     # 20%+ down, closed on a weak bar
EXIT_TIMEOUT = "loss_cap_timeout"      # 20%+ down, ran out of bars to recover
EXIT_BACKSTOP = "loss_cap_backstop"    # hard backstop hit


@dataclass
class LossCapRule:
    """Thresholds are positive percentages of loss from entry (10.0 = 10% down)."""

    enabled: bool = True
    review_loss_pct: float = 10.0
    exit_loss_pct: float = 20.0
    exit_within_bars: int = 2
    backstop_loss_pct: float = 25.0
    # Daily engine only: when True the flat stop_loss_pct is switched off and
    # this rule is the only loss exit. With a 5% flat stop left on, the 10%
    # and 20% levels are never reached, so the rule would never do anything.
    replaces_fixed_stop: bool = True

    def __post_init__(self) -> None:
        if not (0 < self.review_loss_pct < self.exit_loss_pct < self.backstop_loss_pct):
            raise ValueError(
                "loss cap thresholds must satisfy 0 < review < exit < backstop, got "
                f"{self.review_loss_pct}/{self.exit_loss_pct}/{self.backstop_loss_pct}"
            )
        if self.exit_within_bars < 0:
            raise ValueError("exit_within_bars must be >= 0")

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["LossCapRule"]:
        """Build from a yaml `loss_cap:` block. Returns None when missing or disabled."""
        if not cfg or not cfg.get("enabled", False):
            return None
        return cls(
            enabled=True,
            review_loss_pct=float(cfg.get("review_loss_pct", 10.0)),
            exit_loss_pct=float(cfg.get("exit_loss_pct", 20.0)),
            exit_within_bars=int(cfg.get("exit_within_bars", 2)),
            backstop_loss_pct=float(cfg.get("backstop_loss_pct", 25.0)),
            replaces_fixed_stop=bool(cfg.get("replaces_fixed_stop", True)),
        )

    def new_trade(self) -> "LossCapState":
        return LossCapState(self)


class LossCapState:
    """Per-trade state for one open position. Call `update()` once per bar close."""

    def __init__(self, rule: LossCapRule) -> None:
        self.rule = rule
        self.closing = False
        self.bars_in_closing = 0

    def update(self, loss_pct: float, setup_alive: bool) -> Optional[str]:
        """
        loss_pct: current loss from entry as a positive number (12.0 = 12% down).
                  Negative means the trade is in profit.
        setup_alive: True if the setup still shows a chance of gain on this bar.

        Returns an exit reason string when the trade should close on this bar,
        otherwise None.
        """
        rule = self.rule

        if loss_pct >= rule.backstop_loss_pct:
            return EXIT_BACKSTOP

        if self.closing:
            self.bars_in_closing += 1
            if not setup_alive:
                return EXIT_WEAK
            if self.bars_in_closing >= rule.exit_within_bars:
                return EXIT_TIMEOUT
            return None

        if loss_pct >= rule.exit_loss_pct:
            self.closing = True
            self.bars_in_closing = 0
            if not setup_alive or rule.exit_within_bars == 0:
                return EXIT_WEAK if not setup_alive else EXIT_TIMEOUT
            return None

        if loss_pct >= rule.review_loss_pct and not setup_alive:
            return EXIT_REVIEW

        return None


def loss_pct_from_entry(direction: str, entry_price: float, price: float) -> float:
    """Loss from entry as a positive percentage (negative when in profit)."""
    raw = (price - entry_price) / entry_price * 100
    return -raw if direction == "long" else raw

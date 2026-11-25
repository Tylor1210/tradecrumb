"""
A short, fixed list of changes to the dip strategy, run side by side.

Each one targets what sank the plain rules: a few trades losing 50 to 97%,
and money parked for months in positions going nowhere. The settings are
round numbers picked up front, not tuned on the results, and the CLI shows
every variant on an earlier and a later period separately so a rule that
only worked once stands out.

  base        the video's rules as coded in rules.py
  stop        sell a trade that falls 25% below what was paid
  reclaim     after the signal, wait for a close back above the 50-day
              average before buying (no catching the falling knife)
  trend       only start trades while SPY is above its 200-day average
  time        at 3 months, sell a trade that's still below what was paid
  all four    stop + reclaim + trend + time
  all four, small   same, but each trade gets 8% of the account (2% at risk
              with the 25% stop) and up to 10 trades run at once
"""

from dataclasses import dataclass, field, replace
from typing import Optional

from signal_engine.leaps_dip.simulate import TradeRules


@dataclass
class Variant:
    key: str
    label: str
    trade: dict = field(default_factory=dict)   # TradeRules overrides
    market_filter: bool = False
    max_positions: Optional[int] = None
    slot_pct: Optional[float] = None

    def trade_rules(self, base: TradeRules) -> TradeRules:
        return replace(base, **self.trade)


def default_variants(cfg: Optional[dict] = None) -> list[Variant]:
    cfg = cfg or {}
    stop = float(cfg.get("stop_loss", 0.25))
    months = float(cfg.get("time_stop_months", 3))
    risk = float(cfg.get("risk_pct", 0.02))
    small_max = int(cfg.get("small_max_positions", 10))
    all_four = {"stop_loss": stop, "reclaim_sma": True, "time_stop_months": months}
    return [
        Variant("base", "Video rules"),
        Variant("stop", f"+ {stop * 100:.0f}% stop", {"stop_loss": stop}),
        Variant("reclaim", "+ wait for 50-day", {"reclaim_sma": True}),
        Variant("trend", "+ SPY over 200-day", market_filter=True),
        Variant("time", f"+ {months:.0f} mo time stop", {"time_stop_months": months}),
        Variant("all", "All four", dict(all_four), market_filter=True),
        Variant(
            "all_small", f"All four, {risk / stop * 100:.0f}% slots",
            dict(all_four), market_filter=True,
            max_positions=small_max, slot_pct=risk / stop,
        ),
    ]

"""
Signal weight grid search.

Sweeps combinations of:
  - technical / sentiment / political weights
  - hold period (1d, 5d, 10d, 21d)
  - buy threshold (0.40 → 0.65)

Outputs a ranked CSV of all combinations sorted by win_rate_pct.
The top-ranked configs hitting ≥70% win rate are candidates for paper trading.

Usage:
  uv run signal-engine backtest --grid-search

The signal_generator_fn parameter is a callable that, given a set of weights
and a threshold, returns buy signal Series per ticker. This decouples the
grid search from the signal pipeline so you can inject pre-computed signals.
"""

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

from signal_engine.backtest.engine import BacktestConfig, BacktestEngine
from signal_engine.backtest.metrics import BacktestMetrics

logger = logging.getLogger(__name__)

# Type alias for the signal generator function
# Signature: (weights: dict, hold_days: int, threshold: float) -> dict[str, pd.Series]
SignalGeneratorFn = Callable[[dict, int, float], dict[str, pd.Series]]


@dataclass
class GridSearchConfig:
    """Parameter grid for the weight sweep."""

    weight_technical_range: list[float]
    weight_sentiment_range: list[float]
    weight_political_range: list[float]
    hold_period_range: list[int]
    buy_threshold_range: list[float]

    # Fixed backtest parameters
    start_date: str
    end_date: str
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    initial_capital: float = 100_000.0
    position_size_pct: float = 5.0

    @classmethod
    def from_yaml(cls, config: dict) -> "GridSearchConfig":
        """Build from the grid_search section of backtest.yaml."""
        gs = config.get("grid_search", {})
        return cls(
            weight_technical_range=gs.get("weight_technical_range", [0.4]),
            weight_sentiment_range=gs.get("weight_sentiment_range", [0.35]),
            weight_political_range=gs.get("weight_political_range", [0.25]),
            hold_period_range=gs.get("hold_period_range", [5]),
            buy_threshold_range=gs.get("buy_threshold_range", [0.55]),
            start_date=config.get("start_date", "2020-01-01"),
            end_date=config.get("end_date", "2025-01-01"),
            stop_loss_pct=config.get("stop_loss_pct", 5.0),
            take_profit_pct=config.get("take_profit_pct", 10.0),
            initial_capital=config.get("initial_capital", 100_000.0),
            position_size_pct=config.get("position_size_pct", 5.0),
        )


class WeightGridSearch:
    """
    Grid search over signal weights, hold periods, and buy thresholds.

    Each combination is run as a full vectorbt backtest.
    Results are ranked by win_rate_pct descending.
    """

    def __init__(self, grid_config: GridSearchConfig) -> None:
        self.grid = grid_config

    def _weight_combos(self) -> Iterator[dict]:
        """Generate all (tech, sent, pol) weight combinations, normalized to sum=1."""
        seen: set[tuple] = set()
        for w_t, w_s, w_p in itertools.product(
            self.grid.weight_technical_range,
            self.grid.weight_sentiment_range,
            self.grid.weight_political_range,
        ):
            total = w_t + w_s + w_p
            if total == 0:
                continue
            normed = (
                round(w_t / total, 4),
                round(w_s / total, 4),
                round(w_p / total, 4),
            )
            if normed in seen:
                continue
            seen.add(normed)
            yield {
                "weight_technical": normed[0],
                "weight_sentiment": normed[1],
                "weight_political": normed[2],
            }

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        signal_generator_fn: SignalGeneratorFn,
        output_csv: Path = Path("reports/grid_search_results.csv"),
    ) -> pd.DataFrame:
        """
        Run the full grid search.

        Args:
            price_data:          Dict of ticker → OHLCV DataFrame
            signal_generator_fn: Callable that generates buy signals given weights + params
            output_csv:          Where to save the results CSV

        Returns:
            DataFrame of results, sorted by win_rate_pct descending
        """
        weight_combos = list(self._weight_combos())
        total_runs = (
            len(weight_combos)
            * len(self.grid.hold_period_range)
            * len(self.grid.buy_threshold_range)
        )

        logger.info("Grid search: %d combinations to test", total_runs)
        print(f"\n🔬 Grid search: {total_runs} combinations\n")

        results = []
        run_idx = 0

        for weights in weight_combos:
            for hold_days in self.grid.hold_period_range:
                for threshold in self.grid.buy_threshold_range:
                    run_idx += 1
                    if run_idx % 20 == 0 or run_idx == 1:
                        pct = run_idx / total_runs * 100
                        print(f"  [{run_idx:4d}/{total_runs}]  {pct:.1f}%  "
                              f"tech={weights['weight_technical']:.2f} "
                              f"sent={weights['weight_sentiment']:.2f} "
                              f"pol={weights['weight_political']:.2f} "
                              f"hold={hold_days}d threshold={threshold:.2f}")

                    try:
                        signals = signal_generator_fn(weights, hold_days, threshold)

                        bt_config = BacktestConfig(
                            start_date=self.grid.start_date,
                            end_date=self.grid.end_date,
                            hold_period_days=hold_days,
                            stop_loss_pct=self.grid.stop_loss_pct,
                            take_profit_pct=self.grid.take_profit_pct,
                            initial_capital=self.grid.initial_capital,
                            position_size_pct=self.grid.position_size_pct,
                        )

                        engine = BacktestEngine(bt_config)
                        bt_result = engine.run(price_data, signals)
                        metrics = bt_result.get_metrics()

                        row = {
                            "weight_technical": weights["weight_technical"],
                            "weight_sentiment": weights["weight_sentiment"],
                            "weight_political": weights["weight_political"],
                            "hold_period_days": hold_days,
                            "buy_threshold": threshold,
                            **metrics.to_dict(),
                        }

                    except Exception as exc:
                        logger.debug("Grid run failed (tech=%.2f sent=%.2f pol=%.2f hold=%dd): %s",
                                     weights["weight_technical"], weights["weight_sentiment"],
                                     weights["weight_political"], hold_days, exc)
                        row = {
                            "weight_technical": weights["weight_technical"],
                            "weight_sentiment": weights["weight_sentiment"],
                            "weight_political": weights["weight_political"],
                            "hold_period_days": hold_days,
                            "buy_threshold": threshold,
                            "error": str(exc),
                        }

                    results.append(row)

        df = pd.DataFrame(results)

        if "win_rate_pct" in df.columns:
            df = df.sort_values("win_rate_pct", ascending=False).reset_index(drop=True)

        # Flag configs that meet the 70% target
        if "win_rate_pct" in df.columns:
            df["meets_70pct_target"] = df["win_rate_pct"] >= 70.0

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

        meets_target = df.get("meets_70pct_target", pd.Series()).sum() if "meets_70pct_target" in df.columns else 0
        print(f"\n{'═' * 55}")
        print(f"  Grid search complete: {len(df)} results")
        print(f"  Configs meeting ≥70% win rate: {meets_target}")
        print(f"  Results saved → {output_csv}")
        print(f"{'═' * 55}\n")

        return df

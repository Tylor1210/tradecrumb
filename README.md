# signal-engine 🔬

A Python-based market sentiment and signal research system for identifying potential stock trade setups and LEAPS options candidates.

> ⚠️ **RESEARCH & PAPER TRADING ONLY** — No live order execution is implemented or wired. All Alpaca integration uses the paper trading API endpoint exclusively. Live money is never touched.

---

## Features

| Module | What it does |
|--------|-------------|
| 📊 **Technical signals** | SMA-20/50/200, RSI, MACD, support/resistance breakouts, relative volume |
| 📰 **Sentiment signals** | Finnhub news + VADER NLP; recency-weighted aggregation per ticker |
| 🏛️ **Political triggers** | Keyword/entity matching — Trump mentions, tariffs, defense contracts, geopolitical risk |
| ⚖️ **Composite scorer** | Weighted, fully inspectable signal combination with line-by-line breakdown |
| 📈 **Backtesting** | vectorbt-powered sim with win rate, max drawdown, Sharpe; ✅/❌ against 70% target |
| 🔬 **Grid search** | Sweep hundreds of weight × hold-period combos in seconds |
| 📝 **Paper trading** | Alpaca paper API + SQLite trade log (entry/exit/outcome per trade) |
| 🌿 **LEAPS screener** | Relative strength, earnings trend, macro theme alignment; options data stubbed |

---

## Prerequisites

- Python **3.11+**
- [`uv`](https://github.com/astral-sh/uv) — install once:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

---

## Setup

### 1. Install dependencies

```bash
cd signal-engine
uv sync
```

> **Note on vectorbt**: vectorbt uses `numba` for JIT compilation which requires LLVM. If install fails, try: `pip install llvmlite --pre` first, or install with `--no-build-isolation`.

### 2. Configure API keys

```bash
cp .env.example .env
# Now edit .env with your real credentials
```

| Variable | Where to get it | Required for |
|----------|----------------|--------------|
| `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` | [app.alpaca.markets](https://app.alpaca.markets/) | Market data (scan, backtest) |
| `ALPACA_PAPER_API_KEY` + `ALPACA_PAPER_SECRET_KEY` | Alpaca → Paper account | Paper trading |
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io/) free tier | News + sentiment |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org/) free tier | Political news sweep |
| `POLYGON_API_KEY` | [polygon.io](https://polygon.io/) free tier | LEAPS options data |

### 3. Verify

```bash
uv run signal-engine --help
```

---

## Usage

### Daily Signal Scan

```bash
# Scan today's signals
uv run signal-engine scan

# Scan a specific historical date
uv run signal-engine scan --date 2024-06-01
```

**Output**: Ranked table — ticker | composite score | BUY/SELL/HOLD | tech/sentiment/political breakdown | triggered rules (plain English).

### Backtesting

```bash
# Single backtest with current config/backtest.yaml settings
uv run signal-engine backtest

# Run full grid search (sweeps weight × hold-period combinations)
uv run signal-engine backtest --grid-search
```

**Per-trade loss cap (10% review / 20% get out).** Configured under `loss_cap:` in `config/backtest.yaml` and `config/weekend_swing.yaml` (logic in `src/signal_engine/risk/loss_cap.py`). Under 10% down nothing changes. At 10%+ down a trade stays open only while its setup still shows a chance of gain, and closes on the first bar it doesn't. At 20%+ down it starts closing, on the next weak bar or within `exit_within_bars` bars. A 25% backstop closes it no matter what. Checks run on closes, so a gap can overshoot a level; reports show the worst single trade so that's visible.

```bash
# Old exits vs the loss cap, side by side, with win rate vs the 60-70% bar
uv run signal-engine backtest --compare-exits
uv run signal-engine weekend-backtest --compare-exits
```

Results are saved to `reports/`. The win rate output clearly shows ✅ (≥70%) or ❌ (<70%) against the target before you consider activating paper trading.

**Tuning workflow:**
1. Run `uv run signal-engine backtest --grid-search`
2. Open `reports/grid_search_results.csv` — sort by `win_rate_pct` descending
3. Copy the best-performing weights into `config/signals.yaml`
4. Re-run single backtest to confirm, then move to paper trading

### Paper Trading

```bash
# Dry run — compute signals but don't submit orders
uv run signal-engine paper --dry-run

# Full paper trading session (submits to Alpaca paper API)
uv run signal-engine paper
```

All trades are logged to `data/paper_trades.db` (SQLite). Columns: `ticker`, `entry_date`, `entry_price`, `exit_date`, `exit_price`, `outcome_pct`, `signal_breakdown_json`.

### LEAPS Screener

```bash
uv run signal-engine leaps
```

Screens watchlist for long-horizon options candidates using relative strength vs SPY, earnings trends, and macro theme alignment. Options chain data is stubbed — see `src/signal_engine/leaps/options_stub.py` and `src/signal_engine/leaps/README_LEAPS.md` to connect a data source.

---

## Running Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=signal_engine --cov-report=term-missing
```

---

## Configuration Files

All configuration lives in `config/` — no code changes needed for tuning.

| File | Purpose |
|------|---------|
| `watchlist.yaml` | Tickers to monitor + sector/theme tags |
| `signals.yaml` | Signal weights, thresholds, all political keywords |
| `backtest.yaml` | Date range, hold periods, stop-loss, grid search params |
| `paper_trading.yaml` | Paper position sizing, risk management |

### Adding a Ticker

Edit `config/watchlist.yaml`:
```yaml
- symbol: XYZ
  name: "Company Name"
  sector: technology
  themes: [AI_infrastructure, semiconductors]
  public: true
```

### Tuning Signal Weights

Edit `config/signals.yaml`:
```yaml
weights:
  technical: 0.40   # adjust these
  sentiment: 0.35
  political: 0.25
```
Then rerun backtest to see the effect.

### Adding Political Keywords

Edit the `political_keywords` section of `config/signals.yaml`. Each category maps to a sentiment score in `category_sentiment` (+1.0 = strongly bullish, -1.0 = strongly bearish).

---

## Project Structure

```
signal-engine/
├── config/
│   ├── watchlist.yaml          # tickers to monitor
│   ├── signals.yaml            # weights + political keywords
│   ├── backtest.yaml           # backtest parameters
│   └── paper_trading.yaml      # paper trading settings
│
├── src/signal_engine/
│   ├── data/
│   │   ├── alpaca_client.py    # price + volume data (Alpaca API)
│   │   ├── news_client.py      # Finnhub + NewsAPI adapters
│   │   └── cache.py            # Parquet-backed local cache
│   │
│   ├── signals/
│   │   ├── technical.py        # MA, RSI, MACD, breakout signals
│   │   ├── sentiment.py        # news sentiment scoring
│   │   ├── political.py        # keyword/entity trigger detector
│   │   └── composite.py        # weighted score aggregator + ranker
│   │
│   ├── backtest/
│   │   ├── engine.py           # vectorbt wrapper
│   │   ├── metrics.py          # win rate, drawdown, Sharpe
│   │   └── grid_search.py      # weight + hold period sweep
│   │
│   ├── paper_trading/
│   │   ├── broker.py           # Alpaca paper order manager
│   │   ├── tracker.py          # SQLite trade log
│   │   └── reporter.py         # daily P&L summary (Rich tables)
│   │
│   ├── leaps/
│   │   ├── candidate_screener.py  # long-horizon signal logic
│   │   ├── options_stub.py        # stubbed options chain client
│   │   └── README_LEAPS.md        # data source docs + activation guide
│   │
│   └── cli.py                  # CLI entry points
│
├── tests/
│   ├── test_technical_signals.py
│   ├── test_sentiment_signals.py
│   ├── test_composite_scorer.py
│   ├── test_political_trigger.py
│   └── fixtures/               # sample data for tests
│
└── notebooks/                  # optional exploratory analysis
```

---

## Upgrading the Political Trigger to LLM

The `PoliticalTriggerDetector` is built with upgrade hooks. Search for `# TODO(upgrade): LLM classifier` in [`src/signal_engine/signals/political.py`](src/signal_engine/signals/political.py) to find the exact insertion point for a Gemini Flash or similar classifier.

---

## Win Rate Target

The system targets **≥70% win rate** (profitable trades / total trades, where even +0.01% = a win) before signals are trusted for paper trading. The backtest output includes:
- ✅ or ❌ indicator against the 70% bar
- Per-config results in the grid search CSV
- Equity curve HTML saved to `reports/`

**This tool is for research and paper trading only. No live brokerage connection is implemented.**

## Market regime and the weekly paper cycle

```bash
uv run signal-engine regime                    # fear/greed (0-100) + buying/selling pressure, with drivers
uv run signal-engine weekend-regime-study      # backtest tagged by regime; shows what the policy would change
uv run signal-engine weekly-cycle --phase auto # scan / trigger+submit / stop enforcement / review, by time of week
uv run signal-engine weekly-cycle --print-cron # crontab lines that run it unattended (Alpaca PAPER only)
```

Weekly reviews land in `reports/weekly/<monday>.md`. The regime policy (`config/market_regime.yaml`)
is off until the study shows it helps in both halves of the data.

# LEAPS Module — Options Data Source Guide

This module screens for long-horizon (LEAPS, 1yr+ expiration) options candidates.
The candidate **ranking** works fully without options data. The options chain data
is only needed to populate IV, strikes, and Greeks for the final selection step.

---

## Current Status

| Feature | Status |
|---------|--------|
| Candidate screening (RS, earnings, themes) | ✅ Fully functional |
| Options chain (IV, strikes, Greeks) | 🔌 Stub — connect a source below |

---

## Activating Options Data

Edit your `.env` file to add credentials for ONE of the following:

### Option 1: Polygon.io ⭐ (Recommended)

**Free tier**: 5 API calls/minute, end-of-day options data.
**Paid tiers**: Real-time data, higher rate limits.

```bash
# .env
POLYGON_API_KEY=your_polygon_api_key_here
```

Free signup: https://polygon.io/

**What you get free:**
- End-of-day options snapshots (strikes, expiry, OI, IV, Greeks)
- Historical options data
- Contract reference data

**Implementation hook**: `options_stub.py` → `_get_chain_polygon()`

---

### Option 2: Alpaca Options API

**Status**: Limited beta as of mid-2026. Requires an upgraded brokerage account.
Not available on paper-only or free accounts.

```bash
# .env
ALPACA_OPTIONS_ENABLED=true
# Uses your existing ALPACA_API_KEY and ALPACA_SECRET_KEY
```

Contact Alpaca at: https://alpaca.markets/ to request options access.

**Docs**: https://docs.alpaca.markets/reference/optionsnapshots

**Implementation hook**: `options_stub.py` → `_get_chain_alpaca()`

---

### Option 3: Tradier Brokerage API

**Free paper account** available. Provides real-time options chains.

```bash
# .env
TRADIER_API_KEY=your_tradier_key
TRADIER_ACCOUNT_ID=your_account_id
```

Signup: https://developer.tradier.com/

**Implementation hook**: `options_stub.py` → `_get_chain_tradier()`

---

## What Happens Without Options Data

The `LEAPSScreener.screen()` still returns a fully scored `LEAPSCandidate` with:
- LEAPS score (0-1) based on RS, earnings, theme alignment, trend
- Suggested strike (5% OTM from current price — informational only)
- Suggested min DTE = 365 days

`options_data_available` will be `False` and `suggested_strike` is an estimate only.
When you connect a data source, `options_data_available` becomes `True` and the
actual options chain data populates `OptionsContract` objects with real IV and Greeks.

---

## Implementation Steps (Polygon.io example)

1. Get free API key at https://polygon.io/
2. Add `POLYGON_API_KEY=your_key` to `.env`
3. Open `src/signal_engine/leaps/options_stub.py`
4. Implement `_get_chain_polygon()` — see the docstring for the exact endpoint and response mapping
5. Test: `uv run signal-engine leaps`

The `OptionsChain` dataclass is already fully defined — you just need to populate it.

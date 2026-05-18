# TSX Stock Picker — Project Plan

## Overview

Quantitative stock picker for the TSX (Toronto Stock Exchange) using XGBoost and Double Machine Learning.

- **Data source:** yfinance only (free, no API keys)
- **Model:** XGBoost regressor + classifier ensemble
- **Causal inference:** Double Machine Learning (Chernozhukov 2018)
- **Style:** Functions only, no classes

---

## Architecture

```
picker.py
├── Data Acquisition        fetch_prices(), fetch_macro(), fetch_fundamentals(),
│                           fetch_earnings_surprise(), fetch_dividend_history(),
│                           fetch_quarterly_financials(), get_ohlcv()
├── Constraint Filtering    apply_constraints(), _apply_concentration_limits()
├── Feature Engineering     compute_monthly_features(), build_panel(),
│                           compute_earnings_surprise_feature(),
│                           compute_dividend_growth_feature(),
│                           compute_revenue_growth_feature(),
│                           smart_impute(),
│                           cross_sectional_normalize(), apply_momentum_pca()
├── Models                  fit_models(), ensemble_predict()
├── Double Machine Learning estimate_dml_alpha(), apply_dml_adjustment()
├── Regime Detection        detect_regime()
├── Rebalancing Band        apply_rebalancing_band()
│                           (enforces required_sectors min-1/max-2)
├── Position Sizing         risk_parity_weights()
├── Walk-Forward Backtest   walk_forward()
├── Prediction              predict_now()
└── Output                  print_picks(), print_backtest()
```

---

## Key Components

### XGBoost Ensemble

| Model | Purpose | Blend |
|-------|---------|-------|
| Regressor | Predict next-month return | 50% |
| Classifier | Predict top-quintile membership | 50% |

Both use time-decay sample weights (half-life 12 months).

### Double Machine Learning

Estimates the **causal** effect of momentum signals on returns:
1. Predict Y (return) from X (features) → residual E_Y
2. Predict T (treatment signal) from X → residual E_T
3. OLS: E_Y ~ E_T → theta (causal effect)

Uses `TimeSeriesSplit` with gap=1 to prevent temporal leakage.

### Constraints

- **Universe:** focused on 4 sectors (Financials, Energy, Industrials, Utilities) — see `TSX_UNIVERSE` in picker.py.
- **Per-sector caps:** `required_sectors` enforces **min 1 / max 2 picks per sector** via `apply_rebalancing_band`. Vol-spike anti-anomaly filter exempts current holdings.
- **Liquidity:** min ADV $1M, price $2–$400
- **Fundamentals:** PE 0–150, ROE 0–200%, market cap > $800M
- **Concentration:** max_per_gics=2, max_per_style=4, max_per_type=5, max 2 gold miners, max 1 base metal
- **Turnover:** rank buffer 18, hold_bonus 0.05, max 4 changes/month
- **Risk:** VIX scaling, drawdown halt at -15%, regime overrides only `top_n` (BEAR=4, BULL=8, NEUTRAL=8) so per-sector caps stay honored.

### Features

Default: `USE_MOMENTUM_PCA = True` collapses the four raw momentum
horizons into two PCA components (`mom_pc1`, `mom_pc2`). Toggle the
flag in `picker.py` to A/B against the raw-momentum baseline.

| Category | Features | Notes |
|----------|----------|-------|
| Momentum | mom_pc1, mom_pc2 (PCA-compressed) | raw mom_{1,3,6,12}m fall back when flag off |
| Volatility | 20d, 60d, ratio | |
| Technical | RSI, Bollinger Z, 52w high ratio | |
| Volume | ADV rank | |
| Macro (broad) | Oil (CL=F), CAD/USD, US 10Y rates, TSX, gold, VIX | |
| Macro (sector proxies) | natgas (NG=F), carbon (KRBN), transport (IYT), utilities (XLU), inflation (TIP), CAD bonds (XBB.TO) | yfinance ETF/futures proxies for the per-sector spec — pipeline/utility regime, freight, electricity demand, inflation, BOC-rate proxy |
| Per-sub-industry growth | dividend_growth_yoy, revenue_growth_yoy | XGBoost auto-routes via sector_code splits — pipelines/utilities lean on div growth, services/AI infra on revenue growth |
| Fundamental | ROE, PE, div yield, EV/EBITDA, debt/equity | sector-median imputed (PIT disabled, see history) |
| Sector | GICS code | not normalized |
| Earnings surprise | most recent Surprise(%) within 3-month window | DML treatment only, not a model feature |

All numeric features get cross-sectional rank normalization (per
month, to `[-1, +1]`) into `*_norm` companion columns; originals
stay intact so `fwd_ret` keeps raw return units.

---

## Usage

```bash
cd ~/Documents/stock-picker-claude
source venv/bin/activate

python picker.py pick           # Current month top picks
python picker.py backtest       # Walk-forward backtest (~2-3 min)
python picker.py both           # Both

python smoke_test.py            # End-to-end pipeline sanity check (~30s)
python diagnose_holdings.py     # Why each holding did/didn't make picks
python diagnose_constraints.py  # Which constraint rejected a given ticker
```

### Setup

```bash
cp portfolio_config.example.py portfolio_config.py
# edit portfolio_config.py with your TSX holdings
```

`portfolio_config.py` is gitignored — real holdings stay local. When
non-empty, the rebalancing band prefers keeping them across runs.

---

## What Was Removed (from legacy picker_ca.py)

- LightGBM, PyTorch MLP, MC Dropout inference
- FMP API, SimFin API, SEDI insider scraping
- Black-Litterman optimization
- Sensitivity analysis grid search
- HTML/PDF report generation, email sending
- All class-based structures

---

## Potential Enhancements

_All planned enhancements complete. See "Tried and Rejected" below for items that didn't survive measurement._

### Tried and Rejected

- **PIT fundamentals (2026-05-16)**: Wiring `compute_pit_fundamentals` into `build_panel` regressed the backtest by ~3pp annualized (Sharpe 0.83 → 0.76). yfinance's 5-8-quarter limit means PIT data only covers the last ~12-15 months of each training window; the resulting mixed-coverage signal trained worse than uniform sector-median imputation. The function is preserved in `picker.py` for future use with a deeper fundamentals source.

- **Spec-coverage feature additions (2026-05-17)**: Added yield-curve slope (`^TYX`/`^IRX`), credit-spread proxy (`HYG`/`LQD`), refining-margin (`RB=F` - `CL=F`), `^GSPC`/`^IXIC`/`XLK` equity-beta proxies, and per-ticker P/B to close gaps in the per-sector spec coverage. Regressed Sharpe from 1.65 to 1.56 (full additions) and 1.54 (selective keep that dropped only the 1-stock-only additions). Reverted — at the current sample size (~31 tickers × 84 months split across 4 sector models), more features add noise faster than signal. The tickers stay commented in `MACRO_TICKERS` as a record so the experiment isn't accidentally redone.

---

## Dependencies

```
pandas, numpy, scikit-learn, xgboost, yfinance, scipy
lxml  (required by yfinance.get_earnings_dates HTML scraper)
torch, lightgbm (optional, not used in picker.py)
matplotlib, seaborn, plotly, jupyter (for analysis)
```

---

## Session History

### 2026-05-17 — focused universe + per-sector logic

Pivoted from the 95-name curated universe to a 28-name focused list
across 4 sectors (Financials, Energy, Industrials, Utilities), then
layered in the per-sub-industry logic from the user's spec.

**Universe pivot**

- Briefly expanded to ~223 names via TSX Composite (`expand_universe.py`
  → `tsx_extended.py`) — runtime ~3 min, two new picks surfaced. The
  expansion script and `tsx_extended.py` stay in the repo for future
  re-runs if the focused universe gets widened again.
- Replaced `TSX_UNIVERSE` with the focused 28 + XIU.TO + NA.TO + EQB.TO
  (= 31 tickers). `tsx_extended.py` is now imported only for profile
  lookup, not for universe expansion.
- Per-user override: `STOCK_PROFILE["CLS.TO"] = ("Industrials", ...)`
  even though yfinance classifies it as Information Technology.

**Per-sector rule (min 1 / max 2)**

- `CONSTRAINTS["required_sectors"]` lists the 4 active sectors;
  `max_per_gics=2` enforces the per-sector ceiling.
- `apply_rebalancing_band` rewritten as two phases:
  1. Guarantee 1 pick per required sector (highest-scoring there,
     holdings boosted by `hold_bonus`).
  2. Fill remaining slots up to `top_n` by score, capped per sector.
- `detect_regime` no longer overrides `max_per_gics` — only `top_n`
  modulates (BEAR=4, BULL=8). The sector caps stay invariant across
  regimes.
- Side effect: bank-heavy holdings (7 banks) get trimmed to 2 in picks
  — the model now implicitly recommends rebalancing.

**Per-sector signals (yfinance-proxies path)**

After feasibility check on SimFin/FMP (SimFin has none of the
sector-specific macros; FMP has only CPI and US PMI), chose the
yfinance-only path with ETF/futures proxies:

- New macro tickers: `NG=F`, `KRBN`, `IYT`, `XLU`, `TIP`, `XBB.TO` —
  each contributes a `*_mom_1m` feature.
- New per-ticker growth features:
  - `dividend_growth_yoy` from `yfinance.Ticker.dividends` (TTM/TTM-1).
    92.7% panel coverage.
  - `revenue_growth_yoy` from quarterly income statements with 45-day
    lag. ~5% coverage (yfinance only returns 5-8 quarters per ticker).
- **No hard-coded sub-industry signal routing**: per the design
  discussion, XGBoost splits on `sector_code` naturally and picks the
  right signal per leaf. If the model ever fails to auto-route
  correctly, a `SIGNAL_MAP` would be the explicit fallback.

**Backtest impact** (focused 4-sector universe, 47-month walk-forward):

| Metric | Pre (95-ticker) | Post (focused + new features) |
|---|---|---|
| Sharpe | 1.10 | **1.15** |
| Max drawdown | −15.8% | **−8.5%** |
| Excess vs XIU.TO | +9.8%/yr | +1.0%/yr |
| Top features | mom_pc1/2, sector_code | mom_pc1/2, vol_20d, sector_code, **rev_growth_yoy (#5)**, **div_growth_yoy (#9)** |

Lower excess is expected for the narrower universe; the model is more
defensive (much smaller drawdown for similar Sharpe). Both new growth
signals land in the top-10 importances, confirming XGBoost is using
them per the auto-routing assumption.

**Operational changes**

- Vol-spike anti-anomaly filter exempts current holdings (so a held
  stock doesn't get force-sold over a post-earnings volume spike).
- TSX universe cleanup: TECK.TO → TECK-B.TO, ATA.TO → ATS.TO renames;
  ERF.TO/SSL.TO/AND.TO removed (delisted).
- XIU.TO short-circuited in `fetch_earnings_surprise` (ETF, no
  earnings — silences yfinance warning).
- `lxml` added to requirements.txt (needed by yfinance earnings scraper).
- `expand_universe.py` + `tsx_extended.py` retained as the re-runnable
  path back to a larger universe if needed.
- `diagnose_holdings.py` and `diagnose_constraints.py` were the tools
  that surfaced which constraint filters were silently dropping
  holdings (vol_spike, max_per_type cap on "bank" sub_type).

### 2026-05-16 — initial productionization

Took the rewrite from "compiles" to "produces realistic numbers with a
known baseline." Final backtest (47 months, 2022-05 → 2026-03):

| Metric | Value |
|---|---|
| Portfolio annualized | **+25.6%** |
| Benchmark (XIU.TO) annualized | +15.8% |
| Excess | **+9.8%** annualized |
| Sharpe | **1.10** |
| Max drawdown | −15.8% |
| Hit rate | 55.3% |

**Bugs fixed (these will rot the next rewrite if reintroduced):**

1. `apply_constraints` crashed on multi-ticker `yf.download` output —
   `price_df[ticker]` returns an OHLCV DataFrame, not a Close Series,
   so `last_price` comparisons hit ambiguous truthiness. Fix: route
   all 4 sites through `get_ohlcv(price_df, ticker)`.
2. **The big one**: `cross_sectional_normalize` was overwriting
   `mom_1m` to a `[-1, +1]` rank *before* `add_labels` derived
   `fwd_ret` from it. The XGBoost regressor was being trained to
   predict ranks, and `walk_forward` compounded those ranks as if
   they were monthly returns — XIU.TO appeared to return +200%/yr in
   2024. Fix: `add_labels` runs before normalize; normalize writes to
   `_norm` suffix columns instead of mutating originals.

**Enhancements landed:**

- **PCA momentum compression** (`USE_MOMENTUM_PCA = True`): biggest
  single win, Sharpe 0.83 → 1.10. Four collinear momentum horizons
  were giving XGBoost more redundant noise than it could deduplicate.
- **Earnings surprise as DML treatment**: pulls `Surprise(%)` from
  `yfinance.get_earnings_dates` (needs `lxml`). Sharpe 0.78 → 0.83.
- **`portfolio_config.py`**: gitignored, populated from
  `.example.py`. Enables the rebalancing band's incumbency
  preservation. Empty list = fresh ranking every run.
- **Vol-spike filter exempts current holdings**: anti-pump filters
  apply to *new* picks, not stocks you already own. RY/ENB were
  getting dropped post-earnings without this.
- **Smoke test** (`smoke_test.py`): asserts the invariants that would
  catch both major bugs (Series-vs-scalar, rank-as-return). Run after
  any non-trivial change.
- **Diagnostic scripts** (`diagnose_holdings.py`,
  `diagnose_constraints.py`): re-runnable when a holding drops
  unexpectedly. The vol-spike fix above came directly from these.

**TSX universe cleanup:** 95 → 93 active tickers. Renamed
TECK.TO → TECK-B.TO, ATA.TO → ATS.TO; removed ERF.TO, SSL.TO, AND.TO
(all acquired/delisted by 2024-2026).

**Tried and rejected:** PIT fundamentals (see "Tried and Rejected"
above) — yfinance's quarterly history is too shallow to be useful.

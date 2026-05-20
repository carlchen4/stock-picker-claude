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

### Candidate features — companion-script second-pass scan (2026-05-20)

A re-read of `financials_dml_picker-2.py` and `monthly_rank.py` (beyond
the 8-step integration plan in the session history) surfaced four
factor/construction ideas that `picker.py` did **not** have. Listed in
rough ROI order (#1 now landed, #2-4 still open):

1. ✅ **Short-term reversal factor `rev_1m`** — **landed 2026-05-20**.
   Added `rev_1m = -mom_1m` as a standalone column kept OUT of
   `_RAW_MOMENTUM` (so the momentum PCA does not absorb it) and added to
   `_BASE_SECTOR_FEATURES`, giving every sector model a clean 1-month
   signal independent of the `mom_pc1/pc2` mixture. Backtest (same
   47-month walk-forward): Sharpe, annualized (+28.0%) and excess
   (+12.1%) all **held flat**, but max drawdown improved −8.2% → −7.6%
   and hit rate 63.8% → 66.0%. `rev_1m_norm` lands #8 in feature
   importance (0.0557, above high_52w_ratio and adv_20d_rank), and the
   per-year P&L shifted (2022 −3.9% → −0.0%, 2024 +34% → +43%),
   confirming it changes selection rather than no-op'ing. Net: a mild
   risk-side win with no return cost, so kept. (NB: for the pure tree
   models `rev_1m == -mom_1m` is split-equivalent to `mom_1m`; the gain
   is from restoring 1m as its own dimension outside the PCA.)
2. **12−1 momentum (skip most recent month)** (`monthly_rank.py:256`,
   `(1+ret).rolling(12).prod().shift(1)`). `picker.py`'s `mom_12m` =
   `pct_change(12)` *includes* the last month, so short-term reversal
   contaminates the 12-month momentum signal. Switching to the
   skip-a-month construction is the academic standard and a cheap
   quality improvement to the momentum inputs.
3. **Feature-drift health check** (`monthly_rank.py:570-583`):
   flag this month as unreliable when any feature exceeds the training
   set's 95th percentile × 1.5. `picker.py`'s health_check has only 4
   tests (sector coverage / signal strength / DML significance / data
   completeness) — this is the concrete implementation for the
   still-open integration-plan Step 5.
4. **Data hygiene — `ffill(limit=3)`** (`monthly_rank.py:44,891`):
   cap forward-fill at 3 months when aligning prices/macro to the
   monthly index, so a stale value can't be carried indefinitely.
   Minor; fold into whatever touches the alignment next.

### Tried and Rejected

- **LGB ensemble — Step 2 (2026-05-20)**: Added LightGBM as a second tree model, blending XGB+LGB 50/50 on rank-normalized predictions per sector (`fit_sector_ensemble`/`predict_sector_ensemble`). Also tried an ElasticNet feature gate inside the blend, which dropped features XGB found useful (Sharpe 1.91 → 1.52), so EN was backed out and the blend reduced to 2-model. Even the clean 2-model blend regressed everything vs the single-XGB-per-sector baseline (same 47-month walk-forward): Sharpe **1.92 → 1.62**, annualized +28.0% → +22.9%, excess +12.1% → +7.1%, hit rate 63.8% → 55.3% (only max drawdown improved marginally, −8.2% → −7.1%). At ~31 tickers × 84 months split across 4 sector models, a correlated second tree learner adds variance faster than it diversifies error. Reverted entirely — `picker.py` keeps the single XGBoost regressor per sector. The companion `monthly_rank.py` LGB+XGB+EN ensemble does not port over at this sample size.

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

### 2026-05-19 — companion-file analysis (financials_dml_picker-2.py + monthly_rank.py)

User wrote two standalone scripts and asked for a comparison against
`picker.py` to identify what to port back. Both files are
Financials-focused (one 23-stock multi-category, one 6-bank tight) and
implement DML-PLR with different formulations. The strengths are
genuinely useful and worth integrating; the weaknesses are documented
so the integration avoids the pitfalls.

**`financials_dml_picker-2.py` (614 lines, 23 Canadian financials)**

*Strengths:*
- **ETF-baseline DML**: ZEB.TO dropped from ticker dummies → each
  stock's α is measured *relative to the sector ETF*. Much cleaner
  causal interpretation than picker.py's momentum-as-treatment.
- **HC3 robust standard errors** via `statsmodels.OLS` → per-stock
  t-stats and p-values, enabling "significant α" filtering.
- **Real macro data**: BOC Valet (V39079 rates, V122538 group for
  2y/10y bond yields), FRED (`CPALCY01CAM661N` Canadian CPI YoY,
  `BAMLC0A0CM` US IG OAS). All three are gaps `picker.py` only has
  proxies for.
- **Rolling 24m β to both XIC and ACWI** as per-stock features —
  captures time-varying market sensitivity.
- **Excess returns** as target (`fwd_ret − rf_monthly`) — standard
  alpha analysis framing.
- **HTML report output** with category badges, p < 0.05 highlights,
  final score bars. Operational nicety for sharing.
- **5-category profile** (Big Bank / Insurance / Alt Asset / Asset
  Mgr / Alt Fin) finer than picker.py's bank/insurance/holding
  sub_types.

*Weaknesses:*
- **No walk-forward backtest** — one-shot fit on all history,
  predicts only the latest month. The "0.60·DML + 0.40·XGB"
  weighting is unjustified without historical measurement.
- **Look-ahead in rolling β during cross-fit** — β is computed on
  full history, then cross-fitting partitions the panel. Training
  folds see β values that incorporate test-period observations.
  Subtle but real time-series leakage.
- **No liquidity/quality filters** — universe lets in everything
  that downloads. `SII.TO`, `LB.TO` etc. trade thin; picker.py's
  `min_adv`, `vol_spike`, and fundamental-band filters would catch
  these.
- **MultiOutputRegressor with one-hot ticker dummies** for the D
  residualization — 22 sparse one-hot columns × XGBoost is hard to
  fit; could overfit with this many output dimensions on ~120
  monthly observations.
- **No position sizing** — picks are output as a list, equal-weight
  implicit. No risk parity.

**`monthly_rank.py` (927 lines, 6 Canadian Big Banks)**

*Strengths:*
- **DML-PLR with sector factor**: cleaner formulation than
  `financials_dml_picker-2.py`. D = sector return (scalar),
  X = macros, residualize both via cross-fit, closed-form
  θ̂ = (D̃′Ỹ) / (D̃′D̃). Score-based standard error gives
  significance test for the sector-β estimate itself.
- **3-model ensemble** (LightGBM 40% + XGBoost 40% + ElasticNet
  20%). EN doubles as a feature-selection gate — only features with
  non-zero EN coefficients enter the tree models.
- **Cross-sectional z-score normalization** per date (preserves
  magnitude info that picker.py's percentile-rank normalization
  drops).
- **Quarterly fundamentals computed PIT** with 7-day parquet cache
  (`.cache_fundamentals.parquet`): ROE, NIM proxy
  (NII/total_assets), efficiency (opex/revenue), asset growth,
  equity growth. All lagged by 1 quarter (91 days) to prevent
  forward-looking bias.
- **PIT P/B**: `price / (equity / shares_outstanding)` using
  historical balance sheet — better than picker.py's current
  snapshot from `yfinance.Ticker.info`. ⚠️ **Caveat (2026-05-20
  scan):** `monthly_rank.py:277-289` uses the *current*
  `sharesOutstanding` applied across all of history. Buybacks/issuance
  make historical share counts ≠ today's, so this P/B carries a hidden
  look-ahead bias. The equity numerator is point-in-time but the share
  denominator is not — when porting (Step 4), source historical shares
  or drop P/B rather than copy this construction.
- **5-test health check** at predict time: model agreement
  (LGB vs XGB Spearman), signal strength (max |α_z|), feature
  normality (within training 95% percentile), DML significance
  (p < 0.10), data completeness. Failing 2+ tests = "建议观望".
- **Walk-forward validation metrics**: IC mean, ICIR, win rate,
  L/S Sharpe, max drawdown. Sparkline ASCII chart for cumulative
  L/S returns.
- **Rank history tracking** in `rank_history.csv` with month-over-
  month delta (↑ / ↓ / →) printed in the ranking output.
- **Embargo** of 6 months between train and test in walk-forward.

*Weaknesses:*
- **CRITICAL BUG**: `dml_strip()` uses `np.random.permutation(n)` for
  cross-fitting fold assignment (line ~386). On time-series data this
  randomly mixes future observations into training folds — a real
  leakage. Should use `TimeSeriesSplit` or `KFold` over sorted dates.
- **Missing imports**: `lgb`, `norm`, `spearmanr` are referenced but
  never imported — the file as-shipped doesn't actually execute.
  Must add `import lightgbm as lgb`, `from scipy.stats import norm,
  spearmanr` before running.
- **Rolling β with `for i in range(window, len(...))` is past-only**
  for the live signal, but the walk-forward validation re-uses the
  same β series — meaning a train period at date t sees β values
  that depended on returns ≤ t but were *constructed* on the full
  panel. Minor but worth tightening.
- **Static train/test split** `TRAIN_END = "2022-12-31"` for the
  single-pass `train_and_rank()` — only the explicit `--validate`
  flag runs walk-forward.
- **EN gate fixes `l1_ratio` to [0.3, 0.5, 0.7, 0.9]** but doesn't
  grid-search `alpha`. Default `alphas=None` does an internal path,
  but the sparsity level chosen depends entirely on CV and may not
  be optimal.
- **Universe = 6 banks** — too tight to generalize. Insurance,
  asset managers, alt finance need their own treatment.
- **No earnings_surprise** or other DML treatment beyond sector
  factor.

**Integration plan into picker.py** (ranked by ROI):

1. **ETF-baseline DML-PLR per sector** (highest value): ZEB/XFN for
   Financials, XEG for Energy, ZIN for Industrials, XUT for Utilities
   as sector-factor treatments. Closed-form θ̂ per sector via
   `TimeSeriesSplit` cross-fitting. Use as `alpha_target = fwd_ret
   − θ̂·sector_etf_ret` for the per-sector XGBoost target.
2. **3-model ensemble (LGB + XGB + EN)** with EN doubling as feature
   gate — reduces single-model noise.
3. **Rolling 24m β** (strict past-only): equity_β, sector_β,
   cad_β as per-ticker time-varying features.
4. **Quarterly fundamentals + parquet cache**: revisit PIT
   fundamentals with `monthly_rank.py`-style ROE/NIM/efficiency. The
   prior PIT attempt regressed; caching + cleaner derivation may flip
   the result. **Do not** copy `monthly_rank.py`'s PIT P/B as-is — it
   uses current `sharesOutstanding` over all history (look-ahead, see
   caveat above); use historical shares or skip P/B.
5. **5-test health check** at `predict_now` output — add the
   feature-drift test (this-month feature > train 95th pct × 1.5) that
   `picker.py`'s current 4-test check is missing.
6. **IC / ICIR / win-rate / L/S Sharpe** in `walk_forward` results.
7. **HC3 standard errors** for per-stock alpha significance gating.
8. **Rank history file** for month-over-month change reporting.

User picked **Step 1 (ETF-baseline DML + health checks) first** —
landed in commit `4f7b0ef`.

**Step 2 (LGB ensemble) — attempted then rejected (2026-05-20):** built
the XGB+LGB blend (+ a tried-and-dropped ElasticNet gate), measured it
against the single-XGB baseline, and it regressed across the board
(Sharpe 1.92 → 1.62). Reverted; see "Tried and Rejected" above for the
full numbers. Remaining integration-plan items (rolling β, quarterly
fundamentals, IC/ICIR metrics, HC3 SEs, rank history) are still open.

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

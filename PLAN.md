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
2. ❌ **12−1 momentum (skip most recent month)** — **tried, rejected
   2026-05-20**. Changed `mom_12m` from `pct_change(12)` to
   `monthly.shift(1).pct_change(12)` (= `price[t-1]/price[t-13]-1`, the
   Jegadeesh-Titman skip-a-month construction; `monthly_rank.py:256`).
   It raised raw return (+28.0% → +29.7% ann, excess +12.1% → +13.9%)
   but **regressed every risk metric**: Sharpe 1.92 → 1.88, max drawdown
   −7.6% → −8.1%, hit rate 66.0% → 63.8%. The skip-a-month signal tilts
   selection toward higher-beta/more-volatile names — more return per
   unit of nothing, worse return per unit of risk. Reverted to keep the
   Sharpe-first discipline consistent (same reason the LGB ensemble was
   dropped). Could revisit as a *separate* `mom_12_1` feature alongside
   `mom_12m` rather than replacing it.
3. ✅ **Feature-drift health check** — **landed 2026-05-20**. Added as
   `health_check`'s 5th test (sector coverage / signal strength / DML
   significance / data completeness / **feature drift**). Adapted, not
   copied, from `monthly_rank.py:570-583`: that one flags a feature if
   *any single stock* exceeds the train 95th pct × 1.5, which on
   picker.py's 31-name universe with right-tailed raw features (ADV,
   vol, growth) fired on 8/13 features — pure noise. Instead picker.py
   compares **this month's cross-sectional mean** of each *raw* feature
   (not the `_norm` rank columns, which are bounded [-1,1] and can't
   drift) against the distribution of monthly cross-sectional means in
   training, flagging |z| > 3 (~99.7%). This is robust to a single
   outlier name and fires only on genuine market-regime shifts. Live
   run flags exactly one: `rev_growth_yoy` (z=7.6), correctly surfacing
   that low-coverage feature's instability. Closes integration-plan
   Step 5.
4. **Data hygiene — `ffill(limit=3)`** (`monthly_rank.py:44,891`):
   cap forward-fill at 3 months when aligning prices/macro to the
   monthly index, so a stale value can't be carried indefinitely.
   Minor; fold into whatever touches the alignment next.

### Candidate features — picker_ca.py scan (legacy predecessor, 2026-05-20)

`picker_ca.py` is the pre-rewrite predecessor (6643 lines). Most of its
extras were intentionally dropped (see "What Was Removed") or have since
been disproven here (LGB). A scan for things that fit the current
yfinance-only / functions-only design AND address this session's
recurring lesson (adding features overfits the small sample) surfaced:

1. ✅ **VIF collinearity diagnostic** — **landed 2026-05-20**. Added
   `compute_vif` + `run_vif_diagnostic` and a `python picker.py vif`
   mode (needs `statsmodels`, now in requirements.txt). Computes VIF on
   the RAW columns the model consumes (maps `_norm`→base; rank-normalized
   columns are [-1,1]-bounded and hide collinearity). **Findings**: 6
   features show severe collinearity (VIF>10), all technical/volatility:
   high_52w_ratio 65.8, vol_ratio 41.0, rsi_14 39.7, vol_20d 28.8,
   vol_60d 27.0, vix_level 17.7. By contrast mom_pc1/pc2 sit at 1.5/1.3
   — proof the momentum PCA worked — and rev_1m at 2.5. This explains
   why every add-feature experiment this session regressed: the feature
   space is already saturated with redundant price-derived technicals,
   so new features just add noise. **Correction (same day)**: the obvious
   follow-up — PCA-compress the vol family like the momentum PCA — was
   tried and *failed* (see Tried and Rejected). High VIF does NOT imply
   "compress this" for a TREE model: collinearity hurts *linear* models,
   but XGBoost is robust to it, and the vol family's term-structure
   detail (esp. vol_ratio) stays useful as split signal. Read VIF here as
   "why adding features doesn't help" (saturated space), NOT as a
   compress-it to-do list.
2. ✅ **Segmented evaluation** (`evaluate_segments`) — **landed
   2026-05-20, closes Step 6**. `walk_forward(return_perstock=True)` now
   also returns a per-(month,ticker) frame; `evaluate_segments` prints
   per-year RankIC, per-sector RankIC, and pick turnover after the
   backtest. Pure diagnostic — model/Sharpe unchanged.

   **⚠️ Major finding — what actually drives the ~1.9 Sharpe:** the
   cross-sectional stock-picking signal is *weak and regime-dependent*.
   Full-sample RankIC = **+0.029** (ICIR 0.17; industry "useful" is
   ~0.03–0.05), and it is wildly uneven by year: 2023 +0.065 / 2026
   +0.120 (good) vs 2024 +0.010 / 2025 +0.002 (≈ zero). Per-sector IC is
   tiny everywhere (Energy +0.014, Financials +0.005, Industrials +0.042;
   all ICIR < 0.1). So the +28%/yr, Sharpe-1.9 backtest is driven mainly
   by **structural sector/beta exposure (the min-1/max-2 per-sector band)
   plus the 2023–24 regime**, NOT by precise cross-sectional alpha.

   This explains why every feature experiment this session regressed or
   no-op'd: the edge isn't in stock-selection precision, so tuning
   selection features (LGB, betas, vol PCA, 12-1) can't move returns that
   come from elsewhere. **Live implication:** the result leans on the
   2023–24 regime repeating; 2025 already lagged the benchmark (excess
   −4.2%, IC ≈ 0). Treat it as a structured sector-tilt strategy, not a
   reliable selection-alpha engine. Turnover ~54%/mo.
3. ✅ **Monthly email report to self** — **landed 2026-05-20**,
   *user-requested*. After `pick` runs, `build_report_text` formats a
   plain-text report (picks + weights + regime + top features) and
   `send_report_email` sends it via `smtplib.SMTP_SSL("smtp.gmail.com",
   465)` to **carlchenyiqing@gmail.com** (the user's preferred inbox —
   NOT the `carl.chen@myyahoo.com` in the env userEmail context).
   - Credentials live in **`email_config.py`** (gitignored, with
     `email_config.example.py` as the committed template): `EMAIL_FROM`,
     `EMAIL_TO`, `EMAIL_APP_PASSWORD` (Gmail app password). Absent or
     placeholder values → send is skipped, so a fresh clone still runs.
   - `.gitignore` also covers `tsx_report_*.html/.pdf` artifacts.
   - Verified with a live test send. Plain text only — no HTML/PDF port
     needed. ⚠️ The app password used was shared in chat; rotate it.

### ETF-fallback dynamic threshold (picker_ca.py:2868), considered

`calculate_etf_threshold` retreats to holding XIU.TO when the month's
average ensemble score is below the historical 20th percentile — a
data-driven "信号太弱就持基准" guard, complementary to the existing
regime/drawdown-halt logic. Logged as a future option (not selected for
write-up this round). PIT as-of constraints (`apply_constraints_asof`)
were also reviewed but are limited by yfinance's shallow PIT data (same
constraint that benched the PIT-fundamentals experiment).

### Diagnosed weaknesses & proposed fixes (from Step 6, 2026-05-20)

Step 6 (segmented evaluation) showed the model is a **sector-tilt /
regime strategy, not a selection-alpha engine** (full-sample RankIC
0.029, regime-dependent, per-sector ICIR < 0.1). The fixes below target
the ROOT — the mismatch between the modeling target (cross-sectional
fwd_ret ranking) and the actual return source (sector exposure + regime).
They deliberately do NOT add selection features; this session's 4 failed
experiments proved that's optimizing the wrong layer. **All unverified —
each needs an A/B against the Sharpe-1.92 baseline before adoption.**

| # | Weakness (evidence) | Proposed fix | Risk |
|---|---|---|---|
| 1+3 | Selection signal weak; returns come from sector tilt, not stock-picking (RankIC 0.029 vs +28%/yr) | **Re-target to sector rotation**: predict each sector ETF's *relative* forward return, then hold the chosen sector(s) via the existing min-1/max-2 band (equal-weight or by sector score) instead of ranking individual names. Optimizes the layer that actually drives returns. | Largest change (rewrites the target); but it's the only fix aimed at the real edge |
| 2 | Financials (largest sector, 7 banks) internal IC ≈ 0 (+0.005) | Stop selecting *within* Financials — the big banks are too homogeneous (beta-convergent). Hold a bank basket (top-2 equal-weight, or ZEB/XFN ETF) and spend modeling effort on sectors with some signal (Industrials IC +0.042). | Low; mostly a constraint/sizing change |
| 4 | Edge is regime-dependent (2024/2025 IC ≈ 0, yet still fully invested) | Implement the **ETF-fallback** (`calculate_etf_threshold`, already scoped above): when rolling RankIC / mean score is weak, retreat to XIU.TO instead of force-picking in a no-skill regime. Directly cushions the 2024-25 failure mode. | Low; picker_ca has a reference impl, pure overlay |
| 5 | Turnover 54%/mo on a weak signal = noise trading / wasted cost | Trade less when the signal is weak: raise `rank_buffer`/`hold_bonus`, or move to quarterly rebalancing. Testable immediately with existing params. | Low; parameter tuning, fast A/B |
| (root) | Only ~8 names per sector model — too few for cross-sectional ranking | Cautiously widen the universe (31 → ~95/223, the `expand_universe.py` path) to give ranking room. | Medium; PLAN history shows the focused universe was *more* defensive — must A/B, don't assume |

**Strategic fork** (these are project-level directions, not tweaks):
- **A — Go with the grain (recommended):** re-target to sector rotation
  (#1) + ETF-fallback (#4). Optimize the real edge instead of fighting a
  weak selection signal.
- **B — Rescue selection:** widen the universe (#5) and re-check RankIC.
  Works against the diagnosis; only worth it if RankIC actually lifts.
- **C — Accept & de-risk:** keep it as a sector-beta strategy, add the
  ETF-fallback to cap regime risk, and lower return expectations.

**Suggested starting point: fix #4 (ETF-fallback).** Lowest risk (a pure
overlay, reference impl exists), directly addresses the most painful
finding (regime dependence), and doesn't touch the modeling target — so
it's reversible and measurable on its own before committing to the
bigger sector-rotation rewrite (fork A).

### Tried and Rejected

- **Volatility-family PCA (2026-05-20)**: VIF flagged vol_20d/vol_60d/vol_ratio as severely collinear (41/29/27), so — by analogy with the successful momentum PCA — compressed them into a single `vol_pc1` (`apply_vol_pca`, `USE_VOL_PCA` flag). Regressed all five metrics: Sharpe 1.92 → 1.66, annualized +28.0% → +23.9%, excess +12.1% → +8.1%, max drawdown −7.6% → −8.2%, hit rate 66.0% → 53.2%. **Lesson (corrects the VIF write-up's first instinct):** high VIF does not justify compression for a *tree* model. Collinearity inflates variance in *linear* estimators; XGBoost splits are unaffected by it, and the short/long vol distinction plus vol_ratio's term-structure carry real signal that one PC discards. The momentum PCA worked for different reasons (4 horizons far more redundant; model was at Sharpe 0.83 with room to gain), not a general "compress high-VIF families" rule. Reverted.

- **Rolling 24m betas — Step 3 (2026-05-20)**: Added three per-ticker, strictly past-only rolling betas (`equity_beta` vs ^GSPTSE, `sector_beta` vs the sector ETF, `cad_beta` vs CADUSD=X) via `compute_rolling_betas`, wired into `_BASE_SECTOR_FEATURES`. The implementation was correct — betas computed with right-aligned `rolling(24)` so each (ticker, month) value uses only data ≤ that month (avoiding the full-panel look-ahead the companion scripts have), validated by equity_beta mean ≈0.98 / sector_beta ≈0.88. But it regressed **all five metrics** vs the rev_1m baseline: Sharpe 1.92 → 1.83, annualized +28.0% → +26.4%, excess +12.1% → +10.6%, max drawdown −7.6% → −8.8%, hit rate 66.0% → 57.4%. Same lesson as the spec-coverage and LGB experiments — at ~31 tickers × 84 months across 4 sector models, adding 3 features × 4 models (one of them, `cad_beta`, very noisy at range [−6, +10]) overfits faster than the betas inform. A feature-budget problem, not a bug. Reverted entirely.

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
3. ❌ **Rolling 24m β** (strict past-only): equity_β, sector_β,
   cad_β as per-ticker time-varying features. **Tried, rejected
   2026-05-20** — regressed all five metrics (see Tried and Rejected).
4. **Quarterly fundamentals + parquet cache**: revisit PIT
   fundamentals with `monthly_rank.py`-style ROE/NIM/efficiency. The
   prior PIT attempt regressed; caching + cleaner derivation may flip
   the result. **Do not** copy `monthly_rank.py`'s PIT P/B as-is — it
   uses current `sharesOutstanding` over all history (look-ahead, see
   caveat above); use historical shares or skip P/B.
5. ✅ **5-test health check** at `predict_now` output — **done
   2026-05-20**. Feature-drift test added (regime-shift variant, see
   "Candidate features" #3 above); health_check now runs all 5 tests.
6. ✅ **IC / ICIR / win-rate** in `walk_forward` results — **done
   2026-05-20** via `evaluate_segments` (per-year + per-sector RankIC,
   ICIR, hit-rate, turnover). See "Candidate features" #2 for the major
   finding it surfaced (weak, regime-dependent cross-sectional signal).
7. **HC3 standard errors** for per-stock alpha significance gating.
8. ✅ **Rank history file** for month-over-month change reporting —
   **done 2026-05-20**. `predict_now` writes every candidate's
   score-rank to `rank_history.csv` (gitignored run artifact) each run,
   idempotent per month, and prints ↑/↓/→/NEW deltas vs the most recent
   prior month in the "ALL STOCKS RANKED BY SCORE" block. Adapted from
   `monthly_rank.py`'s rank tracking. Pure reporting — does not touch
   the model or features, so backtest/Sharpe are unaffected.

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

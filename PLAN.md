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

Both use time-decay sample weights (half-life **6 months**, updated 2026-05-24 — see Experiment C).

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
- PDF report generation (HTML email re-added 2026-05-23, see Operations #10)
- All class-based structures

---

## Potential Enhancements

### ⚠️ Quantitative-rigor audit & gaps (2026-05-22)

Audited `picker.py` against a comprehensive QA checklist. **Full 9-model
comparison this session** (per-sector, same pipeline): ExtraTrees **2.09**
(mean of 7 seeds, std 0.05) > xgb 1.98 > rf 1.86 > catboost 1.74 >
LGB 1.62 > histgb 1.53 > gbdt/adaboost/linear 1.00. **Bagging > boosting
> linear** is now firmly established.

**Critical honesty — multiple-testing overfitting (the headline gap):**
this session ran ~15 experiments (9 models + 6 feature variants) and
reports the *maximum* (ExtraTrees 2.09). That is exactly the bias that
**Deflated Sharpe Ratio / PBO** correct for. The deflated Sharpe is
almost certainly < 2.09; how much of the +0.11 over XGB is real vs
selection noise is UNQUANTIFIED.

**→ Switched to ExtraTrees as default (2026-05-22).** Re-run on same
47-month walk-forward confirmed Sharpe **2.12** (ann. +33.0%, excess
+17.2%, max DD -8.2%, hit rate 63.8%). `MODEL_KIND = "extratrees"` in
picker.py.

**DSR / PBO audit completed (2026-05-22) — STRONG result:**
Implemented `compute_psr`, `compute_dsr`, `compute_sharpe_ci`, and
`print_overfit_report` in picker.py (shown in `backtest` output).
- Skewness −0.13, excess kurtosis −0.12 (near-normal)
- Bootstrap 95% CI for ann. Sharpe: **[1.17, 3.34]**
- PSR P(true SR > 0): **100.0%**
- DSR P(true SR > E[max₁₅]): **99.3%** (n_trials=15 experiments)
- PBO proxy (1 − DSR): **0.7%** ← selection-bias overfit risk
- Verdict: **STRONG — survives 15-trial selection correction**

The Sharpe of 2.12 is NOT selection noise. With only 47 months and
near-normal returns, the signal-to-noise ratio is high enough that the
DSR correction barely moves the needle. The previous caveat ("do NOT
hard-switch until deflated") is now resolved — ExtraTrees is confirmed
as the production default. The
same caveat deflates the 1.92 baseline and every "best" result here.

**Overfitting / leakage gaps (highest priority):**
- ⚠️ **Survivorship bias** — **documented known limitation (2026-05-23)**.
  Universe is currently-listed TSX names; yfinance cannot supply historical
  point-in-time constituents including delistings. ERF/SSL/AND were manually
  removed (already delisted by backtest start). Bias direction: modest upward
  tilt on historical Sharpe (surviving firms outperform delisted ones on
  average). Magnitude: small for a 4-sector focused universe over 47 months
  with only 3 known delistings removed. Cannot be fixed with yfinance;
  would require a paid PIT constituent database (e.g. Compustat, Refinitiv).
  Accepted as irremediable given the yfinance-only constraint.
- ❌ **Deflated Sharpe / PBO / White's Reality Check / FDR** — none;
  required to judge whether this session's "best" picks are real.
- ❌ **Embargo / Purging** — `walk_forward` has no gap between train/test;
  Purged K-Fold / CPCV (López de Prado) would be leakage-safe.
- ❌ **Nested CV** — hyperparameters hand-tuned over all history.
- ⚠️ **Look-ahead** — mostly avoided (past-only betas, monthly train),
  but momentum-PCA / cross-sectional-normalize fit on the full panel.
- ✅ Data/label leakage (fwd_ret = shift(-1), features point-in-time),
  walk-forward / rolling-window / TimeSeriesSplit, forward/paper testing
  (OOS log) — these are in place.

**Recommended priority (only the few that decide credibility):**
1. **Deflated Sharpe + PBO** — quantify how much of 2.09 / 1.92 is
   selection overfit. Highest priority — it audits every conclusion here.
2. **Survivorship-bias fix** — point-in-time constituents incl.
   delistings (hard with yfinance; may be a known-limitation note).
3. **Purged K-Fold + embargo** — stricter leakage-safe validation.
   (permutation importance + SHAP = Operations #8, explanation layer,
   lower priority than correctness above.)

**Other metric gaps (nice-to-have, low priority):** Sortino / Calmar /
Information Ratio, quantile-decile analysis, classification metrics
(AUC/F1 — clf exists but unreported), regression metrics (RMSE/R²),
bootstrap / Monte Carlo, parameter-stability. Mostly reporting niceties
for a 31-name monthly strategy, not correctness issues.

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

   **⚠️ Major finding + same-day correction (random-score control):**
   Full-sample RankIC is low (+0.029, ICIR 0.17) and regime-dependent
   (2023 +0.065 / 2026 +0.120 good; 2024 +0.010 / 2025 +0.002 ≈ zero;
   per-sector ICIR all < 0.1). That *looked* like "the edge is sector/
   regime, not stock-picking" — and an earlier version of this note said
   so. **A control experiment refuted that.** Replacing model scores with
   random ones (same sector band, regime, universe —
   `walk_forward(score_mode="random")`) collapses excess return from
   +12.1% to **+1.5%** and Sharpe from 1.92 to **1.32**. So stock
   selection contributes ~**+10.6pp excess and +0.60 Sharpe — it IS the
   main alpha source**, not sector tilt.

   Why is RankIC low yet selection so valuable? The strategy only buys
   the **head** (top-1/2 per sector). The model's alpha is concentrated
   in picking the best few names; its ordering of the mid/bottom is
   noisy. Full-cross-section RankIC is therefore a systematically
   pessimistic gauge for a buy-the-top-k strategy. **Methodology lesson:
   don't judge selection value by RankIC alone — run the random-score
   control.**

   Reconciling with the failed feature experiments: selection works, but
   the feature space is saturated (VIF 6 severe; LGB/betas/vol-PCA/12-1
   all regressed). Current features already extract the head alpha;
   adding more can't squeeze out more. **Live implication:** 2025 still
   lagged (excess −4.2%, IC ≈ 0) — regime is a real return-variance risk,
   but that's volatility, not selection failure.
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

### Diagnosed weaknesses & proposed fixes (Step 6 + random-score control, 2026-05-20)

**Corrected reading after the random-score control:** stock selection
IS the main alpha (~**+10.6pp excess, +0.60 Sharpe** vs random scores);
the low RankIC 0.029 just reflects that the edge lives in the **head**
(top-1/2 per sector), which full-cross-section RankIC under-measures. So
the fixes are NOT "abandon selection" — they're "selection works, but the
feature space is saturated (VIF + 4 failed add-feature experiments), so
improve it by changing the data / objective / risk overlay, not by adding
features." All unverified — A/B against Sharpe 1.92 before adopting.

| # | Weakness (evidence) | Proposed fix | Risk |
|---|---|---|---|
| 1 | Feature space saturated — more features can't extract more head-alpha (VIF 6 severe; LGB/β/vol-PCA/12-1 all regressed) | Change the *objective*, not the features: train on top-quintile membership or a pairwise/rank loss so the model optimizes head precision (where the alpha is) rather than full-cross-section ranking. | Medium; reshapes the target |
| 2 | Financials (7 banks) internal IC ≈ 0 (+0.005) | Big banks too homogeneous to rank — hold a bank basket (top-2 EW or ZEB/XFN) and steer selection effort to sectors with signal (Industrials +0.042). | Low; constraint/sizing |
| 3 (root) | Only ~8 names/sector — little room for head-selection | Widen universe (31 → ~95/223, `expand_universe.py`). Now well-motivated: selection works, more candidates = more good names to surface. | Medium; focused universe was more defensive historically — A/B |
| 4 | Regime risk: 2025 excess −4.2%, IC ≈ 0 | Light regime de-risking — scale DOWN exposure when signal is weak, rather than retreat to XIU.TO (retreating to the index would forfeit the selection alpha that does work). | Low-med; sizing overlay |
| 5 | ~~Turnover 54%/mo~~ | ~~Raise rank_buffer/hold_bonus or go quarterly~~ | Removed — not tracked |

**On #2, confirmed by direct scoring (2026-05-21):** predicting the
latest month, the model gave all 7 banks the *identical* score (0.542,
spread 0.000) — XGBoost lands these highly-correlated names in a single
leaf, so within Financials it isn't really selecting at all; which 2
banks survive is decided by tie-break, not skill. Stronger, more concrete evidence than the IC≈0.005
average — supports holding a bank basket / ETF instead of ranking banks.

**Dropped:** the earlier "re-target to sector rotation" / "ETF-fallback
to XIU.TO" plan. The control experiment shows selection is the alpha, so
abandoning it or retreating to the index would discard the edge. Regime
handling should reduce *size*, not switch off selection.

**Update (2026-05-20): #3 was tried and rejected** (see Tried and
Rejected — both full and selective widening underperformed 1.92). With
that, **every avenue for lifting selection has now been tested and
failed**: 4 add-feature experiments (LGB / 12-1 / rolling-β / vol-PCA)
and 2 universe-widening variants. **Sharpe 1.92 on the focused 31-name
universe is a hard ceiling for this design + data.** Selection is real
(+10.6pp vs random) but already maxed out by the current features on the
current universe.

Remaining options are no longer "tweaks" — they are genuinely different
projects: (1) a different/deeper data source (richer fundamentals, the
PIT path yfinance can't support); (2) a different objective (rank/
top-quintile loss — #1 in the table, untested); (3) regime sizing (#4)
to cut the 2025-type drawdown without touching selection; or (4) just
**accept 1.82 Sharpe as a finished, validated result** and run it.

**Decision (2026-05-23): Accept 1.82 Sharpe and move to pure operations.**
Every experiment (LGB, 12-1, betas, universe widening, bank basket,
feature pruning) regressed — the model is at a local optimum on this
31-name / 4-sector / yfinance-only design. The 1.82 Sharpe has
DSR 99.3% (not selection noise). Most valuable next step: accumulate
a real forward OOS record via `picks_log.csv`, not more tuning.

**Update (2026-05-24): vol_ratio removal improved Sharpe 1.81 → 1.99.**
OOS permutation IC for vol_ratio was -0.0153 (strongest negative feature).
Removing it freed the vol_60d signal (+0.0046→+0.0164) and mom_pc2 (-0.0051→+0.0122).
Hit rate 57%→66%, IR 0.81→0.97, Vol 13.7%→12.8%. DSR still STRONG (96.7%). Accepted.

### Operations / production readiness (2026-05-20)

Direction chosen once model tuning hit the 1.92 ceiling: make the
validated model produce reliable, **executable** monthly output rather
than chase more Sharpe.

1. ✅ **Actionable monthly report** — done. `pick` prints/emails a
   SELL/BUY/HOLD diff vs `CURRENT_HOLDINGS` (`diff_holdings`) plus a
   one-line signal-reliability verdict from health_check
   (`_health_summary`), not just a picks list. stdout and email share
   one body (`_format_report`). Live run from the 7-bank holdings:
   SELL 5 banks, BUY 5 (energy/industrials/utilities), HOLD 2 — the
   sector diversification the diagnostics implied.
2. ✅ **Convenience run script** — `run_monthly.sh` (executable) does
   cd + venv-activate + `python picker.py pick` + append to
   `logs/monthly.log`, all in one command. User opted for **manual**
   monthly runs for now (no launchd/cron) — just run `./run_monthly.sh`.
   Automating later is a one-step add: a launchd plist calling this
   script (StartCalendarInterval, monthly).
3. ✅ **Data robustness** — done 2026-05-20. `check_data_health` runs
   right after the download and reports download rate, benchmark
   presence, and per-required-sector coverage (`print_health_check` now
   takes a `title`). main() aborts the run on hard failures (download
   rate < 85% or missing XIU.TO benchmark) instead of silently producing
   picks off partial yfinance data. Verified: a healthy 31/31 run passes
   the gate and is unaffected (Sharpe 1.92).
4. ✅ **OOS track record** — done 2026-05-20. `pick` logs each month's
   picks + weights + a XIU benchmark row to `picks_log.csv` (gitignored,
   idempotent per month); `backfill_realized` fills each pick's
   next-month return from prices once it matures; `oos_track_record`
   reports realized portfolio vs XIU since logging began, shown in the
   report/email. **This is the one thing that genuinely adds information
   from here:** the 1.92 is walk-forward (historical); this accumulates a
   real forward out-of-sample record to confirm — or not — that the
   backtest holds up live. First run shows 1 mo (+2.0% vs XIU +0.8%);
   it grows one month per run, judge after ~6+.
5. ✅ **Regression tests for the new ops functions** — **done
   2026-05-21**. Added `unit_checks()` as section `[0]` of
   `smoke_test.py`: 12 fast, no-download assertions over `diff_holdings`
   (SELL/BUY/HOLD split), `_health_summary` (verdict wording),
   `compute_rank_deltas` (↑/↓/→/NEW + no-history), `log_picks` /
   `oos_track_record` (XIU row, per-month idempotency, portfolio-vs-XIU
   math), and `check_data_health` (benchmark present/missing). Runs
   first so a broken ops function fails in seconds before the ~30s
   end-to-end pipeline. Guards the monthly run against a silent
   regression shipping a bad report. Engineering completeness: done.
6. ⬜ **Richer OOS performance metrics (deferred — needs data)** — once
   `picks_log.csv` has accumulated **~6+ months of realized (matured)**
   rows, extend `oos_track_record` from the current cum-return / win-rate
   summary to also report **OOS Sharpe, max drawdown, and rolling
   IC/ICIR** — the same metrics the backtest shows, but on real
   out-of-sample months, so the live record is comparable apples-to-
   apples with the 1.92 backtest. **Not now:** with ~1 month logged these
   are pure noise. Trigger: revisit after ~6 monthly runs (check distinct
   `as_of` months in `picks_log.csv`).
7. ✅ **Usage-driven UX refinements (2026-05-21)** — landed as the user
   started running it monthly:
   - **Actionable report for any holdings state**: empty CURRENT_HOLDINGS
     now prints an explicit `ACTIONS (initial build — no current
     holdings): BUY all (N)` list (not just a Target portfolio).
   - **`AUTO_ROLL_HOLDINGS`**: when CURRENT_HOLDINGS is empty, seed
     current holdings from last run's picks (`last_logged_picks`) so the
     portfolio rolls month to month with no hand-editing — for a
     follow-the-model workflow. CURRENT_HOLDINGS still overrides.
   - **Named drift in the reliability line**: `_health_summary` shows the
     failed check's detail, e.g. `Feature drift: rev_growth_yoy(z=7.6)`
     instead of an opaque `(Feature drift)`.
   - **Last-6-months table in `print_backtest`**: per-month port/bench/
     excess after the yearly table, so recent month-to-month behavior is
     visible (pseudo-OOS — params tuned over all history, reads
     optimistic vs true forward OOS).
   All covered by `smoke_test` `[0]` unit assertions where applicable.
8. ✅ **Permutation importance — landed 2026-05-23**. `_perm_importance_fold`
   computes OOS RankIC drop per feature on each walk-forward test fold;
   `walk_forward(return_importance=True)` accumulates and averages across
   folds; `print_permutation_importance` displays the ranked table;
   `save_perm_importance`/`load_perm_importance` cache to
   `perm_importance.json` so `pick` uses OOS importance instead of
   train-set gain. New mode: `python picker.py importance`.
   **Bug fixes (2026-05-23):**
   - `sector_code` excluded from permutation — it is only a routing key in
     `predict_sector_models` (mask by code → correct model), not a model
     feature; shuffling it destroys routing, not signal.
   - Macro/time-series features (`vix_level`, `rate_chg_3m`, `oil_mom_1m`,
     etc.) now skipped when `df_eval[feat].nunique() <= 1` within a fold:
     all stocks share the same macro value in any single month, so
     cross-sectional permutation measures nothing. These features operate
     on the time-series axis (which months to be defensive), not the
     cross-sectional axis (which stock to buy), and can't be measured by
     IC permutation. Display filtered to `importance > 0` only.
   **True feature contributions (post-fix):** `high_52w_ratio` (+0.0085),
   `mom_pc2` (+0.0058), `vol_60d` (+0.0051), `adv_20d_rank` (+0.0035),
   `mom_pc1` (+0.0001). Consistent with VIF findings: alpha from head
   selection via a few technical cross-sectional signals.

9. ✅ **SHAP per-pick drivers — landed 2026-05-23**. `_compute_shap_for_models`
   uses `shap.TreeExplainer` on the ExtraTrees ensemble (50% regressor +
   50% classifier, blended); `compute_shap_current` / `python picker.py shap`
   prints the global SHAP table for the current month. `predict_now` now
   returns `shap_by_ticker` (8th return value); the `pick` report and email
   show per-stock top-3 SHAP drivers with sign. New mode: `python picker.py shap`.

10. ✅ **HTML email with visualizations — landed 2026-05-23**. `build_report_html`
    generates a card-based HTML report: regime + reliability badges (color-coded
    green/amber/red), SELL/BUY/HOLD action table with color tags, portfolio table
    with per-stock weight bar, score, action tag, and SHAP drivers (↑green/↓red),
    feature importance bar chart, OOS track record box. `send_report_email` now
    sends `MIMEMultipart("alternative")` with both HTML and plain-text fallback.
    Feature names use human-readable labels (e.g. `high_52w_ratio` → "52-Week
    High Proximity", `rsi_14` → "RSI (14-day)") in both email and SHAP drivers.

11. ✅ **Remove turnover and transaction-cost logic — landed 2026-05-24**. Deleted
    all cost/turnover code: `hold_bonus`, `rank_buffer`, `score_tolerance`,
    `max_turnover`, `cooldown_months`, `tx_cost_bps`, `cost_bps`, `tcost`,
    `port_ret_net`, pick-turnover reporting in `evaluate_segments`, and the
    `has_cost` print block in `print_backtest`. `_wf_metrics` simplified to
    3 params; `run_sensitivity` now sweeps only `embargo_months`. The
    sector min-1/max-2 rebalancing band (`apply_rebalancing_band`) is kept —
    that is portfolio construction, not turnover control.

### 实验接受标准（2026-05-24 起）

**主指标：IR ≥ 1.06**（基线 1.08 − 0.02）。Sharpe 作为副指标参考，不单独作为接受/拒绝依据。DSR 须保持 MODERATE（≥ 80%）以上。

### Tried and Accepted

- **half_life=6m (Experiment C, 2026-05-24)**: Half-life sweep over {6,9,12,18,24} months.
  half_life=6m was best (Δ Sharpe +0.112 vs baseline 12m in the sweep). Full backtest
  with half_life=6m: **Sharpe 1.99 → 2.13**, Ann. +25.4% → +27.0%, Max DD −7.7% → −8.3%,
  Hit Rate 66% → 68.1%, IR 0.97 → 1.08. Quintile monotonicity: No → **Yes ✓** (first time).
  DSR STRONG (95.6%). `walk_forward(half_life=6)` is now the default. The shorter half-life
  upweights recent data more aggressively — ExtraTrees with smaller train windows benefits
  from recency emphasis. Max DD traded off slightly (−0.6pp) but Sharpe gain (+0.14) justified.

- **vol_ratio removal (2026-05-24)**: OOS permutation IC was -0.0153 (strongest negative
  feature — shuffling it *improved* OOS ranking). `vol_ratio = vol_20d / vol_60d` was redundant
  given both components are present, and was actively suppressing `vol_60d`'s signal.
  Removed from `_BASE_SECTOR_FEATURES` and `FEATURE_COLS`. Result: **Sharpe 1.81 → 1.99**,
  Hit Rate 57% → 66%, IR 0.81 → 0.97, Vol 13.7% → 12.8%. DSR STRONG (96.7%). WRC 99.7%.
  `vol_60d` importance: +0.0046 → +0.0164; `mom_pc2`: -0.0051 → +0.0122.

### Adopted (2026-05-24)

- **train_months=30（rolling window 缩短）**: trainwindow 扫描（half_life=6）发现 30m 在所有窗口中 IR 最高（24m:1.34，**30m:1.55**，36m:0.89，42m:1.07，48m:1.29）。完整 backtest：IR **0.92→1.55**，Sharpe 2.04→2.21，DSR MODERATE→**STRONG**（97.6%），PBO 6.8%→**2.4%**，逐年全面改善（2022 excess +5.1% vs -1.0%，2023 +30.8% vs +19.0%）。根因：half_life=6m 下，>30 个月前的数据权重 <3%，添加这些低权重月份只增加噪声。默认值已从 36 改为 30，并入 main。

### Tried and Rejected

- **多种子集成 — Multi-Seed Ensemble（2026-05-24）**: 对 ExtraTrees Regressor + Classifier 各训练 3 个种子（42, 123, 456），预测时取平均，以减少极端随机切割带来的方差。结果：IR **1.08→0.79**��Sharpe 2.13→1.90，2022 年 excess **+1.1%→-7.0%**（显著恶化）。平均多个种子的 rank-normalized 分数稀释了信号：seed=42 是已优化局部最优的结构成分，其他种子的分布与之发散，平均后信噪比下降。ranking IC 略有提升（Spearman 0.034→0.045），但组合级别 IR 大幅下降，说明不同种子选出了不同的股票但质量更差。已丢弃（未并入 main）。

- **Integration Plan #4 — PIT 年报基��面（2026-05-24）**: 两个子实验，全部拒绝。
  根因诊断：`roe`/`pe_ratio`/`div_yield`/`debt_equity` 在 panel 里一直是全 NaN（`fetch_fundamentals` 从未被 `build_panel` 调用），ExtraTrees 静默忽略这些列的内容（sklearn 1.6.1 接受全 NaN 列，importance=0），但这些列的**存在**影响 `max_features=0.7` 的随机采样分布，是模型局部最优的结构组成部分。
  - **子实验 A — PIT 年报 ROE + asset_growth_yoy**: 新增 `fetch_annual_fundamentals` + `compute_pit_annual_features`，用 5 年年报 BS 计算逐股 roe_annual 替换 `roe`，新增 `asset_growth_yoy`。结果：IR **1.08→0.60**，Sharpe 2.13→1.79，hit rate 68.1%→53.2%，quintile 单调性 Yes→No。`roe_annual` importance 升至 +0.0036（确实被使用），但把"板块时序信号"（稳定噪声）替换为"逐股截面信号"，打乱了所有依赖 `roe` 的分裂节点。最大跌幅实验之一。
  - **子实验 B — snapshot 合并进 panel**: 将 `fetch_fundamentals` snapshot 通过 merge 写入 panel，每只股票在所有月份持有恒定的当前快照值（轻微 look-ahead）。结果：IR **1.08→0.92**（低于阈值 1.06），Sharpe 2.13→2.04，hit rate 68.1%→57.4%。把全 NaN（ExtraTrees 忽略）变成恒定逐股值（改变 max_features 采样）同样破坏局部最优。
  - **结论（2026-05-24）**：这些 NaN 占位符既不能删（Sharpe 1.92→1.86，2026-05-21 feature pruning），也不能赋真实值（两种方式均回归）。模型已在这些占位符存在的前提下收敛，无法通过基本面路径改进。

- **Experiment D — bb_zscore removal (2026-05-24)**: OOS permutation IC was −0.0035 under
  half_life=6m baseline. Removed from `_BASE_SECTOR_FEATURES` (kept in FEATURE_COLS).
  Result: Sharpe 2.13 → 1.92, Ann. +27.0% → +23.7%, 2025 excess +0.3% → **−8.5%** (large),
  DSR STRONG → **MODERATE** (95.6%→94.1%), PBO 4.4%→5.9%, Quintile mono Yes→No. Side-effect:
  mom_pc1 IC dropped from −0.0001 to −0.0070, revealing that bb_zscore suppresses spurious
  mom_pc1 signals. Pattern: same stabilizer role as vol_20d and rev_growth_yoy. Reverted.
  **4/4 negative-IC feature removals have failed** — feature set is a fragile local optimum.

- **Experiment B — rev_growth_yoy removal from Industrials (2026-05-24)**: OOS IC was 0.0000
  across 47 months, and yfinance coverage is ~5% for Industrials names. Removing it from
  `SECTOR_FEATURES["Industrials"]` (keeping in FEATURE_COLS) regressed: Sharpe 1.99 → 1.92
  (below threshold 1.97), Sortino 3.37 → 3.26. Side-effect: vol_20d IC dropped from −0.0054
  to −0.0089, suggesting rev_growth_yoy suppresses spurious vol_20d signals in Industrials
  despite its near-zero IC. Reverted.

- **Experiment A — vol_20d removal (2026-05-24)**: OOS permutation IC was −0.0054 (negative —
  shuffling slightly improved ranking). Removed from `_BASE_SECTOR_FEATURES` and `FEATURE_COLS`.
  Result: Sharpe 1.99 → 1.92 (below threshold 1.97). Max DD improved (−7.7% → −6.5%), Sortino
  improved, but Sharpe missed. Side-effect: `div_growth_yoy` IC became −0.0118 (strongly negative),
  suggesting vol_20d stabilizes the model by suppressing spurious div_growth_yoy signals. Reverted.

- **Hyperparameter OAT sweep (2026-05-24)**: Tested `n_estimators` ∈ {100,200,300,500},
  `max_depth` ∈ {3,4,5,6,None}, `min_samples_leaf` ∈ {5,10,15,20}, `max_features` ∈
  {0.5,0.6,0.7,0.8} one-at-a-time vs default (300/5/10/0.7). OAT winners: depth=6
  (+0.21), leaf=20 (+0.19), feat=0.6 (+0.14). Combined all three → net Sharpe
  **1.82→1.74**, DSR **STRONG→MODERATE** (97.5%→93.0%), PBO 2.5%→7.0%, 2025 excess
  +0.6%→−8.3%. OAT improvements do NOT combine: the three params have interaction
  effects and (5,10,0.7) is a stable local optimum. Reverted. **Lesson: for this
  sample size, OAT is unreliable — individual improvements cancel when combined.**

- **Bank basket — ZEB.TO replacing individual banks (2026-05-23)**: Added
  ZEB.TO (BMO Equal Weight Banks ETF) to TSX_UNIVERSE as `bank_etf` sub_type;
  modified `apply_rebalancing_band` to collapse all 7 individual "bank"
  sub_type tickers into ZEB.TO when present. Motivation: all 7 banks score
  identically (0.542, spread=0), so the Financials bank slots are decided by
  tie-break, not skill.
  Result: Sharpe **1.82 → 1.74 net** (−0.08), Max DD −7.1% → −8.1%, DSR
  99.3% → 94.9%. Hit rate improved 59.6% → 63.8% (sole bright spot). Same
  regression pattern as all prior experiments. Reverted — 31-name universe is
  the hard ceiling. Interesting side-finding: with ZEB.TO, permutation
  importance became more meaningful (adv_20d_rank +0.016, vol_60d +0.014,
  high_52w_ratio +0.012, sector_code +0.010, mom_pc2 +0.009, rev_1m +0.008,
  mom_pc1 +0.006 all show real IC drops vs everything ≈0 before), suggesting
  ZEB.TO freed up feature discrimination in non-Financials sectors.

- **Feature pruning — dead features (2026-05-21)**: Removed the per-sector models' apparently-useless inputs — `sector_code` (a *constant column* inside a per-sector model, so importance is necessarily 0) and the sector-median-imputed fundamentals (`roe`/`pe_ratio`/`div_yield`/`debt_equity`, plus `rev_growth_yoy` at 2.9% coverage). Per-sector counts dropped 22/22/18/21 → 17/17/13/16. Expected Sharpe to hold flat (these carry ~0 importance); instead it **regressed**: Sharpe 1.92 → 1.86, annualized +28.0% → +26.8%, hit rate 66.0% → 59.6% (drawdown slightly better −7.6% → −7.2%; RankIC ~unchanged at 0.027). Likely `colsample_bytree=0.7` changes which columns each tree samples when the feature count changes, so even dropping zero-importance columns perturbs the model. **Strongest evidence yet that this feature set is a fragile local optimum** — adding (LGB/12-1/β/vol-PCA), compressing (vol-PCA), and now even *removing dead features* all regress. Answers the "too many features?" question definitively: it's not about the count — the set is a tuned optimum that doesn't tolerate changes at this sample size. Reverted. (Aside: `reg.feature_importances_` read 0 across the board in a standalone diagnostic — a feature_importances_ extraction quirk, not a model problem; the model clearly works given Sharpe 1.92.)

- **Universe widening — fix #3 (2026-05-20)**: With selection shown to be real alpha (random-score control: +10.6pp) and the feature space saturated, tried giving the model more candidates in the 4 required sectors. Two variants, both worse than the focused 31-name universe (Sharpe 1.92): (a) full widen to 53 names (Financials 12→16, Energy 8→15, Industrials 6→13, Utilities 4→8) → Sharpe **1.40**, excess +7.2%, hit 51%; (b) selective widen to 42 (only Energy + Utilities — the sectors whose per-sector RankIC looked better under (a)) → Sharpe **1.57**, excess +10.0%, but the worst drawdown −10.7% and RankIC 0.005. The focused universe is genuinely the better config (confirms the earlier "more defensive" note). One upside seen under (a): returns spread more evenly across years (2025 excess −4.2% → +10.0%), i.e. less regime-dependent — but not worth the Sharpe hit. **Methodology lesson:** per-sector RankIC is unstable and universe-composition-dependent (Energy IC went +0.039 under full widen → −0.009 under selective widen, same names), so a single backtest's per-sector RankIC is NOT a reliable basis for decisions — the "selectively widen Energy/Utilities" call was built on that noise. Reverted to 31.

- **Volatility-family PCA (2026-05-20)**: VIF flagged vol_20d/vol_60d/vol_ratio as severely collinear (41/29/27), so — by analogy with the successful momentum PCA — compressed them into a single `vol_pc1` (`apply_vol_pca`, `USE_VOL_PCA` flag). Regressed all five metrics: Sharpe 1.92 → 1.66, annualized +28.0% → +23.9%, excess +12.1% → +8.1%, max drawdown −7.6% → −8.2%, hit rate 66.0% → 53.2%. **Lesson (corrects the VIF write-up's first instinct):** high VIF does not justify compression for a *tree* model. Collinearity inflates variance in *linear* estimators; XGBoost splits are unaffected by it, and the short/long vol distinction plus vol_ratio's term-structure carry real signal that one PC discards. The momentum PCA worked for different reasons (4 horizons far more redundant; model was at Sharpe 0.83 with room to gain), not a general "compress high-VIF families" rule. Reverted.

- **Rolling 24m betas — Step 3 (2026-05-20)**: Added three per-ticker, strictly past-only rolling betas (`equity_beta` vs ^GSPTSE, `sector_beta` vs the sector ETF, `cad_beta` vs CADUSD=X) via `compute_rolling_betas`, wired into `_BASE_SECTOR_FEATURES`. The implementation was correct — betas computed with right-aligned `rolling(24)` so each (ticker, month) value uses only data ≤ that month (avoiding the full-panel look-ahead the companion scripts have), validated by equity_beta mean ≈0.98 / sector_beta ≈0.88. But it regressed **all five metrics** vs the rev_1m baseline: Sharpe 1.92 → 1.83, annualized +28.0% → +26.4%, excess +12.1% → +10.6%, max drawdown −7.6% → −8.8%, hit rate 66.0% → 57.4%. Same lesson as the spec-coverage and LGB experiments — at ~31 tickers × 84 months across 4 sector models, adding 3 features × 4 models (one of them, `cad_beta`, very noisy at range [−6, +10]) overfits faster than the betas inform. A feature-budget problem, not a bug. Reverted entirely.

- **LGB ensemble — Step 2 (2026-05-20)**: Added LightGBM as a second tree model, blending XGB+LGB 50/50 on rank-normalized predictions per sector (`fit_sector_ensemble`/`predict_sector_ensemble`). Also tried an ElasticNet feature gate inside the blend, which dropped features XGB found useful (Sharpe 1.91 → 1.52), so EN was backed out and the blend reduced to 2-model. Even the clean 2-model blend regressed everything vs the single-XGB-per-sector baseline (same 47-month walk-forward): Sharpe **1.92 → 1.62**, annualized +28.0% → +22.9%, excess +12.1% → +7.1%, hit rate 63.8% → 55.3% (only max drawdown improved marginally, −8.2% → −7.1%). At ~31 tickers × 84 months split across 4 sector models, a correlated second tree learner adds variance faster than it diversifies error. Reverted entirely — `picker.py` keeps the single XGBoost regressor per sector. The companion `monthly_rank.py` LGB+XGB+EN ensemble does not port over at this sample size.

- **PIT fundamentals (2026-05-16)**: Wiring `compute_pit_fundamentals` into `build_panel` regressed the backtest by ~3pp annualized (Sharpe 0.83 → 0.76). yfinance's 5-8-quarter limit means PIT data only covers the last ~12-15 months of each training window; the resulting mixed-coverage signal trained worse than uniform sector-median imputation. The function is preserved in `picker.py` for future use with a deeper fundamentals source.

- **Spec-coverage feature additions (2026-05-17)**: Added yield-curve slope (`^TYX`/`^IRX`), credit-spread proxy (`HYG`/`LQD`), refining-margin (`RB=F` - `CL=F`), `^GSPC`/`^IXIC`/`XLK` equity-beta proxies, and per-ticker P/B to close gaps in the per-sector spec coverage. Regressed Sharpe from 1.65 to 1.56 (full additions) and 1.54 (selective keep that dropped only the 1-stock-only additions). Reverted — at the current sample size (~31 tickers × 84 months split across 4 sector models), more features add noise faster than signal. The tickers stay commented in `MACRO_TICKERS` as a record so the experiment isn't accidentally redone.

- **Macro / rate features — assessed, not expanding (2026-05-21)**: Asked whether to add macro/rate signals (interest rates etc.). The panel already carries 13 macro features incl. US 10Y (`rate_chg_3m`), the REAL BoC overnight rate via Valet API (`boc_rate_chg_3m`), a Canadian bond ETF (`cad_bond_mom_1m`), and inflation (`tips_mom_1m`) — rates are well covered. Decided NOT to add more, for a structural reason beyond the spec-coverage regression above: **a macro value is identical across all stocks within a month**, so it cannot discriminate *which* stock outperforms (a cross-sectional question) — it only moves timing / regime (time-series). The model's edge is cross-sectional selection (+10.6pp vs the random-score control), so extra macro is just cross-sectional noise — exactly why spec-coverage (incl. the yield-curve slope) regressed. Macro already contributes where it legitimately can: indirectly via `sector_code` splits (rates→banks/utilities, oil→energy), per the per-sector spec. Deeper "real" macro (BoC 2y/10y curve, FRED CPI, IG OAS) would hit the same cross-sectional wall.

### 2026-06-01 — intraday cron not firing: diagnosed + offset fix

Picked up the 2026-05-31 intraday cloud work (logged below). Found the
core feature was effectively dead: the `intraday snapshot` workflow was
`active` but had **only one run ever** (a manual `workflow_dispatch` on
05-31). Across a full trading day (Mon 06-01, inside the `13-21 UTC`
window), **zero scheduled runs fired** (`gh run list -e schedule` empty).

Ruled out the usual suspects: repo is public, **not a fork** (forks
disable schedules), Actions enabled (`all`), workflow `active`, only 16
days old (not the 60-day-inactivity auto-disable), YAML valid. Manual
dispatch (`gh workflow run intraday.yml`) succeeded end-to-end —
regenerated `docs/intraday.html` and committed a fresh snapshot — so the
**pipeline is fine; the failure is purely GitHub's scheduler**.

Most-likely cause: the cron hit the **top-of-hour `:00` boundary**, the
busiest tick, which GitHub explicitly warns is the most likely to be
delayed/dropped — and `*/5` fires on `:00` every hour.

**Attempted fix (insufficient):** offset cron `*/5` → `2/5` (fires
`:02,…,:57`, dodging `:00`) + re-push to nudge re-registration
(`ea203c4`). Monitored another ~20 min across several `2/5` ticks —
**still zero scheduled runs.** So it's not the `:00` boundary; GitHub's
scheduler simply isn't running this repo's cron at all (~30 h, zero
fires), while `workflow_dispatch` works every time.

**Resolution — external pinger (chosen 2026-06-01).** Validated the
exact dispatch call the pinger uses: `POST
…/actions/workflows/intraday.yml/dispatches {"ref":"main"}` → **HTTP
204**, run queued instantly. Reliable near-real-time updates now come
from a free external scheduler (cron-job.org) POSTing that endpoint on a
real 5-min timer — no Mac needed. Full setup (token scope, headers,
body, verify) in **`INTRADAY_PINGER.md`**.

- The in-repo `2/5` cron stays as a harmless free backup; the
  market-closed freeze (`a2519c6`) prevents duplicate commits if both
  ever fire. User completes the cron-job.org + fine-grained-PAT setup
  (the one manual, browser-only step).
- Side note: Actions annotation warns `actions/checkout@v4` +
  `setup-python@v5` run on **Node 20, force-migrated to Node 24 on
  2026-06-16**. v4/v5 are already latest; the runtime swap is GitHub-side
  and should be transparent. Flagged, not changed.

### 2026-05-31 — intraday cloud auto-sync (was undocumented until 06-01)

Built a no-Mac-needed near-real-time view: GitHub Actions reruns the
intraday monitor on a schedule, regenerates `docs/intraday.html`, and
commits it, so opening GitHub Pages on a phone shows ~5-min-granularity
quotes. Reconstructed here from the commit log.

- **`intraday_monitor.py` + `intraday.yml`** (`1bb804d`): `--once` mode,
  resilient picker import with a cloud fallback, SVG line charts with
  turning-point markers; workflow installs only lightweight deps
  (yfinance/pandas/numpy/scipy), regenerates + commits the page.
- **Cloud picks sourcing**: pick-logs are gitignored (private), so the
  cloud run can't read them. Two-step resolution: derive the ticker list
  from the already-public `docs/data.json` + `docs/data_us.json`
  (`a67f229`), then later persist the resolved picks to a dedicated
  public `docs/intraday_picks.json` on local runs so the cloud reads the
  full 14-name CA+US list decoupled from stale `data.json` (`806f06d`).
- **Market-closed freeze** (`a2519c6`): `--once` skips the rewrite unless
  a ticker's latest bar is today (ET), so weekends/holidays/pre-market
  keep the last session's static page and the Action produces no commit
  churn. (This is why the 05-31 16:14 manual run committed nothing — it
  was a Sunday.)
- **Mobile snapshot page** (`ad57090`) + `company_names.json` for name
  lookup in lean cloud runs.
- **Housekeeping (same day):** added `test_picker.py` — 24 plain-assert
  tests (no pytest dep) over the deterministic core (config integrity,
  RSI bounds, normalize, sector encoding, rebalancing-band +
  concentration constraints, DML theta recovery Y=1.5D→1.49), wired into
  `run.sh test` (`9920d49`). Removed orphaned `fetch_annual_fundamentals`
  (−47 lines, dead after the PIT-annual removal) and completed
  `requirements.txt` (added `pyarrow` for the cache parquet engine and
  `shap` for per-pick drivers — both used but unlisted) (`f9e7555`).

### 2026-05-22 — Quantitative rigor sprint

Full rigor audit and metric expansion. All changes are additive (no model/Sharpe change); the canonical backtest is now:
**ExtraTrees + DML, embargo=1m, walk-forward 47 months (2022-05 → 2026-03).**

**Model switch: XGBoost → ExtraTrees**
- 9-model comparison confirmed ExtraTrees as best (Sharpe 2.12 gross, no embargo).
- `MODEL_KIND = "extratrees"` in picker.py; banner updated dynamically.

**DSR / PBO audit (implemented in `print_overfit_report`):**
- Functions: `compute_psr`, `compute_dsr`, `compute_sharpe_ci`.
- n_trials=15 (9 models + 6 feature variants this project tested).
- Result: DSR **97.6%**, PBO proxy **2.4%**, bootstrap 95% CI **[1.05, 2.76]**.
- Verdict: STRONG — the Sharpe is not selection noise.

**Embargo (1 month, default on):**
- `walk_forward(embargo_months=1)` excludes the last training month before test
  to prevent label overlap. Sharpe 2.12 → **1.93 gross** (the 0.19 gap = leakage
  that was real; now corrected). Max drawdown improved −8.2% → −7.1%.

**Expanded backtest metrics (all in `print_backtest`):**

| Metric | Value |
|--------|-------|
| Sharpe | **1.86** |
| Sortino | **3.17** |
| Calmar | **3.22** |
| Max Drawdown | **−7.5%** |
| Ann. Volatility | **13.0%** |
| Tracking Error | **10.0%** |
| Information Ratio | **0.84** |
| Hit Rate | **61.7%** |
| Profit Factor | **1.71** |
| Expectancy | **+0.59%/mo** |
| Beta (vs XIU) | **0.70** (defensive) |
| Treynor Ratio | **0.35** |
| Ann. Return | **+24.3%** |
| Excess | **+8.4%/yr** |

**Prediction quality (`evaluate_prediction_quality` — new function):**
Cross-sectional per-month metrics averaged over 47 test months.
Binary label = top-quintile fwd_ret (≥ 80th pct); predicted = above-median score.

| Metric | Value | Notes |
|--------|-------|-------|
| AUC-ROC | **0.554 ±0.154** | +0.054 lift vs random |
| AUC-PR | 0.311 ±0.116 | vs ~0.20 random baseline |
| Brier Score | 0.318 ±0.054 | lower = better |
| MCC | 0.052 ±0.213 | near-zero, positive |
| F1 | 0.321 ±0.117 | |
| Precision | 0.221 ±0.082 | |
| Recall | **0.585 ±0.212** | catches 58% of top-quintile |
| Balanced Accuracy | 0.532 ±0.133 | |
| Directional Accuracy | **0.507 ±0.089** | ≈ random |

**Key interpretation:** Directional accuracy ≈ random and cross-sectional IC = 0.038
are NOT contradictions — they confirm the earlier random-score control finding.
The model's alpha is concentrated in **head selection** (which 1-2 names per sector
to buy), not in predicting every stock's return direction. Full-cross-section metrics
systematically understate the strategy's value.

**Remaining open items:**
- ✅ **Permutation importance** (Operations #8) — done 2026-05-23; sector_code + macro features fixed 2026-05-23.
- ✅ **CPCV / WRC / FDR** — done 2026-05-23 (see rigor sprint below).
- ✅ **SHAP** (Operations #9) — done 2026-05-23.
- ✅ **HTML email** (Operations #10) — done 2026-05-23.
- ⚠️ **Survivorship bias** — documented as known limitation 2026-05-23. yfinance-only constraint; irremediable without paid PIT data.

---

## Quantitative Rigor Checklist

Complete reference for ML-finance validation. Status: ✅ implemented · ⚠️ partial · ⬜ planned / deferred · ❌ known gap · N/A not applicable to this strategy.

### 时间序列验证

| 方法 | 状态 | 说明 |
|------|------|------|
| Walk-forward validation | ✅ | `walk_forward()`, 36m train rolling window, 47 test months |
| Rolling window validation | ✅ | 同上（固定长度滚动训练窗口） |
| Expanding window validation | ✅ | 已实现 (2026-05-24)；`walk_forward(expanding=True)` 从 panel 起点扩展训练集，min_train 保持最小 fold 大小约束 |
| Purged K-Fold cross-validation | ❌ | 已知缺口；train/test 之间无 gap，López de Prado CPCV 可修复 |
| Combinatorial Purged Cross-Validation (CPCV) | ✅ | 已实现 (2026-05-23)；`python picker.py rigor`，15 路径，均值 Sharpe 1.35，100% 路径 > 0 |
| TimeSeriesSplit | ✅ | DML 残差化交叉拟合中使用 (`gap=1`) |
| Out-of-sample test | ✅ | `picks_log.csv` 实时积累真实前向 OOS（2026-05 起） |
| Out-of-time validation | ✅ | 同上；walk-forward 最后一段为 out-of-time |
| Forward testing / Paper trading | ✅ | `oos_track_record()` 每月跟踪实际 vs XIU |

### 预测能力指标（分类器）

| 指标 | 状态 | 说明 |
|------|------|------|
| Accuracy | ✅ | `evaluate_prediction_quality` (2026-05-22)；Balanced Accuracy 0.532 |
| Precision | ✅ | 同上；0.221 (±0.082) |
| Recall | ✅ | 同上；0.585 (±0.212) |
| F1-score | ✅ | 同上；0.321 (±0.117) |
| AUC-ROC | ✅ | 同上；0.554 (±0.154)，lift +0.054 vs random |
| AUC-PR | ✅ | 同上；0.311 (±0.116) |
| Log Loss | ✅ | 已实现 (2026-05-24)；`evaluate_prediction_quality` 中 log_loss，clip to [1e-7, 1-1e-7] |
| Brier Score | ✅ | 同上；0.318 (±0.054) |
| Matthews Correlation Coefficient (MCC) | ✅ | 同上；0.052 (±0.213) |
| Balanced Accuracy | ✅ | 同上；0.532 (±0.133) |

> 当前 ensemble 50% 依赖 classifier；以上指标加入 `evaluate_segments` 或 `picker.py importance` 模式即可。

### 回归预测指标

| 指标 | 状态 | 说明 |
|------|------|------|
| Mean Squared Error (MSE) | N/A | score 非收益单位，RMSE/MSE 无意义；RankIC 是等价的排名 R² |
| Root Mean Squared Error (RMSE) | N/A | 同上 |
| Mean Absolute Error (MAE) | N/A | 同上 |
| Mean Absolute Percentage Error (MAPE) | N/A | 月收益含零/负值，MAPE 无意义 |
| R-squared | ⚠️ | RankIC=0.038 → 截面 R²≈0.001；极低符合预期（alpha 集中在 head） |
| Out-of-sample R² | ⚠️ | 同上；真实 OOS 积累后可算 |
| Directional Accuracy | ✅ | `evaluate_prediction_quality` (2026-05-22)；0.507 (±0.089) ≈ random（alpha 在排名不在方向） |
| Hit Rate | ✅ | 已实现（pick 月份中跑赢 benchmark 的比例） |

### 排序 / 选股能力指标

| 指标 | 状态 | 说明 |
|------|------|------|
| Information Coefficient (IC) | ✅ | `evaluate_segments` per-year / per-sector |
| Rank Information Coefficient (Rank IC) | ✅ | 同上（Spearman） |
| ICIR (IC Information Ratio) | ✅ | 同上 |
| Spearman Rank Correlation | ✅ | Rank IC 的底层实现 |
| Kendall Tau | ✅ | 已实现 (2026-05-23)；`evaluate_segments` 同时报告 Spearman + Kendall τ |
| Top-minus-bottom return | ✅ | 已实现；score 五分位 → 月均收益 + 单调性检验 (`evaluate_segments`)，upgraded from tertile (2026-05-24) |
| Long-short spread | N/A | 策略只做多；不适用 |
| Quantile return analysis | ✅ | 已实现 (2026-05-24)；五分位 (Q1–Q5) score→return 单调性检验，`evaluate_segments` 输出 |
| Decile portfolio analysis | N/A | 31 只股票分十档意义有限；五分位已足够 |
| Lift chart | ⬜ | 可视化选股提升效果；低优先级 |
| Cumulative gains chart | ✅ | 已实现 (2026-05-24)；ASCII 双折线 equity curve (portfolio █ vs benchmark ░) 附 print_backtest 输出，_print_equity_curve() |

> **关键发现 (2026-05-20):** random-score control 证明选股贡献 +10.6pp 超额 / +0.60 Sharpe。全截面 Rank IC 低（0.056）但不代表选股无效——alpha 集中在 head（top-1/2 per sector），全截面 IC 系统低估该策略的选股价值。

### 回测表现指标

| 指标 | 状态 | 说明 |
|------|------|------|
| Sharpe Ratio | ✅ | **副指标**（2026-05-24 降级），当前 **2.13** (ExtraTrees, vol_ratio removed, half_life=6m) |
| Sortino Ratio | ✅ | 已实现 (2026-05-22)；当前 3.03 |
| Calmar Ratio | ✅ | 已实现 (2026-05-22)；当前 3.25 |
| Maximum Drawdown | ✅ | 当前 **-8.3%** (half_life=6m 后轻微扩大) |
| Annualized Return | ✅ | 当前 **+27.0%** |
| Annualized Volatility | ✅ | 已实现 (2026-05-22)；当前 12.7% |
| Cumulative Return | ✅ | 当前 **+155.0%** |
| Win Rate | ✅ | 当前 **68.1%** |
| Profit Factor | ✅ | `print_backtest` (2026-05-22)；**1.89** |
| Expectancy | ✅ | 同上；**+0.78%/月** |
| Alpha (vs benchmark) | ✅ | 超额收益 **+11.2%/yr** vs XIU.TO |
| Beta | ✅ | 同上；0.66（低 beta，防御性） |
| Treynor Ratio | ✅ | 同上；0.41 |
| Information Ratio | ✅ | **主指标**（2026-05-24 升级）；当前 **1.08**；实验接受阈值：IR ≥ **1.06**（基线 − 0.02），Sharpe 作为副指标参考 |
| Tracking Error | ✅ | 已实现 (2026-05-22)；当前 10.4% |

### 稳健性检验

| 检验 | 状态 | 说明 |
|------|------|------|
| Transaction cost analysis | ❌ | 已移除 (2026-05-24)；用户不需要交易费用建模 |
| Slippage analysis | ❌ | 已移除；不需要 |
| Turnover analysis | ❌ | 已移除 (2026-05-24)；用户不关心换手率 |
| Sensitivity analysis | ✅ | 已实现 (2026-05-23)；现仅扫描 `embargo_months`；`python picker.py sensitivity` |
| Parameter stability test | ✅ | 已实现 (2026-05-24)；OAT sweep 17 组合，`python picker.py hptest`。**结论：OAT 单参改善不可叠加**，(300, 5, 10, 0.7) 是稳定局部最优。小样本(47mo)下 OAT 结果本身含大量噪声，不可靠 |
| Feature importance stability | ✅ | 已实现 (2026-05-23)；OOS permutation importance (IC drop) 替代 train-set gain importance；`perm_importance.json` 缓存跨 fold 均值 |
| Permutation importance | ✅ | 已实现 (2026-05-23)；OOS RankIC drop，`perm_importance.json` 缓存，`pick` 模式使用 |
| SHAP value stability | ✅ | 已实现 (2026-05-23)；`_compute_shap_for_models` TreeExplainer，per-pick top-3 驱动因子显示在 email + stdout；`python picker.py shap` 输出全局重要性 |
| Bootstrap test | ✅ | 已实现 (2026-05-23)；WRC (White's Reality Check) bootstrap + `compute_sharpe_ci` 95% CI in `print_overfit_report` |
| Monte Carlo simulation | ⬜ | 低优先级；DSR/WRC bootstrap 已提供置信区间 |
| Stress testing | ⚠️ | 2022 熊市已通过 per-year 表格覆盖（-0.4% vs bench +2.1%）；未做正式系统压力测试 |
| Regime analysis | ✅ | `detect_regime()` + per-year IC breakdown |
| Subperiod analysis | ✅ | per-year / last-6-month 表格 |

### 防止过拟合 / 数据泄露

| 检查 | 状态 | 说明 |
|------|------|------|
| Look-ahead bias check | ⚠️ | fwd_ret=shift(-1) ✅；momentum-PCA / normalize 在全 panel 拟合 ⚠️ |
| Survivorship bias check | ⚠️ | 已知局限 (2026-05-23)；yfinance 无法提供历史成分股；3只退市股已手动移除；偏差方向小幅偏高；yfinance-only 约束下无法修复 |
| Data leakage check | ✅ | features 使用历史数据，标签 shift(-1) |
| Label leakage check | ✅ | `add_labels` 在 normalize 之前运行 |
| Embargo period | ✅ | 已实现 (2026-05-22)；`walk_forward(embargo_months=1)` 默认开启；Sharpe 2.12→1.93（差值=泄漏修正） |
| Purging | ❌ | 未实现；Purged K-Fold / CPCV 可修复 |
| Nested cross-validation | ❌ | 超参手调于全历史；nested CV 可修复 |
| White's Reality Check | ✅ | 已实现 (2026-05-23)；bootstrap demeaned null，**98.6%**，与 DSR 一致 |
| Deflated Sharpe Ratio (DSR) | ✅ | 已实现；`compute_dsr(n_trials=35)` → **91.0%**，MODERATE (n_trials 更新为实际实验次数 2026-05-24) |
| Probability of Backtest Overfitting (PBO) | ⚠️ | PBO proxy (1−DSR) = **9.0%**（n_trials=35 诚实估计）；真正 CPCV-based PBO 需多策略对比，未实现 |
| False Discovery Rate (FDR) control | ✅ | 已实现 (2026-05-23)；BH α=0.05，`rigor` 模式；0/29 特征显著（特征空间饱和，符合预期） |

> **优先级排序（影响可信度）：**
> 1. ✅ DSR + PBO — **更新 (2026-05-24)**：n_trials=35（实际实验次数），DSR **91.0%** MODERATE，PBO **9.0%**。真实 OOS Sharpe 预估 ~1.5–1.8；2.13 为乐观上界。
> 2. ✅ Survivorship bias — **已知局限 (2026-05-23)**：yfinance 限制，irremediable
> 3. ✅ Embargo (1m) — **已完成 (2026-05-22)**：Sharpe 2.12→1.93（差值=泄漏修正量）
> 4. ❌ Transaction cost — **已移除 (2026-05-24)**；用户不需要
> 5. ✅ 补充指标 — **已完成 (2026-05-22)**：当前 Sortino 3.03、Calmar 3.25、Vol 12.7%、IR 1.08、TE 10.4%
> 6. ✅ CPCV / WRC / FDR / Permutation importance — **已完成 (2026-05-23)**：`python picker.py rigor`；WRC 98.6%、CPCV 均值 1.35（15/15 路径 > 0）、FDR 0/29（饱和）

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
  1. Guarantee 1 pick per required sector (highest-scoring).
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

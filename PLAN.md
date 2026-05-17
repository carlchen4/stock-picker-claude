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
├── Data Acquisition        fetch_prices(), fetch_macro(), fetch_fundamentals()
├── Constraint Filtering    apply_constraints(), _apply_concentration_limits()
├── Feature Engineering     compute_monthly_features(), build_panel(), smart_impute()
├── Models                  fit_models(), ensemble_predict()
├── Double Machine Learning estimate_dml_alpha(), apply_dml_adjustment()
├── Regime Detection        detect_regime()
├── Rebalancing Band        apply_rebalancing_band()
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

### Constraints (30+ rules)

- **Liquidity:** min ADV $1M, price $2–$400
- **Fundamentals:** PE 0–150, ROE 0–200%, market cap > $800M
- **Concentration:** max 4 per GICS sector, 2 gold miners, 1 base metal
- **Turnover:** rank buffer 18, score tolerance 1.5%, max 4 changes/month
- **Risk:** VIX scaling, drawdown halt at -15%

### Features (23 total)

| Category | Features |
|----------|----------|
| Momentum | 1m, 3m, 6m, 12m |
| Volatility | 20d, 60d, ratio |
| Technical | RSI, Bollinger Z, 52w high ratio |
| Volume | ADV rank |
| Macro | Oil, CAD, rates, TSX, gold, VIX |
| Fundamental | ROE, PE, div yield, EV/EBITDA, debt/equity |
| Sector | GICS code |

---

## Usage

```bash
cd ~/Documents/stock-picker-claude
source venv/bin/activate

python picker.py pick        # Current month top picks
python picker.py backtest    # Walk-forward backtest
python picker.py both        # Both
```

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

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Portfolio config | `portfolio_config.py` for current holdings tracking | Medium |
| PIT fundamentals | Enrich yfinance quarterly data with better lag handling | Low |
| Collinearity reduction | PCA on correlated momentum features | Low |

---

## Dependencies

```
pandas, numpy, scikit-learn, xgboost, yfinance, scipy
torch, lightgbm (optional, not used in picker.py)
matplotlib, seaborn, plotly, jupyter (for analysis)
```

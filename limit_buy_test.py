"""
Limit-order dip-buy test for FALLING picks.
Instead of predicting the 5-day low, place a limit order X% below the
month-end close, good-til 5 business days; if unfilled, market-buy on day 5.

Trade-off measured: deeper limit = cheaper fill but lower fill rate
(you risk missing the trade while the stock runs away).

Fills are approximated on daily CLOSE <= limit (conservative — real
intraday lows would fill more often, so true savings are a bit better).

Usage: python3 limit_buy_test.py        # TSX
       python3 limit_buy_test.py us      # US tech
"""
import sys, numpy as np
from collections import defaultdict

if len(sys.argv) > 1 and sys.argv[1].lower() == "us":
    import picker_us; MARKET = "US Tech"
else:
    MARKET = "TSX"
import picker
from picker import (TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS, fetch_prices,
    build_panel, smart_impute, add_labels, apply_momentum_pca,
    cross_sectional_normalize, walk_forward, get_ohlcv,
    USE_MOMENTUM_PCA, _RAW_MOMENTUM)

FALL = -0.02      # falling if trailing 5d return < -2%
HOLD_DAYS = 5     # limit order good-til 5 business days
LIMITS = [0.5, 1.0, 1.5, 2.0, 3.0]   # % below close

print(f"\n{'='*60}\n  Limit-order dip-buy — {MARKET} (falling picks)\n{'='*60}\n")
all_t = TSX_UNIVERSE + list(MACRO_TICKERS.values())
price_df = fetch_prices(all_t, years=7)
panel = build_panel(price_df, price_df, TSX_UNIVERSE)
feats = [c for c in FEATURE_COLS if c in panel.columns]
panel = smart_impute(panel, feats); panel = add_labels(panel)
if USE_MOMENTUM_PCA:
    panel = apply_momentum_pca(panel)
    feats = [c for c in FEATURE_COLS if c in panel.columns and c not in _RAW_MOMENTUM]
panel, mfeats = cross_sectional_normalize(panel, feats)
_, perstock = walk_forward(panel, mfeats, return_perstock=True)
sel = perstock[perstock["is_selected"]].copy()

# entry/immediate-1 counting unfilled as market-buy on day HOLD_DAYS
res_all = {L: [] for L in LIMITS}      # net entry vs immediate (with fallback)
res_filled = {L: [] for L in LIMITS}   # entry vs immediate among filled only
fill_rate = {L: [] for L in LIMITS}
n = 0
for _, row in sel.iterrows():
    close, _ = get_ohlcv(price_df, row["ticker"])
    if close is None or len(close) == 0: continue
    past = close.loc[:row["date"]]
    if len(past) < 6: continue
    p0 = float(past.iloc[-1])
    if p0 <= 0: continue
    if p0/float(past.iloc[-6]) - 1.0 >= FALL: continue   # only falling picks
    fut = close.loc[row["date"]:].iloc[1:HOLD_DAYS+1].values.astype(float)
    if len(fut) < HOLD_DAYS: continue
    n += 1
    for L in LIMITS:
        limit = p0 * (1 - L/100)
        hit = fut.min() <= limit
        fill_rate[L].append(1.0 if hit else 0.0)
        if hit:
            res_filled[L].append(limit/p0 - 1.0)
            res_all[L].append(limit/p0 - 1.0)
        else:
            res_all[L].append(fut[-1]/p0 - 1.0)   # market buy day 5

print(f"  Falling pick-months: {n}\n")
print(f"  {'Limit':>7}{'fill rate':>11}{'avg entry(all)':>16}{'avg entry(filled)':>19}")
print(f"  {'─'*7}{'─'*11}{'─'*16}{'─'*19}")
for L in LIMITS:
    fr = np.mean(fill_rate[L])
    allm = np.mean(res_all[L])
    fim = np.mean(res_filled[L]) if res_filled[L] else float('nan')
    print(f"  -{L:>4.1f}%{fr:>10.0%}{allm:>+15.2%}{fim:>+18.2%}")
print(f"\n  immediate buy = 0.00% baseline.  'avg entry(all)' counts misses as")
print(f"  market-buy on day {HOLD_DAYS}.  'filled' = average only when the dip hit.\n")

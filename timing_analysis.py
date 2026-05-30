"""
Execution Timing Analysis
=========================
Measures the empirical cost of executing N business days after month-end
for the stocks that the picker would have selected.

Usage:
    python3 timing_analysis.py

Takes ~2 min (one walk-forward run + price lookups).
"""

import sys
import numpy as np
import pandas as pd
from collections import defaultdict

# Re-use everything from picker.py
sys.path.insert(0, ".")
from picker import (
    TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize,
    walk_forward, get_ohlcv,
    USE_MOMENTUM_PCA, _RAW_MOMENTUM,
)


def analyze_execution_timing(panel, feature_cols, price_df, max_lag=10):
    """
    For every historically selected pick, compute:
      delay_cost[N] = price(month-end + N bdays) / price(month-end) - 1

    Positive = stock rose after month-end → buying late costs alpha.
    Negative = stock fell → buying later is actually cheaper.

    Returns dict {lag: {mean, median, pct_positive, n}}.
    """
    print("  Running walk-forward (one pass, ~1 min)...", flush=True)
    results, perstock = walk_forward(panel, feature_cols, return_perstock=True)
    selected = perstock[perstock["is_selected"]].copy()
    n_months = selected["date"].nunique()
    print(f"  {len(selected)} pick-months across {n_months} months  "
          f"({len(selected) / n_months:.1f} picks/month avg)")

    delay_costs = defaultdict(list)

    for _, row in selected.iterrows():
        month_end = row["date"]
        ticker    = row["ticker"]

        close, _ = get_ohlcv(price_df, ticker)
        if close is None or len(close) == 0:
            continue

        past = close.loc[:month_end]
        if len(past) == 0:
            continue
        p0 = float(past.iloc[-1])
        if p0 <= 0:
            continue

        # Daily prices strictly after month-end close
        future = close.loc[month_end:].iloc[1:]

        for lag in range(1, max_lag + 1):
            if len(future) < lag:
                break
            p_lag = float(future.iloc[lag - 1])
            delay_costs[lag].append(p_lag / p0 - 1.0)

    stats = {}
    for lag in range(1, max_lag + 1):
        vals = delay_costs.get(lag, [])
        if not vals:
            continue
        arr = np.array(vals)
        stats[lag] = {
            "mean":         float(arr.mean()),
            "median":       float(np.median(arr)),
            "pct_positive": float((arr > 0).mean()),
            "std":          float(arr.std()),
            "n":            len(arr),
        }
    return stats


def print_timing_report(stats):
    print("\n" + "═" * 80)
    print("  EXECUTION TIMING ANALYSIS  (historically selected picks only)")
    print("  Delay cost = price(month-end + N days) / price(month-end) - 1")
    print("  + = stock rose after month-end → you pay more by waiting")
    print("  - = stock fell → waiting actually saves money")
    print("═" * 80)
    print(f"  {'Lag':>5}  {'Mean':>8}  {'Median':>8}  {'Std':>7}  "
          f"{'% rose':>7}  {'N':>5}  Note")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*20}")

    for lag, s in sorted(stats.items()):
        ann = s["mean"] * 12  # rough annualised cost
        if abs(s["mean"]) < 0.0008:
            note = "negligible"
        elif s["mean"] > 0.004:
            note = f"≈{ann:+.1%}/yr cost"
        elif s["mean"] < -0.004:
            note = f"≈{ann:+.1%}/yr (buy later OK)"
        else:
            note = f"≈{ann:+.1%}/yr"
        print(f"  +{lag:>2}bd  {s['mean']:>+8.3%}  {s['median']:>+8.3%}  "
              f"{s['std']:>7.3%}  {s['pct_positive']:>6.1%}  {s['n']:>5}  {note}")

    print("═" * 80)

    # Cumulative cost summary
    print("\n  Cumulative cost of waiting (mean, compounded):")
    cum = 1.0
    for lag, s in sorted(stats.items()):
        cum *= (1 + s["mean"])
        print(f"    Execute at month-end + {lag:>2} bdays:  {cum - 1:>+7.3%} total missed")

    # Recommendation
    means = {lag: s["mean"] for lag, s in stats.items()}
    print("\n  ──────────────────────────────────────────")
    day1 = means.get(1, 0)
    day2 = means.get(2, 0)
    day5 = means.get(5, 0)
    print(f"  Day +1:  {day1:+.3%}  │  Day +2:  {day2:+.3%}  │  Day +5:  {day5:+.3%}")

    if day2 < 0.002:
        print("\n  VERDICT: cost of a 1-2 day delay is small — "
              "2nd trading day is fine.")
    elif day1 < 0.001:
        print("\n  VERDICT: execute on day +1 (month-end close available next morning).")
    else:
        print("\n  VERDICT: execute ASAP after month-end — delay is costly.")


def main():
    print("\n" + "═" * 60)
    print("  Execution Timing Analysis")
    print("  (uses same 7-year price history as picker.py)")
    print("═" * 60 + "\n")

    all_tickers = TSX_UNIVERSE + list(MACRO_TICKERS.values())
    print("  [1/3] Downloading price data...")
    price_df = fetch_prices(all_tickers, years=7)

    print("  [2/3] Building feature panel...")
    panel = build_panel(price_df, price_df, TSX_UNIVERSE)
    available = [c for c in FEATURE_COLS if c in panel.columns]
    panel = smart_impute(panel, available)
    panel = add_labels(panel)

    if USE_MOMENTUM_PCA:
        panel = apply_momentum_pca(panel)
        available = [c for c in FEATURE_COLS
                     if c in panel.columns and c not in _RAW_MOMENTUM]
    else:
        available = [c for c in available if c not in ("mom_pc1", "mom_pc2")]

    panel, model_features = cross_sectional_normalize(panel, available)

    print("  [3/3] Analysing execution timing...")
    stats = analyze_execution_timing(panel, model_features, price_df, max_lag=10)
    print_timing_report(stats)


if __name__ == "__main__":
    main()

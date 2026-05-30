"""
Conditional Entry-Timing Analysis — "don't catch a falling knife"
=================================================================
For every historically selected pick, split by whether the stock was
FALLING or RISING at month-end, then compare entry rules:

  IMMEDIATE     buy at month-end close (t0)
  +1d / +3d     buy N business days later (fixed wait)
  FIRST_GREEN   buy on the first day that closes above the prior close
  RECLAIM_SMA5  buy on the first day the close climbs back above its 5-day SMA
  (BEST_10d)    oracle lower bound: the cheapest close in the next 10 days

"Falling" = trailing 5-day return at month-end < -2%.

Entry metric = entry_price / immediate_price - 1
  negative  -> you bought cheaper than buying immediately (waiting helped)
  positive  -> you paid more (waiting hurt)

Usage:  python3 entry_timing.py            # TSX
        python3 entry_timing.py us         # US tech (imports picker_us first)
"""

import sys
import numpy as np
import pandas as pd
from collections import defaultdict

if len(sys.argv) > 1 and sys.argv[1].lower() == "us":
    import picker_us  # noqa: applies US config overrides onto picker
    MARKET = "US Tech"
else:
    MARKET = "TSX"

import picker
from picker import (
    TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize,
    walk_forward, get_ohlcv, USE_MOMENTUM_PCA, _RAW_MOMENTUM,
)

FALL_THRESHOLD = -0.02   # trailing 5-day return below this = "falling"
HORIZON = 10             # business days to look ahead for an entry


def _entry_prices(close, t0, horizon=HORIZON):
    """Return dict of entry prices for each rule, given daily close series."""
    past = close.loc[:t0]
    if len(past) < 6:
        return None
    p0 = float(past.iloc[-1])
    if p0 <= 0:
        return None
    r5 = p0 / float(past.iloc[-6]) - 1.0          # trailing 5-bday return
    sma5_past = past.tail(5).mean()

    future = close.loc[t0:].iloc[1:horizon + 1]    # next up-to-10 closes
    if len(future) < 3:
        return None
    fut = future.values.astype(float)

    rules = {"IMMEDIATE": p0}
    rules["+1d"] = fut[0]
    rules["+3d"] = fut[2] if len(fut) >= 3 else fut[-1]

    # FIRST_GREEN: first day close > prior close (prior = p0 then chained)
    prev = p0
    entry = fut[-1]
    for px in fut:
        if px > prev:
            entry = px
            break
        prev = px
    rules["FIRST_GREEN"] = entry

    # RECLAIM_SMA5: first day rolling 5-day SMA (incl that day) is rising AND
    # close is above the trailing SMA5 measured at month-end.
    entry = fut[-1]
    for px in fut:
        if px > sma5_past:
            entry = px
            break
    rules["RECLAIM_SMA5"] = entry

    rules["BEST_10d"] = float(np.min(fut))         # oracle lower bound

    return p0, r5, rules


def main():
    print(f"\n{'='*64}\n  Conditional Entry-Timing — {MARKET}\n{'='*64}\n")
    all_t = TSX_UNIVERSE + list(MACRO_TICKERS.values())
    print("  [1/3] Downloading prices...")
    price_df = fetch_prices(all_t, years=7)
    print("  [2/3] Building panel + walk-forward...")
    panel = build_panel(price_df, price_df, TSX_UNIVERSE)
    feats = [c for c in FEATURE_COLS if c in panel.columns]
    panel = smart_impute(panel, feats)
    panel = add_labels(panel)
    if USE_MOMENTUM_PCA:
        panel = apply_momentum_pca(panel)
        feats = [c for c in FEATURE_COLS if c in panel.columns and c not in _RAW_MOMENTUM]
    panel, mfeats = cross_sectional_normalize(panel, feats)
    _, perstock = walk_forward(panel, mfeats, return_perstock=True)
    sel = perstock[perstock["is_selected"]].copy()

    print("  [3/3] Measuring entry rules...\n")
    # buckets: 'fall' and 'rise', each maps rule -> list of (entry/imm - 1)
    buckets = {"fall": defaultdict(list), "rise": defaultdict(list)}
    counts = {"fall": 0, "rise": 0}

    for _, row in sel.iterrows():
        close, _ = get_ohlcv(price_df, row["ticker"])
        if close is None or len(close) == 0:
            continue
        res = _entry_prices(close, row["date"])
        if res is None:
            continue
        p0, r5, rules = res
        bucket = "fall" if r5 < FALL_THRESHOLD else "rise"
        counts[bucket] += 1
        for name, px in rules.items():
            buckets[bucket][name].append(px / p0 - 1.0)

    rule_order = ["IMMEDIATE", "+1d", "+3d", "FIRST_GREEN", "RECLAIM_SMA5", "BEST_10d"]
    for bucket, label in [("fall", f"FALLING at month-end (5d ret < {FALL_THRESHOLD:.0%})"),
                          ("rise", "RISING / flat at month-end")]:
        print(f"  ── {label}  —  {counts[bucket]} pick-months ──")
        print(f"     {'Rule':<14}{'avg entry vs immed':>20}{'% cheaper':>12}")
        for name in rule_order:
            vals = np.array(buckets[bucket].get(name, []))
            if len(vals) == 0:
                continue
            cheaper = (vals < -0.0005).mean()
            tag = "  <- oracle (not tradeable)" if name == "BEST_10d" else ""
            print(f"     {name:<14}{vals.mean():>+19.2%}{cheaper:>11.0%}{tag}")
        print()

    print("  Reading: negative 'avg entry vs immed' = waiting got a cheaper price.")
    print("  Compare a tradeable rule (FIRST_GREEN / RECLAIM_SMA5 / +Nd) against")
    print("  IMMEDIATE (0%) and the BEST_10d oracle floor.\n")


if __name__ == "__main__":
    main()

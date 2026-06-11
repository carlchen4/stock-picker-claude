"""
Existing-feature IC audit — measure the standalone predictive power of every
feature ALREADY in the model (FEATURE_COLS), the same way feature_scan.py
measures *candidate* features.

For each feature: per-month cross-sectional Rank-IC vs forward return,
averaged across months, with a t-stat (mean / SE), two-sided p, and a
Benjamini-Hochberg FDR flag. This tells us which of the features we already
trust actually carry a signal that clears the t>2 / FDR bar — and which are
dead weight riding along.

Usage:  python3 feature_ic_existing.py        # TSX
        python3 feature_ic_existing.py us      # US tech
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

if len(sys.argv) > 1 and sys.argv[1].lower() == "us":
    import picker_us  # noqa: applies US overrides onto picker
    MARKET = "US"
else:
    MARKET = "TSX"
import picker
from picker import (TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS, fetch_prices,
                    build_panel, add_labels)


def main():
    print(f"\n{'='*66}\n  Existing-feature IC audit — {MARKET}\n{'='*66}")
    all_t = TSX_UNIVERSE + list(MACRO_TICKERS.values())
    price_df = fetch_prices(all_t, years=7)
    panel = build_panel(price_df, price_df, TSX_UNIVERSE)
    panel = add_labels(panel)
    panel = panel.dropna(subset=["fwd_ret"])

    feats = [c for c in FEATURE_COLS if c in panel.columns and c != "sector_code"]
    if "sector_code" in FEATURE_COLS:
        feats.append("sector_code")  # keep but it's categorical — IC meaningless, flag it
    print(f"  {len(panel)} stock-months, {panel['date'].nunique()} months, "
          f"{len(feats)} features\n")

    results = []
    for f in feats:
        ics = []
        for d, g in panel.groupby("date"):
            sub = g[[f, "fwd_ret"]].dropna()
            if len(sub) >= 5 and sub[f].nunique() > 1:
                ic, _ = spearmanr(sub[f], sub["fwd_ret"])
                if not np.isnan(ic):
                    ics.append(ic)
        if len(ics) < 10:
            continue
        ics = np.array(ics)
        mean_ic = ics.mean()
        t = mean_ic / (ics.std(ddof=1) / np.sqrt(len(ics)) + 1e-12)
        p = 2 * (1 - norm.cdf(abs(t)))
        results.append((f, mean_ic, t, p, len(ics)))

    # BH-FDR across all audited features
    ps = sorted(r[3] for r in results)
    m = len(ps)
    bh_thresh = 0.0
    for i, pv in enumerate(ps, 1):
        if pv <= (i / m) * 0.05:
            bh_thresh = pv

    print(f"  {'Feature':<22}{'mean IC':>9}{'t':>7}{'p':>8}{'mo':>5}  sig?")
    print("  " + "-"*58)
    for f, ic, t, p, n in sorted(results, key=lambda x: -abs(x[2])):
        if f == "sector_code":
            sig = "(categorical)"
        elif p <= bh_thresh and bh_thresh > 0:
            sig = "FDR" + ("***" if p < .001 else "**" if p < .01 else "*")
        elif abs(t) >= 2:
            sig = "t>2"
        else:
            sig = ""
        print(f"  {f:<22}{ic:>+9.4f}{t:>7.2f}{p:>8.3f}{n:>5}  {sig}")
    print("  " + "-"*58)
    nsig = sum(1 for f, ic, t, p, n in results
               if f != "sector_code" and p <= bh_thresh and bh_thresh > 0)
    ngt2 = sum(1 for f, ic, t, p, n in results
               if f != "sector_code" and abs(t) >= 2)
    print(f"\n  {nsig} FDR-significant, {ngt2} with |t|>=2 "
          f"(of {len([r for r in results if r[0]!='sector_code'])} features)")
    print("  Note: IC sign can be negative for genuine signals (e.g. reversal,")
    print("  low-vol). |t| is the strength; sign tells direction.\n")


if __name__ == "__main__":
    main()

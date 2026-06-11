"""
Should we delete the 5 all-NaN fundamental features? — CORRECTED A/B backtest.

v1 of this test was FLAWED: it removed the dead cols only from FEATURE_COLS,
but SECTOR_FEATURES still listed them and build_panel still created the raw
columns, so _resolve_sector_features fell back to the raw (unnormalized) roe
column and the per-sector models KEPT using them → bit-identical result (a
false "safe to delete").

picker.py line ~1228 records a prior careful experiment:
  "Tried (2026-05-21): remove dead features entirely (Sharpe 1.92->1.86) — rejected."
i.e. fully removing them DROPS Sharpe — the NaN placeholders act as a mild
regularizer via max_features sampling. This corrected test removes them from
ALL THREE places (FEATURE_COLS + SECTOR_FEATURES + raw panel columns) to
reproduce the true effect before any deletion.

Usage:  python3 drop_dead_test.py
"""
import warnings
warnings.filterwarnings("ignore")
import copy
import numpy as np
import picker
from picker import (TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS, fetch_prices,
                    build_panel, add_labels, smart_impute,
                    cross_sectional_normalize, walk_forward)

DEAD = ["roe", "pe_ratio", "div_yield", "ev_ebitda", "debt_equity"]


def metrics(results):
    r = results["port_ret"].values
    T = len(results)
    cum_p = (1 + results["port_ret"]).cumprod().iloc[-1] - 1
    cum_b = (1 + results["bench_ret"]).cumprod().iloc[-1] - 1
    ann_p = (1 + cum_p) ** (12 / T) - 1
    ann_b = (1 + cum_b) ** (12 / T) - 1
    sharpe = ann_p / (r.std(ddof=1) * np.sqrt(12))
    ex = (results["port_ret"] - results["bench_ret"]).values
    ir = (ann_p - ann_b) / (ex.std(ddof=1) * np.sqrt(12))
    return sharpe, ir


def run(price_df, truly_remove):
    panel = build_panel(price_df, price_df, TSX_UNIVERSE)
    sec_feat_backup = copy.deepcopy(picker.SECTOR_FEATURES)
    try:
        if truly_remove:
            # 1) drop raw columns so _resolve_sector_features can't fall back
            panel = panel.drop(columns=[c for c in DEAD if c in panel.columns])
            # 2) strip them from every sector feature list
            for sec in picker.SECTOR_FEATURES:
                picker.SECTOR_FEATURES[sec] = [
                    f for f in picker.SECTOR_FEATURES[sec] if f not in DEAD]
            drop = DEAD
        else:
            drop = []
        avail = [c for c in FEATURE_COLS if c in panel.columns and c not in drop]
        p = smart_impute(panel.copy(), avail)
        p = add_labels(p)
        avail = [c for c in avail if c not in ("mom_pc1", "mom_pc2")]
        p, feats = cross_sectional_normalize(p, avail)
        results, _, _ = walk_forward(p, feats, return_perstock=True,
                                     return_importance=True)
        return metrics(results) + (len(feats),)
    finally:
        picker.SECTOR_FEATURES.clear()
        picker.SECTOR_FEATURES.update(sec_feat_backup)


def main():
    print(f"\n{'='*60}\n  DROP-DEAD-FEATURE A/B (CORRECTED) — TSX\n{'='*60}")
    print(f"  dead (all-NaN) features: {DEAD}")
    print("  removing from ALL THREE: FEATURE_COLS + SECTOR_FEATURES + raw panel\n")
    all_t = TSX_UNIVERSE + list(MACRO_TICKERS.values())
    price_df = fetch_prices(all_t, years=7)

    sh0, ir0, nf0 = run(price_df, truly_remove=False)
    sh1, ir1, nf1 = run(price_df, truly_remove=True)

    print(f"  {'variant':<28}{'#feat':>6}{'Sharpe':>9}{'IR':>8}")
    print("  " + "-"*51)
    print(f"  {'baseline (keep dead)':<28}{nf0:>6}{sh0:>9.3f}{ir0:>8.3f}")
    print(f"  {'truly lean (remove all 3)':<28}{nf1:>6}{sh1:>9.3f}{ir1:>8.3f}")
    print(f"  {'Δ (lean − base)':<28}{nf1-nf0:>6}{sh1-sh0:>+9.3f}{ir1-ir0:>+8.3f}")
    print("  " + "-"*51)
    print("\n  If Δ Sharpe is negative, the prior 2026-05-21 finding holds:")
    print("  the NaN placeholders are a mild regularizer — DO NOT delete.\n")


if __name__ == "__main__":
    main()

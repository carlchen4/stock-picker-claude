"""
Final adoption verdict for SKEW60 — DSR / WRC / CPCV, baseline vs +SKEW60.

candidate_backtest.py showed +SKEW60 beats baseline on Sharpe (+0.087) and IR
(+0.060) — the only candidate to clear gate 1. This runs the de-overfitting
gate 2: Deflated Sharpe, White's Reality Check, and CPCV, for BOTH variants,
without permanently touching the model. SKEW60 is adopted ONLY if it keeps DSR
in the same band (no degradation) and CPCV stays positive.

Usage:  python3 rigor_candidate.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import picker
from picker import (TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS, fetch_prices,
                    build_panel, add_labels, smart_impute,
                    cross_sectional_normalize, walk_forward,
                    print_overfit_report, compute_cpcv, print_cpcv_report)
from feature_scan import candidate_panel


def build(panel, extra):
    avail = [c for c in FEATURE_COLS if c in panel.columns] + list(extra)
    p = smart_impute(panel.copy(), avail)
    p = add_labels(p)
    avail = [c for c in avail if c not in ("mom_pc1", "mom_pc2")]
    return cross_sectional_normalize(p, avail)


def main():
    print(f"\n{'='*60}\n  RIGOR VERDICT — baseline vs +SKEW60 (TSX)\n{'='*60}")
    all_t = TSX_UNIVERSE + list(MACRO_TICKERS.values())
    price_df = fetch_prices(all_t, years=7)
    panel = build_panel(price_df, price_df, TSX_UNIVERSE)
    base = add_labels(panel.copy()).dropna(subset=["fwd_ret"])
    mkt = MACRO_TICKERS.get("tsx") or picker.BENCHMARK_TICKER
    cand = candidate_panel(price_df,
                           [t for t in TSX_UNIVERSE if t != picker.BENCHMARK_TICKER],
                           sorted(base["date"].unique()), mkt)
    panel = panel.merge(cand[["date", "ticker", "SKEW60"]],
                        on=["date", "ticker"], how="left")

    for name, extra in [("BASELINE", []), ("+SKEW60", ["SKEW60"])]:
        print(f"\n{'#'*60}\n#  {name}\n{'#'*60}")
        p, feats = build(panel, extra)
        results, _, _ = walk_forward(p, feats, return_perstock=True,
                                     return_importance=True)
        r = results["port_ret"].values
        wf_sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)
        print_overfit_report(results)          # DSR / PSR / WRC
        print(f"\n  CPCV for {name} (15 paths)...")
        paths = compute_cpcv(p, feats)
        print_cpcv_report(paths, wf_sharpe=wf_sharpe)

    print(f"\n{'='*60}")
    print("  DECISION: adopt SKEW60 only if its DSR stays in the same band")
    print("  as baseline AND CPCV median stays positive. Otherwise skip and")
    print("  record the conclusion.\n")


if __name__ == "__main__":
    main()

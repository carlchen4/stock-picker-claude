"""
Candidate-feature BACKTEST — the real adoption test.

IC-scan was only the cheap filter. This actually wires each candidate feature
into the model, runs the full walk-forward, and compares Sharpe/IR to the
baseline. A candidate is worth keeping ONLY if it beats baseline AND would
survive DSR (run `rigor` after, if anything looks promising).

Tests, each as its own walk-forward:
  baseline                (FEATURE_COLS as-is)
  +SKEW60                 (the only candidate with t>2 on TSX)
  +DIST52LOW, +MAXDD60    (next strongest by |IC|)
  +TOP3                   (SKEW60+DIST52LOW+MAXDD60 together)

Usage:  python3 candidate_backtest.py        # TSX
        python3 candidate_backtest.py us      # US tech
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np

if len(sys.argv) > 1 and sys.argv[1].lower() == "us":
    import picker_us  # noqa
    MARKET = "US"
else:
    MARKET = "TSX"
import picker
from picker import (TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS, fetch_prices,
                    build_panel, add_labels, smart_impute,
                    cross_sectional_normalize, walk_forward, _RAW_MOMENTUM)
from feature_scan import candidate_panel


def metrics(results):
    """Sharpe (ann) + IR (vs benchmark) from a walk_forward results frame."""
    r = results["port_ret"].values
    T = len(results)
    cum_p = (1 + results["port_ret"]).cumprod().iloc[-1] - 1
    cum_b = (1 + results["bench_ret"]).cumprod().iloc[-1] - 1
    ann_p = (1 + cum_p) ** (12 / T) - 1
    ann_b = (1 + cum_b) ** (12 / T) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann_p / ann_vol if ann_vol > 0 else 0
    ex = (results["port_ret"] - results["bench_ret"]).values
    te = ex.std(ddof=1) * np.sqrt(12)
    ir = (ann_p - ann_b) / te if te > 0 else float("nan")
    return sharpe, ir, ann_p - ann_b, T


def run(panel_aug, extra_feats):
    """One walk-forward with FEATURE_COLS + extra_feats."""
    avail = [c for c in FEATURE_COLS if c in panel_aug.columns] + list(extra_feats)
    p = smart_impute(panel_aug.copy(), avail)
    p = add_labels(p)
    avail = [c for c in avail if c not in ("mom_pc1", "mom_pc2")]
    p, model_features = cross_sectional_normalize(p, avail)
    results, _, _ = walk_forward(p, model_features,
                                 return_perstock=True, return_importance=True)
    return metrics(results)


def main():
    print(f"\n{'='*60}\n  Candidate-feature BACKTEST — {MARKET}\n{'='*60}")
    all_t = TSX_UNIVERSE + list(MACRO_TICKERS.values())
    price_df = fetch_prices(all_t, years=7)
    panel = build_panel(price_df, price_df, TSX_UNIVERSE)

    # compute candidate columns on the panel's month-ends and merge in
    base = add_labels(panel.copy()).dropna(subset=["fwd_ret"])
    mkt = MACRO_TICKERS.get("tsx") or picker.BENCHMARK_TICKER
    cand = candidate_panel(price_df, [t for t in TSX_UNIVERSE if t != picker.BENCHMARK_TICKER],
                           sorted(base["date"].unique()), mkt)
    cand_cols = [c for c in cand.columns if c not in ("date", "ticker")]
    panel = panel.merge(cand[["date", "ticker"] + cand_cols], on=["date", "ticker"], how="left")

    experiments = [
        ("baseline", []),
        ("+SKEW60", ["SKEW60"]),
        ("+DIST52LOW", ["DIST52LOW"]),
        ("+MAXDD60", ["MAXDD60"]),
        ("+TOP3", ["SKEW60", "DIST52LOW", "MAXDD60"]),
    ]
    # keep only candidates that actually got computed
    experiments = [(n, [f for f in fs if f in cand_cols]) for n, fs in experiments]

    print(f"  candidates available: {cand_cols}\n")
    print(f"  {'experiment':<14}{'Sharpe':>8}{'IR':>8}{'ann.excess':>12}{'mo':>5}")
    print("  " + "-"*47)
    base_sh = base_ir = None
    for name, fs in experiments:
        try:
            sh, ir, ex, T = run(panel, fs)
        except Exception as e:
            print(f"  {name:<14}  ERROR: {e}")
            continue
        if name == "baseline":
            base_sh, base_ir = sh, ir
            tag = ""
        else:
            d_sh = sh - base_sh if base_sh is not None else 0
            d_ir = ir - base_ir if base_ir is not None else 0
            tag = f"  ΔSh {d_sh:+.3f} ΔIR {d_ir:+.3f}"
        print(f"  {name:<14}{sh:>8.3f}{ir:>8.3f}{ex:>+12.1%}{T:>5}{tag}")
    print("  " + "-"*47)
    print("  Adoption bar: must beat baseline on BOTH Sharpe & IR, then")
    print("  survive `python3 picker.py rigor` (DSR/CPCV/PBO). Otherwise skip.\n")


if __name__ == "__main__":
    main()

"""End-to-end smoke test for picker.py.

Runs the full pipeline against a small subset of TSX_UNIVERSE and asserts
invariants that would have caught the bugs found during development:

  - apply_constraints crash on multi-ticker yf.download output
    (Series-vs-scalar truthiness error)
  - rank-as-return bug in walk_forward (fwd_ret holding [-1, +1] ranks
    instead of true monthly returns, inflating benchmark to +200%/yr)

Usage:
    python smoke_test.py
"""
import sys
import numpy as np

import picker

# Small liquid subset — keeps downloads fast (~30s).
SMOKE_UNIVERSE = [
    "XIU.TO",  # benchmark
    "RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO",
    "CNQ.TO", "SU.TO", "ENB.TO", "TRP.TO",
    "ABX.TO", "FNV.TO", "WPM.TO",
    "BCE.TO", "T.TO", "RCI-B.TO",
]


def fail(msg):
    print(f"  FAIL: {msg}")
    sys.exit(1)


def check(cond, msg):
    if not cond:
        fail(msg)
    print(f"  ok: {msg}")


def main():
    print("\n=== Smoke test: picker.py pipeline ===\n")

    print("[1] Fetching prices...")
    all_tickers = SMOKE_UNIVERSE + list(picker.MACRO_TICKERS.values())
    price_df = picker.fetch_prices(all_tickers, years=5)
    check(not price_df.empty, "price_df non-empty")

    print("\n[2] Building panel...")
    panel = picker.build_panel(price_df, price_df, SMOKE_UNIVERSE)
    check(len(panel) > 0, f"panel has rows ({len(panel)})")
    check("mom_1m" in panel.columns, "mom_1m feature present")
    check("earnings_surprise" in panel.columns, "earnings_surprise column present")
    # earnings_surprise is 0-filled where no recent earnings, so absolute
    # values can be 0; but the column must be numeric and contain at
    # least some non-zero entries across the panel.
    nonzero_share = (panel["earnings_surprise"].abs() > 0.01).mean()
    check(nonzero_share > 0.0, f"earnings_surprise has non-zero entries ({nonzero_share:.1%})")

    print("\n[3] Pipeline (impute -> labels -> normalize)...")
    available = [c for c in picker.FEATURE_COLS if c in panel.columns]
    panel = picker.smart_impute(panel, available)
    panel = picker.add_labels(panel)
    panel, model_features = picker.cross_sectional_normalize(panel, available)

    # mom_1m must stay in raw return units (medians ~0.01, well below 0.5)
    mom_1m_median = panel["mom_1m"].abs().median()
    check(mom_1m_median < 0.30,
          f"mom_1m looks like raw returns (|median|={mom_1m_median:.4f})")

    # fwd_ret must be raw returns too — this is the bug we hit before
    fwd_median = panel["fwd_ret"].abs().median()
    check(fwd_median < 0.30,
          f"fwd_ret looks like raw returns (|median|={fwd_median:.4f})")

    # _norm columns must exist and be bounded to [-1, +1]
    check("mom_1m_norm" in panel.columns, "mom_1m_norm column created")
    norm_max = panel["mom_1m_norm"].abs().max()
    check(norm_max <= 1.0 + 1e-6, f"mom_1m_norm bounded in [-1, +1] (max={norm_max:.4f})")

    print("\n[4] Walk-forward backtest...")
    results = picker.walk_forward(panel, model_features, train_months=24)
    check(len(results) > 6, f"walk-forward produced months ({len(results)})")

    # Benchmark returns must be in plausible monthly-return range
    bench_max_monthly = results["bench_ret"].abs().max()
    check(bench_max_monthly < 0.40,
          f"benchmark monthly returns plausible (max |ret|={bench_max_monthly:.4f})")

    # Annualized benchmark return should be within real-world TSX range
    cum_bench = (1 + results["bench_ret"]).prod()
    ann_bench = cum_bench ** (12 / len(results)) - 1
    check(-0.30 < ann_bench < 0.50,
          f"benchmark annualized return plausible ({ann_bench:+.1%})")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()

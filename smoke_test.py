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


def unit_checks():
    """Fast, no-download unit tests for the ops/reporting functions added
    this session (diff_holdings, _health_summary, compute_rank_deltas,
    log_picks / oos_track_record, check_data_health). Guards the monthly
    run against a silent regression in these.
    """
    import os
    import tempfile
    import shutil
    import pandas as pd

    print("[0] Unit checks (ops functions, no download)...")

    # diff_holdings — SELL/BUY/HOLD split vs current holdings
    sell, buy, hold = picker.diff_holdings(["A", "B", "C"], ["B", "C", "D"])
    check(sell == ["D"] and buy == ["A"] and hold == ["B", "C"],
          f"diff_holdings split (sell={sell} buy={buy} hold={hold})")
    s2, b2, h2 = picker.diff_holdings(["A", "B"], [])
    check(b2 == ["A", "B"] and not s2 and not h2,
          "diff_holdings with no holdings -> all BUY")

    # _health_summary — verdict wording from check tuples
    check(picker._health_summary([]) == "", "_health_summary empty -> ''")
    check("all checks passed" in picker._health_summary(
        [("a", True, ""), ("b", True, "")]).lower(),
        "_health_summary all-ok verdict")
    check("caution" in picker._health_summary(
        [("a", True, ""), ("b", False, "x")]).lower(),
        "_health_summary one-warning verdict")

    # _format_report — initial-build (no holdings) vs rebalance (holdings)
    _latest = pd.DataFrame({"ticker": ["A", "B"], "score": [1.0, 0.5]})
    rep_new = " ".join(picker._format_report(
        ["A", "B"], {"A": 0.5, "B": 0.5}, _latest, [("f", 0.1)],
        "NEUTRAL", checks=[], holdings=[])).lower()
    check("initial build" in rep_new and "buy all" in rep_new,
          "_format_report shows initial-build BUY list when no holdings")
    rep_reb = " ".join(picker._format_report(
        ["A", "B"], {"A": 0.5, "B": 0.5}, _latest, [("f", 0.1)],
        "NEUTRAL", checks=[], holdings=["B", "C"])).lower()
    check("vs current holdings" in rep_reb and "sell" in rep_reb,
          "_format_report shows SELL/BUY/HOLD when holding")

    d = tempfile.mkdtemp()
    try:
        # compute_rank_deltas — no history / up / down / flat / NEW
        rpath = os.path.join(d, "rank.csv")
        check(picker.compute_rank_deltas(["A", "B"], "2026-01-31", path=rpath) == {},
              "compute_rank_deltas no history -> {}")
        picker.save_rank_history(["A", "B", "C"], [0.9, 0.5, 0.1], "2026-01-31", path=rpath)
        dl = picker.compute_rank_deltas(["B", "A", "C"], "2026-02-28", path=rpath)
        check(dl["B"] == "↑1" and dl["A"] == "↓1" and dl["C"] == "→",
              f"compute_rank_deltas up/down/flat ({dl})")
        check(picker.compute_rank_deltas(["A", "D"], "2026-02-28", path=rpath).get("D") == "NEW",
              "compute_rank_deltas NEW for unseen ticker")

        # log_picks idempotency + XIU benchmark row; oos_track_record math
        opath = os.path.join(d, "picks.csv")
        picker.log_picks(["A", "B"], {"A": 0.5, "B": 0.5}, {"A": 1, "B": 1}, "2026-01-31", path=opath)
        n1 = len(pd.read_csv(opath))
        picker.log_picks(["A", "B"], {"A": 0.5, "B": 0.5}, {"A": 1, "B": 1}, "2026-01-31", path=opath)
        check(n1 == 3 and len(pd.read_csv(opath)) == n1,
              "log_picks logs picks+XIU row and is idempotent per month")
        pd.DataFrame({
            "as_of": pd.to_datetime(["2026-01-31"] * 3),
            "ticker": ["A", "B", "XIU.TO"], "weight": [0.5, 0.5, 0.0],
            "score": [1, 1, float("nan")], "fwd_realized": [0.04, 0.02, 0.01],
        }).to_csv(opath, index=False)
        oos_txt = " ".join(picker.oos_track_record(opath)).lower()
        check("portfolio" in oos_txt and "xiu" in oos_txt,
              "oos_track_record reports portfolio vs XIU")

        # check_data_health — benchmark present / missing
        idx = pd.date_range("2024-01-01", periods=60, freq="ME")
        cols = pd.MultiIndex.from_product([["XIU.TO", "RY.TO", "CNQ.TO"], ["Close"]])
        pdf = pd.DataFrame(np.random.rand(60, 3) + 10, index=idx, columns=cols)
        ok_map = {l: ok for l, ok, _ in
                  picker.check_data_health(pdf, ["XIU.TO", "RY.TO", "CNQ.TO"], ["Financials"])}
        check(ok_map.get("Benchmark XIU.TO") is True,
              "check_data_health benchmark present -> ok")
        miss = {l: ok for l, ok, _ in picker.check_data_health(
            pdf.drop(columns=[("XIU.TO", "Close")]), ["XIU.TO", "RY.TO"], [])}
        check(miss.get("Benchmark XIU.TO") is False,
              "check_data_health missing benchmark -> fail")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print()


def main():
    print("\n=== Smoke test: picker.py pipeline ===\n")
    unit_checks()

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
    # New per-sub-industry growth signals
    check("div_growth_yoy" in panel.columns, "div_growth_yoy column present")
    check("rev_growth_yoy" in panel.columns, "rev_growth_yoy column present")
    div_pct = (panel["div_growth_yoy"].abs() > 0.001).mean()
    rev_pct = (panel["rev_growth_yoy"].abs() > 0.001).mean()
    check(div_pct > 0.1, f"div_growth_yoy populated ({div_pct:.1%})")
    # rev_growth_yoy is intrinsically sparse — yfinance only returns
    # 5-8 quarters per ticker, so monthly-aligned coverage stays in the
    # 3-8% range. Backtest decides whether it's net useful or noise.
    check(rev_pct > 0.02, f"rev_growth_yoy populated ({rev_pct:.1%})")
    # New macro proxies
    for col in ["natgas_mom_1m", "carbon_mom_1m", "transport_mom_1m"]:
        check(col in panel.columns, f"{col} macro feature present")

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

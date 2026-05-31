"""
Basic unit tests for picker.py — run with:  python3 test_picker.py
(No pytest needed: plain asserts + a tiny runner.)

Covers deterministic core logic with SYNTHETIC data (no network):
config integrity · RSI · cross-sectional normalize · sector encoding ·
selection constraints · DML theta recovery · picker_us config overrides.

NOTE: picker_us mutates picker's module globals, so its test runs LAST.
"""
import numpy as np
import pandas as pd
import picker

_passed, _failed = 0, 0

def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"  PASS  {name}")
    else:
        _failed += 1; print(f"  FAIL  {name}")


# ── 1. Config integrity ──────────────────────────────────────────────
def test_config():
    sectors = set(picker.SECTOR_NAME_TO_CODE)
    bad = [t for t, p in picker.STOCK_PROFILE.items()
           if p[0] not in sectors and p[0] != "Unknown"]
    check("STOCK_PROFILE sectors all known", not bad)
    check("SECTOR_FEATURES keys ⊆ sectors", set(picker.SECTOR_FEATURES) <= sectors)
    req = picker.CONSTRAINTS.get("required_sectors") or []
    check("required_sectors all known", all(s in sectors for s in req))
    check("FEATURE_COLS no duplicates",
          len(picker.FEATURE_COLS) == len(set(picker.FEATURE_COLS)))
    check("benchmark in universe", picker.BENCHMARK_TICKER in picker.TSX_UNIVERSE)


# ── 2. compute_rsi ───────────────────────────────────────────────────
def test_rsi():
    # realistic noisy trends (strictly-monotonic series → loss/gain=0 → NaN RSI,
    # which never happens with real prices)
    rng = np.random.RandomState(1)
    up = pd.Series(100 + np.cumsum(rng.randn(80) * 0.5 + 0.8))    # net up, some down days
    r = picker.compute_rsi(up).dropna()
    check("RSI bounded 0–100", len(r) > 0 and (r >= 0).all() and (r <= 100).all())
    check("RSI high on uptrend", r.iloc[-1] > 60)
    down = pd.Series(200 + np.cumsum(rng.randn(80) * 0.5 - 0.8))  # net down
    rf = picker.compute_rsi(down).dropna()
    check("RSI low on downtrend", len(rf) > 0 and rf.iloc[-1] < 40)


# ── 3. cross_sectional_normalize ─────────────────────────────────────
def test_xsnorm():
    df = pd.DataFrame({
        "date": ["2025-01-31"] * 5,
        "ticker": list("ABCDE"),
        "feat": [10.0, 20, 30, 40, 50],
    })
    out, cols = picker.cross_sectional_normalize(df, ["feat"])
    n = out["feat_norm"]
    check("xsnorm returns _norm col", "feat_norm" in cols)
    check("xsnorm in [-1,1]", (n >= -1).all() and (n <= 1).all())
    check("xsnorm monotone (max row highest)", n.iloc[4] == n.max() and n.iloc[0] == n.min())
    # ties -> 0
    df2 = pd.DataFrame({"date": ["d"] * 3, "ticker": list("XYZ"), "feat": [5.0, 5, 5]})
    out2, _ = picker.cross_sectional_normalize(df2, ["feat"])
    check("xsnorm ties → 0", (out2["feat_norm"] == 0).all())


# ── 4. encode_sector ─────────────────────────────────────────────────
def test_encode_sector():
    code = picker.encode_sector("RY.TO")
    check("RY.TO encodes to Financials code",
          code == picker.SECTOR_NAME_TO_CODE.get("Financials"))
    check("unknown ticker → 0", picker.encode_sector("ZZZ.TO") == 0)


# ── 5. apply_rebalancing_band (selection constraints) ────────────────
def test_rebalancing_band():
    # real TSX tickers so STOCK_PROFILE sector lookup works
    picks = ["RY.TO", "TD.TO", "BMO.TO",          # Financials
             "CNQ.TO", "SU.TO", "ENB.TO",          # Energy
             "CNR.TO", "CP.TO",                     # Industrials
             "FTS.TO", "EMA.TO"]                    # Utilities
    scores = [0.9, 0.85, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]
    sel = picker.apply_rebalancing_band(picks, scores, current_holdings=[])
    C = picker.CONSTRAINTS
    check("≤ top_n picks", len(sel) <= C["top_n"])
    # per-sector cap
    from collections import Counter
    sec = Counter(picker.STOCK_PROFILE.get(t, ("?",))[0] for t in sel)
    check("≤ max_per_gics per sector", all(v <= C["max_per_gics"] for v in sec.values()))
    # ≥1 per required sector
    req = C.get("required_sectors") or []
    check("≥1 per required sector", all(sec.get(s, 0) >= 1 for s in req))


# ── 6. _apply_concentration_limits ───────────────────────────────────
def test_concentration():
    C = dict(picker.CONSTRAINTS)
    many_fin = ["RY.TO", "TD.TO", "BMO.TO", "CM.TO", "BNS.TO", "NA.TO"]
    sel = picker._apply_concentration_limits(many_fin, C)
    from collections import Counter
    sec = Counter(picker.STOCK_PROFILE.get(t, ("?",))[0] for t in sel)
    check("concentration caps Financials at max_per_gics",
          sec.get("Financials", 0) <= C["max_per_gics"])


# ── 7. DML theta recovery (synthetic Y = 1.5·D + g(X) + noise) ───────
def test_dml_theta():
    rng = np.random.RandomState(0)
    rows = []
    dates = pd.date_range("2023-01-31", periods=12, freq="ME")
    for d in dates:
        for _ in range(25):
            f1, f2 = rng.randn(), rng.randn()
            D = rng.randn() * 0.05
            Y = 1.5 * D + 0.3 * f1 * 0.01 + rng.randn() * 0.01
            rows.append({"date": d, "fwd_ret": Y, "sector_etf_ret": D,
                         "f1": f1, "f2": f2})
    sec_df = pd.DataFrame(rows)
    res = picker.estimate_sector_dml_theta(sec_df, ["f1", "f2"])
    check("DML returns a result", res is not None)
    if res is not None:
        check(f"DML theta ≈ 1.5 (got {res['theta']:.2f})", abs(res["theta"] - 1.5) < 0.5)


# ── 8. picker_us config overrides (LAST — mutates picker globals) ────
def test_picker_us_last():
    import picker_us  # noqa: applies overrides onto picker
    check("US benchmark = QQQ", picker.BENCHMARK_TICKER == "QQQ")
    check("US has 3 sectors", set(picker.SECTOR_NAME_TO_CODE) ==
          {"Semiconductors", "Cloud", "Hardware"})
    check("US no Software", "Software" not in picker.SECTOR_NAME_TO_CODE)
    check("US dashboard file separated", picker.DASHBOARD_FILE == "data_us.json")


if __name__ == "__main__":
    print("Running picker unit tests...\n")
    for t in [test_config, test_rsi, test_xsnorm, test_encode_sector,
              test_rebalancing_band, test_concentration, test_dml_theta,
              test_picker_us_last]:   # picker_us LAST
        print(f"[{t.__name__}]")
        try:
            t()
        except Exception as e:
            _failed += 1
            print(f"  ERROR {t.__name__}: {e}")
        print()
    print(f"{'='*40}\n  {_passed} passed, {_failed} failed\n{'='*40}")
    import sys
    sys.exit(1 if _failed else 0)

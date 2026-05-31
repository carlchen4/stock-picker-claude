"""
US Tech Stock Picker — picker_us.py

Override wrapper for picker.py. Imports picker then replaces all config
variables before calling picker.main(). Any improvement made to picker.py
(walk-forward logic, CPCV, reporting, etc.) is automatically inherited here.

Universe: 20 large-cap US tech stocks across 3 sub-sectors (no Software)
Benchmark: QQQ (NASDAQ-100 ETF)
"""

import sys
import os

# Must import picker before overriding its module-level variables
import picker

import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════

picker.BENCHMARK_TICKER = "QQQ"
picker.REPORT_LABEL = "US Tech"
picker.DASHBOARD_FILE = "data_us.json"   # separate dashboard from TSX (docs/us.html)
picker.DASHBOARD_URL = "https://carlchen4.github.io/stock-picker-claude/us.html"
picker.USE_NEWS_SENTIMENT = True         # AV news sentiment on US picks (6/run, 24h cached)

# Plain-English company names for report readability (US universe).
picker.COMPANY_NAMES = {
    "MSFT": "Microsoft", "AMZN": "Amazon", "META": "Meta Platforms",
    "GOOGL": "Alphabet (Google)", "ANET": "Arista Networks",
    "PLTR": "Palantir Technologies", "NET": "Cloudflare", "SNOW": "Snowflake",
    "NVDA": "NVIDIA", "AMD": "Advanced Micro Devices", "AVGO": "Broadcom",
    "QCOM": "Qualcomm", "TXN": "Texas Instruments", "AMAT": "Applied Materials",
    "MU": "Micron Technology", "LRCX": "Lam Research",
    "TSM": "Taiwan Semiconductor (TSMC)",
    "AAPL": "Apple", "TSLA": "Tesla", "ARM": "Arm Holdings", "INTC": "Intel",
    "QQQ": "Invesco QQQ (NASDAQ-100 ETF)",
}

# ══════════════════════════════════════════════════════════════════
# UNIVERSE  (benchmark first, then stocks by sub-sector)
# ══════════════════════════════════════════════════════════════════

picker.TSX_UNIVERSE = [
    "QQQ",   # NASDAQ-100 ETF (benchmark)
    # Cloud / Internet (8, incl. MSFT)
    "MSFT", "AMZN", "META", "GOOGL", "ANET", "PLTR", "NET", "SNOW",
    # Semiconductors (9)
    "NVDA", "AMD", "AVGO", "QCOM", "TXN", "AMAT", "MU", "LRCX", "TSM",
    # Hardware / Mixed (4)
    "AAPL", "TSLA", "ARM", "INTC",
]

# ══════════════════════════════════════════════════════════════════
# STOCK PROFILE  (sector, style, sub_type)
# ══════════════════════════════════════════════════════════════════

picker.STOCK_PROFILE = {
    # Cloud / Internet (MSFT moved here)
    "MSFT":  ("Cloud", "core",   "hyperscaler"),
    # Semiconductors
    "NVDA":  ("Semiconductors", "growth", "gpu"),
    "AMD":   ("Semiconductors", "growth", "cpu"),
    "AVGO":  ("Semiconductors", "value",  "broadband"),
    "QCOM":  ("Semiconductors", "value",  "mobile"),
    "TXN":   ("Semiconductors", "value",  "analog"),
    "AMAT":  ("Semiconductors", "core",   "equipment"),
    "MU":    ("Semiconductors", "value",  "memory"),
    "LRCX":  ("Semiconductors", "core",   "equipment"),
    "TSM":   ("Semiconductors", "core",   "foundry"),
    # Cloud / Internet
    "AMZN":  ("Cloud", "core",   "hyperscaler"),
    "META":  ("Cloud", "core",   "social"),
    "GOOGL": ("Cloud", "core",   "search"),
    "ANET":  ("Cloud", "growth", "networking"),
    "PLTR":  ("Cloud", "growth", "analytics"),
    "NET":   ("Cloud", "growth", "cdn"),
    "SNOW":  ("Cloud", "growth", "data_platform"),
    # Hardware / Mixed
    "AAPL":  ("Hardware", "core",  "consumer"),
    "TSLA":  ("Hardware", "growth", "ev"),
    "ARM":   ("Hardware", "growth", "cpu_ip"),
    "INTC":  ("Hardware", "value",  "cpu"),
    # Benchmark
    "QQQ":   ("Unknown", "etf",    "benchmark"),
}

# ══════════════════════════════════════════════════════════════════
# SECTOR ENCODING
# ══════════════════════════════════════════════════════════════════

picker.SECTOR_NAME_TO_CODE = {
    "Semiconductors": 1, "Cloud": 2, "Hardware": 3,
}

# ══════════════════════════════════════════════════════════════════
# MACRO TICKERS  (remove Canadian ETFs, add US-specific)
# ══════════════════════════════════════════════════════════════════

picker.MACRO_TICKERS = {
    # Regime detection index (used by detect_regime via MACRO_TICKERS["tsx"])
    "tsx":        "SPY",
    # US tech / broad equity
    "qqq":        "QQQ",
    "sp500":      "SPY",
    "usd_index":  "UUP",         # USD index ETF (PowerShares)
    # Cross-asset
    "vix":        "^VIX",
    "gold":       "GC=F",
    "oil":        "CL=F",
    "us10y":      "^TNX",
    "inflation":  "TIP",
    # Sector ETFs for DML-PLR treatment (one per sub-sector)
    "etf_semi":   "SOXX",        # iShares Semiconductor ETF
    "etf_cloud":  "WCLD",        # WisdomTree Cloud Computing ETF
    "etf_hw":     "XLK",         # SPDR Technology Select Sector ETF
}

# ══════════════════════════════════════════════════════════════════
# SECTOR ETF  (DML-PLR treatment per sub-sector)
# ══════════════════════════════════════════════════════════════════

picker.SECTOR_ETF = {
    "Semiconductors": "SOXX",
    "Cloud":          "WCLD",
    "Hardware":       "XLK",
}

# ══════════════════════════════════════════════════════════════════
# SECTOR FEATURES  (features per sub-sector model)
# ══════════════════════════════════════════════════════════════════

# Start from the base technical/momentum features, strip Canadian macro
_BASE = [f for f in picker._BASE_SECTOR_FEATURES]  # copy

picker.SECTOR_FEATURES = {
    "Semiconductors": _BASE + [
        "rate_chg_3m",
        "vix_level",
        "qqq_mom_1m",
        "oil_mom_1m",           # energy cost proxy (fabs)
        "fed_rate_chg_3m",
        "roe", "pe_ratio", "div_yield", "debt_equity",
        "sector_code",
    ],
    "Cloud": _BASE + [
        "rate_chg_3m",
        "vix_level",
        "qqq_mom_1m",
        "tips_mom_1m",
        "fed_rate_chg_3m",
        "roe", "pe_ratio", "div_yield", "debt_equity",
        "sector_code",
    ],
    "Hardware": _BASE + [
        "rate_chg_3m",
        "vix_level",
        "qqq_mom_1m",
        "usd_index_mom_1m",
        "fed_rate_chg_3m",
        "roe", "pe_ratio", "div_yield", "debt_equity",
        "sector_code",
    ],
}

# ══════════════════════════════════════════════════════════════════
# FEATURE_COLS  (global model feature list)
# Strip Canadian features; add US equivalents.
# ══════════════════════════════════════════════════════════════════

_CA_ONLY = {
    "cad_bond_mom_1m", "boc_rate_chg_3m",
    "cad_mom_1m", "tsx_mom_1m",
    "natgas_mom_1m", "carbon_mom_1m",
    "transport_mom_1m", "util_mom_1m",
}

picker.FEATURE_COLS = (
    [c for c in picker.FEATURE_COLS if c not in _CA_ONLY]
    + ["qqq_mom_1m", "sp500_mom_1m", "usd_index_mom_1m", "fed_rate_chg_3m"]
)

# ══════════════════════════════════════════════════════════════════
# CONSTRAINTS  (same keys as picker.py; USD-denominated thresholds)
# ══════════════════════════════════════════════════════════════════

picker.CONSTRAINTS = {
    # Liquidity & Price — keys keep the *_cad names because picker.py's
    # apply_constraints() reads those exact keys; values are USD here.
    "min_adv_cad":          50_000_000,    # USD daily $ volume
    "min_price_cad":        5.0,           # USD
    "max_price_cad":        10_000.0,      # USD — NVDA/AVGO can exceed $500
    # Fundamentals
    "min_mktcap_cad":       10_000_000_000,
    "min_pe":               0.0,
    # max_pe raised 300→600: growth tech routinely has PE>300 (ARM ~415,
    # PLTR/AMD spike). 300 was a TSX-financials carry-over that silently
    # excluded ARM every month despite high scores. 600 = sanity cap only.
    "max_pe":               600.0,
    "min_roe":              -5.0,          # allow temporary losses
    "max_roe":              10.0,          # NVDA ROE can be very high
    # History
    "min_listing_days":     252,
    # Concentration
    "max_per_gics":         2,
    "max_per_style":        5,
    "max_per_type":         6,
    "required_sectors":     ["Semiconductors", "Cloud", "Hardware"],
    "max_single_alloc":     0.25,
    # (gold/base-metal/energy sub caps removed — never accessed for a
    #  tech-only universe; picker.py short-circuits before reading them.)
    # Quality — vol_spike_sigma raised 3.0→4.0: tech routinely spikes on
    # earnings; 3σ over-excluded freshly-reported names (non-holdings only).
    "vol_spike_sigma":      4.0,
    "vol_spike_min_days":   2,
    # Portfolio
    "top_n":                6,
}

# ══════════════════════════════════════════════════════════════════
# FED RATE  (replaces BOC Valet API — uses ^IRX via yfinance)
# ══════════════════════════════════════════════════════════════════

def _fetch_fed_rate(years=7):
    """Monthly US 13-week T-bill rate from ^IRX (yfinance), as BOC replacement."""
    import yfinance as yf
    cache_file = picker._cache_path("us_fed_rate.parquet")
    cached = picker._cache_load(cache_file, max_age_hours=24)
    if cached is not None:
        return cached["rate"]
    hist = yf.Ticker("^IRX").history(period=f"{years}y")
    if hist.empty:
        return pd.Series(dtype=float)
    rate = hist["Close"] / 100
    # Strip timezone so index is tz-naive, matching panel dates
    if rate.index.tz is not None:
        rate.index = rate.index.tz_localize(None)
    result = rate.resample("ME").last()
    picker._cache_save(cache_file, result.to_frame("rate"))
    return result


def _get_fed_rate_features(dates):
    """fed_rate_chg_3m: 3-month change in US short-rate (replaces boc_rate_chg_3m)."""
    result = pd.DataFrame(index=dates)
    rate = _fetch_fed_rate()
    if rate.empty:
        result["fed_rate_chg_3m"] = np.nan
        return result
    monthly = rate.resample("ME").last()
    # Ensure tz-naive to match panel dates
    if monthly.index.tz is not None:
        monthly.index = monthly.index.tz_localize(None)
    result["fed_rate_chg_3m"] = monthly.diff(3).reindex(dates, method="ffill", limit=3)
    return result


picker.fetch_boc_overnight_rate = _fetch_fed_rate
picker.get_boc_features = _get_fed_rate_features

# ══════════════════════════════════════════════════════════════════
# P/S RATIO  (Price-to-Sales TTM, injected via build_panel override)
# ══════════════════════════════════════════════════════════════════

def _attach_ps_ratio(panel, price_df):
    """Add ps_ratio_ttm column to panel using quarterly revenue + current shares.

    FCF yield was tested (2026-05-30) and rejected: ΔIR +0.01, near-zero
    permutation importance — noise. P/S alone is the kept fundamental.
    """
    import yfinance as yf
    from datetime import timedelta

    rows = []
    for ticker in panel["ticker"].unique():
        if ticker == picker.BENCHMARK_TICKER:
            continue
        qf = picker.fetch_quarterly_financials(ticker)
        if qf is None or qf.empty or "revenue" not in qf.columns:
            continue
        rev = qf["revenue"].dropna().sort_index()
        if len(rev) < 4:
            continue
        # PIT lag: only use revenue published ≥45 days ago
        rev.index = rev.index + timedelta(days=45)
        ttm = rev.rolling(4, min_periods=4).sum()

        # Shares outstanding — current value (large-cap stable enough for ranking)
        try:
            info = yf.Ticker(ticker).fast_info
            shares = getattr(info, "shares", None)
            if not shares or shares <= 0:
                shares = yf.Ticker(ticker).info.get("sharesOutstanding", 0)
            if not shares or shares <= 0:
                continue
        except Exception:
            continue

        # Monthly closing price for this ticker
        try:
            px_series, _ = picker.get_ohlcv(price_df, ticker)
            if px_series is None:
                continue
            monthly_px = px_series.resample("ME").last()
        except Exception:
            continue

        for date in panel.loc[panel["ticker"] == ticker, "date"].unique():
            rev_at_date = ttm[ttm.index <= date]
            px_at_date  = monthly_px[monthly_px.index <= date]
            if rev_at_date.empty or px_at_date.empty:
                continue
            ttm_val = rev_at_date.iloc[-1]
            if pd.isna(ttm_val) or ttm_val <= 0:
                continue
            ps = (px_at_date.iloc[-1] * shares) / ttm_val
            rows.append({"date": date, "ticker": ticker, "ps_ratio_ttm": ps})

    if rows:
        ps_df = pd.DataFrame(rows)
        panel = panel.merge(ps_df, on=["date", "ticker"], how="left")
    else:
        panel["ps_ratio_ttm"] = np.nan
    return panel


_orig_build_panel = picker.build_panel

def _build_panel_us(price_df, macro_df, universe):
    panel = _orig_build_panel(price_df, macro_df, universe)
    print("  Attaching P/S ratio (TTM)...")
    panel = _attach_ps_ratio(panel, price_df)
    return panel

picker.build_panel = _build_panel_us

# Add ps_ratio_ttm to feature list and sector features
picker.FEATURE_COLS = picker.FEATURE_COLS + ["ps_ratio_ttm"]
for sec in picker.SECTOR_FEATURES:
    picker.SECTOR_FEATURES[sec] = picker.SECTOR_FEATURES[sec] + ["ps_ratio_ttm"]

# ══════════════════════════════════════════════════════════════════
# PICKS LOG  (separate file so US picks don't mix with TSX log)
# ══════════════════════════════════════════════════════════════════

picker.PICKS_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "picks_log_us.csv"
)

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    picker.main()


if __name__ == "__main__":
    main()

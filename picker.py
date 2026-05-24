"""
TSX Stock Picker — ExtraTrees + Double Machine Learning
═══════════════════════════════════════════════════════
Data source: yfinance only (free, no API keys needed)
Model: ExtraTrees regression + classification ensemble
Causal: Double Machine Learning (Chernozhukov 2018)

Usage:
    python picker.py pick          # Current month top picks
    python picker.py backtest      # Walk-forward backtest
    python picker.py both          # Both

Install:
    pip install -r requirements.txt
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau
import sys
import os
import json
import time
from itertools import combinations
from scipy.stats import ttest_1samp
from concurrent.futures import ThreadPoolExecutor

try:
    # portfolio_config.py is gitignored; copy from portfolio_config.example.py
    # and edit to declare your real holdings. Falls back to no holdings
    # so a fresh clone still runs.
    from portfolio_config import CURRENT_HOLDINGS
except ImportError:
    CURRENT_HOLDINGS = []

# When True, replace mom_{1,3,6,12}m features with 2 PCA components.
# Set False to keep the raw momentum features (the original baseline).
USE_MOMENTUM_PCA = True

# When True and portfolio_config.CURRENT_HOLDINGS is empty, treat last
# run's picks as the current holdings — auto-rolls the portfolio month to
# month so holdings never need hand-editing. Valid ONLY if the model's
# picks are actually executed each month; if your real positions diverge,
# fill CURRENT_HOLDINGS (it takes precedence) or set this False.
AUTO_ROLL_HOLDINGS = True

# When True, train one XGBoost+DML per required sector (Financials,
# Energy, Industrials, Utilities), each on its curated feature subset
# from SECTOR_FEATURES below. Set False to use a single global model
# trained on all stocks with the full feature set.
USE_SECTOR_MODELS = True

# Mirrors encode_sector's sector_map; lifted to module scope so the
# per-sector functions can look up codes from sector names.
SECTOR_NAME_TO_CODE = {
    "Financials": 1, "Energy": 2, "Materials": 3, "Industrials": 4,
    "ConsumerDisc": 5, "ConsumerStaples": 6, "Technology": 7,
    "Communication": 8, "Utilities": 9, "RealEstate": 10, "HealthCare": 11,
}

# Sector ETFs used as DML-PLR treatments. Each sector's XGBoost model
# is trained on alpha residuals (fwd_ret - theta * sector_etf_ret),
# stripping out the sector beta and forcing the model to learn pure
# idiosyncratic alpha.
SECTOR_ETF = {
    "Financials":  "XFN.TO",   # iShares S&P/TSX Capped Financials
    "Energy":      "XEG.TO",   # iShares S&P/TSX Capped Energy
    "Industrials": "ZIN.TO",   # BMO Equal Weight Industrials
    "Utilities":   "XUT.TO",   # iShares S&P/TSX Capped Utilities
}

# Per-sector feature subsets (the X for each sector's model). Listed
# as BASE names — the resolver picks the _norm counterpart at run-time
# where cross_sectional_normalize has produced one. Each sector gets
# the broad momentum/technical/volume features plus its curated set
# of macro control variables from the per-sector spec.
_BASE_SECTOR_FEATURES = [
    "mom_pc1", "mom_pc2",
    "rev_1m",  # short-term reversal, independent of the momentum PCs
    "vol_20d", "vol_60d",
    "rsi_14", "bb_zscore", "high_52w_ratio",
    "adv_20d_rank",
]

SECTOR_FEATURES = {
    "Financials": _BASE_SECTOR_FEATURES + [
        "rate_chg_3m",       # US 10Y change
        "cad_bond_mom_1m",   # XBB.TO — Canadian bond ETF, BOC sensitivity
        "boc_rate_chg_3m",   # REAL Bank of Canada overnight rate (3mo change)
        "vix_level",         # market volatility
        "cad_mom_1m",        # CAD/USD
        "tsx_mom_1m",        # equity market beta
        "tips_mom_1m",       # CPI proxy
        "roe", "pe_ratio", "div_yield", "debt_equity",
        "sector_code",
    ],
    "Energy": _BASE_SECTOR_FEATURES + [
        "oil_mom_1m",        # WTI
        "natgas_mom_1m",     # Henry Hub
        "carbon_mom_1m",     # KRBN
        "cad_mom_1m",        # CAD/USD
        "rate_chg_3m",       # 10Y
        "tsx_mom_1m",        # sector beta proxy
        "div_growth_yoy",    # for pipelines (ENB, TRP)
        "roe", "pe_ratio", "div_yield", "debt_equity",
        "sector_code",
    ],
    "Industrials": _BASE_SECTOR_FEATURES + [
        "transport_mom_1m",  # IYT — freight / PMI proxy
        "cad_mom_1m",        # CAD/USD
        "rate_chg_3m",       # rates
        "rev_growth_yoy",    # for services / AI infra (WSP, TRI, CLS)
        "roe", "pe_ratio", "debt_equity",
        "sector_code",
    ],
    "Utilities": _BASE_SECTOR_FEATURES + [
        "rate_chg_3m",       # US 10Y
        "cad_bond_mom_1m",   # XBB Canadian bond proxy
        "boc_rate_chg_3m",   # REAL BOC overnight rate (3mo change)
        "util_mom_1m",       # XLU — electricity demand proxy
        "tips_mom_1m",       # CPI proxy
        "div_growth_yoy",    # for regulated (H.TO)
        "roe", "pe_ratio", "div_yield", "debt_equity",
        "sector_code",
    ],
}

# ══════════════════════════════════════════════════════════════════
# UNIVERSE
# ══════════════════════════════════════════════════════════════════

TSX_UNIVERSE = [
    "XIU.TO",  # TSX 60 ETF (benchmark)
    # Financials (12) — includes NA.TO and EQB.TO (held but were missing)
    "RY.TO", "TD.TO", "BMO.TO", "CM.TO", "BNS.TO",
    "NA.TO", "EQB.TO",
    "MFC.TO", "SLF.TO", "FFH.TO", "BAM.TO", "BN.TO",
    # Energy (8)
    "CNQ.TO", "SU.TO", "CVE.TO", "ARX.TO", "TOU.TO",
    "ENB.TO", "TRP.TO", "IMO.TO",
    # Industrials (6)
    "CNR.TO", "CP.TO", "WSP.TO", "TRI.TO", "WCN.TO", "CLS.TO",
    # Utilities (4)
    "FTS.TO", "H.TO", "EMA.TO", "AQN.TO",
]

# Focused 4-sector universe; the TSX Composite extension is NOT merged
# in. STOCK_PROFILE still pulls from tsx_extended for any of the
# extended tickers that are in the curated list (BAM, BN, TOU, TRI,
# WCN, CLS), so concentration limits work correctly.
try:
    from tsx_extended import EXTENDED_PROFILES
except ImportError:
    EXTENDED_PROFILES = {}

# Macro proxy tickers (all via yfinance)
MACRO_TICKERS = {
    "oil": "CL=F",
    "cad_usd": "CADUSD=X",
    "us10y": "^TNX",
    "tsx": "^GSPTSE",
    "gold": "GC=F",
    "vix": "^VIX",
    # Sector-relevant yfinance proxies (per the per-sector spec):
    "natgas": "NG=F",        # Henry Hub futures — gas / pipeline relevance
    "carbon": "KRBN",        # KraneShares Global Carbon ETF
    "transport": "IYT",      # iShares Transportation ETF — freight proxy / PMI lag
    "utilities_etf": "XLU",  # XLU — utility-sector sentiment, electricity demand
    "inflation": "TIP",      # TIPS ETF — market-implied inflation
    "cad_bonds": "XBB.TO",   # Canadian aggregate bond ETF — BOC rate proxy
    # Sector ETFs for DML-PLR baseline (NOT used as model features —
    # build_panel maps these to a per-row sector_etf_ret column).
    "etf_fin": "XFN.TO",
    "etf_eng": "XEG.TO",
    "etf_ind": "ZIN.TO",
    "etf_uti": "XUT.TO",
    # Tried-and-rejected (kept commented for the record): ^TYX, ^IRX,
    # HYG, LQD, RB=F, ^GSPC, ^IXIC, XLK plus P/B per-ticker. Adding
    # yield-curve-slope, credit-spread, refining-margin, sp500/nasdaq
    # beta, and P/B regressed backtest from Sharpe 1.65 to 1.54 even
    # in the selective-keep variant — more features = noisier
    # per-sector models at this sample size.
}

# Stock sector/style classification
STOCK_PROFILE = {
    # Financials
    "RY.TO": ("Financials", "core", "bank"), "TD.TO": ("Financials", "core", "bank"),
    "BNS.TO": ("Financials", "value", "bank"), "BMO.TO": ("Financials", "core", "bank"),
    "CM.TO": ("Financials", "value", "bank"), "NA.TO": ("Financials", "growth", "bank"),
    "MFC.TO": ("Financials", "value", "insurance"), "SLF.TO": ("Financials", "core", "insurance"),
    "GWO.TO": ("Financials", "value", "insurance"), "POW.TO": ("Financials", "value", "holding"),
    "IFC.TO": ("Financials", "growth", "insurance"), "FFH.TO": ("Financials", "value", "insurance"),
    "IGM.TO": ("Financials", "value", "asset_mgmt"), "IAG.TO": ("Financials", "value", "insurance"),
    "EQB.TO": ("Financials", "growth", "bank"),
    # Energy
    "CNQ.TO": ("Energy", "core", "oil_gas"), "SU.TO": ("Energy", "core", "oil_gas"),
    "ENB.TO": ("Energy", "value", "pipeline"), "TRP.TO": ("Energy", "value", "pipeline"),
    "CVE.TO": ("Energy", "value", "oil_gas"), "IMO.TO": ("Energy", "value", "oil_gas"),
    "PPL.TO": ("Energy", "value", "pipeline"), "ARX.TO": ("Energy", "growth", "oil_gas"),
    "WCP.TO": ("Energy", "growth", "oil_gas"), "TVE.TO": ("Energy", "value", "oil_gas"),
    "BIR.TO": ("Energy", "value", "oil_gas"), "PSK.TO": ("Energy", "value", "royalty"),
    "KEL.TO": ("Energy", "growth", "oil_gas"), "PEY.TO": ("Energy", "value", "oil_gas"),
    "FRU.TO": ("Energy", "value", "royalty"),
    # Materials — Gold
    "ABX.TO": ("Materials", "core", "gold"), "FNV.TO": ("Materials", "growth", "gold_royalty"),
    "WPM.TO": ("Materials", "growth", "gold_royalty"), "AEM.TO": ("Materials", "core", "gold"),
    "K.TO": ("Materials", "core", "gold"), "AGI.TO": ("Materials", "growth", "gold"),
    "CG.TO": ("Materials", "growth", "gold"), "OR.TO": ("Materials", "growth", "gold_royalty"),
    "ELD.TO": ("Materials", "value", "gold"),
    "BTO.TO": ("Materials", "growth", "gold"),
    # Materials — Base
    "FM.TO": ("Materials", "core", "base_metal"), "IVN.TO": ("Materials", "growth", "base_metal"),
    "LUN.TO": ("Materials", "core", "base_metal"), "TECK-B.TO": ("Materials", "core", "diversified"),
    "HBM.TO": ("Materials", "growth", "base_metal"), "CS.TO": ("Materials", "value", "base_metal"),
    "CCL-B.TO": ("Materials", "core", "packaging"), "WFG.TO": ("Materials", "core", "forestry"),
    # Industrials
    "CNR.TO": ("Industrials", "core", "rail"), "CP.TO": ("Industrials", "core", "rail"),
    "WSP.TO": ("Industrials", "growth", "engineering"), "TIH.TO": ("Industrials", "growth", "industrial"),
    "CAE.TO": ("Industrials", "growth", "aerospace"), "STN.TO": ("Industrials", "growth", "engineering"),
    "TFII.TO": ("Industrials", "growth", "transport"), "ATS.TO": ("Industrials", "value", "industrial"),
    "GFL.TO": ("Industrials", "growth", "waste"), "RBA.TO": ("Industrials", "growth", "auction"),
    # Consumer Discretionary
    "DOL.TO": ("ConsumerDisc", "core", "retail"), "ATZ.TO": ("ConsumerDisc", "growth", "retail"),
    "QSR.TO": ("ConsumerDisc", "core", "restaurant"), "MG.TO": ("ConsumerDisc", "value", "auto"),
    "LNR.TO": ("ConsumerDisc", "growth", "retail"), "BYD.TO": ("ConsumerDisc", "growth", "gaming"),
    "PBH.TO": ("ConsumerDisc", "core", "pharma_retail"),
    # Consumer Staples
    "SAP.TO": ("ConsumerStaples", "core", "grocery"), "ATD.TO": ("ConsumerStaples", "growth", "convenience"),
    "L.TO": ("ConsumerStaples", "core", "grocery"), "MFI.TO": ("ConsumerStaples", "value", "food"),
    "EMP-A.TO": ("ConsumerStaples", "value", "grocery"), "WN.TO": ("ConsumerStaples", "value", "food"),
    "NWC.TO": ("ConsumerStaples", "value", "retail"),
    # Technology
    "SHOP.TO": ("Technology", "growth", "ecommerce"), "CSU.TO": ("Technology", "growth", "software"),
    "OTEX.TO": ("Technology", "value", "software"), "ENGH.TO": ("Technology", "growth", "software"),
    "DSG.TO": ("Technology", "growth", "software"), "KXS.TO": ("Technology", "growth", "software"),
    "GIB-A.TO": ("Technology", "value", "it_services"), "LSPD.TO": ("Technology", "growth", "software"),
    # Communication
    "BCE.TO": ("Communication", "value", "telecom"), "T.TO": ("Communication", "value", "telecom"),
    "RCI-B.TO": ("Communication", "core", "telecom"),
    # Utilities
    "FTS.TO": ("Utilities", "value", "utility"), "EMA.TO": ("Utilities", "value", "utility"),
    "AQN.TO": ("Utilities", "value", "utility"), "CU.TO": ("Utilities", "value", "utility"),
    "H.TO": ("Utilities", "value", "utility"),
    # REITs
    "REI-UN.TO": ("RealEstate", "value", "reit"), "HR-UN.TO": ("RealEstate", "value", "reit"),
    "CAR-UN.TO": ("RealEstate", "growth", "reit"), "AP-UN.TO": ("RealEstate", "core", "reit"),
    "GRT-UN.TO": ("RealEstate", "value", "reit"), "DIR-UN.TO": ("RealEstate", "value", "reit"),
    # Healthcare
    "WELL.TO": ("HealthCare", "growth", "health_services"),
    "CTC-A.TO": ("ConsumerDisc", "core", "retail"),
}

# Merge in the auto-generated extension profiles. Curated entries above
# take precedence (dict.update only fills missing keys via this pattern).
for _t, _p in EXTENDED_PROFILES.items():
    STOCK_PROFILE.setdefault(_t, _p)

# Per-user overrides for sector classification. yfinance/Wikipedia
# classify CLS.TO (Celestica) under Information Technology, but the
# user includes it in their Industrials bucket — honor that here so
# the required_sectors / per-sector caps see it as Industrials.
STOCK_PROFILE["CLS.TO"] = ("Industrials", "core", "other")

# Bank basket: individual Big Banks all score identically (single leaf in
# ExtraTrees). ZEB.TO represents the bank sector; individual banks are kept
# in the universe for model training but excluded at pick time.
BANK_BASKET_TICKER = "ZEB.TO"

# ══════════════════════════════════════════════════════════════════
# CONSTRAINTS
# ══════════════════════════════════════════════════════════════════

CONSTRAINTS = {
    # Liquidity & Price
    "min_adv_cad": 1_000_000,
    "min_price_cad": 2.0,
    "max_price_cad": 400.0,
    # Fundamentals
    "min_mktcap_cad": 800_000_000,
    "min_pe": 0.0,
    "max_pe": 150.0,
    "min_roe": 0.0,
    "max_roe": 2.0,
    # History
    "min_listing_days": 252,
    # Concentration. User rule: picks restricted to the 4 sectors
    # below, with at least 1 and at most 2 from each. top_n=8 covers
    # the 2-per-sector maximum (4 sectors x 2).
    "max_per_gics": 2,
    "max_per_style": 4,
    "max_per_type": 5,
    "required_sectors": ["Financials", "Energy", "Industrials", "Utilities"],
    "max_single_alloc": 0.25,
    "max_gold_mining": 2,
    "max_base_metals": 1,
    "max_energy_sub": 2,
    # Risk
    "dd_halt_threshold": -0.15,
    "dd_halt_scale": 0.50,
    "vix_scale_threshold": 25.0,
    "vix_scale_factor": 0.70,
    # Quality
    "vol_spike_sigma": 3.0,
    "vol_spike_min_days": 2,
    "min_confidence": 0.15,
    # Portfolio
    "top_n": 8,
}

FEATURE_COLS = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    "rev_1m",  # short-term reversal (-mom_1m); independent of momentum PCA
    "mom_pc1", "mom_pc2",  # only present when USE_MOMENTUM_PCA is True
    "vol_20d", "vol_60d",
    "rsi_14", "bb_zscore", "high_52w_ratio",
    "adv_20d_rank",
    "oil_mom_1m", "cad_mom_1m", "rate_chg_3m",
    "tsx_mom_1m", "gold_mom_1m", "vix_level",
    # Sector-relevant macro proxies (yfinance ETFs / futures)
    "natgas_mom_1m", "carbon_mom_1m", "transport_mom_1m",
    "util_mom_1m", "tips_mom_1m", "cad_bond_mom_1m",
    # External macro (BOC Valet API): real BOC overnight rate change
    "boc_rate_chg_3m",
    # Per-sub-industry growth signals
    "div_growth_yoy", "rev_growth_yoy",
    "roe", "pe_ratio", "div_yield",
    "ev_ebitda", "debt_equity",
    "sector_code",
]

# Raw momentum features get dropped from the model's feature list when
# PCA is enabled (the PCs replace them). The DML stage still needs a
# momentum treatment though, so picker.py routes that through mom_pc1
# in PCA mode and mom_1m otherwise.
_RAW_MOMENTUM = ["mom_1m", "mom_3m", "mom_6m", "mom_12m"]

# ══════════════════════════════════════════════════════════════════
# DATA ACQUISITION (yfinance only)
# ══════════════════════════════════════════════════════════════════

def safe_float(val, default=np.nan):
    """Safely convert yfinance's inconsistent types to float."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════
# LOCAL DATA CACHE  (avoids repeated yfinance downloads within a day)
# ═══════════════════════════════════════════════���══════════════════

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _cache_load(path, max_age_hours=24):
    """Return cached DataFrame/Series if file exists and is fresh; else None."""
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) / 3600 > max_age_hours:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _cache_save(path, obj):
    try:
        if isinstance(obj, pd.Series):
            obj = obj.to_frame("value")
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(path)
    except Exception:
        pass


def fetch_prices(tickers, years=7):
    """Download daily OHLCV for all tickers via yfinance."""
    cache_file = _cache_path(f"prices_{len(tickers)}_{years}y.parquet")
    cached = _cache_load(cache_file, max_age_hours=20)
    if cached is not None:
        return cached
    start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, auto_adjust=True,
                       progress=False, threads=True, group_by="ticker")
    _cache_save(cache_file, data)
    return data


def get_ohlcv(price_df, ticker):
    """Return (close, volume) Series for ticker.

    Handles both layouts that yf.download produces: a MultiIndex
    (ticker, field) when multiple tickers are requested, and flat
    columns ("Close", "Volume", ...) for a single ticker.
    Returns (None, None) if the ticker isn't present.
    """
    try:
        if isinstance(price_df.columns, pd.MultiIndex):
            close = price_df[(ticker, "Close")].dropna()
            volume = (price_df[(ticker, "Volume")].dropna()
                      if (ticker, "Volume") in price_df.columns else None)
        else:
            close = price_df["Close"].dropna()
            volume = (price_df["Volume"].dropna()
                      if "Volume" in price_df.columns else None)
        return close, volume
    except (KeyError, TypeError):
        return None, None


def check_data_health(price_df, universe, required_sectors=None):
    """Download-completeness diagnostic; returns (label, ok, detail) tuples.

    Catches the silent-failure mode where yfinance returns partial data:
    flags a low overall download rate, a missing benchmark, and any
    required sector left with too few names for its per-sector model to
    be meaningful. main() aborts the run on the hard failures rather than
    quietly producing picks off broken data.
    """
    avail = set()
    for t in universe:
        close, _ = get_ohlcv(price_df, t)
        if close is not None and len(close) > 20:
            avail.add(t)
    missing = [t for t in universe if t not in avail]

    checks = []
    rate = len(avail) / max(len(universe), 1)
    detail = f"{len(avail)}/{len(universe)} tickers ({rate:.0%})"
    if missing:
        detail += f"; missing: {missing}"
    checks.append(("Download rate", rate >= 0.85, detail))

    bench_ok = "XIU.TO" in avail
    checks.append(("Benchmark XIU.TO", bench_ok,
                   "present" if bench_ok else "MISSING — excess/benchmark invalid"))

    for sec in (required_sectors or []):
        n = sum(1 for t in avail
                if t != "XIU.TO" and STOCK_PROFILE.get(t, ("",))[0] == sec)
        checks.append((f"{sec} coverage", n >= 2,
                       f"{n} names" + ("" if n >= 2 else " (need >=2)")))
    return checks


def fetch_macro(years=7):
    """Download macro indicator time series."""
    tickers = list(MACRO_TICKERS.values())
    start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, auto_adjust=True,
                       progress=False, threads=True)
    return data


def fetch_fundamentals(tickers):
    """Fetch current fundamentals from yfinance for constraint filtering."""
    cache_file = _cache_path("fundamentals.parquet")
    cached = _cache_load(cache_file, max_age_hours=24)
    if cached is not None:
        missing = [t for t in tickers if t not in cached.index]
        if not missing:
            return cached.loc[tickers]

    def _fetch_one(t):
        try:
            info = yf.Ticker(t).info
            return t, {
                "pe": safe_float(info.get("trailingPE")),
                "fwd_pe": safe_float(info.get("forwardPE")),
                "roe": safe_float(info.get("returnOnEquity")),
                "mktcap": safe_float(info.get("marketCap")),
                "div_yield": safe_float(info.get("dividendYield", 0)),
                "debt_equity": safe_float(info.get("debtToEquity")),
                "ev_ebitda": safe_float(info.get("enterpriseToEbitda")),
                "pb": safe_float(info.get("priceToBook")),
                "sector": info.get("sector", ""),
                "name": info.get("shortName", t),
            }
        except Exception:
            return t, {}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = dict(ex.map(lambda t: _fetch_one(t), tickers))

    df = pd.DataFrame(results).T
    _cache_save(cache_file, df)
    return df


def fetch_earnings_surprise(ticker):
    """Fetch historical earnings surprises (%) from yfinance."""
    if ticker == "XIU.TO":
        return pd.Series(dtype=float)
    cache_file = _cache_path(f"{ticker.replace('.', '_')}_earnings.parquet")
    cached = _cache_load(cache_file, max_age_hours=24)
    if cached is not None:
        return cached["value"].astype(float)
    try:
        df = yf.Ticker(ticker).get_earnings_dates()
        if df is None or df.empty or "Surprise(%)" not in df.columns:
            result = pd.Series(dtype=float)
        else:
            s = df["Surprise(%)"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            result = s.sort_index()
    except Exception:
        result = pd.Series(dtype=float)
    _cache_save(cache_file, result)
    return result


def compute_earnings_surprise_feature(ticker, monthly_dates, window_months=3):
    """Align earnings surprises onto monthly_dates.

    For each month-end, returns the most recent earnings surprise within
    the past `window_months` months, or 0.0 if none. Zero represents
    "no recent surprise to react to" — keeps the column dense so DML's
    dropna doesn't shrink the training set when earnings data is sparse.
    """
    surprises = fetch_earnings_surprise(ticker)
    result = pd.Series(0.0, index=monthly_dates)
    if surprises.empty:
        return result
    window = pd.Timedelta(days=31 * window_months)
    for dt in monthly_dates:
        recent = surprises[(surprises.index <= dt) & (surprises.index > dt - window)]
        if not recent.empty:
            result.loc[dt] = float(recent.iloc[-1])
    return result


def fetch_dividend_history(ticker):
    """Fetch the dividend payment history from yfinance."""
    cache_file = _cache_path(f"{ticker.replace('.', '_')}_divs.parquet")
    cached = _cache_load(cache_file, max_age_hours=168)  # 7 days — dividends change quarterly
    if cached is not None:
        return cached["value"].astype(float)
    try:
        divs = yf.Ticker(ticker).dividends
        if divs is None or divs.empty:
            result = pd.Series(dtype=float)
        else:
            divs.index = pd.to_datetime(divs.index).tz_localize(None)
            result = divs.sort_index()
    except Exception:
        result = pd.Series(dtype=float)
    _cache_save(cache_file, result)
    return result


def compute_dividend_growth_feature(ticker, monthly_dates):
    """Year-over-year trailing-12-month dividend growth, aligned monthly.

    For each month-end t:
      TTM_t   = sum of dividends paid in (t-12mo, t]
      TTM_t-1 = sum of dividends paid in (t-24mo, t-12mo]
      growth  = TTM_t / TTM_t-1 - 1

    Returns a Series indexed by monthly_dates, 0.0 where no prior
    TTM exists (non-payer or insufficient history). Used as a feature
    so XGBoost can route by sub-industry (e.g., pipelines like ENB
    where the dividend signal dominates).
    """
    divs = fetch_dividend_history(ticker)
    result = pd.Series(0.0, index=monthly_dates)
    if divs.empty:
        return result
    year = pd.Timedelta(days=365)
    for dt in monthly_dates:
        ttm = divs[(divs.index > dt - year) & (divs.index <= dt)].sum()
        prior = divs[(divs.index > dt - 2 * year) & (divs.index <= dt - year)].sum()
        if prior > 0:
            result.loc[dt] = float(ttm / prior - 1.0)
    return result


def compute_revenue_growth_feature(ticker, monthly_dates):
    """Year-over-year quarterly revenue growth, aligned monthly.

    Pulls quarterly revenue from yfinance, applies a 45-day reporting
    lag (only data published by date t can be used), and forwards the
    most recent YoY growth rate at each monthly date.

    Returns 0.0 where no prior-year quarter is available.
    """
    qf = fetch_quarterly_financials(ticker)
    result = pd.Series(0.0, index=monthly_dates)
    if qf is None or qf.empty or "revenue" not in qf.columns:
        return result
    rev = qf["revenue"].dropna().sort_index()
    if len(rev) < 5:
        return result
    # Apply reporting lag so each monthly date only sees revenue that
    # was actually published by then.
    rev.index = rev.index + timedelta(days=45)
    yoy = rev / rev.shift(4) - 1.0
    yoy = yoy.dropna()
    if yoy.empty:
        return result
    aligned = yoy.reindex(monthly_dates, method="ffill", limit=3)
    return aligned.fillna(0.0)


def fetch_boc_overnight_rate(years=7):
    """Fetch the Bank of Canada overnight money market rate.

    Pulls from the BOC Valet API (free, no key). Returns a daily Series
    indexed by date. Empty Series on any failure (network, parse, missing
    observations).

    The overnight rate is BOC's primary policy lever — a cleaner signal
    for Canadian monetary policy than the XBB.TO bond-ETF proxy, since
    XBB moves with the whole curve (and inversely to rate changes).
    """
    import requests
    start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    series_id = "V39079"  # Overnight money market financing rate
    url = (f"https://www.bankofcanada.ca/valet/observations/"
           f"{series_id}/json?start_date={start}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        records = []
        for obs in observations:
            d = obs.get("d")
            v = obs.get(series_id, {}).get("v")
            if d and v:
                try:
                    records.append((pd.to_datetime(d), float(v)))
                except ValueError:
                    continue
        if not records:
            return pd.Series(dtype=float)
        df = pd.DataFrame(records, columns=["date", "rate"])
        return df.set_index("date")["rate"].sort_index()
    except Exception:
        return pd.Series(dtype=float)


def get_boc_features(dates):
    """Compute monthly BOC overnight rate features aligned to dates.

    Returns a DataFrame with `boc_rate_chg_3m` (3-month change in the
    monthly-average overnight rate). Mirrors the shape of the existing
    `rate_chg_3m` US-10Y feature. Empty frame on fetch failure — the
    column ends up NaN and `smart_impute` fills it cross-sectionally.
    """
    rate = fetch_boc_overnight_rate()
    result = pd.DataFrame(index=dates)
    if rate.empty:
        result["boc_rate_chg_3m"] = np.nan
        return result
    monthly = rate.resample("ME").last()
    result["boc_rate_chg_3m"] = monthly.diff(3).reindex(dates, method="ffill", limit=3)
    return result


def fetch_quarterly_financials(ticker):
    """Fetch quarterly financials from yfinance for PIT fundamental features."""
    cache_file = _cache_path(f"{ticker.replace('.', '_')}_qfin.parquet")
    cached = _cache_load(cache_file, max_age_hours=168)  # 7 days — quarterly updates
    if cached is not None:
        return cached
    try:
        tk = yf.Ticker(ticker)
        inc = tk.quarterly_income_stmt
        bal = tk.quarterly_balance_sheet
        if inc is None or inc.empty:
            return None
        records = []
        for col in inc.columns[:8]:
            rec = {"date": col}
            rec["net_income"] = safe_float(inc.loc["Net Income", col]) if "Net Income" in inc.index else np.nan
            rec["revenue"] = safe_float(inc.loc["Total Revenue", col]) if "Total Revenue" in inc.index else np.nan
            rec["ebitda"] = safe_float(inc.loc["EBITDA", col]) if "EBITDA" in inc.index else np.nan
            if bal is not None and col in bal.columns:
                rec["total_equity"] = safe_float(bal.loc["Total Equity Gross Minority Interest", col]) if "Total Equity Gross Minority Interest" in bal.index else np.nan
                rec["total_debt"] = safe_float(bal.loc["Total Debt", col]) if "Total Debt" in bal.index else np.nan
            records.append(rec)
        result = pd.DataFrame(records).set_index("date").sort_index()
    except Exception:
        return None
    _cache_save(cache_file, result)
    return result


# ══════════════════════════════════════════════════════════════════
# CONSTRAINT FILTERING
# ══════════════════════════════════════════════════════════════════

def apply_constraints(candidates, fundamentals_df, price_df, mode="pick",
                      current_holdings=None, constraints=None):
    """
    Filter candidates based on constraints.
    mode="pick": use live fundamentals
    mode="backtest": use only price-based constraints (no look-ahead)
    """
    C = constraints or CONSTRAINTS
    passed = []

    for ticker in candidates:
        if ticker == "XIU.TO":
            continue  # benchmark only

        prices, vol = get_ohlcv(price_df, ticker)
        if prices is None or len(prices) < C["min_listing_days"]:
            continue

        last_price = prices.iloc[-1]
        if last_price < C["min_price_cad"] or last_price > C["max_price_cad"]:
            continue

        if vol is not None:
            adv = (vol.tail(20) * prices.tail(20)).mean()
            if adv < C["min_adv_cad"]:
                continue

            # Volume-spike anti-anomaly filter only applies to new picks.
            # If we already own the ticker, recent earnings or news that
            # produced a spike isn't a reason to forcibly sell — the band
            # gets to decide based on score instead.
            is_holding = current_holdings and ticker in current_holdings
            if not is_holding:
                vol_series = vol.tail(60)
                vol_mean = vol_series.mean()
                vol_std = vol_series.std()
                if vol_std > 0:
                    spike_days = (vol_series > vol_mean + C["vol_spike_sigma"] * vol_std).sum()
                    if spike_days >= C["vol_spike_min_days"]:
                        continue

        # Fundamental checks (pick mode only)
        if mode == "pick" and ticker in fundamentals_df.index:
            row = fundamentals_df.loc[ticker]
            pe = safe_float(row.get("pe"))
            if not np.isnan(pe) and (pe < C["min_pe"] or pe > C["max_pe"]):
                continue
            roe = safe_float(row.get("roe"))
            if not np.isnan(roe) and (roe < C["min_roe"] or roe > C["max_roe"]):
                continue
            mktcap = safe_float(row.get("mktcap"))
            if not np.isnan(mktcap) and mktcap < C["min_mktcap_cad"]:
                continue

        passed.append(ticker)

    # Concentration limits
    passed = _apply_concentration_limits(passed, C, current_holdings)
    return passed


def _apply_concentration_limits(candidates, C, current_holdings=None):
    """Enforce sector/style/type concentration caps."""
    selected = []
    gics_count = {}
    style_count = {}
    type_count = {}
    gold_count = 0
    base_count = 0
    energy_sub_count = {}

    # Give priority to current holdings
    priority = []
    rest = []
    for t in candidates:
        if current_holdings and t in current_holdings:
            priority.append(t)
        else:
            rest.append(t)

    for t in priority + rest:
        profile = STOCK_PROFILE.get(t, ("Unknown", "core", "other"))
        gics, style, sub_type = profile

        # Check limits
        if gics_count.get(gics, 0) >= C["max_per_gics"]:
            continue
        if style_count.get(style, 0) >= C["max_per_style"]:
            continue
        if type_count.get(sub_type, 0) >= C["max_per_type"]:
            continue
        if sub_type in ("gold", "gold_royalty") and gold_count >= C["max_gold_mining"]:
            continue
        if sub_type == "base_metal" and base_count >= C["max_base_metals"]:
            continue

        # Update counts
        gics_count[gics] = gics_count.get(gics, 0) + 1
        style_count[style] = style_count.get(style, 0) + 1
        type_count[sub_type] = type_count.get(sub_type, 0) + 1
        if sub_type in ("gold", "gold_royalty"):
            gold_count += 1
        if sub_type == "base_metal":
            base_count += 1

        selected.append(t)

    return selected


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def compute_rsi(close, window=14):
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_time_decay_weights(n_samples, half_life_months=12):
    """Exponential decay weights for sample weighting in training."""
    decay = np.log(2) / half_life_months
    weights = np.exp(-decay * np.arange(n_samples)[::-1])
    return weights / weights.sum() * n_samples


def compute_monthly_features(price_df, ticker):
    """Compute monthly technical features for a single ticker."""
    close, vol = get_ohlcv(price_df, ticker)
    if close is None or len(close) < 252:
        return None

    # Resample to month-end
    monthly = close.resample("ME").last().dropna()
    if len(monthly) < 13:
        return None

    feats = pd.DataFrame(index=monthly.index)

    # Momentum
    feats["mom_1m"] = monthly.pct_change(1)
    feats["mom_3m"] = monthly.pct_change(3)
    feats["mom_6m"] = monthly.pct_change(6)
    feats["mom_12m"] = monthly.pct_change(12)

    # Short-term reversal: negative of the most recent month's return.
    # Kept OUT of _RAW_MOMENTUM so the momentum PCA does not absorb it —
    # this gives each sector model a clean 1-month signal independent of
    # the mom_pc1/pc2 mixture. (For the tree models rev_1m == -mom_1m is
    # split-equivalent to mom_1m; the point is restoring 1m as its own
    # dimension. Reversal naming kept for semantics / future linear use.)
    feats["rev_1m"] = -feats["mom_1m"]

    # Volatility (annualized from daily returns)
    daily_ret = close.pct_change()
    feats["vol_20d"] = daily_ret.rolling(20).std().resample("ME").last() * np.sqrt(252)
    feats["vol_60d"] = daily_ret.rolling(60).std().resample("ME").last() * np.sqrt(252)
    feats["vol_ratio"] = feats["vol_20d"] / feats["vol_60d"].replace(0, np.nan)

    # RSI
    rsi = compute_rsi(close)
    feats["rsi_14"] = rsi.resample("ME").last()

    # Bollinger Z-score
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_z = (close - ma20) / std20.replace(0, np.nan)
    feats["bb_zscore"] = bb_z.resample("ME").last()

    # 52-week high ratio
    high_52w = close.rolling(252).max()
    feats["high_52w_ratio"] = (close / high_52w.replace(0, np.nan)).resample("ME").last()

    # Volume rank (average daily volume, normalized later)
    if vol is not None and len(vol) > 20:
        adv_20 = vol.rolling(20).mean()
        feats["adv_20d_rank"] = adv_20.resample("ME").last()
    else:
        feats["adv_20d_rank"] = np.nan

    # Earnings surprise (DML treatment, not in FEATURE_COLS — used by
    # estimate_dml_alpha as a causal signal but never fed to the
    # ensemble as a feature).
    feats["earnings_surprise"] = compute_earnings_surprise_feature(ticker, monthly.index)

    # Per-sub-industry signals: dividend growth (pipelines, regulated
    # utilities) and revenue growth (professional services, AI infra).
    # Added as features so XGBoost can route them per sub_type via
    # tree splits — no hard-coded routing needed.
    feats["div_growth_yoy"] = compute_dividend_growth_feature(ticker, monthly.index)
    feats["rev_growth_yoy"] = compute_revenue_growth_feature(ticker, monthly.index)

    feats["ticker"] = ticker
    return feats


def get_macro_features(macro_df, dates):
    """Extract macro features aligned to monthly dates."""
    result = pd.DataFrame(index=dates)

    for name, ticker in MACRO_TICKERS.items():
        try:
            close = macro_df[(ticker, "Close")].dropna() if (ticker, "Close") in macro_df.columns else macro_df[ticker]["Close"].dropna()
            monthly = close.resample("ME").last()

            if name == "oil":
                result["oil_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
            elif name == "cad_usd":
                result["cad_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
            elif name == "us10y":
                result["rate_chg_3m"] = monthly.diff(3).reindex(dates, method="ffill", limit=3)
            elif name == "tsx":
                result["tsx_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
            elif name == "gold":
                result["gold_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
            elif name == "vix":
                result["vix_level"] = monthly.reindex(dates, method="ffill", limit=3)
            elif name == "natgas":
                result["natgas_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
            elif name == "carbon":
                result["carbon_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
            elif name == "transport":
                result["transport_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
            elif name == "utilities_etf":
                result["util_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
            elif name == "inflation":
                result["tips_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
            elif name == "cad_bonds":
                result["cad_bond_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill", limit=3)
        except (KeyError, TypeError):
            pass

    return result



def encode_sector(ticker):
    """Numeric sector code for the model."""
    profile = STOCK_PROFILE.get(ticker, ("Unknown", "core", "other"))
    return SECTOR_NAME_TO_CODE.get(profile[0], 0)


def _attach_sector_etf_forward_returns(panel, macro_df):
    """Add `sector_etf_ret` column = ETF forward 1m return for the row's sector.

    For each (date, ticker), looks up the ticker's sector, then the
    forward 1-month return of that sector's ETF (XFN/XEG/ZIN/XUT).
    Forward return aligns with `fwd_ret`: at month t, both represent
    the realized return from t to t+1.
    """
    frames = []
    for sector, etf in SECTOR_ETF.items():
        code = SECTOR_NAME_TO_CODE.get(sector)
        if code is None:
            continue
        try:
            if isinstance(macro_df.columns, pd.MultiIndex):
                close = macro_df[(etf, "Close")].dropna()
            else:
                close = macro_df[etf]["Close"].dropna()
        except (KeyError, TypeError):
            continue
        monthly = close.resample("ME").last()
        fwd_ret = monthly.pct_change().shift(-1).rename("sector_etf_ret")
        f = fwd_ret.to_frame().reset_index().rename(columns={"index": "date", "Date": "date"})
        f["sector_code"] = code
        frames.append(f)

    if not frames:
        panel["sector_etf_ret"] = np.nan
        return panel

    etf_df = pd.concat(frames, ignore_index=True)
    panel = panel.merge(etf_df, on=["date", "sector_code"], how="left")
    return panel


def build_panel(price_df, macro_df, tickers):
    """Build the full feature panel for all tickers and months."""
    print("  Building feature panel...")
    all_frames = []

    for t in tickers:
        feat = compute_monthly_features(price_df, t)
        if feat is None:
            continue
        all_frames.append(feat)

    if not all_frames:
        return pd.DataFrame()

    panel = pd.concat(all_frames, axis=0)
    panel = panel.reset_index().rename(columns={"index": "date", "Date": "date"})

    # Macro features
    all_dates = panel["date"].unique()
    macro_feat = get_macro_features(macro_df, pd.DatetimeIndex(all_dates))
    panel = panel.merge(macro_feat, left_on="date", right_index=True, how="left")

    # External macro: real BOC overnight rate (from BOC Valet API). Fetched
    # once; falls through to NaN -> sector-median if the API is down.
    boc_feat = get_boc_features(pd.DatetimeIndex(all_dates))
    panel = panel.merge(boc_feat, left_on="date", right_index=True, how="left")

    # Sector code
    panel["sector_code"] = panel["ticker"].map(encode_sector)

    # Sector ETF FORWARD return per row (used as DML-PLR treatment).
    # Aligned to fwd_ret horizon: at month t, sector_etf_ret = ETF return
    # from t to t+1 (matching how add_labels constructs fwd_ret).
    panel = _attach_sector_etf_forward_returns(panel, macro_df)

    # Fundamental placeholders. compute_pit_fundamentals is intentionally
    # NOT wired up here: yfinance only returns 5-8 quarters of history
    # per ticker, so PIT data would only populate the last ~12-15 months
    # of each training window. That mixed-coverage signal hurt backtest
    # results (-3pp annualized) versus uniform sector-median imputation.
    # The function stays defined for future use if a deeper fundamentals
    # source becomes available.
    for col in ["roe", "pe_ratio", "div_yield", "ev_ebitda", "debt_equity"]:
        if col not in panel.columns:
            panel[col] = np.nan

    return panel


def add_labels(panel):
    """Add next-month return as label (regression target and classification)."""
    panel = panel.sort_values(["ticker", "date"])
    panel["fwd_ret"] = panel.groupby("ticker")["mom_1m"].shift(-1)
    panel = panel.dropna(subset=["fwd_ret"])
    return panel


def smart_impute(panel, feature_cols):
    """Cross-sectional median imputation by sector."""
    for col in feature_cols:
        if col in panel.columns and panel[col].isna().any():
            panel[col] = panel.groupby(["date", "sector_code"])[col].transform(
                lambda x: x.fillna(x.median())
            )
            # Global median fallback
            panel[col] = panel[col].fillna(panel[col].median())
    return panel


def apply_momentum_pca(panel, n_components=2):
    """Replace correlated momentum features with PCA components.

    Adds mom_pc1..mom_pcN columns to panel via PCA fit on the full
    panel's mom_{1,3,6,12}m columns. Originals stay in place because
    fwd_ret derives from mom_1m and the DML stage still needs a
    momentum signal — they just get filtered out of the model's
    feature list by main().

    The full-panel fit introduces a tiny look-ahead (a row's PC
    coordinates depend on the panel's overall covariance, including
    rows after that date). For a feature-space rotation rather than
    target leakage, this is a standard compromise; refit-per-window
    PCA would be more rigorous but adds significant complexity in
    walk_forward.
    """
    from sklearn.decomposition import PCA

    mom_cols = ["mom_1m", "mom_3m", "mom_6m", "mom_12m"]
    available = [c for c in mom_cols if c in panel.columns]
    if len(available) < n_components:
        return panel

    X = panel[available].fillna(0).values
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X)
    for i in range(n_components):
        panel[f"mom_pc{i+1}"] = pcs[:, i]
    return panel


def cross_sectional_normalize(panel, feature_cols, suffix="_norm"):
    """Add cross-sectional rank-normalized columns (per month) to panel.

    Writes new columns named `<col>{suffix}` rather than mutating the
    originals so downstream code (e.g. add_labels deriving fwd_ret from
    mom_1m) can still see raw values. sector_code passes through
    unchanged because it's a categorical, not a feature to normalize.

    Returns (panel, model_feature_cols), where model_feature_cols is the
    list of column names to feed the model.
    """
    model_cols = []
    for col in feature_cols:
        if col not in panel.columns:
            continue
        if col == "sector_code":
            model_cols.append(col)
            continue
        new_col = f"{col}{suffix}"
        panel[new_col] = panel.groupby("date")[col].transform(
            lambda x: (x.rank(pct=True) - 0.5) * 2
        )
        model_cols.append(new_col)
    return panel, model_cols


# ══════════════════════════════════════════════════════════════════
# MODELS — XGBoost
# ══════════════════════════════════════════════════════════════════

def make_xgb_regressor():
    """XGBoost regressor for return prediction."""
    return xgb.XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def make_xgb_classifier(pos_weight=1.0):
    """XGBoost classifier for top-quintile prediction."""
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=1.5,
        reg_lambda=3.0,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


# Model family for the per-sector learners. "xgb" is the baseline; the
# comparison harness flips this to test rf / extratrees / histgb / linear.
MODEL_KIND = "extratrees"

# ExtraTrees hyperparameters — read by make_regressor/make_classifier so that
# the hptest mode can temporarily override them without touching model code.
ET_HP = dict(n_estimators=300, max_depth=5, min_samples_leaf=10, max_features=0.7)


def make_regressor(kind=None):
    """Regressor for the requested model family (defaults to MODEL_KIND)."""
    kind = kind or MODEL_KIND
    if kind == "xgb":
        return make_xgb_regressor()
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, max_depth=5,
            min_samples_leaf=10, max_features=0.7, n_jobs=-1, random_state=42)
    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**ET_HP, n_jobs=-1, random_state=42)
    if kind == "histgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, max_depth=3,
            learning_rate=0.04, l2_regularization=1.0, random_state=42)
    if kind == "linear":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    if kind == "gbdt":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=300, max_depth=3,
            learning_rate=0.04, subsample=0.8, random_state=42)
    if kind == "adaboost":
        from sklearn.ensemble import AdaBoostRegressor
        return AdaBoostRegressor(n_estimators=200, learning_rate=0.05,
            random_state=42)
    if kind == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=300, depth=3, learning_rate=0.04,
            l2_leaf_reg=3.0, random_state=42, verbose=False)
    raise ValueError(f"unknown MODEL_KIND: {kind}")


def make_classifier(kind=None, pos_weight=1.0):
    """Top-quintile classifier for the requested model family."""
    kind = kind or MODEL_KIND
    if kind == "xgb":
        return make_xgb_classifier(pos_weight=pos_weight)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, max_depth=5,
            min_samples_leaf=10, max_features=0.7, class_weight="balanced",
            n_jobs=-1, random_state=42)
    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**ET_HP, class_weight="balanced",
            n_jobs=-1, random_state=42)
    if kind == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, max_depth=3,
            learning_rate=0.05, l2_regularization=1.0, random_state=42)
    if kind == "linear":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
    if kind == "gbdt":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=300, max_depth=3,
            learning_rate=0.05, subsample=0.8, random_state=42)
    if kind == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(n_estimators=200, learning_rate=0.05,
            random_state=42)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05,
            l2_leaf_reg=3.0, random_state=42, verbose=False, auto_class_weights="Balanced")
    raise ValueError(f"unknown MODEL_KIND: {kind}")


def fit_models(X_train, y_train, sample_weights=None):
    """Train a regressor + top-quintile classifier ensemble (MODEL_KIND)."""
    reg = make_regressor()
    reg.fit(X_train, y_train, sample_weight=sample_weights)

    y_cls = (y_train > y_train.quantile(0.8)).astype(int)
    pos_w = (len(y_cls) - y_cls.sum()) / max(y_cls.sum(), 1)
    clf = make_classifier(pos_weight=pos_w)
    clf.fit(X_train, y_cls, sample_weight=sample_weights)

    return reg, clf


def ensemble_predict(reg, clf, X):
    """50/50 blend of regression rank and classification probability."""
    pred_reg = reg.predict(X)
    pred_cls = clf.predict_proba(X)[:, 1]

    # Rank-normalize each
    from scipy.stats import rankdata
    rank_reg = rankdata(pred_reg) / len(pred_reg)
    rank_cls = rankdata(pred_cls) / len(pred_cls)

    return 0.5 * rank_reg + 0.5 * rank_cls


def _resolve_sector_features(base_features, panel_columns):
    """Map base feature names to their normalized counterparts.

    `cross_sectional_normalize` writes <name>_norm columns; for
    non-normalized columns (sector_code, fundamentals, growth signals
    that aren't in FEATURE_COLS) the base name passes through.
    """
    resolved = []
    for f in base_features:
        norm = f + "_norm"
        if norm in panel_columns:
            resolved.append(norm)
        elif f in panel_columns:
            resolved.append(f)
    return resolved


def estimate_sector_dml_theta(sec_df, features, treatment_col="sector_etf_ret",
                                outcome_col="fwd_ret", n_splits=5):
    """Closed-form DML-PLR estimate of sector beta theta.

    Model: Y = theta * D + g(X) + epsilon
      Y = fwd_ret (per-stock forward return)
      D = sector ETF forward return (same horizon, scalar per date)
      X = sector feature controls (the per-sector subset)

    Steps:
      1. Cross-fit g(X) and h(X) on TimeSeriesSplit-by-date folds
      2. Residualize: Y_tilde = Y - g(X), D_tilde = D - h(X)
      3. Closed form: theta_hat = (D_tilde . Y_tilde) / (D_tilde . D_tilde)
      4. Score-based SE: var = mean(psi^2) / (J^2 * n), J = mean(D_tilde^2),
         psi = D_tilde * (Y_tilde - theta * D_tilde)

    Returns dict{theta, se, t_stat, p_value, n_obs} or None on failure
    (insufficient samples, degenerate residuals).
    """
    from scipy.stats import norm
    # Only require non-NaN outcome and treatment; XGBoost handles NaN
    # features natively (the fundamentals columns roe/pe_ratio/etc.
    # are intentionally all-NaN placeholders, so requiring them dense
    # would empty the panel).
    df = sec_df.dropna(subset=[outcome_col, treatment_col])
    if len(df) < 50:
        return None
    # Drop feature columns that are entirely NaN within this sector —
    # they add nothing and break XGBoost's column-pruning.
    features = [f for f in features if df[f].notna().any()]
    if not features:
        return None

    # Keep dates as a homogeneous datetime64 array so np.isin works
    # between unique sorted dates and the per-row date column. Going
    # through np.array(sorted(.unique())) downgraded to object/Timestamp
    # and silently produced all-False masks.
    dates = np.sort(df["date"].unique())
    if len(dates) < 4:
        return None
    n_splits = max(2, min(n_splits, len(dates) - 2))

    Y = df[outcome_col].values.astype(float)
    D = df[treatment_col].values.astype(float)
    X = df[features].values.astype(float)
    df_dates = df["date"].values

    Y_tilde = np.full(len(df), np.nan)
    D_tilde = np.full(len(df), np.nan)

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)
    for tr_di, te_di in tscv.split(dates):
        tr_dates, te_dates = dates[tr_di], dates[te_di]
        tr_mask = np.isin(df_dates, tr_dates)
        te_mask = np.isin(df_dates, te_dates)
        if not tr_mask.any() or not te_mask.any():
            continue
        try:
            my = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                  verbosity=0, n_jobs=-1, random_state=42)
            my.fit(X[tr_mask], Y[tr_mask])
            Y_tilde[te_mask] = Y[te_mask] - my.predict(X[te_mask])
            md = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                  verbosity=0, n_jobs=-1, random_state=42)
            md.fit(X[tr_mask], D[tr_mask])
            D_tilde[te_mask] = D[te_mask] - md.predict(X[te_mask])
        except Exception:
            continue

    valid = ~(np.isnan(Y_tilde) | np.isnan(D_tilde))
    if valid.sum() < 30 or np.var(D_tilde[valid]) < 1e-12:
        return None

    yt = Y_tilde[valid]
    dt = D_tilde[valid]

    theta = float(np.dot(dt, yt) / np.dot(dt, dt))
    psi = dt * (yt - theta * dt)
    J = float(np.mean(dt ** 2))
    var = float(np.mean(psi ** 2)) / (J ** 2 * len(yt) + 1e-12)
    se = float(np.sqrt(var))
    t_stat = float(theta / (se + 1e-12))
    p_value = float(2 * (1 - norm.cdf(abs(t_stat))))

    return {"theta": theta, "se": se, "t_stat": t_stat,
            "p_value": p_value, "n_obs": int(valid.sum())}


def fit_sector_models(train_df, sample_weights=None, min_samples=50):
    """Train one XGBoost ensemble per sector using SECTOR_FEATURES.

    For each sector with a sector ETF mapping, first estimates theta via
    DML-PLR (estimate_sector_dml_theta) and trains the XGBoost on alpha
    residuals (fwd_ret - theta * sector_etf_ret) instead of raw fwd_ret.
    The model learns pure idiosyncratic alpha; sector beta is stripped
    upfront. If DML fails (too few samples, degenerate residuals), the
    sector falls back to the raw fwd_ret target.

    Returns (models, dml_stats) where:
      models[sector_name] = (regressor, classifier, feature_cols)
      dml_stats[sector_name] = {theta, se, t_stat, p_value, n_obs}
        (only present for sectors where DML succeeded)
    """
    models = {}
    dml_stats = {}
    for sector_name, base_features in SECTOR_FEATURES.items():
        code = SECTOR_NAME_TO_CODE.get(sector_name)
        if code is None:
            continue
        sec_mask = (train_df["sector_code"] == code).values
        sec_df = train_df.loc[sec_mask]
        if len(sec_df) < min_samples:
            continue
        feats = _resolve_sector_features(base_features, train_df.columns)
        if not feats:
            continue

        # ETF-baseline DML for this sector (if the ETF column is available)
        stats = None
        if sector_name in SECTOR_ETF and "sector_etf_ret" in sec_df.columns:
            stats = estimate_sector_dml_theta(sec_df, feats)

        if stats is not None:
            theta = stats["theta"]
            etf_ret = sec_df["sector_etf_ret"].fillna(0.0).values
            y = sec_df["fwd_ret"].values - theta * etf_ret
            dml_stats[sector_name] = stats
        else:
            y = sec_df["fwd_ret"].values

        X = sec_df[feats].values
        w = sample_weights[sec_mask] if sample_weights is not None else None
        try:
            reg, clf = fit_models(
                pd.DataFrame(X, columns=feats),
                pd.Series(y),
                sample_weights=w,
            )
            models[sector_name] = (reg, clf, feats)
        except Exception:
            continue
    return models, dml_stats


def health_check(latest_df, train_df, sec_models, sec_dml_stats, dml_p_threshold=0.10):
    """Diagnose this month's prediction reliability.

    Five tests, mirroring monthly_rank.py's health-check pattern:
      1. Sector coverage: every required_sector has a trained model
      2. Signal strength: max |z-score within sector| >= 0.5
      3. DML significance: at least 2 sectors have p < threshold
      4. Data completeness: every candidate has a non-NaN score
      5. Feature drift: this month's RAW feature values stay within the
         training distribution (|z| <= train 95th pct x 1.5)

    Test 5 deliberately uses the raw feature columns, not the _norm
    ones: cross_sectional_normalize produces per-month rank values that
    are bounded in [-1, 1] by construction and so can never "drift". The
    raw columns (vol_20d, mom_1m, ...) carry real units and reveal when
    the current month is in a market state the models never trained on.

    Returns list of (label, ok, detail) tuples for display.
    """
    checks = []

    expected = set(SECTOR_FEATURES.keys())
    trained = set(sec_models.keys())
    missing = expected - trained
    detail = f"trained {len(trained)}/{len(expected)} sectors"
    if missing:
        detail += f" (missing: {sorted(missing)})"
    checks.append(("Sector coverage", not missing, detail))

    max_z = 0.0
    if "score" in latest_df.columns:
        for code in latest_df["sector_code"].unique():
            mask = (latest_df["sector_code"] == code).values
            scores = latest_df.loc[mask, "score"].dropna().values
            if len(scores) < 2:
                continue
            std = scores.std()
            if std > 0:
                z = float(np.max(np.abs((scores - scores.mean()) / std)))
                max_z = max(max_z, z)
    checks.append(("Signal strength", max_z >= 0.5,
                   f"max |z-score within sector| = {max_z:.2f} (>= 0.5)"))

    n_sig = sum(1 for st in sec_dml_stats.values()
                if st["p_value"] < dml_p_threshold)
    checks.append(("DML significance", n_sig >= 2,
                   f"{n_sig}/{len(sec_dml_stats)} sector betas significant "
                   f"at p<{dml_p_threshold}"))

    n_total = len(latest_df)
    n_ok = int(latest_df["score"].notna().sum()) if "score" in latest_df.columns else 0
    checks.append(("Data completeness", n_ok == n_total and n_total > 0,
                   f"{n_ok}/{n_total} candidates have scores"))

    # Feature drift: for each RAW (non-_norm, non-categorical) feature
    # the sector models consume, compare THIS MONTH's cross-sectional
    # mean against the distribution of monthly cross-sectional means in
    # the training period. Using the monthly mean (not a single stock's
    # max) keeps one outlier name from tripping the alarm — it flags a
    # genuine market-regime shift (e.g. a market-wide vol spike) the
    # models never trained on. |z| > 3 (~99.7%) keeps it to real shifts.
    base_feats = set()
    for _reg, _clf, feats in sec_models.values():
        for f in feats:
            base = f[:-5] if f.endswith("_norm") else f
            if base != "sector_code":
                base_feats.add(base)
    drift = []
    for f in sorted(base_feats):
        if f not in train_df.columns or f not in latest_df.columns:
            continue
        tr_means = train_df.groupby("date")[f].mean()
        tr_means = tr_means[np.isfinite(tr_means)]
        latest_mean = latest_df[f].mean()
        if len(tr_means) < 24 or tr_means.std() <= 0 or not np.isfinite(latest_mean):
            continue
        z = abs((latest_mean - tr_means.mean()) / tr_means.std())
        if z > 3.0:
            drift.append(f"{f}(z={z:.1f})")
    detail = (f"{len(drift)} regime-shifted (|z|>3): {', '.join(drift)}"
              if drift else "all features within train range")
    checks.append(("Feature drift", len(drift) == 0, detail))

    return checks


def print_health_check(checks, title="Health check"):
    print(f"\n  {title}:")
    for label, ok, detail in checks:
        marker = "OK" if ok else "!!"
        print(f"    [{marker}] {label}: {detail}")
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    if n_fail == 0:
        print("    All checks passed.")
    else:
        print(f"    {n_fail} warning(s).")


def predict_sector_models(test_df, sector_models):
    """Score test_df rows via each row's sector model.

    Returns np.ndarray of scores positionally aligned to test_df. Rows
    whose sector lacks a trained model get NaN — the caller can either
    drop them or fill with 0 (excluded from picks via rebalancing band).
    """
    scores = np.full(len(test_df), np.nan)
    for sector_name, (reg, clf, feats) in sector_models.items():
        code = SECTOR_NAME_TO_CODE.get(sector_name)
        if code is None:
            continue
        mask = (test_df["sector_code"] == code).values
        if not mask.any():
            continue
        X = test_df.loc[mask, feats].values
        scores[mask] = ensemble_predict(reg, clf, X)
    return scores


# ══════════════════════════════════════════════════════════════════
# PERMUTATION IMPORTANCE
# ══════════════════════════════════════════════════════════════════

PERM_IMPORTANCE_FILE = "perm_importance.json"


def _perm_importance_fold(sec_models, test_df, n_repeats=5, rng=None):
    """Permutation importance on one walk-forward test fold.

    Shuffles each feature in test_df one at a time, re-scores all
    non-benchmark stocks via predict_sector_models, and measures the
    RankIC drop vs the baseline (higher drop = feature matters more).
    Returns {feature: mean_ic_drop} or {} when the fold is too small.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    mask = test_df["ticker"] != "XIU.TO"
    df_eval = test_df[mask].copy().reset_index(drop=True)

    if len(df_eval) < 5 or df_eval["fwd_ret"].isna().all():
        return {}

    # Collect all features used across all sector models (dedup, preserve order)
    seen = set()
    all_feats = []
    for _, (_, _, feats) in sec_models.items():
        for f in feats:
            if f not in seen:
                all_feats.append(f)
                seen.add(f)

    # Baseline RankIC
    base_scores = predict_sector_models(df_eval, sec_models)
    valid = ~np.isnan(base_scores) & ~df_eval["fwd_ret"].isna().values
    if valid.sum() < 5:
        return {}
    base_ic, _ = spearmanr(base_scores[valid], df_eval.loc[valid, "fwd_ret"])
    if np.isnan(base_ic):
        return {}

    importance = {}
    for feat in all_feats:
        if feat not in df_eval.columns:
            continue
        # sector_code is only a routing key in predict_sector_models, not a
        # model feature — shuffling it destroys sector routing rather than
        # measuring signal contribution, so skip it.
        if feat == "sector_code":
            continue
        # Macro/time-series features (oil price, VIX, rate changes, etc.) have
        # identical values for all stocks in a given month. Permuting within a
        # single-month fold shuffles identical values → IC drop is always 0,
        # which is not a real importance signal. Skip unmeasurable features.
        if df_eval[feat].nunique() <= 1:
            continue
        orig = df_eval[feat].values.copy()
        drops = []
        for _ in range(n_repeats):
            df_eval[feat] = orig[rng.permutation(len(orig))]
            shuf_scores = predict_sector_models(df_eval, sec_models)
            ic, _ = spearmanr(shuf_scores[valid], df_eval.loc[valid, "fwd_ret"])
            if not np.isnan(ic):
                drops.append(base_ic - ic)
        df_eval[feat] = orig  # restore
        importance[feat] = float(np.mean(drops)) if drops else 0.0

    return importance


def save_perm_importance(importance_dict, path=PERM_IMPORTANCE_FILE):
    """Cache permutation importance to JSON so predict_now can use it."""
    try:
        with open(path, "w") as f:
            json.dump(importance_dict, f, indent=2)
    except Exception:
        pass


def load_perm_importance(path=PERM_IMPORTANCE_FILE):
    """Load cached permutation importance, or None if absent/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# DOUBLE MACHINE LEARNING (Chernozhukov 2018)
# ══════════════════════════════════════════════════════════════════

def estimate_dml_alpha(panel, feature_cols, treatment_col, outcome_col="fwd_ret", n_splits=5):
    """
    Estimate causal effect of treatment on outcome, controlling for features.

    Steps:
      1. Predict Y from X -> residual E_Y
      2. Predict T from X -> residual E_T
      3. OLS: E_Y ~ E_T -> theta (causal effect)

    Uses TimeSeriesSplit to prevent temporal leakage.
    """
    df = panel.dropna(subset=[treatment_col, outcome_col] + feature_cols).copy()
    if len(df) < 100:
        return 0.0

    X = df[feature_cols].values
    Y = df[outcome_col].values
    T = df[treatment_col].values

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)

    residual_y = np.zeros(len(df))
    residual_t = np.zeros(len(df))
    mask = np.zeros(len(df), dtype=bool)

    for train_idx, test_idx in tscv.split(X):
        # Model Y ~ X
        model_y = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            verbosity=0, n_jobs=-1, random_state=42
        )
        model_y.fit(X[train_idx], Y[train_idx])
        residual_y[test_idx] = Y[test_idx] - model_y.predict(X[test_idx])

        # Model T ~ X
        model_t = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            verbosity=0, n_jobs=-1, random_state=42
        )
        model_t.fit(X[train_idx], T[train_idx])
        residual_t[test_idx] = T[test_idx] - model_t.predict(X[test_idx])

        mask[test_idx] = True

    # OLS on residuals
    e_y = residual_y[mask]
    e_t = residual_t[mask]

    if np.std(e_t) < 1e-8:
        return 0.0

    theta = np.sum(e_t * e_y) / np.sum(e_t ** 2)
    return np.clip(theta, -0.20, 0.20)


def apply_dml_adjustment(scores, panel_latest, dml_thetas):
    """Apply DML causal adjustments to ensemble scores."""
    adjusted = scores.copy()
    for treatment_col, theta in dml_thetas.items():
        if treatment_col in panel_latest.columns and abs(theta) > 0.01:
            signal = panel_latest[treatment_col].values
            signal_norm = (signal - np.nanmean(signal)) / (np.nanstd(signal) + 1e-8)
            adjusted += theta * signal_norm
    return adjusted


# ══════════════════════════════════════════════════════════════════
# MARKET REGIME DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_regime(macro_df):
    """Detect market regime: BULL / NEUTRAL / BEAR."""
    try:
        vix = macro_df[("^VIX", "Close")].dropna()
    except (KeyError, TypeError):
        try:
            vix = macro_df["^VIX"]["Close"].dropna()
        except (KeyError, TypeError):
            return "NEUTRAL", {}

    try:
        tsx = macro_df[("^GSPTSE", "Close")].dropna()
    except (KeyError, TypeError):
        try:
            tsx = macro_df["^GSPTSE"]["Close"].dropna()
        except (KeyError, TypeError):
            return "NEUTRAL", {}

    current_vix = vix.iloc[-1]
    tsx_ma200 = tsx.rolling(200).mean().iloc[-1]
    tsx_current = tsx.iloc[-1]

    # Regime modulates only top_n; max_per_gics stays at 2 per the
    # user's 1-2-per-sector rule. BEAR shrinks to 4 (1 per sector);
    # BULL allows the full 8 (2 per sector). NEUTRAL = default top_n.
    if current_vix > 25 and tsx_current < tsx_ma200:
        regime = "BEAR"
        adjustments = {"top_n": 4, "min_confidence": 0.20}
    elif current_vix < 15 and tsx_current > tsx_ma200:
        regime = "BULL"
        adjustments = {"top_n": 8, "min_confidence": 0.10}
    else:
        regime = "NEUTRAL"
        adjustments = {}

    return regime, adjustments


# ══════════════════════════════════════════════════════════════════
# REBALANCING BAND
# ══════════════════════════════════════════════════════════════════

def apply_rebalancing_band(new_picks, new_scores, current_holdings, constraints=None):
    """Pick top stocks under sector min/max constraints.

    Rules enforced (in order):
      1. Every sector in `required_sectors` gets at least 1 pick
         (highest-scoring candidate from that sector).
      2. Remaining slots filled by score, capped at `max_per_gics` per
         sector.
      3. Total picks <= `top_n`.

    Falls back to plain top_n if `required_sectors` is empty (legacy).
    """
    C = constraints or CONSTRAINTS
    if not new_picks:
        return []

    top_n = C["top_n"]
    max_per_gics = C["max_per_gics"]
    required_sectors = C.get("required_sectors") or []

    score_dict = dict(zip(new_picks, new_scores))

    ranked = sorted(score_dict.items(), key=lambda x: -x[1])

    def sector_of(t):
        return STOCK_PROFILE.get(t, ("Unknown",))[0]

    if not required_sectors:
        # Legacy: no sector requirements — return top_n by score
        return [t for t, _ in ranked[:top_n]]

    final = []
    sector_count = {}

    # Phase 1: guarantee 1 per required sector (highest-scoring there)
    for sector in required_sectors:
        for t, _ in ranked:
            if t in final:
                continue
            if sector_of(t) == sector:
                final.append(t)
                sector_count[sector] = 1
                break

    # Phase 2: fill remaining slots by score, capping each sector
    for t, _ in ranked:
        if len(final) >= top_n:
            break
        if t in final:
            continue
        sec = sector_of(t)
        if sec not in required_sectors:
            continue  # universe should already exclude these, defensive
        if sector_count.get(sec, 0) >= max_per_gics:
            continue
        final.append(t)
        sector_count[sec] = sector_count.get(sec, 0) + 1

    return final[:top_n]


# ══════════════════════════════════════════════════════════════════
# POSITION SIZING — Risk Parity
# ══════════════════════════════════════════════════════════════════

def risk_parity_weights(tickers, price_df, lookback=60):
    """Inverse-volatility weighting."""
    vols = {}
    for t in tickers:
        close, _ = get_ohlcv(price_df, t)
        if close is None:
            vols[t] = 0.20  # default 20% vol
            continue
        daily_ret = close.pct_change().tail(lookback)
        vols[t] = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 20 else 0.20

    inv_vol = {t: 1.0 / max(v, 0.05) for t, v in vols.items()}
    total = sum(inv_vol.values())
    weights = {t: v / total for t, v in inv_vol.items()}

    # Cap max allocation
    max_alloc = CONSTRAINTS["max_single_alloc"]
    for t in weights:
        weights[t] = min(weights[t], max_alloc)

    # Renormalize
    total = sum(weights.values())
    weights = {t: v / total for t, v in weights.items()}
    return weights


def estimate_portfolio_te(picks, weights, price_df, benchmark="XIU.TO", lookback=252):
    """Estimate annualized tracking error from last `lookback` trading days.

    Equal-weights any ticker missing from the weights dict. Returns NaN if
    fewer than 30 overlapping days are available.
    """
    try:
        bench_close, _ = get_ohlcv(price_df, benchmark)
        if bench_close is None:
            return np.nan
        bench_ret = bench_close.pct_change().dropna().tail(lookback)

        port_daily = None
        for ticker in picks:
            close, _ = get_ohlcv(price_df, ticker)
            if close is None:
                continue
            ret = close.pct_change().dropna().tail(lookback)
            w = weights.get(ticker, 1.0 / max(len(picks), 1))
            port_daily = ret * w if port_daily is None else port_daily.add(ret * w, fill_value=0)

        if port_daily is None:
            return np.nan

        aligned = pd.concat([port_daily.rename("port"), bench_ret.rename("bench")],
                            axis=1).dropna()
        if len(aligned) < 30:
            return np.nan

        return (aligned["port"] - aligned["bench"]).std() * np.sqrt(252)
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════

def walk_forward(panel, feature_cols, train_months=36, min_train=24,
                 return_perstock=False, score_mode="model",
                 embargo_months=1, return_importance=False,
                 return_raw_importance=False, expanding=False,
                 half_life=6):
    """
    Walk-forward backtester with rolling window (default) or expanding window.

    expanding=False (default): rolling window of train_months months.
    expanding=True: use all data from the start of the panel up to the
        embargo cutoff; min_train still enforces the minimum fold size.

    embargo_months: months excluded between end of training and test date.
    Prevents label leakage — the last training row's fwd_ret overlaps with
    the test period without the gap.

    When return_perstock=True, also returns a per-(month, ticker)
    DataFrame (date, ticker, score, fwd_ret, sector_code, is_selected)
    for segmented RankIC diagnostics. Default False keeps the original
    single-DataFrame return for existing callers (smoke_test, etc.).

    score_mode="random" replaces model scores with random values (a
    diagnostic baseline that isolates the stock-picking contribution from
    the sector-constraint structure + regime); "model" (default) uses the
    trained models.

    return_importance=True accumulates per-fold permutation importance
    (OOS RankIC drop per feature) and returns it as a third value alongside
    results_df and perstock_df. Only implemented for USE_SECTOR_MODELS=True.
    """
    panel = panel.sort_values("date")
    dates = sorted(panel["date"].unique())

    results = []
    perstock = []
    holdings = []
    fold_importances = {}  # feat -> list of per-fold IC drops
    _perm_rng = np.random.RandomState(42)

    mode_str = "expanding" if expanding else f"train={train_months}m"
    print(f"  Walk-forward: {len(dates)} months, {mode_str}, embargo={embargo_months}m")

    for i in range(train_months, len(dates) - 1):
        # Expanding window: always train from the beginning; rolling: last train_months
        train_start = 0 if expanding else max(0, i - train_months)
        # Embargo: exclude the `embargo_months` months immediately before
        # the test date so no training label overlaps with the test period.
        train_end = max(train_start, i - embargo_months)
        train_dates = dates[train_start:train_end]
        test_date = dates[i]

        train_df = panel[panel["date"].isin(train_dates)].copy()
        test_df = panel[panel["date"] == test_date].copy()

        if len(train_df) < min_train * 5 or len(test_df) < 3:
            continue

        # Time decay weights
        n = len(train_df)
        weights = compute_time_decay_weights(n, half_life_months=half_life)

        # Fit and predict — per-sector or global depending on the flag
        try:
            if USE_SECTOR_MODELS:
                sec_models, _dml_stats = fit_sector_models(train_df, sample_weights=weights)
                scores = predict_sector_models(test_df, sec_models)
                scores = np.nan_to_num(scores, nan=0.0)
                if return_importance:
                    fold_imp = _perm_importance_fold(sec_models, test_df, rng=_perm_rng)
                    for feat, drop in fold_imp.items():
                        fold_importances.setdefault(feat, []).append(drop)
            else:
                reg, clf = fit_models(
                    pd.DataFrame(train_df[feature_cols].values, columns=feature_cols),
                    pd.Series(train_df["fwd_ret"].values),
                    sample_weights=weights
                )
                scores = ensemble_predict(reg, clf, test_df[feature_cols].values)
        except Exception:
            continue

        # Diagnostic baseline: replace model scores with random ones to
        # isolate how much the model's stock-picking actually contributes
        # vs the sector-constraint structure + regime. Per-month seed for
        # reproducibility.
        if score_mode == "random":
            scores = np.random.RandomState(1000 + i).rand(len(test_df))

        # Rank and select
        test_df = test_df.copy()
        test_df["score"] = scores
        test_df = test_df.sort_values("score", ascending=False)

        # Apply rebalancing band
        top_tickers = test_df["ticker"].tolist()
        top_scores = test_df["score"].tolist()
        picks = apply_rebalancing_band(top_tickers, top_scores, holdings)

        # DML adjustments: collect causal alphas from multiple treatments.
        # Momentum treatment routes through whichever momentum column is
        # actually in the feature set (mom_pc1_norm under PCA mode,
        # mom_1m_norm otherwise). Earnings_surprise is raw % and sits
        # outside the feature set as a pure treatment.
        if "mom_pc1_norm" in feature_cols:
            mom_treatment = "mom_pc1_norm"
        elif "mom_1m_norm" in feature_cols:
            mom_treatment = "mom_1m_norm"
        else:
            mom_treatment = None
        candidate_treatments = []
        if mom_treatment is not None:
            candidate_treatments.append(mom_treatment)
        if "earnings_surprise" in train_df.columns:
            candidate_treatments.append("earnings_surprise")

        dml_thetas = {}
        if len(train_df) > 200:
            for t in candidate_treatments:
                theta = estimate_dml_alpha(train_df, feature_cols, t)
                if abs(theta) > 0.01:
                    dml_thetas[t] = theta

        if dml_thetas:
            print(f"    {test_date.strftime('%Y-%m')}: DML alphas = {dml_thetas}")
            pick_mask = test_df["ticker"].isin(picks)
            dml_adj = apply_dml_adjustment(
                test_df.loc[pick_mask, "score"].values,
                test_df.loc[pick_mask],
                dml_thetas
            )
            pick_df = test_df.loc[pick_mask].copy()
            pick_df["adj_score"] = dml_adj
            picks = pick_df.sort_values("adj_score", ascending=False)["ticker"].tolist()[:CONSTRAINTS["top_n"]]

        holdings = picks

        if return_perstock:
            ps = test_df[test_df["ticker"] != "XIU.TO"][
                ["date", "ticker", "score", "fwd_ret", "sector_code"]].copy()
            ps["is_selected"] = ps["ticker"].isin(picks)
            perstock.append(ps)

        # Record actual returns. fwd_ret was computed from raw monthly
        # returns BEFORE cross_sectional_normalize rewrote mom_1m to a rank,
        # so it's the only column that still holds true forward returns.
        pick_rets = test_df[test_df["ticker"].isin(picks)]["fwd_ret"]
        bench_row = test_df[test_df["ticker"] == "XIU.TO"]["fwd_ret"]

        port_ret_gross = pick_rets.mean() if len(pick_rets) > 0 else 0.0
        results.append({
            "date": test_date,
            "picks": picks,
            "port_ret": port_ret_gross,
            "bench_ret": bench_row.iloc[0] if len(bench_row) > 0 else 0,
            "n_picks": len(picks),
        })

    results_df = pd.DataFrame(results)
    importance_dict = {f: float(np.mean(v)) for f, v in fold_importances.items()}
    perstock_df = pd.concat(perstock, ignore_index=True) if perstock else pd.DataFrame()

    if return_raw_importance:
        # Return raw per-fold lists for FDR computation
        if return_perstock and return_importance:
            return results_df, perstock_df, importance_dict, fold_importances
        if return_importance:
            return results_df, importance_dict, fold_importances
        return results_df, fold_importances

    if return_perstock and return_importance:
        return results_df, perstock_df, importance_dict
    if return_perstock:
        return results_df, perstock_df
    if return_importance:
        return results_df, importance_dict
    return results_df


# ══════════════════════════════════════════════════════════════════
# COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
# ══════════════════════════════════════════════════════════════════

def compute_cpcv(panel, feature_cols, n_folds=6, embargo_months=1):
    """Combinatorial Purged Cross-Validation (López de Prado 2018).

    Divides T months into n_folds groups; for each of C(n_folds, 2) = 15
    paths, trains on (n_folds−2) groups (purged around test boundaries)
    and tests on the remaining 2. Fits ONE model per path (not rolling).
    Returns a list of per-path dicts with Sharpe, annualized return, max DD.

    Purpose: detect overfitting that walk-forward may miss. Walk-forward
    reuses adjacent train/test splits; CPCV tests arbitrary time-period
    combinations, giving a distribution of out-of-sample Sharpe ratios.
    """
    panel = panel.sort_values("date")
    all_dates = sorted(panel["date"].unique())
    T = len(all_dates)

    fold_size = T // n_folds
    groups = [list(all_dates[i * fold_size: (i + 1) * fold_size if i < n_folds - 1 else T])
              for i in range(n_folds)]

    path_results = []

    for test_ids in combinations(range(n_folds), 2):
        test_set = set()
        for gid in test_ids:
            test_set.update(groups[gid])
        test_sorted = sorted(test_set)

        # Purge: exclude dates within embargo_months of any test-group boundary
        boundaries = [groups[gid][0] for gid in test_ids] + \
                     [groups[gid][-1] for gid in test_ids]
        embargo_days = embargo_months * 32

        train_dates = [
            d for d in all_dates
            if d not in test_set
            and all(abs((d - b).days) > embargo_days for b in boundaries)
        ]

        if len(set(train_dates)) < 24:
            continue

        train_df = panel[panel["date"].isin(set(train_dates))].copy()
        if len(train_df) < 24 * 5:
            continue

        n = len(train_df)
        weights = compute_time_decay_weights(n)

        try:
            if USE_SECTOR_MODELS:
                sec_models, _ = fit_sector_models(train_df, sample_weights=weights)
            else:
                reg, clf = fit_models(
                    pd.DataFrame(train_df[feature_cols].values, columns=feature_cols),
                    pd.Series(train_df["fwd_ret"].values),
                    sample_weights=weights
                )
        except Exception:
            continue

        monthly_rets = []
        holdings = []

        for test_date in test_sorted:
            test_df = panel[panel["date"] == test_date].copy()
            if len(test_df) < 3:
                continue
            try:
                if USE_SECTOR_MODELS:
                    scores = predict_sector_models(test_df, sec_models)
                else:
                    scores = ensemble_predict(reg, clf, test_df[feature_cols].values)
                scores = np.nan_to_num(scores, nan=0.0)
            except Exception:
                continue

            test_df = test_df.copy()
            test_df["score"] = scores
            ranked = test_df.sort_values("score", ascending=False)
            picks = apply_rebalancing_band(
                ranked["ticker"].tolist(), ranked["score"].tolist(), holdings
            )

            holdings = picks

            rets = test_df[test_df["ticker"].isin(picks)]["fwd_ret"]
            monthly_rets.append(rets.mean() if len(rets) > 0 else 0.0)

        if len(monthly_rets) < 8:
            continue

        r = np.array(monthly_rets)
        sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0.0
        ann_ret = (1 + r.mean()) ** 12 - 1
        cum = np.cumprod(1 + r)
        roll_max = np.maximum.accumulate(cum)
        max_dd = float(np.min((cum - roll_max) / roll_max))

        path_results.append({
            "test_groups": test_ids,
            "n_months": len(monthly_rets),
            "sharpe": round(sharpe, 3),
            "ann_ret": round(ann_ret, 4),
            "max_dd": round(max_dd, 4),
        })

    return path_results


def print_cpcv_report(path_results, wf_sharpe=None):
    """Print CPCV Sharpe distribution across combinatorial paths."""
    if not path_results:
        print("  No CPCV paths with enough data.")
        return
    sharpes = [p["sharpe"] for p in path_results]
    print(f"\n{'═' * 60}")
    print(f"  COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)")
    print(f"  {len(path_results)} paths  |  each: 1 model fit, ~28 test months")
    print(f"{'═' * 60}")
    print(f"  {'Path':<12}  {'Months':>6}  {'Sharpe':>8}  {'Ann.Ret':>8}  {'MaxDD':>8}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
    for p in sorted(path_results, key=lambda x: -x["sharpe"]):
        grps = f"G{p['test_groups'][0]}&G{p['test_groups'][1]}"
        print(f"  {grps:<12}  {p['n_months']:>6}  {p['sharpe']:>8.3f}  "
              f"{p['ann_ret']:>7.1%}  {p['max_dd']:>7.1%}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'Mean':<12}  {'':>6}  {np.mean(sharpes):>8.3f}")
    print(f"  {'Std':<12}  {'':>6}  {np.std(sharpes):>8.3f}")
    print(f"  {'Min':<12}  {'':>6}  {np.min(sharpes):>8.3f}")
    print(f"  {'Max':<12}  {'':>6}  {np.max(sharpes):>8.3f}")
    pct_positive = 100 * np.mean([s > 0 for s in sharpes])
    pct_above1 = 100 * np.mean([s > 1.0 for s in sharpes])
    print(f"  Sharpe > 0:   {pct_positive:.0f}% of paths")
    print(f"  Sharpe > 1.0: {pct_above1:.0f}% of paths")
    if wf_sharpe is not None:
        print(f"  Walk-forward: {wf_sharpe:.3f} (for comparison)")
    print(f"{'═' * 60}")
    print(f"  Note: CPCV fits ONE model per path (not rolling). Sharpe")
    print(f"  distribution shows generalization across time periods.")


# ══════════════════════════════════════════════════════════════════
# FDR — FALSE DISCOVERY RATE ON FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════

def compute_fdr_importance(fold_importances, alpha=0.05):
    """Benjamini-Hochberg FDR correction on permutation importance.

    For each feature, runs a one-sample t-test (H0: mean IC drop = 0)
    on the per-fold IC drops accumulated during walk_forward. Applies
    the BH procedure at the given alpha level.

    Returns list of (feature, mean_drop, p_value, bh_significant) sorted
    by mean IC drop descending.
    """
    results = []
    for feat, drops in fold_importances.items():
        drops_arr = np.array(drops)
        if len(drops_arr) < 3:
            results.append((feat, float(np.mean(drops_arr)), 1.0, False))
            continue
        _, p = ttest_1samp(drops_arr, 0.0, alternative="greater")
        results.append((feat, float(np.mean(drops_arr)), float(p), False))

    # Benjamini-Hochberg
    results.sort(key=lambda x: x[2])  # sort by p-value
    m = len(results)
    bh_threshold = [alpha * (i + 1) / m for i in range(m)]
    significant_mask = [results[i][2] <= bh_threshold[i] for i in range(m)]
    # All ranks up to the last significant rank are significant
    last_sig = -1
    for i in range(m - 1, -1, -1):
        if significant_mask[i]:
            last_sig = i
            break
    for i in range(m):
        feat, mean_drop, p, _ = results[i]
        results[i] = (feat, mean_drop, p, i <= last_sig)

    results.sort(key=lambda x: -x[1])  # sort by mean IC drop
    return results


def print_fdr_importance(fdr_results):
    """Print FDR-corrected permutation importance table."""
    if not fdr_results:
        print("  No FDR results.")
        return
    n_sig = sum(1 for _, _, _, s in fdr_results if s)
    print(f"\n{'─' * 62}")
    print(f"  FDR-CORRECTED PERMUTATION IMPORTANCE  (BH α=0.05)")
    print(f"  {n_sig}/{len(fdr_results)} features statistically significant")
    print(f"{'─' * 62}")
    print(f"  {'Feature':<28}  {'IC Drop':>8}  {'p-val':>7}  {'Sig?'}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*7}  {'─'*5}")
    for feat, mean_drop, p, sig in fdr_results:
        name = feat.replace("_norm", "")
        marker = " ✓" if sig else ""
        print(f"  {name:<28}  {mean_drop:>+8.4f}  {p:>7.4f}  {marker}")
    print(f"{'─' * 62}")
    print(f"  ✓ = survives Benjamini-Hochberg FDR correction at α=0.05")
    print(f"  H0 per feature: mean OOS IC drop = 0 (one-sided t-test)")


# ══════════════════════════════════════════════════════════════════
# PREDICTION (Current Month)
# ══════════════════════════════════════════════════════════════════

RANK_HISTORY_FILE = "rank_history.csv"


def load_rank_history(path=RANK_HISTORY_FILE):
    """Load saved monthly score rankings, or None if absent/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None


def compute_rank_deltas(ranked_tickers, as_of, path=RANK_HISTORY_FILE):
    """Map each ticker to its rank change vs the most recent prior month.

    `ranked_tickers` is this month's tickers in score order (rank 1 =
    best). Returns {ticker: "↑2" / "↓1" / "→" / "NEW"}; empty dict if no
    prior month exists yet. Compares against the latest history row whose
    date is strictly before `as_of`, so re-running the same month still
    diffs against last month (not itself).
    """
    hist = load_rank_history(path)
    if hist is None:
        return {}
    as_of = pd.Timestamp(as_of).normalize()
    prior = hist[hist["date"] < as_of]
    if prior.empty:
        return {}
    last = prior[prior["date"] == prior["date"].max()].set_index("ticker")["rank"]
    deltas = {}
    for cur_rank, t in enumerate(ranked_tickers, 1):
        if t not in last.index:
            deltas[t] = "NEW"
            continue
        d = int(last[t]) - cur_rank   # positive = moved up the ranking
        deltas[t] = f"↑{d}" if d > 0 else (f"↓{-d}" if d < 0 else "→")
    return deltas


def save_rank_history(ranked_tickers, scores, as_of, path=RANK_HISTORY_FILE):
    """Append this month's (date, ticker, rank, score) ranking.

    Idempotent: re-running the same month overwrites that month's rows
    rather than duplicating them.
    """
    as_of = pd.Timestamp(as_of).normalize()
    rows = pd.DataFrame({
        "date": as_of,
        "ticker": list(ranked_tickers),
        "rank": range(1, len(ranked_tickers) + 1),
        "score": list(scores),
    })
    hist = load_rank_history(path)
    if hist is not None:
        hist = hist[hist["date"] != as_of]
        rows = pd.concat([hist, rows], ignore_index=True)
    rows.to_csv(path, index=False)


PICKS_LOG_FILE = "picks_log.csv"


def last_logged_picks(path=PICKS_LOG_FILE):
    """Tickers from the most recent month in the picks log (weight > 0).

    Used by AUTO_ROLL_HOLDINGS to treat last run's picks as the current
    holdings — only valid if the model's picks are actually executed each
    month. Excludes the weight==0 XIU benchmark row. Returns [] if no log.
    """
    if not os.path.exists(path):
        return []
    try:
        log = pd.read_csv(path, parse_dates=["as_of"])
    except Exception:
        return []
    if log.empty:
        return []
    last = log[log["as_of"] == log["as_of"].max()]
    return last[last["weight"] > 0]["ticker"].tolist()


def log_picks(picks, weights, scores_by_ticker, as_of, path=PICKS_LOG_FILE):
    """Append this month's picks to the OOS track-record log.

    Records (as_of, ticker, weight, score, fwd_realized=NaN) per pick,
    plus one XIU.TO benchmark row (weight 0). Idempotent per as_of month.
    fwd_realized is filled later by backfill_realized once the next
    month's prices exist — that's what turns this log into a genuine
    out-of-sample record to check against the backtest's 1.92.
    """
    as_of = pd.Timestamp(as_of).normalize()
    tickers = list(picks) + ["XIU.TO"]
    rows = pd.DataFrame({
        "as_of": as_of,
        "ticker": tickers,
        "weight": [weights.get(t, 0.0) for t in picks] + [0.0],
        "score": [scores_by_ticker.get(t, np.nan) for t in picks] + [np.nan],
        "fwd_realized": np.nan,
    })
    if os.path.exists(path):
        try:
            hist = pd.read_csv(path, parse_dates=["as_of"])
            hist = hist[hist["as_of"] != as_of]
            rows = pd.concat([hist, rows], ignore_index=True)
        except Exception:
            pass
    rows.to_csv(path, index=False)


def backfill_realized(price_df, path=PICKS_LOG_FILE):
    """Fill fwd_realized for logged picks whose next-month price now exists.

    fwd_realized = the ticker's return from its as_of month-end to the
    following month-end, from price_df. Rows stay NaN until that data
    arrives, so the log only credits returns that were genuinely OOS at
    pick time.
    """
    if not os.path.exists(path):
        return
    try:
        log = pd.read_csv(path, parse_dates=["as_of"])
    except Exception:
        return
    need = log["fwd_realized"].isna()
    if not need.any():
        return
    updated = False
    for idx in log.index[need]:
        t = log.at[idx, "ticker"]
        d = pd.Timestamp(log.at[idx, "as_of"]).normalize()
        close, _ = get_ohlcv(price_df, t)
        if close is None:
            continue
        m = close.resample("ME").last()
        at, after = m[m.index <= d], m[m.index > d]
        if len(at) and len(after):
            r = after.iloc[0] / at.iloc[-1] - 1
            if np.isfinite(r):
                log.at[idx, "fwd_realized"] = float(r)
                updated = True
    if updated:
        log.to_csv(path, index=False)


def oos_track_record(path=PICKS_LOG_FILE):
    """Summary lines for the realized out-of-sample track record.

    Portfolio monthly OOS return per as_of = weight-weighted fwd_realized
    of that month's picks; benchmark = the logged XIU.TO row. Only matured
    (filled) months count, so this is empty until the first month's picks
    have a realized return.
    """
    if not os.path.exists(path):
        return []
    try:
        log = pd.read_csv(path, parse_dates=["as_of"])
    except Exception:
        return []
    filled = log.dropna(subset=["fwd_realized"])
    picks_f = filled[filled["weight"] > 0]
    if picks_f.empty:
        return ["OOS track record: none matured yet "
                "(picks logged; returns fill in next month)."]
    port = picks_f.groupby("as_of").apply(
        lambda g: (g["weight"] * g["fwd_realized"]).sum() / g["weight"].sum())
    bench = (filled[filled["ticker"] == "XIU.TO"]
             .set_index("as_of")["fwd_realized"].reindex(port.index))
    n = len(port)
    cum = (1 + port).prod() - 1
    hit = (port > 0).mean()
    lines = ["OOS track record (realized, since logging began):"]
    if bench.notna().all():
        bcum = (1 + bench).prod() - 1
        beat = (port.values > bench.values).mean()
        lines.append(f"  {n} mo | portfolio {cum:+.1%} vs XIU {bcum:+.1%} "
                     f"(excess {cum - bcum:+.1%})")
        lines.append(f"  monthly avg {port.mean():+.2%} | positive {hit:.0%} "
                     f"| beat XIU {beat:.0%}")
    else:
        lines.append(f"  {n} mo | portfolio cum {cum:+.1%} | "
                     f"monthly avg {port.mean():+.2%} | positive {hit:.0%}")
    if n < 6:
        lines.append(f"  (only {n} mo of live data — too few to judge; "
                     f"backtest was Sharpe 1.92)")
    return lines


def predict_now(panel, feature_cols, price_df, macro_df, current_holdings=None):
    """Generate current month stock picks."""
    panel = panel.sort_values("date")
    dates = sorted(panel["date"].unique())

    # Use all history for training
    train_df = panel[panel["date"] < dates[-1]].copy()
    latest_df = panel[panel["date"] == dates[-1]].copy()

    if len(train_df) < 100 or len(latest_df) < 3:
        print("  ERROR: Insufficient data for prediction")
        return []

    weights = compute_time_decay_weights(len(train_df))

    sec_dml_stats = {}
    if USE_SECTOR_MODELS:
        sec_models, sec_dml_stats = fit_sector_models(train_df, sample_weights=weights)
        scores = predict_sector_models(latest_df, sec_models)
        scores = np.nan_to_num(scores, nan=0.0)
        print(f"  Sector models trained: {sorted(sec_models.keys())}")
        for s, st in sec_dml_stats.items():
            sig = ("***" if st["p_value"] < 0.01 else
                   "**"  if st["p_value"] < 0.05 else
                   "*"   if st["p_value"] < 0.10 else "")
            print(f"    DML {s:<12} θ={st['theta']:+.3f}  "
                  f"t={st['t_stat']:+.2f}  p={st['p_value']:.3f}{sig}  "
                  f"n={st['n_obs']}")
    else:
        reg, clf = fit_models(
            pd.DataFrame(train_df[feature_cols].values, columns=feature_cols),
            pd.Series(train_df["fwd_ret"].values),
            sample_weights=weights
        )
        scores = ensemble_predict(reg, clf, latest_df[feature_cols].values)

    latest_df = latest_df.copy()
    latest_df["score"] = scores

    checks = []
    if USE_SECTOR_MODELS:
        checks = health_check(latest_df, train_df, sec_models, sec_dml_stats)
        print_health_check(checks)

    # DML adjustment. Momentum treatments route through whichever momentum
    # columns are in the feature set (mom_pc{1,2}_norm under PCA mode,
    # mom_{1,3}m_norm otherwise). Earnings_surprise sits outside the
    # feature set as a pure treatment.
    if "mom_pc1_norm" in feature_cols:
        mom_treatments = [c for c in ("mom_pc1_norm", "mom_pc2_norm") if c in feature_cols]
    else:
        mom_treatments = [c for c in ("mom_1m_norm", "mom_3m_norm") if c in feature_cols]

    dml_thetas = {}
    for treatment in mom_treatments:
        theta = estimate_dml_alpha(train_df, feature_cols, treatment)
        if abs(theta) > 0.01:
            dml_thetas[treatment] = theta
    if "earnings_surprise" in train_df.columns:
        theta = estimate_dml_alpha(train_df, feature_cols, "earnings_surprise")
        if abs(theta) > 0.01:
            dml_thetas["earnings_surprise"] = theta

    if dml_thetas:
        latest_df["score"] = apply_dml_adjustment(
            latest_df["score"].values, latest_df, dml_thetas
        )
        print(f"  DML adjustments: {dml_thetas}")

    # Regime detection
    regime, regime_adj = detect_regime(macro_df)
    constraints = {**CONSTRAINTS, **regime_adj}
    print(f"  Market regime: {regime}")

    # Sort and filter
    latest_df = latest_df.sort_values("score", ascending=False)
    candidates = latest_df["ticker"].tolist()

    # Month-over-month rank changes vs the saved history (Step 8). Compute
    # BEFORE saving this month so the diff is against last month.
    as_of = dates[-1]
    rank_deltas = compute_rank_deltas(candidates, as_of)

    # Full ranked board with month-over-month rank change (vs saved history)
    print("\n  ═══ ALL STOCKS RANKED BY SCORE ═══")
    for i, (ticker, row) in enumerate(latest_df.iterrows(), 1):
        sector = STOCK_PROFILE.get(row["ticker"], ("Unknown",))[0]
        chg = rank_deltas.get(row["ticker"], "")
        print(f"    {i:2d}. {row['ticker']:<8} {sector:<14} "
              f"Score: {row['score']:.3f}  {chg}")

    # Persist this month's full ranking so next run can show deltas.
    save_rank_history(candidates, latest_df["score"].tolist(), as_of)

    # Constraint filter, then rebalancing band → final picks.
    print("  Fetching fundamentals for constraint check...")
    fund_df = fetch_fundamentals(candidates[:30])  # Top 30 only
    filtered = apply_constraints(
        candidates, fund_df, price_df, mode="pick",
        current_holdings=current_holdings, constraints=constraints
    )
    final_picks = apply_rebalancing_band(
        filtered,
        latest_df[latest_df["ticker"].isin(filtered)]["score"].tolist(),
        current_holdings or [],
        constraints=constraints
    )

    # Position sizing
    weights = risk_parity_weights(final_picks, price_df)

    # OOS track record: log this month's picks, then backfill any prior
    # months whose realized returns have now matured (see functions).
    sbt = dict(zip(latest_df["ticker"], latest_df["score"]))
    log_picks(final_picks, weights, sbt, as_of)
    backfill_realized(price_df)

    # Feature importance: prefer OOS permutation importance (honest, computed
    # on walk-forward test folds) over train-set gain (overfits training data,
    # has known all-zero extraction quirk). Fall back to gain if no cached
    # permutation importance exists yet — run `picker.py importance` to build it.
    importance = load_perm_importance()
    _importance_source = "permutation (OOS)"
    if importance is None:
        _importance_source = "gain (train-set) — run `picker.py importance` for OOS"
        importance = {}
        if USE_SECTOR_MODELS:
            for sector_name, (sec_reg, _clf, feats) in sec_models.items():
                for f, imp in zip(feats, sec_reg.feature_importances_):
                    importance[f] = importance.get(f, 0.0) + float(imp)
            n_models = max(len(sec_models), 1)
            importance = {f: v / n_models for f, v in importance.items()}
        else:
            importance = dict(zip(feature_cols, reg.feature_importances_))
    top_features = [(f, v) for f, v in sorted(importance.items(), key=lambda x: -x[1])
                    if v > 0][:10]
    print(f"  Feature importance source: {_importance_source}")

    shap_by_ticker = {}
    if USE_SECTOR_MODELS:
        shap_by_ticker = _compute_shap_for_models(sec_models, latest_df)

    return final_picks, weights, latest_df, top_features, regime, checks, (current_holdings or []), shap_by_ticker


# ══════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════

def diff_holdings(picks, holdings):
    """Split this month's picks vs current holdings into actions.

    Returns (sell, buy, hold):
      sell = held but not in picks (exit)
      buy  = in picks but not held (enter)
      hold = in both               (keep)
    """
    picks_set, hold_set = set(picks), set(holdings or [])
    sell = [t for t in (holdings or []) if t not in picks_set]
    buy = [t for t in picks if t not in hold_set]
    hold = [t for t in picks if t in hold_set]
    return sell, buy, hold


def _health_summary(checks):
    """One-line reliability verdict, naming each failed check + its detail."""
    if not checks:
        return ""
    fails = [(label, detail) for label, ok, detail in checks if not ok]
    n = len(fails)
    verdict = ("OK — all checks passed" if n == 0 else
               "CAUTION — 1 warning" if n == 1 else
               f"LOW CONFIDENCE — {n} warnings")
    if fails:
        verdict += " — " + "; ".join(f"{lbl}: {det}" for lbl, det in fails)
    return verdict


def _format_report(picks, weights, panel_latest, top_features, regime,
                   checks=None, holdings=None, shap_by_ticker=None,
                   te_estimate=None):
    """Build the shared monthly report body as a list of lines.

    Same content for stdout (print_picks) and email (build_report_text):
    a header with signal reliability, the actionable SELL/BUY/HOLD diff
    vs current holdings, the target portfolio with weights, and the top
    feature importances.
    """
    lines = [
        f"TSX STOCK PICKS — {datetime.now().strftime('%Y-%m-%d')}",
        f"Regime: {regime}",
    ]
    hs = _health_summary(checks or [])
    if hs:
        lines.append(f"Signal reliability: {hs}")
    lines.append("=" * 60)

    sell, buy, hold = diff_holdings(picks, holdings)
    bw = ", ".join(f"{t} {weights.get(t, 0):.0%}" for t in buy)
    if holdings:
        lines += [
            "",
            "ACTIONS (vs current holdings):",
            f"  SELL ({len(sell)}): {', '.join(sell) if sell else '—'}",
            f"  BUY  ({len(buy)}): {bw if buy else '—'}",
            f"  HOLD ({len(hold)}): {', '.join(hold) if hold else '—'}",
        ]
    else:
        lines += [
            "",
            "ACTIONS (initial build — no current holdings):",
            f"  BUY all ({len(buy)}): {bw}",
        ]

    lines += ["", "Target portfolio:"]
    for i, ticker in enumerate(picks, 1):
        w = weights.get(ticker, 0)
        profile = STOCK_PROFILE.get(ticker, ("?", "?", "?"))
        row = panel_latest[panel_latest["ticker"] == ticker]
        score = row["score"].iloc[0] if len(row) > 0 else 0
        lines.append(f"  {i}. {ticker:<10} {profile[0]:<14} {profile[1]:<8} "
                     f"Score: {score:.3f}  Weight: {w:.1%}")
        if shap_by_ticker and ticker in shap_by_ticker:
            sv = shap_by_ticker[ticker]
            top3 = sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            drivers = "  ".join(
                f"{f.replace('_norm', '')}({'+'if v>0 else ''}{v:.3f})"
                for f, v in top3 if abs(v) > 0.001
            )
            if drivers:
                lines.append(f"     Drivers: {drivers}")

    if te_estimate is not None and not np.isnan(te_estimate):
        lines.append(f"  Est. Tracking Error: {te_estimate:.1%}/yr  "
                     f"({'high' if te_estimate > 0.12 else 'normal' if te_estimate > 0.07 else 'low'})")

    lines += ["", "Top Features:"]
    for feat, imp in top_features:
        lines.append(f"  {feat:<20} {imp:.4f}")

    oos = oos_track_record()
    if oos:
        lines += [""] + oos
    return lines


def print_picks(picks, weights, panel_latest, top_features, regime,
                checks=None, holdings=None, shap_by_ticker=None,
                te_estimate=None):
    """Print the actionable monthly report to stdout."""
    lines = _format_report(picks, weights, panel_latest, top_features,
                           regime, checks, holdings, shap_by_ticker,
                           te_estimate=te_estimate)
    print("\n" + "═" * 60)
    for ln in lines:
        print(f"  {ln}" if ln else "")
    print("═" * 60)


def build_report_text(picks, weights, panel_latest, top_features, regime,
                      checks=None, holdings=None, shap_by_ticker=None,
                      te_estimate=None):
    """Plain-text actionable monthly report for emailing."""
    return "\n".join(_format_report(picks, weights, panel_latest,
                                    top_features, regime, checks, holdings, shap_by_ticker,
                                    te_estimate=te_estimate))


def build_report_html(picks, weights, panel_latest, top_features, regime,
                      checks=None, holdings=None, shap_by_ticker=None):
    """HTML actionable monthly report for emailing."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    hs = _health_summary(checks or [])
    sell, buy, hold = diff_holdings(picks, holdings)

    # Reliability badge color
    if "LOW CONFIDENCE" in hs:
        badge_color = "#e74c3c"
    elif "CAUTION" in hs:
        badge_color = "#f39c12"
    else:
        badge_color = "#27ae60"
    reliability_text = hs if hs else "OK — all checks passed"

    # Regime badge
    regime_colors = {"BULL": "#27ae60", "BEAR": "#e74c3c", "NEUTRAL": "#3498db"}
    regime_color = regime_colors.get(regime, "#7f8c8d")

    css = """
    <style>
      body { font-family: -apple-system, Arial, sans-serif; background: #f4f6f8;
             color: #2c3e50; margin: 0; padding: 20px; }
      .card { background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
              max-width: 680px; margin: 0 auto 20px; padding: 24px; }
      h1 { font-size: 22px; margin: 0 0 4px; color: #1a252f; }
      .subtitle { color: #7f8c8d; font-size: 13px; margin-bottom: 16px; }
      .badge { display: inline-block; border-radius: 6px; padding: 3px 10px;
               font-size: 12px; font-weight: 600; color: #fff; margin-right: 6px; }
      h2 { font-size: 15px; font-weight: 700; color: #34495e;
           border-bottom: 2px solid #eee; padding-bottom: 6px; margin: 20px 0 12px; }
      table { width: 100%; border-collapse: collapse; font-size: 13px; }
      th { text-align: left; padding: 8px 10px; background: #f8f9fa;
           color: #6c757d; font-weight: 600; border-bottom: 2px solid #dee2e6; }
      td { padding: 8px 10px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
      tr:last-child td { border-bottom: none; }
      .tag-sell { background: #fde8e8; color: #c0392b; border-radius: 4px;
                  padding: 2px 7px; font-weight: 600; font-size: 12px; }
      .tag-buy  { background: #e8f8f0; color: #1e8449; border-radius: 4px;
                  padding: 2px 7px; font-weight: 600; font-size: 12px; }
      .tag-hold { background: #eaf0fb; color: #2471a3; border-radius: 4px;
                  padding: 2px 7px; font-weight: 600; font-size: 12px; }
      .driver-up   { color: #27ae60; font-weight: 600; }
      .driver-down { color: #e74c3c; font-weight: 600; }
      .driver-name { color: #5d6d7e; font-size: 11px; }
      .weight-bar-bg { background: #eef2f7; border-radius: 4px; height: 6px; min-width: 60px; }
      .weight-bar    { background: #2e86c1; border-radius: 4px; height: 6px; }
      .score-num { font-weight: 700; color: #2c3e50; }
      .oos-box { background: #f0f9ff; border-left: 4px solid #3498db;
                 border-radius: 4px; padding: 12px 16px; font-size: 13px; }
      .feat-bar-bg { background: #eef2f7; border-radius: 3px; height: 8px; display: inline-block; width: 120px; vertical-align: middle; }
      .feat-bar    { background: #8e44ad; border-radius: 3px; height: 8px; display: inline-block; }
      .footer { text-align: center; color: #aaa; font-size: 11px; margin-top: 8px; }
    </style>"""

    # ── Header card ──────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{css}</head><body>
    <div class="card">
      <h1>TSX Stock Picks</h1>
      <div class="subtitle">{date_str}</div>
      <span class="badge" style="background:{regime_color}">Regime: {regime}</span>
      <span class="badge" style="background:{badge_color}">{reliability_text}</span>
    </div>"""

    # ── Actions card ─────────────────────────────────────────────────────────
    def action_cells(tickers, tag_class, label):
        if not tickers:
            return f'<span style="color:#aaa">—</span>'
        return " ".join(f'<span class="{tag_class}">{t}</span>' for t in tickers)

    bw_pairs = [(t, weights.get(t, 0)) for t in buy]
    buy_html = " ".join(
        f'<span class="tag-buy">{t} {w:.0%}</span>' for t, w in bw_pairs
    ) if buy else '<span style="color:#aaa">—</span>'
    sell_html = action_cells(sell, "tag-sell", "SELL")
    hold_html = action_cells(hold, "tag-hold", "HOLD")

    html += f"""
    <div class="card">
      <h2>Actions vs Current Holdings</h2>
      <table>
        <tr><th style="width:60px">Action</th><th>Tickers</th></tr>
        <tr><td><span class="tag-sell">SELL {len(sell)}</span></td><td>{sell_html}</td></tr>
        <tr><td><span class="tag-buy">BUY {len(buy)}</span></td><td>{buy_html}</td></tr>
        <tr><td><span class="tag-hold">HOLD {len(hold)}</span></td><td>{hold_html}</td></tr>
      </table>
    </div>"""

    # ── Feature label helper (used in portfolio + features cards) ────────────
    _FEAT_LABELS = {
        "sector_code":       "Sector",
        "high_52w_ratio":    "52-Week High Proximity",
        "mom_pc1":           "Momentum Factor 1",
        "mom_pc2":           "Momentum Factor 2",
        "vol_60d":           "60-Day Volatility",
        "vol_20d":           "20-Day Volatility",
        "vol_ratio":         "Short/Long Vol Ratio",
        "rsi_14":            "RSI (14-day)",
        "bb_zscore":         "Bollinger Band Z-Score",
        "adv_20d_rank":      "Avg Daily Volume Rank",
        "rev_1m":            "1-Month Revenue Change",
        "rev_growth_yoy":    "Revenue Growth (YoY)",
        "div_growth_yoy":    "Dividend Growth (YoY)",
        "div_yield":         "Dividend Yield",
        "roe":               "Return on Equity",
        "pe_ratio":          "Price/Earnings Ratio",
        "debt_equity":       "Debt/Equity Ratio",
        "rate_chg_3m":       "Interest Rate Change (3M)",
        "boc_rate_chg_3m":   "BoC Rate Change (3M)",
        "cad_bond_mom_1m":   "CAD Bond Momentum (1M)",
        "cad_mom_1m":        "CAD/USD Momentum (1M)",
        "tsx_mom_1m":        "TSX Index Momentum (1M)",
        "oil_mom_1m":        "Oil Price Momentum (1M)",
        "natgas_mom_1m":     "Natural Gas Momentum (1M)",
        "carbon_mom_1m":     "Carbon Price Momentum (1M)",
        "tips_mom_1m":       "TIPS Momentum (1M)",
        "vix_level":         "VIX (Market Fear Index)",
    }

    def _feat_label(name):
        clean = name.replace("_norm", "")
        return _FEAT_LABELS.get(clean, clean.replace("_", " ").title())

    # ── Portfolio card ────────────────────────────────────────────────────────
    max_w = max((weights.get(t, 0) for t in picks), default=0.01)
    rows_html = ""
    for i, ticker in enumerate(picks, 1):
        w = weights.get(ticker, 0)
        profile = STOCK_PROFILE.get(ticker, ("?", "?", "?"))
        row = panel_latest[panel_latest["ticker"] == ticker]
        score = row["score"].iloc[0] if len(row) > 0 else 0
        bar_pct = int(w / max_w * 100) if max_w > 0 else 0

        # Determine action tag for this ticker
        if ticker in sell:
            action = '<span class="tag-sell">SELL</span>'
        elif ticker in buy:
            action = '<span class="tag-buy">BUY</span>'
        else:
            action = '<span class="tag-hold">HOLD</span>'

        drivers_html = ""
        if shap_by_ticker and ticker in shap_by_ticker:
            sv = shap_by_ticker[ticker]
            top3 = sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            parts = []
            for f, v in top3:
                if abs(v) <= 0.001:
                    continue
                fname = _feat_label(f)
                cls = "driver-up" if v > 0 else "driver-down"
                arrow = "↑" if v > 0 else "↓"
                parts.append(
                    f'<span class="driver-name">{fname}</span>'
                    f'<span class="{cls}"> {arrow}{abs(v):.3f}</span>'
                )
            drivers_html = "&nbsp; ".join(parts)

        rows_html += f"""
        <tr>
          <td style="font-weight:700;color:#1a252f">{ticker}</td>
          <td>{profile[0]}</td>
          <td>{action}</td>
          <td class="score-num">{score:.3f}</td>
          <td>
            <div style="font-size:12px;font-weight:600;margin-bottom:3px">{w:.1%}</div>
            <div class="weight-bar-bg"><div class="weight-bar" style="width:{bar_pct}%"></div></div>
          </td>
          <td style="font-size:12px">{drivers_html}</td>
        </tr>"""

    html += f"""
    <div class="card">
      <h2>Target Portfolio ({len(picks)} positions)</h2>
      <table>
        <tr>
          <th>Ticker</th><th>Sector</th><th>Action</th>
          <th>Score</th><th>Weight</th><th>SHAP Drivers</th>
        </tr>
        {rows_html}
      </table>
    </div>"""

    # ── Top features card ─────────────────────────────────────────────────────
    if top_features:
        max_imp = max(v for _, v in top_features) or 0.001
        feat_rows = ""
        for feat, imp in top_features[:10]:
            bar_w = int(imp / max_imp * 120)
            feat_rows += f"""
            <tr>
              <td style="font-size:13px">{_feat_label(feat)}</td>
              <td style="font-family:monospace;color:#7f8c8d;font-size:11px">{imp:.4f}</td>
              <td><div class="feat-bar-bg"><div class="feat-bar" style="width:{bar_w}px"></div></div></td>
            </tr>"""
        html += f"""
    <div class="card">
      <h2>Top Feature Importances (Permutation, OOS)</h2>
      <table>
        <tr><th>Feature</th><th>IC Drop</th><th>Importance</th></tr>
        {feat_rows}
      </table>
    </div>"""

    # ── OOS track record card ─────────────────────────────────────────────────
    oos_lines = oos_track_record()
    if oos_lines:
        oos_content = "<br>".join(oos_lines)
        html += f"""
    <div class="card">
      <h2>OOS Track Record (Live)</h2>
      <div class="oos-box">{oos_content}</div>
    </div>"""

    html += """
    <div class="footer">Generated by TSX Stock Picker</div>
    </body></html>"""
    return html


def send_report_email(body, html_body=None, subject=None):
    """Email the report to the inbox configured in email_config.py.

    email_config.py is gitignored (copy from email_config.example.py). If
    it's absent or still has placeholder values, prints a notice and skips
    — so a fresh clone still runs without credentials. Returns True if sent.
    """
    try:
        import email_config as cfg
        from_email, to_email, app_pw = (cfg.EMAIL_FROM, cfg.EMAIL_TO,
                                         cfg.EMAIL_APP_PASSWORD)
    except Exception:
        print("  Email: no email_config.py — skipping send (report printed above).")
        return False
    if "your@" in from_email or "your_gmail" in app_pw:
        print("  Email: email_config.py still has placeholder values — skipping send.")
        return False

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    if subject is None:
        subject = f"TSX Stock Picks — {datetime.now().strftime('%Y-%m-%d')}"
    if html_body:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))
    else:
        msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"], msg["From"], msg["To"] = subject, from_email, to_email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_pw)
            server.sendmail(from_email, to_email, msg.as_string())
        print(f"  Email: report sent to {to_email}")
        return True
    except Exception as e:
        print(f"  Email: send failed ({e}).")
        return False


def compute_vif(panel, feature_cols):
    """Variance inflation factor per feature.

    VIF > 10 = severe collinearity, > 5 = moderate. A high VIF means the
    feature is largely a linear combination of the others (redundant).
    Returns a DataFrame (feature, vif) sorted high-to-low.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = panel[feature_cols].copy()
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]   # drop zero-variance columns
    rows = []
    for i, col in enumerate(X.columns):
        try:
            v = float(variance_inflation_factor(X.values, i))
        except Exception:
            v = np.nan
        rows.append({"feature": col, "vif": v})
    return pd.DataFrame(rows).sort_values("vif", ascending=False, na_position="last")


def run_vif_diagnostic(panel, model_features):
    """Print a VIF collinearity table for the RAW features the model uses.

    Maps each _norm feature back to its base column: rank normalization
    bounds everything to [-1, 1] and masks the underlying collinearity, so
    VIF on the raw columns is the honest view. Pure diagnostic — no model
    or data is changed.
    """
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        print("  VIF needs statsmodels — `pip install statsmodels` and re-run.")
        return None

    base_feats = []
    for f in model_features:
        base = f[:-5] if f.endswith("_norm") else f
        if base == "sector_code":
            continue
        if base in panel.columns and base not in base_feats:
            base_feats.append(base)
    vif_df = compute_vif(panel, base_feats)

    print("\n" + "═" * 56)
    print("  VIF COLLINEARITY DIAGNOSTIC (raw model features)")
    print("═" * 56)
    print(f"  {'feature':<22}{'VIF':>9}   status")
    print("  " + "-" * 48)
    for _, r in vif_df.iterrows():
        v = r["vif"]
        vs = f"{v:.1f}" if pd.notna(v) else "N/A"
        if pd.notna(v) and v > 10:
            status = "✗ severe (>10)"
        elif pd.notna(v) and v > 5:
            status = "~ moderate (>5)"
        else:
            status = "✓ ok"
        print(f"  {r['feature']:<22}{vs:>9}   {status}")
    n_sev = int((vif_df["vif"] > 10).sum())
    n_mod = int(((vif_df["vif"] > 5) & (vif_df["vif"] <= 10)).sum())
    print("  " + "-" * 48)
    print(f"  {n_sev} severe (VIF>10), {n_mod} moderate (5<VIF≤10), "
          f"of {len(vif_df)} features.")
    print("  High-VIF features are drop candidates (less overfit at this "
          "sample size).")
    print("═" * 56)
    return vif_df


def evaluate_segments(perstock):
    """Per-year and per-sector RankIC + pick turnover — anti-overfit checks.

    RankIC = each month's cross-sectional Spearman corr between the model
    score and realized fwd_ret. Answers: is the edge concentrated in a few
    regime years, which sectors carry real signal vs noise, and is monthly
    selection stable? Adapted from picker_ca.py's evaluate_segments
    (picker.py's STOCK_PROFILE/sector_code differs from that file).
    """
    if perstock is None or perstock.empty:
        print("  (no per-stock data for segmented evaluation)")
        return
    v = perstock.dropna(subset=["score", "fwd_ret"]).copy()
    if v.empty:
        return
    v["date"] = pd.to_datetime(v["date"])
    code_to_name = {c: n for n, c in SECTOR_NAME_TO_CODE.items()}
    v["sector"] = v["sector_code"].map(code_to_name).fillna("Other")

    def _ic(g):
        if len(g) < 5:
            return np.nan
        return g["score"].rank().corr(g["fwd_ret"].rank())

    def _ic_kendall(g):
        if len(g) < 5:
            return np.nan
        tau, _ = kendalltau(g["score"].rank(), g["fwd_ret"].rank())
        return tau

    monthly_ic = v.groupby("date").apply(_ic).dropna()
    monthly_ic_k = v.groupby("date").apply(_ic_kendall).dropna()
    monthly_ic.index = pd.to_datetime(monthly_ic.index)
    monthly_ic_k.index = pd.to_datetime(monthly_ic_k.index)

    print("\n" + "═" * 65)
    print("  SEGMENTED EVALUATION (anti-overfit diagnostics)")
    print("═" * 65)

    print("\n  By year — RankIC  (Spearman | Kendall τ)")
    print(f"    {'year':<6}{'months':>7}{'Spearman':>10}{'Kendall τ':>11}{'ICIR':>7}{'hit%':>7}")
    print("    " + "-" * 51)
    for year, g in monthly_ic.groupby(monthly_ic.index.year):
        icm, ics = g.mean(), g.std()
        icir = icm / ics if ics > 0 else 0.0
        km = monthly_ic_k.reindex(g.index).mean()
        print(f"    {year:<6}{len(g):>7}{icm:>+10.3f}{km:>+11.3f}{icir:>+7.2f}"
              f"{(g > 0).mean() * 100:>6.0f}%")
    icm, ics = monthly_ic.mean(), monthly_ic.std()
    icir = icm / ics if ics > 0 else 0.0
    km_all = monthly_ic_k.mean()
    print("    " + "-" * 51)
    print(f"    {'ALL':<6}{len(monthly_ic):>7}{icm:>+10.3f}{km_all:>+11.3f}{icir:>+7.2f}"
          f"{(monthly_ic > 0).mean() * 100:>6.0f}%")

    print("\n  By sector — RankIC")
    print(f"    {'sector':<14}{'months':>7}{'Spearman':>10}{'Kendall τ':>11}{'ICIR':>7}")
    print("    " + "-" * 49)
    for sector, g in v.groupby("sector"):
        sic = g.groupby("date").apply(_ic).dropna()
        sic_k = g.groupby("date").apply(_ic_kendall).dropna()
        if len(sic) < 6:
            continue
        icm, ics = sic.mean(), sic.std()
        icir = icm / ics if ics > 0 else 0.0
        print(f"    {sector:<14}{len(sic):>7}{icm:>+10.3f}{sic_k.mean():>+11.3f}{icir:>+7.2f}")

    # Score quintile → realized return spread (monotonicity check)
    q_labels = ["Q1 (bottom)", "Q2", "Q3", "Q4", "Q5 (top)"]
    quintile_frames = []
    for _date, _g in v.groupby("date"):
        if len(_g) < 10:
            continue
        try:
            _g = _g.copy()
            _g["quintile"] = pd.qcut(_g["score"], 5, labels=q_labels)
            quintile_frames.append(_g.groupby("quintile", observed=True)["fwd_ret"].mean())
        except Exception:
            continue
    if quintile_frames:
        qdf = pd.concat(quintile_frames).groupby(level=0).mean() * 100
        spread = qdf.get("Q5 (top)", 0) - qdf.get("Q1 (bottom)", 0)
        print(f"\n  Score quintile → avg realized return (monotonicity):")
        for lbl in q_labels:
            val = qdf.get(lbl, np.nan)
            print(f"    {lbl:<14} {val:>+6.2f}%/mo  ({val * 12:>+6.1f}%/yr)")
        print(f"    Top−Bottom spread: {spread:>+6.2f}%/mo  ({spread*12:>+6.1f}%/yr)")
        # Monotonicity: is each quintile strictly greater than the previous?
        q_vals = [qdf.get(lbl, np.nan) for lbl in q_labels]
        mono = all(q_vals[i] < q_vals[i+1] for i in range(len(q_vals)-1)
                   if not (np.isnan(q_vals[i]) or np.isnan(q_vals[i+1])))
        print(f"    Monotone (Q1<Q2<Q3<Q4<Q5): {'Yes ✓' if mono else 'No ✗'}")

    print("  Guide: |IC mean| range across years > 0.15, or a sector IC < 0,")
    print("  signals a regime-dependent / noisy edge.")
    print("═" * 65)


def evaluate_prediction_quality(perstock):
    """Classification + regression prediction quality metrics (per-month, averaged).

    Binary label = top-quintile fwd_ret (>= 80th pct cross-sectional).
    Predicted label = above-median score (cross-sectional).
    Directional accuracy = fraction where sign(score vs median) == sign(fwd_ret).
    AUC-ROC / AUC-PR / Brier use continuous scores; MCC / F1 / Prec / Rec use
    the above-median threshold for the binary prediction.
    """
    if perstock is None or perstock.empty:
        return
    try:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, brier_score_loss,
            log_loss, matthews_corrcoef, balanced_accuracy_score,
            f1_score, precision_score, recall_score,
        )
    except ImportError:
        print("  sklearn required — already installed, check import path.")
        return

    v = perstock.dropna(subset=["score", "fwd_ret"]).copy()
    v = v[v["ticker"] != "XIU.TO"]
    if v.empty:
        return

    aucroc_l, aucpr_l, brier_l, logloss_l, mcc_l = [], [], [], [], []
    f1_l, prec_l, rec_l, bacc_l, dacc_l = [], [], [], [], []

    for _, g in v.groupby("date"):
        if len(g) < 5:
            continue
        ret = g["fwd_ret"].values
        score = g["score"].values

        # Top-quintile binary label
        thr = np.percentile(ret, 80)
        y_true = (ret >= thr).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        # Predicted class: above-median score
        y_pred = (score >= np.median(score)).astype(int)

        # Normalized score → [0, 1] for Brier / LogLoss
        rng = score.max() - score.min()
        score_01 = (score - score.min()) / (rng + 1e-12)

        try:
            aucroc_l.append(roc_auc_score(y_true, score))
            aucpr_l.append(average_precision_score(y_true, score))
            brier_l.append(brier_score_loss(y_true, score_01))
            logloss_l.append(log_loss(y_true, np.clip(score_01, 1e-7, 1 - 1e-7)))
            mcc_l.append(matthews_corrcoef(y_true, y_pred))
            f1_l.append(f1_score(y_true, y_pred, zero_division=0))
            prec_l.append(precision_score(y_true, y_pred, zero_division=0))
            rec_l.append(recall_score(y_true, y_pred, zero_division=0))
            bacc_l.append(balanced_accuracy_score(y_true, y_pred))
            # Directional accuracy: does above-median score → positive return?
            dacc_l.append(np.mean((score > np.median(score)) == (ret > 0)))
        except Exception:
            continue

    if not aucroc_l:
        return

    def _fmt(lst):
        return f"{np.mean(lst):.3f} (±{np.std(lst):.3f})" if lst else "n/a"

    print("\n" + "═" * 60)
    print("  PREDICTION QUALITY — CLASSIFICATION & REGRESSION")
    print("═" * 60)
    print(f"  Months evaluated:      {len(aucroc_l)}")
    print()
    print("  — Classifier metrics (top-quintile = positive class) —")
    print(f"  AUC-ROC:               {_fmt(aucroc_l)}")
    print(f"  AUC-PR:                {_fmt(aucpr_l)}")
    print(f"  Brier Score:           {_fmt(brier_l)}  (lower = better)")
    print(f"  Log Loss:              {_fmt(logloss_l)}  (lower = better)")
    print(f"  MCC:                   {_fmt(mcc_l)}")
    print(f"  F1 Score:              {_fmt(f1_l)}")
    print(f"  Precision:             {_fmt(prec_l)}")
    print(f"  Recall:                {_fmt(rec_l)}")
    print(f"  Balanced Accuracy:     {_fmt(bacc_l)}")
    print()
    print("  — Regression / ranking metrics —")
    print(f"  Directional Accuracy:  {_fmt(dacc_l)}")
    auc_base = 0.5
    lift = np.mean(aucroc_l) - auc_base
    print(f"  AUC lift over random:  {lift:+.3f}")
    print("═" * 60)


# ── Deflated Sharpe / Overfitting Audit ──────────────────────────────────────

def compute_psr(monthly_returns, sr_benchmark=0.0):
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark).

    Bailey & López de Prado (2014). Inputs are per-period (monthly) returns;
    sr_benchmark is also in per-period units (pass 0 for the null SR=0 test).
    """
    from scipy.stats import norm
    from scipy.stats import skew as _skew, kurtosis as _kurt
    r = np.asarray(monthly_returns, dtype=float)
    T = len(r)
    if T < 4:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    s = _skew(r)
    k = _kurt(r, fisher=True)  # excess kurtosis
    var = (1 - s * sr + (k - 1) / 4 * sr ** 2) / (T - 1)
    if var <= 0:
        return np.nan
    return float(norm.cdf((sr - sr_benchmark) / np.sqrt(var)))


def compute_dsr(monthly_returns, n_trials):
    """Deflated Sharpe Ratio: PSR with benchmark = E[max of n_trials iid SRs].

    n_trials: number of independent strategies evaluated before selecting the
    best one. Adjusts for selection bias — a Sharpe earned by picking the top
    strategy out of K is worth less than a Sharpe earned without selection.
    Returns probability in [0, 1].
    """
    from scipy.stats import norm
    from scipy.stats import skew as _skew, kurtosis as _kurt
    r = np.asarray(monthly_returns, dtype=float)
    T = len(r)
    if T < 4:
        return np.nan
    if n_trials < 2:
        return compute_psr(monthly_returns, sr_benchmark=0.0)
    sr = r.mean() / r.std(ddof=1)
    s = _skew(r)
    k = _kurt(r, fisher=True)
    var = max((1 - s * sr + (k - 1) / 4 * sr ** 2) / (T - 1), 1e-12)
    gamma = 0.5772156649  # Euler-Mascheroni constant
    # Expected maximum SR under the null (all strategies have true SR = 0)
    e_max = np.sqrt(var) * (
        (1 - gamma) * norm.ppf(1 - 1 / n_trials) +
        gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return compute_psr(monthly_returns, sr_benchmark=e_max)


def compute_sharpe_ci(monthly_returns, confidence=0.95, n_boot=2000,
                      block_size=4, seed=42):
    """Block-bootstrap confidence interval for annualized Sharpe.

    Block bootstrap (block_size months) preserves serial correlation in
    monthly returns. Returns (observed_ann_sharpe, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(monthly_returns, dtype=float)
    T = len(r)
    obs = r.mean() / r.std(ddof=1) * np.sqrt(12)
    n_blocks = int(np.ceil(T / block_size))
    boot = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(T - block_size + 1, 1), size=n_blocks)
        sample = np.concatenate([r[s: s + block_size] for s in starts])[:T]
        sd = sample.std(ddof=1)
        if sd > 0:
            boot.append(sample.mean() / sd * np.sqrt(12))
    alpha = 1 - confidence
    lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return obs, lo, hi


def compute_ir_ci(excess_returns, confidence=0.95, n_boot=2000,
                  block_size=4, seed=43):
    """Block-bootstrap confidence interval for annualized IR.

    excess_returns: monthly (port_ret - bench_ret).
    Returns (observed_ann_ir, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    e = np.asarray(excess_returns, dtype=float)
    T = len(e)
    sd = e.std(ddof=1)
    obs = e.mean() * np.sqrt(12) / sd if sd > 0 else np.nan
    n_blocks = int(np.ceil(T / block_size))
    boot = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(T - block_size + 1, 1), size=n_blocks)
        sample = np.concatenate([e[s: s + block_size] for s in starts])[:T]
        sd_s = sample.std(ddof=1)
        if sd_s > 0:
            boot.append(sample.mean() * np.sqrt(12) / sd_s)
    alpha = 1 - confidence
    lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return obs, lo, hi


def print_overfit_report(results_df, n_trials=35):
    """DSR / PSR / bootstrap CI overfitting audit appended to backtest output.

    n_trials: total independent experiments run before selecting the current
    model (≈15 for this project: 9 model variants + 6 feature variants).
    DSR < 0.80 → probable overfitting; 0.80–0.95 → moderate; > 0.95 → strong.
    PBO proxy = 1 − DSR (selection-bias-adjusted probability of overfitting).
    Note: true PBO requires CPCV across N strategies; this is the single-series
    approximation via the DSR framework.
    """
    from scipy.stats import skew as _skew, kurtosis as _kurt
    r = results_df["port_ret"].values
    T = len(r)
    sr_ann = r.mean() / r.std(ddof=1) * np.sqrt(12)
    s = _skew(r)
    k = _kurt(r, fisher=True)

    e = (results_df["bench_ret"].values
         if "bench_ret" in results_df.columns
         else np.zeros_like(r))
    excess = r - e
    ir_ann = excess.mean() * np.sqrt(12) / excess.std(ddof=1) if excess.std(ddof=1) > 0 else np.nan
    _, ir_ci_lo, ir_ci_hi = compute_ir_ci(excess)

    psr = compute_psr(r, sr_benchmark=0.0)
    dsr = compute_dsr(r, n_trials=n_trials)
    _, ci_lo, ci_hi = compute_sharpe_ci(r)

    # White's Reality Check (bootstrap): test if observed Sharpe exceeds what
    # is expected from the best of n_trials independent random strategies with
    # the same return distribution (null = no real edge).
    # Equivalent to DSR analytically; bootstrap provides a second estimate.
    rng_wrc = np.random.RandomState(42)
    n_boot_wrc = 5000
    # WRC null: demeaned returns → H0 = no edge (expected mean = 0)
    r_null = r - r.mean()
    def _boot_sr_null(r0, T, rng):
        s = rng.choice(r0, size=T, replace=True)
        sd = s.std(ddof=1)
        return (s.mean() / sd * np.sqrt(12)) if sd > 0 else 0.0
    boot_sr_null = np.array([_boot_sr_null(r_null, T, rng_wrc) for _ in range(n_boot_wrc)])
    # Under H0: best of n_trials independent zero-edge strategies
    max_sr_null = np.array([np.max(rng_wrc.choice(boot_sr_null, size=n_trials))
                            for _ in range(n_boot_wrc)])
    wrc_pval = float(np.mean(max_sr_null <= sr_ann))

    print("\n" + "═" * 60)
    print("  DEFLATED SHARPE / OVERFITTING AUDIT")
    print("═" * 60)
    print(f"  Observations (months):        {T}")
    print(f"  Observed Sharpe (ann.):       {sr_ann:.3f}")
    ir_str = f"{ir_ann:.3f}" if not np.isnan(ir_ann) else "n/a"
    print(f"  Observed IR    (ann.):        {ir_str}")
    print(f"  Skewness:                     {s:+.3f}")
    print(f"  Excess kurtosis:              {k:+.3f}")
    print(f"  Bootstrap 95% CI (ann. SR):   [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  Bootstrap 95% CI (ann. IR):   [{ir_ci_lo:.2f}, {ir_ci_hi:.2f}]")
    print(f"  PSR  P(true SR > 0):          {psr:.1%}")
    print(f"  DSR  P(true SR > E[max_{n_trials:02d}]):  {dsr:.1%}")
    pbo = 1.0 - dsr if dsr is not None and not np.isnan(dsr) else np.nan
    print(f"  PBO proxy (1 − DSR):          {pbo:.1%}  ← est. selection-bias overfit risk")
    print(f"  WRC  P(SR > max-of-{n_trials:02d} null):  {wrc_pval:.1%}  ← White's Reality Check (bootstrap)")
    if dsr >= 0.95:
        verdict = f"STRONG — survives {n_trials}-trial selection correction"
    elif dsr >= 0.80:
        verdict = "MODERATE — treat observed Sharpe as optimistic upper bound"
    else:
        verdict = "WEAK — likely overfitting, do NOT rely on this backtest"
    print(f"  Verdict:                      {verdict}")
    print("═" * 60)


def print_monte_carlo_report(results_df, n_sim=5000, seed=42):
    """Bootstrap Monte Carlo: resample monthly returns to show path distribution.

    Resamples the observed monthly returns (with replacement) n_sim times to
    generate alternative equity paths. Reports percentile table for Sharpe, IR,
    max drawdown, and total return, plus tail-risk probabilities and an ASCII
    fan chart showing the P5/P25/P50/P75/P95 equity paths.
    """
    r = results_df["port_ret"].values
    b = results_df["bench_ret"].values
    T = len(r)
    rng = np.random.default_rng(seed)

    sharpes, irs, max_dds, terminals = [], [], [], []
    paths = np.empty((n_sim, T), dtype=float)

    for i in range(n_sim):
        idx = rng.integers(0, T, size=T)
        r_s = r[idx]
        b_s = b[idx]
        excess_s = r_s - b_s
        sd_r = r_s.std(ddof=1)
        sd_e = excess_s.std(ddof=1)
        sharpes.append(r_s.mean() / sd_r * np.sqrt(12) if sd_r > 0 else np.nan)
        irs.append(excess_s.mean() * np.sqrt(12) / sd_e if sd_e > 0 else np.nan)
        cum = (1 + r_s).cumprod()
        dd = float((cum / np.maximum.accumulate(cum) - 1).min())
        max_dds.append(dd)
        terminals.append(float(cum[-1]))
        paths[i] = cum

    sharpes = np.array(sharpes)
    irs    = np.array(irs)
    max_dds = np.array(max_dds)
    terminals = np.array(terminals)

    pcts = [5, 25, 50, 75, 95]
    sr_p  = np.nanpercentile(sharpes, pcts)
    ir_p  = np.nanpercentile(irs, pcts)
    dd_p  = np.nanpercentile(max_dds, pcts)
    tr_p  = np.nanpercentile((terminals - 1) * 100, pcts)

    print("\n" + "═" * 62)
    print(f"  MONTE CARLO  ({n_sim:,} bootstrap paths, {T} months resampled)")
    print("═" * 62)
    print(f"  {'Metric':<14}  {'P5':>7}  {'P25':>7}  {'P50':>7}  {'P75':>7}  {'P95':>7}")
    print(f"  {'─'*14}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    print(f"  {'Sharpe':<14}  {sr_p[0]:>+7.2f}  {sr_p[1]:>+7.2f}  {sr_p[2]:>+7.2f}  {sr_p[3]:>+7.2f}  {sr_p[4]:>+7.2f}")
    print(f"  {'IR':<14}  {ir_p[0]:>+7.2f}  {ir_p[1]:>+7.2f}  {ir_p[2]:>+7.2f}  {ir_p[3]:>+7.2f}  {ir_p[4]:>+7.2f}")
    print(f"  {'Max Drawdown':<14}  {dd_p[0]:>+7.1%}  {dd_p[1]:>+7.1%}  {dd_p[2]:>+7.1%}  {dd_p[3]:>+7.1%}  {dd_p[4]:>+7.1%}")
    print(f"  {'Total Return':<14}  {tr_p[0]:>+6.0f}%  {tr_p[1]:>+6.0f}%  {tr_p[2]:>+6.0f}%  {tr_p[3]:>+6.0f}%  {tr_p[4]:>+6.0f}%")
    print(f"  {'─'*60}")
    print(f"  P(Sharpe > 1):       {np.mean(sharpes > 1.0):.1%}")
    print(f"  P(IR > 0):           {np.mean(irs > 0.0):.1%}")
    print(f"  P(MaxDD < −15%):     {np.mean(max_dds < -0.15):.1%}")
    print(f"  P(MaxDD < −25%):     {np.mean(max_dds < -0.25):.1%}")

    # ASCII fan chart — P5/P25/P50/P75/P95 equity paths
    fan = np.percentile(paths, pcts, axis=0)  # (5, T)
    width = min(T, 48)
    col_idx = np.linspace(0, T - 1, width, dtype=int)
    fan_w = fan[:, col_idx]  # (5, width)

    height = 8
    y_hi = fan_w.max() * 1.05
    y_lo = max(fan_w.min() * 0.95, 0.5)
    y_rng = y_hi - y_lo or 1.0

    def _to_row(val):
        return int((y_hi - val) / y_rng * (height - 1))

    grid = [[' '] * width for _ in range(height)]
    chars = ['·', '░', '█', '▓', '■']   # P5 P25 P50 P75 P95
    for p_i, ch in enumerate(chars):
        for t in range(width):
            row = max(0, min(height - 1, _to_row(fan_w[p_i, t])))
            # only overwrite if higher priority (P50 > P25 > P5 etc.)
            if grid[row][t] in (' ', '·', '░') or ch in ('█', '▓', '■'):
                grid[row][t] = ch

    print(f"\n  Fan  ·=P5  ░=P25  █=P50  ▓=P75  ■=P95  (×1 = initial capital)")
    for i, row in enumerate(grid):
        y_val = y_hi - y_rng * i / (height - 1)
        label = f"{y_val:.1f}×" if i % 2 == 0 else "     "
        print(f"  {label:>5} {''.join(row)}")
    print("═" * 62)


def print_permutation_importance(importance_dict, top_n=20):
    """Print OOS permutation importance table (mean RankIC drop per feature)."""
    if not importance_dict:
        print("  No permutation importance data.")
        return
    ranked = sorted(importance_dict.items(), key=lambda x: -x[1])
    max_drop = max(abs(v) for _, v in ranked) if ranked else 1.0
    print(f"\n{'─' * 58}")
    print(f"  PERMUTATION IMPORTANCE  (OOS RankIC drop, mean over folds)")
    print(f"  Positive = shuffling hurts ranking (feature matters)")
    print(f"  Negative = shuffling helps (feature is noise)")
    print(f"{'─' * 58}")
    print(f"  {'Feature':<28}  {'IC Drop':>8}  Bar")
    print(f"  {'─'*28}  {'─'*8}  {'─'*18}")
    for feat, drop in ranked[:top_n]:
        bar_len = int(abs(drop) / max_drop * 18) if max_drop > 0 else 0
        bar = ("█" if drop >= 0 else "░") * bar_len
        sign = "+" if drop > 0 else (" " if drop == 0 else "-")
        name = feat.replace("_norm", "")
        print(f"  {name:<28}  {sign}{abs(drop):>7.4f}  {bar}")
    print(f"{'─' * 58}")


def print_backtest(results_df):
    """Print backtest summary."""
    if results_df.empty:
        print("  No backtest results.")
        return

    r = results_df["port_ret"].values
    T = len(results_df)
    results_df["excess"] = results_df["port_ret"] - results_df["bench_ret"]
    cum_port = (1 + results_df["port_ret"]).cumprod()
    cum_bench = (1 + results_df["bench_ret"]).cumprod()

    total_ret = cum_port.iloc[-1] - 1
    bench_total = cum_bench.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (12 / T) - 1
    ann_bench = (1 + bench_total) ** (12 / T) - 1
    ann_excess = ann_ret - ann_bench

    # --- core metrics ---
    ann_vol = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = r[r < 0]
    ann_downside_vol = downside.std(ddof=1) * np.sqrt(12) if len(downside) > 1 else np.nan
    sortino = ann_ret / ann_downside_vol if ann_downside_vol and ann_downside_vol > 0 else np.nan
    max_dd = (cum_port / cum_port.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    hit_rate = (results_df["excess"] > 0).mean()

    # --- benchmark-relative metrics ---
    excess_ser = results_df["excess"].values
    tracking_err = excess_ser.std(ddof=1) * np.sqrt(12)
    info_ratio = ann_excess / tracking_err if tracking_err > 0 else np.nan

    # --- profit factor + expectancy (on monthly excess) ---
    wins = excess_ser[excess_ser > 0]
    losses = excess_ser[excess_ser <= 0]
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.nan
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    expectancy = hit_rate * avg_win + (1 - hit_rate) * avg_loss

    # --- beta + treynor (vs benchmark) ---
    bench_r = results_df["bench_ret"].values
    beta = np.cov(r, bench_r)[0, 1] / np.var(bench_r, ddof=1) if np.var(bench_r, ddof=1) > 0 else np.nan
    treynor = ann_ret / beta if (beta and not np.isnan(beta) and beta > 0) else np.nan

    print("\n" + "═" * 60)
    print("  WALK-FORWARD BACKTEST RESULTS")
    print("═" * 60)
    print(f"  Period:              {results_df['date'].iloc[0].strftime('%Y-%m')} to {results_df['date'].iloc[-1].strftime('%Y-%m')}")
    print(f"  Months:              {T}")
    print(f"  Portfolio:           {ann_ret:+.1%} ann. (total {total_ret:+.1%})")
    print(f"  Benchmark:           {ann_bench:+.1%} ann. (total {bench_total:+.1%})")
    print(f"  Excess:              {ann_excess:+.1%} ann.")
    print()
    print(f"  Sharpe:              {sharpe:.2f}")
    print(f"  Sortino:             {sortino:.2f}" if not np.isnan(sortino) else "  Sortino:             n/a")
    print(f"  Calmar:              {calmar:.2f}" if not np.isnan(calmar) else "  Calmar:              n/a")
    print(f"  Max Drawdown:        {max_dd:.1%}")
    print(f"  Ann. Volatility:     {ann_vol:.1%}")
    print(f"  Tracking Error:      {tracking_err:.1%}")
    print(f"  Information Ratio:   {info_ratio:.2f}" if not np.isnan(info_ratio) else "  Information Ratio:   n/a")
    print(f"  Hit Rate:            {hit_rate:.1%}")
    print(f"  Profit Factor:       {profit_factor:.2f}" if not np.isnan(profit_factor) else "  Profit Factor:       n/a")
    print(f"  Expectancy (mo.):    {expectancy:+.2%}")
    print(f"  Beta (vs XIU):       {beta:.2f}" if not np.isnan(beta) else "  Beta (vs XIU):       n/a")
    print(f"  Treynor Ratio:       {treynor:.2f}" if not np.isnan(treynor) else "  Treynor Ratio:       n/a")
    print("═" * 60)

    # Yearly breakdown
    results_df["year"] = results_df["date"].dt.year
    results_df["monthly_excess"] = results_df["port_ret"] - results_df["bench_ret"]
    yearly = results_df.groupby("year").agg(
        port=("port_ret", lambda x: (1 + x).prod() - 1),
        bench=("bench_ret", lambda x: (1 + x).prod() - 1),
        te=("monthly_excess", lambda x: x.std(ddof=1) * np.sqrt(12)),
        n=("port_ret", "count"),
    )
    yearly["excess"] = yearly["port"] - yearly["bench"]
    yearly["ir"] = yearly.apply(
        lambda r: r["excess"] / r["te"] if r["te"] > 0 and r["n"] >= 3 else np.nan, axis=1
    )
    print("\n  Year    Portfolio   Benchmark   Excess      IR")
    print("  " + "-" * 52)
    for year, row in yearly.iterrows():
        ir_str = f"{row['ir']:+.2f}" if not np.isnan(row["ir"]) else "  n/a"
        print(f"  {year}    {row['port']:+.1%}      {row['bench']:+.1%}      {row['excess']:+.1%}    {ir_str}")

    # Recent monthly detail — feel for month-to-month behavior, not just
    # the aggregate Sharpe. (pseudo-OOS: params were tuned over all
    # history, so recent months read optimistic vs a true forward OOS.)
    recent = results_df.tail(6)
    print("\n  Last 6 months   Portfolio   Benchmark   Excess")
    print("  " + "-" * 44)
    for _, row in recent.iterrows():
        d = pd.Timestamp(row["date"]).strftime("%Y-%m")
        ex = row["port_ret"] - row["bench_ret"]
        print(f"  {d}      {row['port_ret']:+.1%}      {row['bench_ret']:+.1%}      {ex:+.1%}")
    print()

    # ASCII equity curve — cumulative growth of $1
    _print_equity_curve(results_df)

    print_overfit_report(results_df)
    print_monte_carlo_report(results_df)


def _print_equity_curve(results_df, width=50):
    """Print a compact two-line ASCII equity curve (portfolio vs benchmark)."""
    port_cum = (1 + results_df["port_ret"]).cumprod().values
    bench_cum = (1 + results_df["bench_ret"]).cumprod().values
    n = len(port_cum)

    # Sample to `width` points evenly; label every ~12 months
    idx = np.linspace(0, n - 1, min(width, n), dtype=int)
    p_vals = port_cum[idx]
    b_vals = bench_cum[idx]
    all_vals = np.concatenate([p_vals, b_vals])
    lo, hi = all_vals.min(), all_vals.max()
    rng = hi - lo or 1.0
    rows = 5  # height in terminal lines

    # Build grid: rows × width chars
    grid = [[" "] * len(idx) for _ in range(rows)]

    def _row(v):
        return rows - 1 - int(round((v - lo) / rng * (rows - 1)))

    for col, (pv, bv) in enumerate(zip(p_vals, b_vals)):
        pr, br = _row(pv), _row(bv)
        if 0 <= pr < rows:
            grid[pr][col] = "█"
        if 0 <= br < rows and br != pr:
            grid[br][col] = "░"

    dates = results_df["date"].values
    first_yr = pd.Timestamp(dates[0]).year
    last_yr = pd.Timestamp(dates[-1]).year

    print("  Equity curve  █ Portfolio  ░ Benchmark")
    for r in range(rows):
        label = f" {lo + (rows - 1 - r) / (rows - 1) * rng:.0%}" if r in (0, rows - 1) else "     "
        print("  " + label + " " + "".join(grid[r]))

    # Year tick marks
    tick_line = [" "] * len(idx)
    for i, ix in enumerate(idx):
        yr = pd.Timestamp(dates[ix]).year
        if yr != (pd.Timestamp(dates[idx[i-1]]).year if i > 0 else yr - 1):
            s = str(yr)
            for k, ch in enumerate(s):
                if i + k < len(tick_line):
                    tick_line[i + k] = ch
    print("  " + " " * 6 + "".join(tick_line))
    print(f"  {first_yr}–{last_yr}  Port: {port_cum[-1]:.0%} total  Bench: {bench_cum[-1]:.0%} total")
    print()


# ══════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════

def _wf_metrics(panel, model_features, embargo_months=1, half_life=12):
    """Run walk_forward; return (sharpe, ir, ann_ret, max_dd, hit_rate)."""
    results = walk_forward(panel, model_features, embargo_months=embargo_months,
                           half_life=half_life)
    if results.empty:
        return (np.nan,) * 5
    r = results["port_ret"].values
    b = results["bench_ret"].values
    excess = r - b
    sr  = r.mean() / r.std(ddof=1) * np.sqrt(12) if r.std(ddof=1) > 0 else np.nan
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    ann_bench = (1 + b).prod() ** (12 / len(b)) - 1
    ann_excess = ann - ann_bench
    te  = excess.std(ddof=1) * np.sqrt(12)
    ir  = ann_excess / te if te > 0 else np.nan
    cum = (1 + r).cumprod()
    dd  = float((cum / np.maximum.accumulate(cum) - 1).min())
    hit = float((r > 0).mean())
    return sr, ir, ann, dd, hit


# ══════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════

def _compute_shap_for_models(sec_models, latest_df):
    """SHAP values for latest_df using already-fitted sec_models. Returns {} if shap unavailable."""
    try:
        import shap as _shap
    except ImportError:
        return {}
    shap_by_ticker = {}
    for sector_name, (reg, clf, feats) in sec_models.items():
        code = SECTOR_NAME_TO_CODE.get(sector_name)
        if code is None:
            continue
        mask = (latest_df["sector_code"] == code).values
        if not mask.any():
            continue
        X = latest_df.loc[mask, feats].values
        tickers = latest_df.loc[mask, "ticker"].values
        try:
            sv_r = _shap.TreeExplainer(reg).shap_values(X)
            sv_c_raw = _shap.TreeExplainer(clf).shap_values(X)
            if isinstance(sv_c_raw, list):
                sv_c = sv_c_raw[1]
            elif hasattr(sv_c_raw, "ndim") and sv_c_raw.ndim == 3:
                sv_c = sv_c_raw[:, :, 1]
            else:
                sv_c = sv_c_raw
            sv = 0.5 * np.array(sv_r) + 0.5 * np.array(sv_c)
            for i, ticker in enumerate(tickers):
                shap_by_ticker[ticker] = {f: float(sv[i, j]) for j, f in enumerate(feats)}
        except Exception:
            continue
    return shap_by_ticker


def compute_shap_current(panel, model_features):
    """Fit models on last 36 months; return SHAP values for current month."""
    try:
        import shap  # noqa: F401 — just verify installed
    except ImportError:
        return None, "shap not installed — run: pip install shap"

    dates = sorted(panel["date"].unique())
    if len(dates) < 37:
        return None, "Not enough data"

    train_df = panel[panel["date"].isin(dates[-37:-1])].copy()
    latest_df = panel[panel["date"] == dates[-1]].copy()
    weights = compute_time_decay_weights(len(train_df))
    sec_models, _ = fit_sector_models(train_df, sample_weights=weights)
    return _compute_shap_for_models(sec_models, latest_df), None


def print_shap_report(shap_by_ticker, picks=None):
    """Print mean |SHAP| ranking and per-pick breakdown."""
    if not shap_by_ticker:
        print("  (no SHAP values)")
        return

    all_feats = sorted({f for sv in shap_by_ticker.values() for f in sv})
    mean_abs = {f: np.mean([abs(sv.get(f, 0.0)) for sv in shap_by_ticker.values()])
                for f in all_feats}
    ranked = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)
    bar_max = ranked[0][1] if ranked[0][1] > 0 else 1.0

    print("\n" + "─" * 60)
    print("  SHAP FEATURE IMPORTANCE  (mean |SHAP|, current month)")
    print("  50% ExtraTrees regressor + 50% classifier, blended")
    print("─" * 60)
    print(f"  {'Feature':<28} {'|SHAP|':>8}  Bar")
    print(f"  {'─'*28} {'─'*8}  {'─'*18}")
    for feat, val in ranked[:15]:
        name = feat.replace("_norm", "")
        bar = "█" * max(1, int(18 * val / bar_max)) if val > 0 else ""
        print(f"  {name:<28} {val:>8.4f}  {bar}")
    print("─" * 60)

    if picks:
        top_feats = [f for f, _ in ranked[:5]]
        col_w = 11
        header = f"  {'Ticker':<10}" + "".join(
            f"  {f.replace('_norm','')[:col_w-2]:>{col_w}}" for f in top_feats)
        print(f"\n  PER-PICK SHAP (top 5 features)")
        print(header)
        print("  " + "─" * (10 + (col_w + 2) * len(top_feats)))
        for ticker in picks:
            if ticker not in shap_by_ticker:
                continue
            sv = shap_by_ticker[ticker]
            row = f"  {ticker:<10}"
            for f in top_feats:
                v = sv.get(f, 0.0)
                row += f"  {v:>+{col_w}.4f}"
            print(row)
        print("─" * 60)


def run_hp_stability(panel, model_features):
    """OAT hyperparameter stability test for ExtraTrees.

    Fixes 3 params at default, varies the 4th across a range, runs a full
    walk-forward backtest for each combo. Returns list of result dicts.
    """
    import copy

    baseline = dict(n_estimators=300, max_depth=5, min_samples_leaf=10, max_features=0.7)

    grid = {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 4, 5, 6, None],
        "min_samples_leaf":[5, 10, 15, 20],
        "max_features":    [0.5, 0.6, 0.7, 0.8],
    }

    rows = []
    total = sum(len(v) for v in grid.values())
    done = 0
    for param, values in grid.items():
        for val in values:
            done += 1
            hp = copy.copy(baseline)
            hp[param] = val
            tag = "*" if val == baseline[param] else " "
            label = f"{param}={val}{tag}"
            print(f"  [{done}/{total}] {label} ...", flush=True)
            # Temporarily override global ET_HP
            old_hp = ET_HP.copy()
            ET_HP.update(hp)
            try:
                sr, ir, ann, dd, hit = _wf_metrics(panel, model_features, embargo_months=1)
            finally:
                ET_HP.clear()
                ET_HP.update(old_hp)
            rows.append({
                "param": param, "value": val,
                "is_baseline": val == baseline[param],
                "sharpe": sr, "ir": ir, "ann_ret": ann, "max_dd": dd, "hit": hit,
            })
    return rows


def print_hp_stability_report(rows):
    """Print hyperparameter stability results grouped by parameter."""
    params = list(dict.fromkeys(r["param"] for r in rows))  # preserve order
    base_row = next(r for r in rows if r["is_baseline"] and r["param"] == "n_estimators")
    baseline_ir = base_row["ir"]

    print("\n" + "═" * 74)
    print("  HYPERPARAMETER STABILITY  (ExtraTrees, primary metric: IR)")
    print(f"  Baseline: n_estimators=300, max_depth=5, min_samples_leaf=10, max_features=0.7")
    print(f"  Baseline IR: {baseline_ir:.2f}  (Sharpe: {base_row['sharpe']:.2f})")
    print("═" * 74)

    for param in params:
        param_rows = [r for r in rows if r["param"] == param]
        print(f"\n  {param}")
        print(f"  {'Value':<12}  {'IR':>6}  {'Sharpe':>7}  {'Ann Ret':>8}  {'MaxDD':>7}  {'Hit%':>6}  {'ΔIR':>7}")
        print("  " + "─" * 62)
        for r in param_rows:
            tag = " ←" if r["is_baseline"] else ""
            delta = r["ir"] - baseline_ir if not np.isnan(r["ir"]) else np.nan
            delta_str = f"{delta:+.2f}" if not np.isnan(delta) else "  n/a"
            val_str = str(r["value"]) if r["value"] is not None else "None"
            print(f"  {val_str:<12}  {r['ir']:>6.2f}  {r['sharpe']:>7.2f}  {r['ann_ret']:>7.1%}  "
                  f"{r['max_dd']:>7.1%}  {r['hit']:>5.1%}  {delta_str:>7}{tag}")
    print("\n" + "═" * 74)


def run_sensitivity(panel, model_features):
    """One-at-a-time sensitivity over embargo_months."""
    base = {"embargo_months": 1}
    grids = {
        "embargo_months": [0, 1, 2, 3],
    }
    rows = []
    total = sum(len(v) for v in grids.values())
    done = 0
    for param, values in grids.items():
        for val in values:
            done += 1
            kwargs = dict(base)
            kwargs[param] = val
            print(f"  [{done}/{total}] {param}={val} ...", end="", flush=True)
            sr, ir, ann, dd, hit = _wf_metrics(panel, model_features, **kwargs)
            print(f"  IR {ir:.3f}  Sharpe {sr:.3f}")
            rows.append({"param": param, "value": val,
                         "baseline": (val == base[param]),
                         "sharpe": sr, "ir": ir, "ann_ret": ann, "max_dd": dd, "hit": hit})
    return rows


def print_sensitivity_report(rows):
    print("\n" + "═" * 76)
    print("  SENSITIVITY ANALYSIS  (one-at-a-time, ★ = baseline, primary metric: IR)")
    print("═" * 76)
    print(f"  {'Parameter':<16} {'Value':>7}  {'IR':>6}  {'Sharpe':>7}  {'Ann.Ret':>8}  {'MaxDD':>7}  {'Hit':>6}")
    print(f"  {'─'*16} {'─'*7}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*6}")
    prev_param = None
    for r in rows:
        if prev_param and r["param"] != prev_param:
            print(f"  {'─'*16} {'─'*7}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*6}")
        prev_param = r["param"]
        star = "★" if r["baseline"] else " "
        val_str = f"{r['value']:.2f}" if isinstance(r["value"], float) else str(r["value"])
        print(f"  {r['param']:<15}{star} {val_str:>7}  "
              f"{r['ir']:>6.3f}  "
              f"{r['sharpe']:>7.3f}  "
              f"{r['ann_ret']:>+8.1%}  "
              f"{r['max_dd']:>7.1%}  "
              f"{r['hit']:>6.1%}")
    print("═" * 76)


def run_halflife_sweep(panel, model_features):
    """Sweep time-decay half-life over {6, 9, 12, 18, 24} months."""
    values = [6, 9, 12, 18, 24]
    baseline = 12
    rows = []
    for i, hl in enumerate(values):
        print(f"  [{i+1}/{len(values)}] half_life={hl}m ...", end="", flush=True)
        sr, ir, ann, dd, hit = _wf_metrics(panel, model_features, half_life=hl)
        print(f"  IR {ir:.3f}  Sharpe {sr:.3f}")
        rows.append({"half_life": hl, "baseline": (hl == baseline),
                     "sharpe": sr, "ir": ir, "ann_ret": ann, "max_dd": dd, "hit": hit})
    return rows


def print_halflife_report(rows):
    print("\n" + "═" * 70)
    print("  HALF-LIFE SWEEP  (time-decay weight, ★ = current default, primary: IR)")
    print("═" * 70)
    print(f"  {'Half-life':>10}  {'IR':>6}  {'Sharpe':>7}  {'Ann.Ret':>8}  {'MaxDD':>7}  {'Hit':>6}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*6}")
    best_ir = max(r["ir"] for r in rows if not np.isnan(r["ir"]))
    for r in rows:
        star = "★" if r["baseline"] else " "
        best = "◀ best" if abs(r["ir"] - best_ir) < 1e-9 else ""
        print(f"  {r['half_life']:>9}m{star}  "
              f"{r['ir']:>6.3f}  "
              f"{r['sharpe']:>7.3f}  "
              f"{r['ann_ret']:>+8.1%}  "
              f"{r['max_dd']:>7.1%}  "
              f"{r['hit']:>6.1%}  {best}")
    print("═" * 70)
    best_row = max(rows, key=lambda r: r["ir"])
    if not best_row["baseline"]:
        delta = best_row["ir"] - next(r["ir"] for r in rows if r["baseline"])
        print(f"\n  Best: half_life={best_row['half_life']}m  "
              f"(Δ IR {delta:+.3f} vs baseline 12m)")
        if delta >= 0.05:
            print(f"  → Meets adoption threshold (ΔIR ≥ 0.05) — consider updating default.")
        else:
            print(f"  → Below adoption threshold (ΔIR < 0.05) — keep default 12m.")
    else:
        print(f"\n  Baseline (12m) is already optimal by IR.")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pick"
    assert mode in ("pick", "backtest", "both", "vif", "importance", "rigor", "sensitivity", "shap", "hptest", "halflife"), \
        f"Usage: python picker.py [pick|backtest|both|vif|importance|rigor|sensitivity|shap|hptest|halflife]"

    print(f"\n{'═' * 60}")
    print(f"  TSX Stock Picker — Mode: {mode.upper()}")
    print(f"  Data source: yfinance | Model: {MODEL_KIND.upper()} + DML")
    print(f"{'═' * 60}\n")

    # Get tickers (exclude benchmark for modeling)
    tickers = [t for t in TSX_UNIVERSE if t != "XIU.TO"]

    # Download data
    print("  [1/5] Downloading price data...")
    all_tickers = TSX_UNIVERSE + list(MACRO_TICKERS.values())
    price_df = fetch_prices(all_tickers, years=7)
    if price_df is None or len(price_df) == 0:
        print("  ERROR: price download returned no data — aborting (retry later).")
        return

    # Data-health gate: catch yfinance partial/failed downloads before
    # they silently produce bad picks. Hard failures abort the run.
    data_checks = check_data_health(price_df, TSX_UNIVERSE,
                                    CONSTRAINTS.get("required_sectors"))
    print_health_check(data_checks, title="Data health")
    fatal = [lbl for lbl, ok, _ in data_checks
             if not ok and lbl in ("Download rate", "Benchmark XIU.TO")]
    if fatal:
        print(f"  ERROR: data health failed ({', '.join(fatal)}) — aborting this run.")
        return

    print("  [2/5] Downloading macro data...")
    macro_df = price_df  # Already included in the download

    # Build features
    print("  [3/5] Computing features...")
    panel = build_panel(price_df, macro_df, TSX_UNIVERSE)
    if panel.empty:
        print("  ERROR: No data available.")
        return

    # Impute raw features, derive fwd_ret from raw mom_1m, then add
    # rank-normalized companion columns (_norm) for the model to consume.
    # Originals stay untouched so fwd_ret keeps real return units.
    available_features = [c for c in FEATURE_COLS if c in panel.columns]
    panel = smart_impute(panel, available_features)
    panel = add_labels(panel)

    if USE_MOMENTUM_PCA:
        panel = apply_momentum_pca(panel)
        # PCs replace the raw momentum features in the model's feature
        # list. Raw mom_*m columns stay on the panel (still needed by
        # fwd_ret and DML treatment routing).
        available_features = [c for c in FEATURE_COLS
                              if c in panel.columns and c not in _RAW_MOMENTUM]
        print("  Momentum: PCA mode (mom_pc1, mom_pc2 replace raw mom_*)")
    else:
        available_features = [c for c in available_features
                              if c not in ("mom_pc1", "mom_pc2")]

    panel, model_features = cross_sectional_normalize(panel, available_features)
    print(f"  Panel: {len(panel)} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} months")

    if mode == "vif":
        run_vif_diagnostic(panel, model_features)
        return

    if mode == "importance":
        print("\n  [4/5] Computing permutation importance (walk-forward OOS)...")
        print("  (This runs the full backtest — ~2-3 min)")
        _, _, importance = walk_forward(
            panel, model_features,
            return_perstock=True, return_importance=True
        )
        print_permutation_importance(importance)
        save_perm_importance(importance)
        print(f"\n  Saved to {PERM_IMPORTANCE_FILE} — `pick` will now use OOS importance.")
        return

    if mode == "rigor":
        print("\n  [4/5] Running quantitative-rigor suite...")
        print("  Step 1/3: Walk-forward + FDR (permutation importance with p-values)...")
        results, _, importance, fold_imps = walk_forward(
            panel, model_features,
            return_perstock=True, return_importance=True, return_raw_importance=True
        )
        wf_sharpe_net = None
        if not results.empty and "port_ret" in results.columns:
            r = results["port_ret"].values
            wf_sharpe_net = r.mean() / r.std(ddof=1) * np.sqrt(12)

        print_overfit_report(results)

        fdr_results = compute_fdr_importance(fold_imps)
        print_fdr_importance(fdr_results)

        print("\n  Step 2/3: CPCV (combinatorial purged cross-validation)...")
        print("  (15 paths × 1 model fit each — ~2 min)")
        cpcv_paths = compute_cpcv(panel, model_features)
        print_cpcv_report(cpcv_paths, wf_sharpe=wf_sharpe_net)

        print("\n  Step 3/3: White's Reality Check — see DSR/WRC above.")
        print("  (WRC bootstrap is included in the overfitting audit output.)")
        print("\n  Nested CV: skipped — ExtraTrees has minimal tunable params;")
        print("  all 9-model / 6-feature experiments converged to same Sharpe")
        print("  ceiling, indicating params are already at the local optimum.")
        return

    if mode == "sensitivity":
        print("\n  [4/5] Running sensitivity analysis (~14 walk-forward runs, ~25 min)...")
        rows = run_sensitivity(panel, model_features)
        print_sensitivity_report(rows)
        return

    if mode == "hptest":
        print("\n  [4/5] Running hyperparameter stability test...")
        print("  OAT scan: n_estimators, max_depth, min_samples_leaf, max_features")
        print("  ~16 walk-forward runs × ~1 min each ≈ 15-20 min total\n")
        rows = run_hp_stability(panel, model_features)
        print_hp_stability_report(rows)
        return

    if mode == "halflife":
        print("\n  [4/5] Running half-life sweep (5 values × ~1 min each ≈ 5 min)...")
        rows = run_halflife_sweep(panel, model_features)
        print_halflife_report(rows)
        return

    if mode == "shap":
        print("\n  [4/5] Computing SHAP values for current month...")
        shap_vals, err = compute_shap_current(panel, model_features)
        if err:
            print(f"  ERROR: {err}")
            return
        print_shap_report(shap_vals)
        return

    if mode in ("backtest", "both"):
        print("\n  [4/5] Running walk-forward backtest...")
        results, perstock, importance = walk_forward(
            panel, model_features,
            return_perstock=True, return_importance=True
        )
        print_backtest(results)
        evaluate_segments(perstock)
        evaluate_prediction_quality(perstock)
        print_permutation_importance(importance)
        save_perm_importance(importance)

    if mode in ("pick", "both"):
        print("\n  [5/5] Generating current picks...")
        holdings = list(CURRENT_HOLDINGS)
        if holdings:
            print(f"  Current holdings from portfolio_config: {len(holdings)} tickers")
        elif AUTO_ROLL_HOLDINGS:
            holdings = last_logged_picks()
            if holdings:
                print(f"  Auto-roll: using last run's {len(holdings)} picks as current holdings")
        result = predict_now(panel, model_features, price_df, macro_df,
                             current_holdings=holdings)
        if result:
            picks, weights, latest_df, top_features, regime, checks, holdings, shap_vals = result
            te_est = estimate_portfolio_te(picks, weights, price_df)
            print_picks(picks, weights, latest_df, top_features, regime, checks, holdings, shap_vals,
                        te_estimate=te_est)
            report = build_report_text(picks, weights, latest_df, top_features,
                                       regime, checks, holdings, shap_vals,
                                       te_estimate=te_est)
            html_report = build_report_html(picks, weights, latest_df, top_features,
                                            regime, checks, holdings, shap_vals)
            send_report_email(report, html_body=html_report)


if __name__ == "__main__":
    main()

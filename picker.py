"""
TSX Stock Picker — XGBoost + Double Machine Learning
═══════════════════════════════════════════════════════
Data source: yfinance only (free, no API keys needed)
Model: XGBoost regression + classification ensemble
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
from scipy.stats import spearmanr
import sys

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

# Per-sector feature subsets (the X for each sector's model). Listed
# as BASE names — the resolver picks the _norm counterpart at run-time
# where cross_sectional_normalize has produced one. Each sector gets
# the broad momentum/technical/volume features plus its curated set
# of macro control variables from the per-sector spec.
_BASE_SECTOR_FEATURES = [
    "mom_pc1", "mom_pc2",
    "vol_20d", "vol_60d", "vol_ratio",
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
    # Turnover control
    "rank_buffer": 18,
    "score_tolerance": 0.015,
    "max_turnover": 4,
    "hold_bonus": 0.05,
    "cooldown_months": 1,
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
    "tx_cost_bps": 10,
}

FEATURE_COLS = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    "mom_pc1", "mom_pc2",  # only present when USE_MOMENTUM_PCA is True
    "vol_20d", "vol_60d", "vol_ratio",
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


def fetch_prices(tickers, years=7):
    """Download daily OHLCV for all tickers via yfinance."""
    start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, auto_adjust=True,
                       progress=False, threads=True, group_by="ticker")
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


def fetch_macro(years=7):
    """Download macro indicator time series."""
    tickers = list(MACRO_TICKERS.values())
    start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, auto_adjust=True,
                       progress=False, threads=True)
    return data


def fetch_fundamentals(tickers):
    """Fetch current fundamentals from yfinance for constraint filtering."""
    results = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            results[t] = {
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
            results[t] = {}
    return pd.DataFrame(results).T


def fetch_earnings_surprise(ticker):
    """Fetch historical earnings surprises (%) from yfinance.

    Returns a Series indexed by announcement date with the Surprise(%)
    value. Empty Series on any failure (delisted ticker, API error,
    missing column). Used as a Double ML treatment to test whether
    earnings beats/misses have causal alpha beyond what the feature set
    already explains (the post-earnings-announcement drift effect).

    ETFs don't have earnings — skip XIU.TO (the benchmark) explicitly
    so yfinance doesn't print "No earnings dates found" noise.
    """
    if ticker == "XIU.TO":
        return pd.Series(dtype=float)
    try:
        df = yf.Ticker(ticker).get_earnings_dates()
        if df is None or df.empty or "Surprise(%)" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Surprise(%)"].dropna()
        if s.empty:
            return pd.Series(dtype=float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)


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
    """Fetch the dividend payment history from yfinance.

    Returns a Series of date -> dividend amount. Empty Series on
    failure or for non-paying tickers.
    """
    try:
        divs = yf.Ticker(ticker).dividends
        if divs is None or divs.empty:
            return pd.Series(dtype=float)
        divs.index = pd.to_datetime(divs.index).tz_localize(None)
        return divs.sort_index()
    except Exception:
        return pd.Series(dtype=float)


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
    aligned = yoy.reindex(monthly_dates, method="ffill")
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
    result["boc_rate_chg_3m"] = monthly.diff(3).reindex(dates, method="ffill")
    return result


def fetch_quarterly_financials(ticker):
    """Fetch quarterly financials from yfinance for PIT fundamental features."""
    try:
        tk = yf.Ticker(ticker)
        inc = tk.quarterly_income_stmt
        bal = tk.quarterly_balance_sheet
        if inc is None or inc.empty:
            return None
        records = []
        for col in inc.columns[:8]:  # Last 8 quarters
            rec = {"date": col}
            rec["net_income"] = safe_float(inc.loc["Net Income", col]) if "Net Income" in inc.index else np.nan
            rec["revenue"] = safe_float(inc.loc["Total Revenue", col]) if "Total Revenue" in inc.index else np.nan
            rec["ebitda"] = safe_float(inc.loc["EBITDA", col]) if "EBITDA" in inc.index else np.nan
            if bal is not None and col in bal.columns:
                rec["total_equity"] = safe_float(bal.loc["Total Equity Gross Minority Interest", col]) if "Total Equity Gross Minority Interest" in bal.index else np.nan
                rec["total_debt"] = safe_float(bal.loc["Total Debt", col]) if "Total Debt" in bal.index else np.nan
            records.append(rec)
        return pd.DataFrame(records).set_index("date").sort_index()
    except Exception:
        return None


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
                result["oil_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
            elif name == "cad_usd":
                result["cad_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
            elif name == "us10y":
                result["rate_chg_3m"] = monthly.diff(3).reindex(dates, method="ffill")
            elif name == "tsx":
                result["tsx_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
            elif name == "gold":
                result["gold_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
            elif name == "vix":
                result["vix_level"] = monthly.reindex(dates, method="ffill")
            elif name == "natgas":
                result["natgas_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
            elif name == "carbon":
                result["carbon_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
            elif name == "transport":
                result["transport_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
            elif name == "utilities_etf":
                result["util_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
            elif name == "inflation":
                result["tips_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
            elif name == "cad_bonds":
                result["cad_bond_mom_1m"] = monthly.pct_change(1).reindex(dates, method="ffill")
        except (KeyError, TypeError):
            pass

    return result


def compute_pit_fundamentals(ticker, monthly_dates):
    """Point-in-time fundamentals with 45-day reporting lag."""
    qf = fetch_quarterly_financials(ticker)
    if qf is None or len(qf) < 4:
        return pd.DataFrame(index=monthly_dates, columns=["roe", "pe_ratio", "div_yield", "ev_ebitda", "debt_equity"])

    # Apply 45-day lag for point-in-time
    qf.index = qf.index + timedelta(days=45)
    qf = qf.sort_index()

    result = pd.DataFrame(index=monthly_dates)

    # TTM metrics via rolling 4-quarter sum
    if "net_income" in qf.columns and "total_equity" in qf.columns:
        ttm_ni = qf["net_income"].rolling(4).sum()
        equity = qf["total_equity"]
        roe_series = (ttm_ni / equity.replace(0, np.nan))
        result["roe"] = roe_series.reindex(monthly_dates, method="ffill")
    else:
        result["roe"] = np.nan

    # Simplified PE, div_yield, ev/ebitda - will be filled cross-sectionally
    result["pe_ratio"] = np.nan
    result["div_yield"] = np.nan
    result["ev_ebitda"] = np.nan

    if "total_debt" in qf.columns and "total_equity" in qf.columns:
        de = qf["total_debt"] / qf["total_equity"].replace(0, np.nan)
        result["debt_equity"] = de.reindex(monthly_dates, method="ffill")
    else:
        result["debt_equity"] = np.nan

    return result


def encode_sector(ticker):
    """Numeric sector code for the model."""
    sector_map = {
        "Financials": 1, "Energy": 2, "Materials": 3, "Industrials": 4,
        "ConsumerDisc": 5, "ConsumerStaples": 6, "Technology": 7,
        "Communication": 8, "Utilities": 9, "RealEstate": 10, "HealthCare": 11,
    }
    profile = STOCK_PROFILE.get(ticker, ("Unknown", "core", "other"))
    return sector_map.get(profile[0], 0)


def build_panel(price_df, macro_df, tickers):
    """Build the full feature panel for all tickers and months."""
    print("  Building feature panel...")
    all_frames = []

    macro_sample = None
    for t in tickers:
        feat = compute_monthly_features(price_df, t)
        if feat is None:
            continue
        if macro_sample is None:
            macro_sample = feat.index
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


def fit_models(X_train, y_train, sample_weights=None):
    """Train XGBoost regressor + classifier ensemble."""
    # Regression
    reg = make_xgb_regressor()
    reg.fit(X_train, y_train, sample_weight=sample_weights)

    # Classification (top quintile)
    y_cls = (y_train > y_train.quantile(0.8)).astype(int)
    pos_w = (len(y_cls) - y_cls.sum()) / max(y_cls.sum(), 1)
    clf = make_xgb_classifier(pos_weight=pos_w)
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


def fit_sector_models(train_df, sample_weights=None, min_samples=50):
    """Train one XGBoost ensemble per sector using SECTOR_FEATURES.

    Returns dict[sector_name] = (regressor, classifier, feature_cols).
    Sectors with fewer than `min_samples` training rows are skipped;
    `predict_sector_models` returns NaN for tickers in skipped sectors
    and the caller decides the fallback (0 score / drop).
    """
    models = {}
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
        X = sec_df[feats].values
        y = sec_df["fwd_ret"].values
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
    return models


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
         (highest-scoring candidate from that sector, holdings boosted
         by hold_bonus).
      2. Remaining slots filled by score, capped at `max_per_gics` per
         sector.
      3. Total picks <= `top_n`.

    Holdings get a hold_bonus added to their score, biasing them
    toward selection without breaking the sector caps. With 7 bank
    holdings and max_per_gics=2 for Financials, only the 2 highest-
    scored banks survive — the other 5 become implicit "trim"
    recommendations.

    Falls back to plain top_n if `required_sectors` is empty (legacy).
    """
    C = constraints or CONSTRAINTS
    if not new_picks:
        return []

    top_n = C["top_n"]
    max_per_gics = C["max_per_gics"]
    required_sectors = C.get("required_sectors") or []
    hold_bonus = C["hold_bonus"]

    score_dict = dict(zip(new_picks, new_scores))
    for h in (current_holdings or []):
        if h in score_dict:
            score_dict[h] += hold_bonus

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


# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════

def walk_forward(panel, feature_cols, train_months=36, min_train=24):
    """
    Walk-forward backtester with rolling window.
    Returns monthly picks and scores for each period.
    """
    panel = panel.sort_values("date")
    dates = sorted(panel["date"].unique())

    results = []
    holdings = []

    print(f"  Walk-forward: {len(dates)} months, train={train_months}m")

    for i in range(train_months, len(dates) - 1):
        train_start = max(0, i - train_months)
        train_dates = dates[train_start:i]
        test_date = dates[i]

        train_df = panel[panel["date"].isin(train_dates)].copy()
        test_df = panel[panel["date"] == test_date].copy()

        if len(train_df) < min_train * 5 or len(test_df) < 3:
            continue

        # Time decay weights
        n = len(train_df)
        weights = compute_time_decay_weights(n)

        # Fit and predict — per-sector or global depending on the flag
        try:
            if USE_SECTOR_MODELS:
                sec_models = fit_sector_models(train_df, sample_weights=weights)
                scores = predict_sector_models(test_df, sec_models)
                scores = np.nan_to_num(scores, nan=0.0)
            else:
                reg, clf = fit_models(
                    pd.DataFrame(train_df[feature_cols].values, columns=feature_cols),
                    pd.Series(train_df["fwd_ret"].values),
                    sample_weights=weights
                )
                scores = ensemble_predict(reg, clf, test_df[feature_cols].values)
        except Exception:
            continue

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

        # Record actual returns. fwd_ret was computed from raw monthly
        # returns BEFORE cross_sectional_normalize rewrote mom_1m to a rank,
        # so it's the only column that still holds true forward returns.
        pick_rets = test_df[test_df["ticker"].isin(picks)]["fwd_ret"]
        bench_row = test_df[test_df["ticker"] == "XIU.TO"]["fwd_ret"]

        results.append({
            "date": test_date,
            "picks": picks,
            "port_ret": pick_rets.mean() if len(pick_rets) > 0 else 0,
            "bench_ret": bench_row.iloc[0] if len(bench_row) > 0 else 0,
            "n_picks": len(picks),
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════
# PREDICTION (Current Month)
# ══════════════════════════════════════════════════════════════════

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

    if USE_SECTOR_MODELS:
        sec_models = fit_sector_models(train_df, sample_weights=weights)
        scores = predict_sector_models(latest_df, sec_models)
        scores = np.nan_to_num(scores, nan=0.0)
        print(f"  Sector models trained: {sorted(sec_models.keys())}")
    else:
        reg, clf = fit_models(
            pd.DataFrame(train_df[feature_cols].values, columns=feature_cols),
            pd.Series(train_df["fwd_ret"].values),
            sample_weights=weights
        )
        scores = ensemble_predict(reg, clf, latest_df[feature_cols].values)

    latest_df = latest_df.copy()
    latest_df["score"] = scores

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

    # Fetch fundamentals for constraint checking
    print("  Fetching fundamentals for constraint check...")
    fund_df = fetch_fundamentals(candidates[:30])  # Top 30 only

    # Apply constraints
    filtered = apply_constraints(
        candidates, fund_df, price_df, mode="pick",
        current_holdings=current_holdings, constraints=constraints
    )

    # Rebalancing band
    final_picks = apply_rebalancing_band(
        filtered,
        latest_df[latest_df["ticker"].isin(filtered)]["score"].tolist(),
        current_holdings or [],
        constraints=constraints
    )

    # Position sizing
    weights = risk_parity_weights(final_picks, price_df)

    # Feature importance. Under sector models, average importance across
    # all trained sector regressors so the user still gets a single top-10
    # view; under the global model, use the single regressor directly.
    importance = {}
    if USE_SECTOR_MODELS:
        for sector_name, (sec_reg, _clf, feats) in sec_models.items():
            for f, imp in zip(feats, sec_reg.feature_importances_):
                importance[f] = importance.get(f, 0.0) + float(imp)
        # Normalize so the sums are comparable to single-model output
        n_models = max(len(sec_models), 1)
        importance = {f: v / n_models for f, v in importance.items()}
    else:
        importance = dict(zip(feature_cols, reg.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

    return final_picks, weights, latest_df, top_features, regime


# ══════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════

def print_picks(picks, weights, panel_latest, top_features, regime):
    """Print formatted stock picks."""
    print("\n" + "═" * 60)
    print(f"  TSX STOCK PICKS — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  Regime: {regime}")
    print("═" * 60)

    for i, ticker in enumerate(picks, 1):
        w = weights.get(ticker, 0)
        profile = STOCK_PROFILE.get(ticker, ("?", "?", "?"))
        row = panel_latest[panel_latest["ticker"] == ticker]
        score = row["score"].iloc[0] if len(row) > 0 else 0
        print(f"  {i}. {ticker:<12} {profile[0]:<14} {profile[1]:<8} "
              f"Score: {score:.3f}  Weight: {w:.1%}")

    print("\n  Top Features:")
    for feat, imp in top_features:
        print(f"    {feat:<20} {imp:.4f}")
    print("═" * 60)


def print_backtest(results_df):
    """Print backtest summary."""
    if results_df.empty:
        print("  No backtest results.")
        return

    results_df["excess"] = results_df["port_ret"] - results_df["bench_ret"]
    cum_port = (1 + results_df["port_ret"]).cumprod()
    cum_bench = (1 + results_df["bench_ret"]).cumprod()

    total_ret = cum_port.iloc[-1] - 1
    bench_ret = cum_bench.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (12 / len(results_df)) - 1
    ann_bench = (1 + bench_ret) ** (12 / len(results_df)) - 1
    sharpe = results_df["port_ret"].mean() / results_df["port_ret"].std() * np.sqrt(12) if results_df["port_ret"].std() > 0 else 0
    max_dd = (cum_port / cum_port.cummax() - 1).min()
    hit_rate = (results_df["excess"] > 0).mean()

    print("\n" + "═" * 60)
    print("  WALK-FORWARD BACKTEST RESULTS")
    print("═" * 60)
    print(f"  Period:          {results_df['date'].iloc[0].strftime('%Y-%m')} to {results_df['date'].iloc[-1].strftime('%Y-%m')}")
    print(f"  Months:          {len(results_df)}")
    print(f"  Portfolio:       {ann_ret:+.1%} ann. (total {total_ret:+.1%})")
    print(f"  Benchmark:       {ann_bench:+.1%} ann. (total {bench_ret:+.1%})")
    print(f"  Excess:          {ann_ret - ann_bench:+.1%} ann.")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd:.1%}")
    print(f"  Hit Rate:        {hit_rate:.1%}")
    print("═" * 60)

    # Yearly breakdown
    results_df["year"] = results_df["date"].dt.year
    yearly = results_df.groupby("year").agg(
        port=("port_ret", lambda x: (1 + x).prod() - 1),
        bench=("bench_ret", lambda x: (1 + x).prod() - 1),
    )
    yearly["excess"] = yearly["port"] - yearly["bench"]
    print("\n  Year    Portfolio   Benchmark   Excess")
    print("  " + "-" * 44)
    for year, row in yearly.iterrows():
        print(f"  {year}    {row['port']:+.1%}      {row['bench']:+.1%}      {row['excess']:+.1%}")
    print()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pick"
    assert mode in ("pick", "backtest", "both"), f"Usage: python picker.py [pick|backtest|both]"

    print(f"\n{'═' * 60}")
    print(f"  TSX Stock Picker — Mode: {mode.upper()}")
    print(f"  Data source: yfinance | Model: XGBoost + DML")
    print(f"{'═' * 60}\n")

    # Get tickers (exclude benchmark for modeling)
    tickers = [t for t in TSX_UNIVERSE if t != "XIU.TO"]

    # Download data
    print("  [1/5] Downloading price data...")
    all_tickers = TSX_UNIVERSE + list(MACRO_TICKERS.values())
    price_df = fetch_prices(all_tickers, years=7)

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

    if mode in ("backtest", "both"):
        print("\n  [4/5] Running walk-forward backtest...")
        results = walk_forward(panel, model_features)
        print_backtest(results)

    if mode in ("pick", "both"):
        print("\n  [5/5] Generating current picks...")
        if CURRENT_HOLDINGS:
            print(f"  Current holdings from portfolio_config: {len(CURRENT_HOLDINGS)} tickers")
        result = predict_now(panel, model_features, price_df, macro_df,
                             current_holdings=CURRENT_HOLDINGS)
        if result:
            picks, weights, latest_df, top_features, regime = result
            print_picks(picks, weights, latest_df, top_features, regime)


if __name__ == "__main__":
    main()

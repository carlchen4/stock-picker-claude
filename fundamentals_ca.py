"""
Canadian Universe — Deep Fundamental Analysis  (standalone, picker-independent)
==============================================================================
Pulls valuation / profitability / growth / financial-health / analyst data
from yfinance for the 30-name TSX universe, ranks names cross-sectionally,
and writes a composite Value/Quality/Growth screen.

This is a FUNDAMENTAL screen — orthogonal to picker.py, which trades on
price/volume momentum. Use it for context, not as a model input.

Usage:  python3 fundamentals_ca.py
Output: console tables + fundamentals_ca.csv
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf

# --- Universe (ticker -> (company, sector)) — self-contained, no picker import ---
UNIVERSE = {
    # Financials
    "RY.TO": ("Royal Bank of Canada", "Financials"),
    "TD.TO": ("TD Bank", "Financials"),
    "BMO.TO": ("Bank of Montreal", "Financials"),
    "CM.TO": ("CIBC", "Financials"),
    "BNS.TO": ("Scotiabank", "Financials"),
    "NA.TO": ("National Bank of Canada", "Financials"),
    "EQB.TO": ("EQB Inc.", "Financials"),
    "MFC.TO": ("Manulife Financial", "Financials"),
    "SLF.TO": ("Sun Life Financial", "Financials"),
    "FFH.TO": ("Fairfax Financial", "Financials"),
    "BAM.TO": ("Brookfield Asset Mgmt", "Financials"),
    "BN.TO": ("Brookfield Corporation", "Financials"),
    # Energy
    "CNQ.TO": ("Canadian Natural Resources", "Energy"),
    "SU.TO": ("Suncor Energy", "Energy"),
    "CVE.TO": ("Cenovus Energy", "Energy"),
    "ARX.TO": ("ARC Resources", "Energy"),
    "TOU.TO": ("Tourmaline Oil", "Energy"),
    "ENB.TO": ("Enbridge", "Energy"),
    "TRP.TO": ("TC Energy", "Energy"),
    "IMO.TO": ("Imperial Oil", "Energy"),
    # Industrials
    "CNR.TO": ("Canadian National Railway", "Industrials"),
    "CP.TO": ("Canadian Pacific Kansas City", "Industrials"),
    "WSP.TO": ("WSP Global", "Industrials"),
    "TRI.TO": ("Thomson Reuters", "Industrials"),
    "WCN.TO": ("Waste Connections", "Industrials"),
    "CLS.TO": ("Celestica", "Industrials"),
    # Utilities
    "FTS.TO": ("Fortis", "Utilities"),
    "H.TO": ("Hydro One", "Utilities"),
    "EMA.TO": ("Emera", "Utilities"),
    "AQN.TO": ("Algonquin Power", "Utilities"),
}

PCT = lambda x: x * 100 if x is not None and not pd.isna(x) else np.nan


def fetch(ticker):
    """Pull a fundamental snapshot from yfinance .info (cached 24h)."""
    from data_cache import cached_info
    info = cached_info(ticker)

    def g(*keys):
        for k in keys:
            v = info.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        return np.nan

    price = g("currentPrice", "regularMarketPrice")
    tgt = g("targetMeanPrice")
    upside = (tgt / price - 1.0) if (price and tgt and price > 0) else np.nan
    return {
        "mktcap_B": g("marketCap") / 1e9 if g("marketCap") == g("marketCap") else np.nan,
        # Valuation
        "PE": g("trailingPE"),
        "FwdPE": g("forwardPE"),
        "PB": g("priceToBook"),
        "PS": g("priceToSalesTrailing12Months"),
        "EV_EBITDA": g("enterpriseToEbitda"),
        "DivYld%": PCT(g("dividendYield")) if g("dividendYield") and g("dividendYield") < 1 else g("dividendYield"),
        "Payout%": PCT(g("payoutRatio")),
        # Profitability
        "ROE%": PCT(g("returnOnEquity")),
        "ROA%": PCT(g("returnOnAssets")),
        "NetMargin%": PCT(g("profitMargins")),
        "OpMargin%": PCT(g("operatingMargins")),
        # Growth
        "RevGrowth%": PCT(g("revenueGrowth")),
        "EarnGrowth%": PCT(g("earningsGrowth", "earningsQuarterlyGrowth")),
        # Health
        "DebtEq": g("debtToEquity"),
        "CurrRatio": g("currentRatio"),
        "Beta": g("beta"),
        # Analyst
        "TgtUpside%": PCT(upside),
        "Reco": g("recommendationKey"),
        "nAnalysts": g("numberOfAnalystOpinions"),
    }


def pctile_rank(s, higher_better=True):
    r = s.rank(pct=True)
    return r if higher_better else 1 - r


def main():
    print(f"\n{'='*70}\n  Canadian Universe — Deep Fundamental Analysis\n{'='*70}")
    print(f"  Pulling yfinance fundamentals for {len(UNIVERSE)} names...\n")
    rows = []
    for t, (name, sector) in UNIVERSE.items():
        d = fetch(t)
        d["Ticker"], d["Company"], d["Sector"] = t, name, sector
        rows.append(d)
        print(f"    {t:<8} {name[:28]:<28} done")

    df = pd.DataFrame(rows).set_index("Ticker")
    cols_front = ["Company", "Sector", "mktcap_B"]
    df = df[cols_front + [c for c in df.columns if c not in cols_front]]

    # --- Composite percentile score: Value + Quality + Growth ---
    value = (pctile_rank(df["PE"], False).fillna(0.5)
             + pctile_rank(df["PB"], False).fillna(0.5)
             + pctile_rank(df["EV_EBITDA"], False).fillna(0.5)) / 3
    quality = (pctile_rank(df["ROE%"], True).fillna(0.5)
               + pctile_rank(df["NetMargin%"], True).fillna(0.5)
               + pctile_rank(df["DebtEq"], False).fillna(0.5)) / 3
    growth = (pctile_rank(df["RevGrowth%"], True).fillna(0.5)
              + pctile_rank(df["EarnGrowth%"], True).fillna(0.5)) / 2
    df["Value"] = (value * 100).round(0)
    df["Quality"] = (quality * 100).round(0)
    df["Growth"] = (growth * 100).round(0)
    df["Composite"] = ((value + quality + growth) / 3 * 100).round(0)

    pd.set_option("display.width", 200, "display.max_columns", 40)

    print(f"\n{'─'*70}\n  VALUATION  (lower P/E,P/B,P/S,EV/EBITDA = cheaper)\n{'─'*70}")
    print(df[["Company", "Sector", "PE", "FwdPE", "PB", "PS", "EV_EBITDA", "DivYld%"]]
          .round(2).to_string())

    print(f"\n{'─'*70}\n  PROFITABILITY & HEALTH\n{'─'*70}")
    print(df[["ROE%", "ROA%", "NetMargin%", "OpMargin%", "DebtEq", "Payout%"]]
          .round(1).to_string())

    print(f"\n{'─'*70}\n  GROWTH & ANALYST\n{'─'*70}")
    print(df[["RevGrowth%", "EarnGrowth%", "TgtUpside%", "Reco", "nAnalysts"]]
          .round(1).to_string())

    print(f"\n{'─'*70}\n  COMPOSITE SCREEN  (0-100 percentile within universe; higher=better)\n{'─'*70}")
    rank = df[["Company", "Sector", "Value", "Quality", "Growth", "Composite"]] \
        .sort_values("Composite", ascending=False)
    print(rank.to_string())

    print(f"\n  By sector — avg composite:")
    print(df.groupby("Sector")["Composite"].mean().round(0).sort_values(ascending=False).to_string())

    df.to_csv("fundamentals_ca.csv")
    print(f"\n  Saved full table -> fundamentals_ca.csv\n")


if __name__ == "__main__":
    main()

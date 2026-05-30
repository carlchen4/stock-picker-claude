"""
Canadian Financials — Fundamental Score & Rank (ALL 12, not just banks)
=======================================================================
Covers banks + insurers + asset managers using CROSS-TYPE common metrics
from yfinance (the only fields comparable across all three business models):
  ROE · P/B · P/E · Dividend yield · Revenue growth · EPS growth · Beta

Type-specific deep metrics live elsewhere:
  • Banks (PCL / CET1 / segments)  -> bank_score_ca.py  (7 banks)
  • Insurers (LICAT / new-biz value) and Asset mgrs (AUM / fee earnings)
    are NOT in yfinance — note only.

Scoring = percentile rank WITHIN the 12 financials, weighted:
  ROE 30 · P/B 20 (cheaper=better) · Rev growth 15 · EPS growth 15 ·
  Div yield 10 · P/E 10 (cheaper=better).   Higher score = better.

Usage: python3 financials_score_ca.py   ->  console + financials_score_ca.csv
"""
import time
import numpy as np
import pandas as pd
import yfinance as yf

# ticker -> (name, type)
UNIV = {
    "RY.TO":  ("Royal Bank of Canada", "Bank"),
    "TD.TO":  ("TD Bank", "Bank"),
    "BMO.TO": ("Bank of Montreal", "Bank"),
    "CM.TO":  ("CIBC", "Bank"),
    "BNS.TO": ("Scotiabank", "Bank"),
    "NA.TO":  ("National Bank", "Bank"),
    "EQB.TO": ("EQB (Equitable)", "Bank"),
    "MFC.TO": ("Manulife", "Insurer"),
    "SLF.TO": ("Sun Life", "Insurer"),
    "FFH.TO": ("Fairfax", "Insurer"),
    "BAM.TO": ("Brookfield Asset Mgmt", "AssetMgr"),
    "BN.TO":  ("Brookfield Corp", "AssetMgr"),
}

WEIGHTS = {"ROE":0.30, "PB":0.20, "RevGr":0.15, "EPSGr":0.15, "DivYld":0.10, "PE":0.10}


def fetch(t):
    try:
        i = yf.Ticker(t).info
    except Exception:
        i = {}
    g = lambda *k: next((i[x] for x in k if i.get(x) is not None
                         and not (isinstance(i[x], float) and np.isnan(i[x]))), np.nan)
    dy = g("dividendYield")
    if dy == dy and dy > 1:   # some feeds give %, some fraction
        dy = dy
    elif dy == dy:
        dy = dy * 100
    return {
        "ROE":   g("returnOnEquity"),
        "PB":    g("priceToBook"),
        "PE":    g("trailingPE"),
        "RevGr": g("revenueGrowth"),
        "EPSGr": g("earningsGrowth", "earningsQuarterlyGrowth"),
        "DivYld": dy,
        "Beta":  g("beta"),
    }


def pr(s, higher=True):
    s = s.astype(float)
    rk = s.rank(pct=True)
    return (rk if higher else 1 - rk).fillna(0.5) * 100


def main():
    print(f"\n{'='*74}\n  Canadian Financials — Fundamental Score (ALL 12)\n{'='*74}")
    rows = []
    for t, (name, typ) in UNIV.items():
        d = fetch(t); d.update(Ticker=t, Name=name, Type=typ)
        rows.append(d); print(f"    {t:<8} {name[:26]:<26} {typ}"); time.sleep(0.2)
    df = pd.DataFrame(rows).set_index("Ticker")

    sc = pd.DataFrame(index=df.index)
    sc["ROE"]   = pr(df["ROE"], True)
    sc["PB"]    = pr(df["PB"], False)
    sc["PE"]    = pr(df["PE"], False)
    sc["RevGr"] = pr(df["RevGr"], True)
    sc["EPSGr"] = pr(df["EPSGr"], True)
    sc["DivYld"]= pr(df["DivYld"], True)
    sc["SCORE"] = sum(sc[m]*w for m, w in WEIGHTS.items())
    out = df.join(sc[["SCORE"]]).sort_values("SCORE", ascending=False)

    pd.set_option("display.width", 200)
    show = out.copy()
    show["ROE%"] = (show["ROE"]*100).round(1)
    show["DivY%"] = show["DivYld"].round(1)
    show["RevGr%"] = (show["RevGr"]*100).round(1)
    cols = ["Name","Type","ROE%","PB","PE","RevGr%","DivY%","SCORE"]
    print(f"\n{'─'*74}\n  RANKED (score = percentile within the 12, weighted)\n{'─'*74}")
    print(show[cols].round({"PB":2,"PE":1,"SCORE":0}).to_string())

    print(f"\n  By type — avg score:")
    print(out.groupby("Type")["SCORE"].mean().round(0).sort_values(ascending=False).to_string())
    out.to_csv("financials_score_ca.csv")
    print(f"\n  Saved -> financials_score_ca.csv")
    print("  NOTE: bank-specific PCL/CET1/segment ranking -> bank_score_ca.py (7 banks)\n")


if __name__ == "__main__":
    main()

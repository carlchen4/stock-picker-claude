"""
US Tech Universe — Fundamental Score & Rank  (standalone, picker-independent)
============================================================================
21-name US tech universe, scored on TECH-appropriate metrics from yfinance:
  GROWTH   35% : revenue growth 20 + earnings growth 15
  QUALITY  35% : gross margin 12 + net margin 11 + ROE 12
  VALUE    30% : P/S 15 + EV/EBITDA 15   (cheaper = better)

Tech weighting favours growth + margins over cheapness (unlike banks).
Score = weighted percentile WITHIN the 21 names; higher = better.

Usage:  python3 fundamentals_us.py   ->  console + fundamentals_us.csv
"""
import time
import numpy as np
import pandas as pd
import yfinance as yf

UNIV = {
    # Cloud / Internet
    "MSFT": ("Microsoft", "Cloud"), "AMZN": ("Amazon", "Cloud"),
    "META": ("Meta", "Cloud"), "GOOGL": ("Alphabet", "Cloud"),
    "ANET": ("Arista", "Cloud"), "PLTR": ("Palantir", "Cloud"),
    "NET": ("Cloudflare", "Cloud"), "SNOW": ("Snowflake", "Cloud"),
    # Semiconductors
    "NVDA": ("NVIDIA", "Semi"), "AMD": ("AMD", "Semi"), "AVGO": ("Broadcom", "Semi"),
    "QCOM": ("Qualcomm", "Semi"), "TXN": ("Texas Instruments", "Semi"),
    "AMAT": ("Applied Materials", "Semi"), "MU": ("Micron", "Semi"),
    "LRCX": ("Lam Research", "Semi"), "TSM": ("TSMC", "Semi"),
    # Hardware
    "AAPL": ("Apple", "Hardware"), "TSLA": ("Tesla", "Hardware"),
    "ARM": ("Arm Holdings", "Hardware"), "INTC": ("Intel", "Hardware"),
}

WEIGHTS = {"RevGr":0.20, "EarnGr":0.15,          # growth 35
           "GrossM":0.12, "NetM":0.11, "ROE":0.12,  # quality 35
           "PS":0.15, "EVEBITDA":0.15}             # value 30 (cheaper better)


def fetch(t):
    from data_cache import cached_info
    i = cached_info(t)
    g = lambda *k: next((i[x] for x in k if i.get(x) is not None
                         and not (isinstance(i[x], float) and np.isnan(i[x]))), np.nan)
    return {
        "RevGr":   g("revenueGrowth"),
        "EarnGr":  g("earningsGrowth", "earningsQuarterlyGrowth"),
        "GrossM":  g("grossMargins"),
        "NetM":    g("profitMargins"),
        "ROE":     g("returnOnEquity"),
        "PS":      g("priceToSalesTrailing12Months"),
        "EVEBITDA": g("enterpriseToEbitda"),
        "PE":      g("trailingPE"),
        "FCF":     g("freeCashflow"),
        "Rev":     g("totalRevenue"),
    }


def pr(s, higher=True):
    s = s.astype(float)
    rk = s.rank(pct=True)
    return (rk if higher else 1 - rk).fillna(0.5) * 100


def main():
    print(f"\n{'='*72}\n  US Tech — Fundamental Score (21 names)\n{'='*72}")
    rows = []
    for t, (name, sub) in UNIV.items():
        d = fetch(t); d.update(Ticker=t, Name=name, Sub=sub)
        rows.append(d); print(f"    {t:<6} {name[:22]:<22} {sub}")
    df = pd.DataFrame(rows).set_index("Ticker")

    sc = pd.DataFrame(index=df.index)
    sc["RevGr"]    = pr(df["RevGr"], True)
    sc["EarnGr"]   = pr(df["EarnGr"], True)
    sc["GrossM"]   = pr(df["GrossM"], True)
    sc["NetM"]     = pr(df["NetM"], True)
    sc["ROE"]      = pr(df["ROE"], True)
    sc["PS"]       = pr(df["PS"], False)
    sc["EVEBITDA"] = pr(df["EVEBITDA"], False)
    sc["SCORE"] = sum(sc[m]*w for m, w in WEIGHTS.items())
    out = df.join(sc[["SCORE"]]).sort_values("SCORE", ascending=False)

    pd.set_option("display.width", 200)
    show = out.copy()
    show["RevGr%"]  = (show["RevGr"]*100).round(0)
    show["GrossM%"] = (show["GrossM"]*100).round(0)
    show["NetM%"]   = (show["NetM"]*100).round(0)
    show["ROE%"]    = (show["ROE"]*100).round(0)
    cols = ["Name","Sub","RevGr%","GrossM%","NetM%","ROE%","PS","EVEBITDA","SCORE"]
    print(f"\n{'─'*72}\n  RANKED (percentile within 21, growth+quality weighted)\n{'─'*72}")
    print(show[cols].round({"PS":1,"EVEBITDA":1,"SCORE":0}).to_string())

    print(f"\n  By sub-sector — avg score:")
    print(out.groupby("Sub")["SCORE"].mean().round(0).sort_values(ascending=False).to_string())

    # ── Rule of 40 — Cloud/SaaS only ────────────────────────────────────
    # rev growth% + FCF margin% >= 40 = healthy growth/profit balance.
    # Applies to software economics only; semis (cyclical) & hardware excluded.
    cloud = out[out["Sub"] == "Cloud"].copy()
    cloud["FCFm"] = cloud["FCF"] / cloud["Rev"]
    cloud["Rule40"] = (cloud["RevGr"].astype(float) + cloud["FCFm"]) * 100
    cloud["Pass"] = np.where(cloud["Rule40"] >= 40, "PASS", "fail")
    # HAND layer: net revenue retention (us_data.csv) — a SaaS-only metric
    # yfinance can't give; hyperscalers/networking don't report it (shown "—").
    nrr = {}
    try:
        hd = pd.read_csv("us_data.csv")
        nrr = dict(zip(hd[hd["kpi"] == "nrr"]["ticker"],
                       hd[hd["kpi"] == "nrr"]["value"]))
    except FileNotFoundError:
        pass
    cloud["NRR%"] = [f"{nrr[t]:.0f}" if t in nrr else "—" for t in cloud.index]
    cloud = cloud.sort_values("Rule40", ascending=False)
    cloud["RevGr%"] = (cloud["RevGr"].astype(float)*100).round(0)
    cloud["FCFm%"]  = (cloud["FCFm"]*100).round(0)
    print(f"\n{'─'*72}\n  CLOUD — Rule of 40 (rev growth%+FCF margin%>=40) + NRR (net rev retention)\n{'─'*72}")
    print(cloud[["Name","RevGr%","FCFm%","Rule40","Pass","NRR%"]].round({"Rule40":0}).to_string())

    out.to_csv("fundamentals_us.csv")
    print(f"\n  Saved -> fundamentals_us.csv\n")


if __name__ == "__main__":
    main()

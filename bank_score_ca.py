"""
Canadian Banks — Fundamental Score & Rank
=========================================
TWO data sources, clean separation:
  • AUTO  (yfinance, no typing): ROE(TTM) · Revenue YoY · Operating leverage~
                                 · BVPS YoY
  • HAND  (you fill each quarter from the earnings reports, in
           bank_data_ca.csv): CET1 % · PCL ($M / bps / YoY%) · segment_quality

  -> Each quarter you ONLY edit bank_data_ca.csv. Run this; it auto-pulls the
     rest from yfinance and recomputes the ranking.

Weights (importance tiers): ROE 25 · PCL 20 · CET1 15 · OperLev 10 ·
                            Segment 10 · RevGrowth 10 · BVPS 10

Usage: python3 bank_score_ca.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

CSV = "bank_data_ca.csv"
NAMES = {"RY.TO":"Royal Bank","CM.TO":"CIBC","NA.TO":"National Bank",
         "TD.TO":"TD","BNS.TO":"Scotiabank","BMO.TO":"BMO","EQB.TO":"EQB"}
WEIGHTS = {"ROE":0.25, "PCL":0.20, "CET1":0.15,
           "OperLev":0.10, "Segment":0.10, "RevGrowth":0.10, "BVPS":0.10}


def _row(df, name, col):
    try:
        v = df.loc[name, col]
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan


def auto_metrics(t):
    """ROE(TTM), Revenue YoY, Operating leverage~, BVPS YoY from yfinance."""
    inc = yf.Ticker(t).quarterly_income_stmt
    bal = yf.Ticker(t).quarterly_balance_sheet
    if inc is None or inc.empty:
        return dict(ROE=np.nan, RevYoY=np.nan, OperLev=np.nan, BVPS=np.nan)
    q = list(inc.columns)[:5]
    eq_names = ["Common Stock Equity", "Stockholders Equity",
                "Total Equity Gross Minority Interest"]
    sh_names = ["Ordinary Shares Number", "Share Issued"]

    ni_ttm = np.nansum([_row(inc, "Net Income", c) for c in q[:4]])
    eq0 = next((_row(bal, n, q[0]) for n in eq_names if pd.notna(_row(bal, n, q[0]))), np.nan)
    roe = ni_ttm/eq0*100 if eq0 else np.nan

    rev0 = _row(inc, "Total Revenue", q[0])
    rev4 = _row(inc, "Total Revenue", q[4]) if len(q) >= 5 else np.nan
    rev_yoy = (rev0/rev4-1)*100 if rev4 else np.nan

    pti0, pti4 = _row(inc, "Pretax Income", q[0]), (_row(inc, "Pretax Income", q[4]) if len(q) >= 5 else np.nan)
    c0, c4 = rev0-pti0, (rev4-pti4 if pd.notna(rev4) and pd.notna(pti4) else np.nan)
    oplev = ((rev0/rev4-1)-(c0/c4-1))*100 if (rev4 and c4) else np.nan

    sh0 = next((_row(bal, n, q[0]) for n in sh_names if pd.notna(_row(bal, n, q[0]))), np.nan)
    bvps0 = eq0/sh0 if sh0 else np.nan
    bvps = np.nan
    if len(q) >= 5:
        eq4 = next((_row(bal, n, q[4]) for n in eq_names if pd.notna(_row(bal, n, q[4]))), np.nan)
        sh4 = next((_row(bal, n, q[4]) for n in sh_names if pd.notna(_row(bal, n, q[4]))), np.nan)
        if eq4 and sh4 and bvps0:
            bvps = (bvps0/(eq4/sh4)-1)*100
    return dict(ROE=roe, RevYoY=rev_yoy, OperLev=oplev, BVPS=bvps)


def pcl_score(r):
    """0-100 from raw PCL facts: improving YoY + low bps = good (higher)."""
    s = 50.0
    yoy = r.get("pcl_yoy_pct")
    if pd.notna(yoy):
        s += -yoy * 1.2          # PCL down (neg yoy) -> score up
    bps = r.get("pcl_bps")
    if pd.notna(bps):
        s += (45 - bps) * 0.8    # below ~45bps good, above bad
    return float(np.clip(s, 0, 100))


def minmax(s):
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    out = (s-lo)/(hi-lo)*100 if hi > lo else s*0+50
    return out.fillna(out.median())


def main():
    hand = pd.read_csv(CSV).set_index("ticker")
    print(f"\n  Hand-filled report data read from {CSV}; pulling yfinance...\n")
    rows = {}
    for t in hand.index:
        m = auto_metrics(t)
        m["CET1"] = float(hand.loc[t, "cet1_pct"])
        m["PCL_q"] = pcl_score(hand.loc[t])
        m["SEG_q"] = float(hand.loc[t, "segment_quality"])
        m["note"] = hand.loc[t, "note"]
        rows[t] = m
    df = pd.DataFrame(rows).T

    sc = pd.DataFrame(index=df.index)
    sc["ROE"]       = minmax(df["ROE"])
    sc["CET1"]      = minmax(df["CET1"])
    sc["RevGrowth"] = minmax(df["RevYoY"])
    sc["OperLev"]   = minmax(df["OperLev"])
    sc["BVPS"]      = minmax(df["BVPS"])
    sc["PCL"]       = df["PCL_q"].astype(float)
    sc["Segment"]   = df["SEG_q"].astype(float)
    sc["SCORE"] = sum(sc[m]*w for m, w in WEIGHTS.items())
    sc = sc.sort_values("SCORE", ascending=False)

    pd.set_option("display.width", 200)
    print(f"{'='*78}\n  Canadian Banks — Fundamental Score & Rank\n{'='*78}")
    print("  Weights: ROE 25 | PCL 20 | CET1 15 | OperLev 10 | Segment 10 | RevGr 10 | BVPS 10")
    print("  AUTO=yfinance(ROE,RevGr,OperLev,BVPS)  HAND=bank_data_ca.csv(CET1,PCL,Segment)\n")
    disp = sc.copy(); disp["Bank"] = [NAMES.get(t, t) for t in disp.index]
    print(disp[["Bank","ROE","PCL","CET1","OperLev","Segment","RevGrowth","BVPS","SCORE"]].round(0).to_string())
    print(f"\n{'─'*78}\n  RANK\n{'─'*78}")
    for i, t in enumerate(sc.index, 1):
        print(f"  {i}. {NAMES.get(t,t):<14} {sc.loc[t,'SCORE']:>5.1f}   {df.loc[t,'note']}")
    print()


if __name__ == "__main__":
    main()

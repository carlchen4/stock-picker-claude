"""Sector fundamental scorecards (CA) — Energy / Industrials / Utilities.

Mirror of bank_score_ca.py for the non-financial sectors of TSX_UNIVERSE.
Each sector is scored on the fundamentals that actually drive it, min-max
normalized within the sector, then weighted into a 0-100 SCORE + rank.

  • Energy:      capital discipline + cash generation (leverage, FCF yield,
                 ROE, EBITDA margin, dividend, revenue growth)
  • Industrials: quality compounding (ROE, operating margin, revenue growth,
                 FCF yield, leverage, EBITDA margin)
  • Utilities:   dividend safety + balance sheet (yield, payout safety,
                 leverage, ROE, growth, EBITDA margin)

All metrics from yfinance .info (no hand-filled data). Net debt/EBITDA and
FCF yield are derived. "Lower is better" metrics (leverage, payout) are
sign-flipped before normalizing so higher SCORE is always better.

Usage: python3 sector_score_ca.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import picker  # for TSX_UNIVERSE, STOCK_PROFILE, company names

NAMES = dict(getattr(picker, "COMPANY_NAMES", {}))
NAMES.update({  # gold names absent from picker.COMPANY_NAMES
    "AEM.TO": "Agnico Eagle", "ABX.TO": "Barrick", "WPM.TO": "Wheaton PM",
    "FNV.TO": "Franco-Nevada",
})

# Sector → list of tickers from the live universe (pipelines kept in Energy)
def sector_tickers(sector):
    return [t for t in picker.TSX_UNIVERSE
            if t != "XIU.TO" and picker.STOCK_PROFILE.get(t, ("",))[0] == sector]

# ── Hand-filled B-tier KPIs (sector_data_ca.csv) ─────────────────────────
# Operating metrics yfinance can't give (AISC, rate-base growth, % contracted,
# breakeven). Loaded as a per-ticker dict; folded into the score as ONE extra
# weighted "moat" metric per sector. Missing/VERIFY rows → neutral (50).
import os
_HAND = {}  # ticker -> {kpi: (value, direction)}
_HAND_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sector_data_ca.csv")
try:
    _hd = pd.read_csv(_HAND_CSV)
    for _, r in _hd.iterrows():
        v = pd.to_numeric(r.get("value"), errors="coerce")
        if pd.notna(v):
            _HAND.setdefault(r["ticker"], {})[r["kpi"]] = (float(v), r["direction"])
except FileNotFoundError:
    pass

# Per sector: weight of the hand "moat" metric + a label. Industrials has no
# hand data yet → weight 0 (unchanged). Energy blends two KPIs onto one
# higher-is-better "structural quality" scale (pct_contracted as-is; oil
# breakeven mapped to 100-BE so lower cost scores higher).
HAND_CFG = {
    "Energy":      (0.15, "HandKPI"),
    "Utilities":   (0.20, "RateBaseG%"),
    "Materials":   (0.20, "UnitCost$"),
}

def _anchor(val, lo, hi):
    """Map val to 0-100 between lo→0 and hi→100 (clip). lo>hi inverts."""
    z = (val - lo)/(hi - lo)
    return float(min(100.0, max(0.0, z*100)))

def hand_quality(sector, ticker):
    """Return (quality_0to100 | nan, display_str) for the ticker's hand KPI.

    Each KPI is mapped to an ABSOLUTE 0-100 score against sensible anchors
    (not cross-ticker min-max) so heterogeneous Energy KPIs — pipeline
    %-contracted vs oil breakeven — sit on one comparable quality scale.
    """
    kv = _HAND.get(ticker, {})
    if sector == "Materials":                      # unit cost, lower=better
        for k in ("AISC", "cash_cost_geo"):
            if k in kv:
                v = kv[k][0]
                return _anchor(v, 1800, 300), f"{v:.0f}"   # $1800→0, $300→100
    elif sector == "Utilities":                    # rate-base CAGR, higher=better
        if "rate_base_cagr" in kv:
            v = kv["rate_base_cagr"][0]
            return _anchor(v, 4, 9), f"{v:.1f}"            # 4%→0, 9%→100
    elif sector == "Energy":
        if "pct_contracted" in kv:                 # pipeline cash-flow certainty
            v = kv["pct_contracted"][0]
            return _anchor(v, 50, 100), f"ct{v:.0f}%"      # 50%→0, 100%→100
        if "wti_breakeven" in kv:                  # oil breakeven, lower=better
            v = kv["wti_breakeven"][0]
            return _anchor(v, 60, 30), f"be${v:.0f}"       # $60→0, $30→100
    return np.nan, "—"

# (metric_key, weight, higher_is_better)
SECTOR_METRICS = {
    "Energy": [
        ("fcf_yield", 0.20, True),    # FCF / market cap
        ("ffo_debt",  0.20, True),    # cash-based credit (fair to pipelines)
        ("roe",       0.15, True),
        ("int_cov",   0.10, True),    # leverage serviceability
        ("nd_ebitda", 0.10, False),   # net debt / EBITDA
        ("ebitda_m",  0.10, True),
        ("div_yield", 0.10, True),
        ("rev_g",     0.05, True),
    ],
    "Industrials": [
        ("roe",       0.25, True),
        ("op_margin", 0.20, True),    # = inverse operating ratio (rails)
        ("rev_g",     0.20, True),
        ("fcf_yield", 0.15, True),
        ("int_cov",   0.10, True),
        ("nd_ebitda", 0.10, False),
    ],
    "Utilities": [
        ("ffo_debt",  0.20, True),    # S&P's key util credit metric (~13% line)
        ("int_cov",   0.15, True),
        ("div_yield", 0.15, True),
        ("payout",    0.15, False),   # lower payout = safer dividend
        ("nd_ebitda", 0.15, False),
        ("eps_g",     0.10, True),    # rate-base growth proxy
        ("roe",       0.10, True),
    ],
    # Gold (STOCK_PROFILE sector = "Materials"). No AISC in yfinance, so
    # proxy mine economics with margin (separates royalty/streamers from
    # producers), cash generation, balance-sheet strength, and growth.
    "Materials": [
        ("fcf_yield", 0.25, True),
        ("ebitda_m",  0.20, True),    # royalty/streamers run ~80%+ margins
        ("roe",       0.20, True),
        ("rev_g",     0.15, True),    # production + gold-price driven
        ("cash_conv", 0.10, True),    # earnings quality (FFO/debt is N/A — net cash)
        ("nd_ebitda", 0.05, False),   # miners carry little debt → low weight
        ("div_yield", 0.05, True),
    ],
}

# Display titles (STOCK_PROFILE sector name → human label)
DISPLAY = {"Materials": "Gold"}

LABELS = {"nd_ebitda":"NetDbt/EBITDA","fcf_yield":"FCF_yld%","roe":"ROE%",
          "ebitda_m":"EBITDA_m%","div_yield":"Div%","rev_g":"RevGr%",
          "op_margin":"OpMgn%","payout":"Payout%","eps_g":"EPSGr%",
          "ffo_debt":"FFO/Debt%","int_cov":"IntCov_x","cash_conv":"CashConv"}


def pull(t):
    tk = yf.Ticker(t)
    i = tk.info
    mc  = i.get("marketCap") or np.nan
    ebd = i.get("ebitda") or np.nan
    debt= i.get("totalDebt") or 0.0
    cash= i.get("totalCash") or 0.0
    fcf = i.get("freeCashflow") or np.nan
    ocf = i.get("operatingCashflow") or np.nan
    ni  = i.get("netIncomeToCommon") or np.nan
    # Compute yield from rate/price (unambiguous across yfinance versions;
    # .dividendYield units vary). Fall back to 0 for non-payers.
    rate = i.get("dividendRate")
    price = i.get("currentPrice") or i.get("regularMarketPrice")
    dy = (100*rate/price if rate and price else 0.0)
    # Interest coverage = EBIT / |interest expense| from the income
    # statement (a separate, slower call). Serviceability of leverage —
    # the metric net-debt/EBITDA can't show. NaN if unavailable.
    int_cov = np.nan
    try:
        fin = tk.financials
        ebit = fin.loc["EBIT"].iloc[0] if "EBIT" in fin.index else np.nan
        ie   = fin.loc["Interest Expense"].iloc[0] if "Interest Expense" in fin.index else np.nan
        if ebit == ebit and ie == ie and ie:
            int_cov = ebit/abs(ie)
    except Exception:
        pass
    return {
        "nd_ebitda": (debt - cash)/ebd if ebd and ebd == ebd else np.nan,
        "fcf_yield": 100*fcf/mc if fcf == fcf and mc == mc else np.nan,
        "ffo_debt":  100*ocf/debt if ocf == ocf and debt else np.nan,  # cash-based credit metric
        "int_cov":   int_cov,
        "cash_conv": ocf/ni if ocf == ocf and ni == ni and ni else np.nan,  # earnings quality
        "roe":       100*(i.get("returnOnEquity") or np.nan),
        "ebitda_m":  100*(i.get("ebitdaMargins") or np.nan),
        "op_margin": 100*(i.get("operatingMargins") or np.nan),
        "div_yield": dy if dy is not None else np.nan,
        "payout":    100*(i.get("payoutRatio") or np.nan),
        "rev_g":     100*(i.get("revenueGrowth") or np.nan),
        "eps_g":     100*(i.get("earningsGrowth") or np.nan),
    }


def minmax(s, higher_better=True):
    s = s.astype(float)
    lo, hi = s.min(), s.max()           # min/max skip NaN
    if hi == lo or pd.isna(hi) or pd.isna(lo):
        return pd.Series(50.0, index=s.index)
    z = (s - lo)/(hi - lo)
    out = (z if higher_better else 1 - z) * 100
    return out.fillna(50.0)             # missing metric → neutral, never NaN-out the SCORE


def score_sector(sector):
    tickers = sector_tickers(sector)
    rows = {t: pull(t) for t in tickers}
    df = pd.DataFrame(rows).T
    metrics = SECTOR_METRICS[sector]
    hand_w, hand_lbl = HAND_CFG.get(sector, (0.0, ""))
    auto_w = 1.0 - hand_w  # auto (yfinance) metrics share the rest of the weight

    sc = pd.Series(0.0, index=df.index)
    for key, w, hib in metrics:
        sc += auto_w * w * minmax(df[key], hib)

    # Hand "moat" metric: one extra weighted term from sector_data_ca.csv
    if hand_w > 0:
        hq = {t: hand_quality(sector, t) for t in df.index}
        df["_hand_raw"] = pd.Series({t: hq[t][0] for t in df.index})
        df["_hand_disp"] = pd.Series({t: hq[t][1] for t in df.index})
        sc += hand_w * df["_hand_raw"].fillna(50.0)  # already 0-100, neutral=50

    df["SCORE"] = sc
    df = df.sort_values("SCORE", ascending=False)

    cols = [m[0] for m in metrics]
    disp = df[cols].copy()
    if hand_w > 0:
        disp[hand_lbl] = df["_hand_disp"]
    disp["SCORE"] = df["SCORE"]
    disp.insert(0, "Name", [NAMES.get(t, t)[:18] for t in disp.index])
    disp.columns = ["Name"] + [LABELS.get(c, c) for c in cols] + \
                   ([hand_lbl] if hand_w > 0 else []) + ["SCORE"]

    wtxt = " | ".join(f"{LABELS.get(k,k)} {int(auto_w*w*100)}" for k, w, _ in metrics)
    if hand_w > 0:
        wtxt += f" | {hand_lbl} {int(hand_w*100)} (hand)"
    title = DISPLAY.get(sector, sector)
    print(f"\n{'='*92}\n  Canadian {title} — Fundamental Score & Rank   (yfinance, {pd.Timestamp.today().date()})")
    print(f"  Weights: {wtxt}\n{'='*92}")
    print(disp.round(1).to_string())
    print(f"\n  RANK: " + "  ".join(
        f"{i+1}.{NAMES.get(t,t).split()[0]}({df.loc[t,'SCORE']:.0f})"
        for i, t in enumerate(df.index)))


if __name__ == "__main__":
    for sec in ["Energy", "Industrials", "Utilities", "Materials"]:
        score_sector(sec)
    print()

"""
Canadian Big-6 Banks — Recent Earnings Trend  (standalone)
==========================================================
Pulls the line items yfinance DOES expose for banks and shows the recent
quarterly trajectory + YoY / QoQ growth.

IMPORTANT — yfinance does NOT carry these bank-critical items; they live
only in each bank's quarterly press release / MD&A (investor-relations
pages or SEDAR+):
    • Provision for Credit Losses (PCL / 准备金)
    • Loan & Deposit balances (贷款 / 存款)
    • Capital Markets segment revenue (资本市场分部)
    • CET1 capital ratio
Use this script for the income-trend it CAN give; pull the four items above
from the actual Q2-2026 reports.

Usage: python3 bank_earnings_ca.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

BANKS = {
    "RY.TO": "Royal Bank of Canada", "TD.TO": "TD Bank",
    "BMO.TO": "Bank of Montreal", "CM.TO": "CIBC",
    "BNS.TO": "Scotiabank", "NA.TO": "National Bank",
}
ROWS = ["Total Revenue", "Net Interest Income", "Net Income", "Diluted EPS"]


def grab(inc, row, col):
    try:
        if row in inc.index and col in inc.columns:
            v = inc.loc[row, col]
            return float(v) if pd.notna(v) else np.nan
    except Exception:
        pass
    return np.nan


def main():
    print(f"\n{'='*72}\n  Canadian Big-6 — Recent Earnings Trend (yfinance line items)\n{'='*72}")
    for t, name in BANKS.items():
        inc = yf.Ticker(t).quarterly_income_stmt
        if inc is None or inc.empty:
            print(f"\n  {t}: no data"); continue
        cols = list(inc.columns)[:5]          # latest 5 quarters
        print(f"\n  ── {name} ({t}) ──  latest quarter: {cols[0].date()}")
        # build per-metric row across quarters (in $B where large)
        hdr = "    " + f"{'metric':<20}" + "".join(f"{str(c.date()):>13}" for c in cols)
        print(hdr); print("    " + "-" * (20 + 13 * len(cols)))
        latest_vals = {}
        for r in ROWS:
            vals = [grab(inc, r, c) for c in cols]
            latest_vals[r] = vals
            def fmt(v):
                if pd.isna(v): return f"{'—':>13}"
                if r == "Diluted EPS": return f"{v:>13.2f}"
                return f"{v/1e9:>12.2f}B"
            print(f"    {r:<20}" + "".join(fmt(v) for v in vals))
        # YoY (latest vs 4 quarters ago) + QoQ (latest vs prior)
        print("    " + "-" * (20 + 13 * len(cols)))
        for r in ROWS:
            v = latest_vals[r]
            yoy = (v[0]/v[4]-1) if len(v) >= 5 and v[4] and v[0] and v[4] != 0 else np.nan
            qoq = (v[0]/v[1]-1) if len(v) >= 2 and v[1] and v[0] and v[1] != 0 else np.nan
            yoy_s = f"{yoy:+.1%}" if pd.notna(yoy) else "n/a"
            qoq_s = f"{qoq:+.1%}" if pd.notna(qoq) else "n/a"
            print(f"    {r:<20}  YoY {yoy_s:>8}   QoQ {qoq_s:>8}")

    print(f"\n{'='*72}")
    print("  NOT in yfinance — pull from each bank's Q2-2026 press release / MD&A:")
    print("    PCL (准备金) · Loans/Deposits · Capital Markets segment · CET1 ratio")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()

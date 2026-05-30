"""
Canadian Big-6 Banks — Deep Fundamental Dashboard (yfinance-computable part)
===========================================================================
Covers the checklist items obtainable from yfinance:
  Revenue · EPS · ROE · BVPS (+growth) · Dividend growth · Operating cash flow
  across a multi-quarter time frame, with YoY.

──────────────────────────────────────────────────────────────────────────
  yfinance / Yahoo CANNOT fetch these (bank-critical) — get from the report:
──────────────────────────────────────────────────────────────────────────
  GAP                               WHERE TO GET IT
  • PCL (准备金 / provision)         earnings press release + MD&A
  • Revenue by segment              MD&A "Strategic business units" table
    (Personal/Commercial/Wealth/     (Capital Markets, Wealth, P&C ...)
     Capital Markets)
  • PPPT (pre-provision pre-tax)    = pretax income + PCL  (need PCL first)
  • CET1 capital ratio              press release "capital" section / OSFI
  • Net interest margin (NIM)       MD&A
  • Efficiency ratio                MD&A (non-int expense / revenue)
  • Gross/impaired loans, deposits  MD&A balance-sheet detail
  • Dividend per share (reliable)   press release — yfinance TTM dividend
                                    growth is windowing-distorted, DO NOT trust

  DATA SOURCES (free, machine-fetchable):
    1. SEC EDGAR 6-K (Canadian banks file as foreign issuers), EX-99.1:
       https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=<cik>&type=6-K
       CIKs:  RY=0001000275   CIBC(CM)=0001045520   BMO=0000927971
              (TD / BNS / NA: EDGAR full-text search by name)
    2. Each bank's investor-relations "quarterly results" page
    3. SEDAR+ (Canadian filings): https://www.sedarplus.ca

  Latest pulled (FY2026 Q2, qtr ended 2026-04-30) — manually from reports:
    CET1:  RY 13.5% · TD 14.3% · CIBC 13.6% · BMO 13.0% · BNS 13.3% · NA 13.5%
    PCL$M: RY 912(-36% YoY) · TD 1001(flat,43bps) · CIBC 605(flat)
           BMO 739(from 1054) · BNS 1217(from 1398,66bps) · NA 233(26bps)
    Read: PCL falling YoY sector-wide = credit fear easing (constructive).

  TODO: add fetch_segment_pcl_cet1(ticker) — pull the bank's latest 6-K
        EX-99.1 from EDGAR and parse out PCL, CET1, and the SBU revenue
        table. This module currently covers the income-trend side only.

Usage: python3 bank_deep_ca.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

BANKS = {
    "RY.TO": "Royal Bank", "TD.TO": "TD", "BMO.TO": "BMO",
    "CM.TO": "CIBC", "BNS.TO": "Scotiabank", "NA.TO": "National Bank",
}


def row(df, name, col):
    try:
        if df is not None and name in df.index and col in df.columns:
            v = df.loc[name, col]
            return float(v) if pd.notna(v) else np.nan
    except Exception:
        pass
    return np.nan


def first_row(df, names, col):
    for n in names:
        v = row(df, n, col)
        if pd.notna(v):
            return v
    return np.nan


def analyze(t, name):
    tk = yf.Ticker(t)
    inc = tk.quarterly_income_stmt
    bal = tk.quarterly_balance_sheet
    cf = tk.quarterly_cashflow
    div = tk.dividends
    if inc is None or inc.empty:
        print(f"\n  {name} ({t}): no data"); return

    q = list(inc.columns)[:5]               # latest 5 quarters (newest first)
    latest = q[0]
    print(f"\n{'='*64}\n  {name} ({t})   fiscal quarter ending {latest.date()}\n{'='*64}")

    # --- Revenue / NII / Net income / EPS over time ---
    print(f"  {'($B unless EPS)':<18}" + "".join(f"{str(c.date()):>12}" for c in q))
    print("  " + "-"*(18+12*len(q)))
    metrics = {
        "Revenue":   ("Total Revenue", 1e9),
        "Net Int Inc": ("Net Interest Income", 1e9),
        "Net Income": ("Net Income", 1e9),
        "Diluted EPS": ("Diluted EPS", 1),
    }
    store = {}
    for label, (rname, scale) in metrics.items():
        vals = [row(inc, rname, c) for c in q]
        store[label] = vals
        cells = "".join((f"{v/scale:>12.2f}" if pd.notna(v) else f"{'—':>12}") for v in vals)
        print(f"  {label:<18}{cells}")

    # --- YoY (latest vs 4 quarters ago) ---
    print("  " + "-"*(18+12*len(q)))
    yoy = {}
    for label in metrics:
        v = store[label]
        g = (v[0]/v[4]-1) if len(v) >= 5 and v[0] and v[4] and v[4] != 0 else np.nan
        yoy[label] = g
        print(f"  {label:<18}YoY {f'{g:+.1%}' if pd.notna(g) else 'n/a':>10}")

    # --- ROE (TTM) ---
    ni_ttm = np.nansum([row(inc, "Net Income", c) for c in q[:4]])
    equity = first_row(bal, ["Common Stock Equity", "Stockholders Equity",
                             "Total Equity Gross Minority Interest"], latest)
    roe = ni_ttm / equity if equity and equity != 0 else np.nan

    # --- BVPS + growth ---
    shares = first_row(bal, ["Ordinary Shares Number", "Share Issued"], latest)
    bvps = equity / shares if shares and shares != 0 else np.nan
    bvps_prior = np.nan
    if len(q) >= 5:
        eq4 = first_row(bal, ["Common Stock Equity", "Stockholders Equity",
                              "Total Equity Gross Minority Interest"], q[4])
        sh4 = first_row(bal, ["Ordinary Shares Number", "Share Issued"], q[4])
        if eq4 and sh4:
            bvps_prior = eq4 / sh4
    bvps_g = (bvps/bvps_prior - 1) if bvps_prior and pd.notna(bvps) else np.nan

    # --- Dividend growth (TTM vs prior TTM) ---
    div_g = np.nan; ttm_div = np.nan
    if div is not None and len(div) > 8:
        d = div.copy(); d.index = pd.to_datetime(d.index).tz_localize(None)
        last_date = d.index.max()
        ttm = d[d.index > last_date - pd.Timedelta(days=365)].sum()
        prior = d[(d.index <= last_date - pd.Timedelta(days=365)) &
                  (d.index > last_date - pd.Timedelta(days=730))].sum()
        ttm_div = ttm
        div_g = (ttm/prior - 1) if prior else np.nan

    # --- Operating cash flow (latest quarter) ---
    ocf = first_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], latest) if cf is not None else np.nan

    # --- Operating leverage (approx): revenue YoY - cost-base YoY ---
    # cost base proxy = Total Revenue - Pretax Income (= expenses + PCL).
    # Positive = revenue growing faster than costs (efficiency improving).
    op_lev = np.nan
    if len(q) >= 5:
        rev0, rev4 = row(inc, "Total Revenue", q[0]), row(inc, "Total Revenue", q[4])
        pti0, pti4 = row(inc, "Pretax Income", q[0]), row(inc, "Pretax Income", q[4])
        cost0 = (rev0 - pti0) if pd.notna(rev0) and pd.notna(pti0) else np.nan
        cost4 = (rev4 - pti4) if pd.notna(rev4) and pd.notna(pti4) else np.nan
        if pd.notna(cost0) and pd.notna(cost4) and cost4 and rev4:
            op_lev = (rev0/rev4 - 1) - (cost0/cost4 - 1)

    print("  " + "-"*(18+12*len(q)))
    print(f"  ROE (TTM):              {f'{roe:.1%}' if pd.notna(roe) else 'n/a':>10}")
    print(f"  BVPS:                   {f'${bvps:.2f}' if pd.notna(bvps) else 'n/a':>10}   YoY {f'{bvps_g:+.1%}' if pd.notna(bvps_g) else 'n/a'}")
    print(f"  Dividend/sh (TTM):      {f'${ttm_div:.2f}' if pd.notna(ttm_div) else 'n/a':>10}   YoY {f'{div_g:+.1%}' if pd.notna(div_g) else 'n/a'}")
    print(f"  Operating CF (Q):       {f'${ocf/1e9:.2f}B' if pd.notna(ocf) else 'n/a (banks: noisy)':>10}")
    print(f"  Operating leverage~:    {f'{op_lev:+.1%}' if pd.notna(op_lev) else 'n/a':>10}   (rev YoY - cost-base YoY, approx)")


def main():
    print(f"\n{'#'*64}\n  Canadian Big-6 — Deep Fundamentals (yfinance-derived)\n{'#'*64}")
    for t, name in BANKS.items():
        analyze(t, name)
    print(f"\n{'#'*64}")
    print("  STILL NEEDED FROM Q2-2026 REPORTS (not in yfinance):")
    print("    • Revenue by product/segment (Personal/Commercial/Wealth/Capital Mkts)")
    print("    • PPPT — pre-provision pre-tax profit (= pretax + PCL)")
    print("    • CET1 capital ratio   • PCL (provision for credit losses)")
    print(f"{'#'*64}\n")


if __name__ == "__main__":
    main()

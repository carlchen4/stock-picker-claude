"""
Feature IC Scanner — MEASURE candidate features before adding any to the model.
=============================================================================
Computes each candidate's standalone cross-sectional Rank-IC vs forward return
(per month, averaged) + t-stat + Benjamini-Hochberg FDR. Does NOT touch the
model — pure measurement, so it adds zero overfitting risk. Only a feature
that shows a real, FDR-significant IC is worth considering (then it must still
pass a full backtest + DSR before adoption).

Candidates (academically motivated, yfinance-computable, NOT already tried):
  MAX21      max daily return last 21d   (Bali 2011 lottery; expect NEG)
  SKEW60     daily-return skewness 60d   (skewness premium; expect NEG)
  BETA60     beta vs index 60d           (BAB low-beta anomaly; expect NEG)
  IDIOVOL60  residual vol vs index 60d   (Ang 2006; expect NEG)
  VOLTREND   20d/60d avg volume - 1      (volume-trend; sign unclear)
Plus VOL60D_EXISTING as a sanity benchmark (already in the model).

Usage:  python3 feature_scan.py        # TSX
        python3 feature_scan.py us      # US tech
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

if len(sys.argv) > 1 and sys.argv[1].lower() == "us":
    import picker_us  # noqa
    MARKET = "US"
else:
    MARKET = "TSX"
import picker
from picker import (TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS, fetch_prices,
                    build_panel, smart_impute, add_labels, get_ohlcv,
                    BENCHMARK_TICKER)


def candidate_panel(price_df, tickers, month_dates, mkt_ticker):
    """Per (date,ticker) candidate features computed from daily data."""
    mkt_close, _ = get_ohlcv(price_df, mkt_ticker)
    mkt_ret = mkt_close.pct_change() if mkt_close is not None else None
    rows = []
    for t in tickers:
        close, vol = get_ohlcv(price_df, t)
        if close is None or len(close) < 70:
            continue
        dret = close.pct_change()
        for d in month_dates:
            h = dret[dret.index <= d]
            if len(h) < 60:
                continue
            w60, w21 = h.tail(60), h.tail(21)
            rec = {"date": d, "ticker": t,
                   "MAX21": float(w21.max()),
                   "SKEW60": float(w60.skew()),
                   "KURT60": float(w60.kurt()),
                   "VOL60D_EXISTING": float(w60.std())}
            # batch 2
            cum = (1 + w60).cumprod()
            rec["MAXDD60"] = float((cum / cum.cummax() - 1).min())
            c60 = close[close.index <= d].tail(252)
            if len(c60) > 20 and c60.min() > 0:
                rec["DIST52LOW"] = float(close[close.index <= d].iloc[-1] / c60.min() - 1)
            cp = close[close.index <= d].tail(60)
            if len(cp) >= 30 and (cp > 0).all():
                y = np.log(cp.values); x = np.arange(len(y))
                r = np.corrcoef(x, y)[0, 1]
                rec["TRENDR2"] = float(r * r)
            if vol is not None:
                dv = (close[close.index <= d] * vol[vol.index <= d]).tail(60)
                ar = h.tail(60).abs()
                aligned = (ar.reindex(dv.index) / dv.replace(0, np.nan)).dropna()
                if len(aligned) > 20:
                    rec["AMIHUD"] = float(aligned.mean() * 1e9)
            # beta / idio-vol vs market
            if mkt_ret is not None:
                m = mkt_ret.reindex(w60.index).dropna()
                s = w60.reindex(m.index)
                if len(m) > 20 and m.var() > 0:
                    beta = float(np.cov(s, m)[0, 1] / m.var())
                    resid = s - beta * m
                    rec["BETA60"] = beta
                    rec["IDIOVOL60"] = float(resid.std())
            if vol is not None:
                vh = vol[vol.index <= d]
                if len(vh) >= 60 and vh.tail(60).mean() > 0:
                    rec["VOLTREND"] = float(vh.tail(20).mean() / vh.tail(60).mean() - 1)
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    print(f"\n{'='*64}\n  Feature IC Scan — {MARKET} (measure, don't add)\n{'='*64}")
    tickers = [t for t in TSX_UNIVERSE if t != BENCHMARK_TICKER]
    all_t = TSX_UNIVERSE + list(MACRO_TICKERS.values())
    price_df = fetch_prices(all_t, years=7)
    panel = build_panel(price_df, price_df, TSX_UNIVERSE)
    panel = add_labels(panel)                       # gives fwd_ret
    base = panel[["date", "ticker", "fwd_ret"]].dropna(subset=["fwd_ret"])

    cand = candidate_panel(price_df, tickers, sorted(base["date"].unique()),
                           MACRO_TICKERS.get("tsx"))
    df = base.merge(cand, on=["date", "ticker"], how="inner")

    feats = ["MAX21", "SKEW60", "KURT60", "BETA60", "IDIOVOL60", "VOLTREND",
             "MAXDD60", "DIST52LOW", "TRENDR2", "AMIHUD", "VOL60D_EXISTING"]
    print(f"  {len(df)} stock-months, {df['date'].nunique()} months\n")

    results = []
    for f in feats:
        if f not in df.columns:
            continue
        ics = []
        for d, g in df.groupby("date"):
            sub = g[[f, "fwd_ret"]].dropna()
            if len(sub) >= 5 and sub[f].nunique() > 1:
                ic, _ = spearmanr(sub[f], sub["fwd_ret"])
                if not np.isnan(ic):
                    ics.append(ic)
        if len(ics) < 10:
            continue
        ics = np.array(ics)
        mean_ic = ics.mean()
        t = mean_ic / (ics.std(ddof=1) / np.sqrt(len(ics)) + 1e-12)
        p = 2 * (1 - norm.cdf(abs(t)))
        results.append((f, mean_ic, t, p, len(ics)))

    # BH-FDR across candidates (exclude the sanity benchmark from correction count)
    cand_only = [r for r in results if r[0] != "VOL60D_EXISTING"]
    ps = sorted([r[3] for r in cand_only])
    m = len(ps)
    bh_thresh = 0.0
    for i, pv in enumerate(ps, 1):
        if pv <= (i / m) * 0.05:
            bh_thresh = pv

    print(f"  {'Feature':<18}{'mean IC':>9}{'t':>7}{'p':>8}{'months':>8}  sig?")
    print("  " + "-"*56)
    for f, ic, t, p, n in sorted(results, key=lambda x: -abs(x[1])):
        sig = "✓ FDR" if (f != "VOL60D_EXISTING" and p <= bh_thresh and bh_thresh > 0) else ""
        tag = " (benchmark)" if f == "VOL60D_EXISTING" else ""
        print(f"  {f:<18}{ic:>+9.4f}{t:>7.2f}{p:>8.3f}{n:>8}  {sig}{tag}")
    print("  " + "-"*56)
    nsig = sum(1 for f, ic, t, p, n in cand_only if p <= bh_thresh and bh_thresh > 0)
    print(f"\n  {nsig}/{len(cand_only)} candidates FDR-significant at α=0.05")
    print("  → only a ✓ feature is worth a full backtest+DSR test before adding.\n")


if __name__ == "__main__":
    main()

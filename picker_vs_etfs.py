"""picker 80/20 (CA+US) vs VCN/XIU/QQQ/SPY/VOO — 同窗口月度对比。"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yfinance as yf


def run():
    import picker
    px = picker.fetch_prices(picker.TSX_UNIVERSE + list(picker.MACRO_TICKERS.values()), years=7)
    p = picker.build_panel(px, px, picker.TSX_UNIVERSE)
    f = [c for c in picker.FEATURE_COLS if c in p.columns]
    p = picker.smart_impute(p, f); p = picker.add_labels(p)
    if picker.USE_MOMENTUM_PCA:
        p = picker.apply_momentum_pca(p)
        f = [c for c in picker.FEATURE_COLS if c in p.columns and c not in picker._RAW_MOMENTUM]
    else:
        f = [c for c in f if c not in ("mom_pc1", "mom_pc2")]
    p, mf = picker.cross_sectional_normalize(p, f)
    return picker.walk_forward(p, mf).set_index("date")["port_ret"]


print("CA picker ..."); ca = run()
import picker_us
print("US picker ..."); us = run()
active = (0.8 * ca + 0.2 * us).dropna()
active.index = active.index.tz_localize(None) if active.index.tz else active.index
start, end = active.index.min(), active.index.max()

def mret(t):
    h = yf.Ticker(t).history(start=start - pd.Timedelta(days=40), end=end + pd.Timedelta(days=5))["Close"]
    h.index = h.index.tz_localize(None)
    return h.resample("ME").last().pct_change()

etfs = {"VCN(加全市场)": "VCN.TO", "XIU(TSX60)": "XIU.TO",
        "QQQ(纳指100)": "QQQ", "SPY(标普500)": "SPY", "VOO(标普500)": "VOO"}
series = {"🟢 Picker 80/20": active}
for nm, t in etfs.items():
    series[nm] = mret(t)

def stats(r):
    r = np.asarray(pd.Series(r).dropna())
    if len(r) < 6: return (np.nan,)*4
    ann = (1+r).prod()**(12/len(r))-1
    sr = r.mean()/r.std(ddof=1)*np.sqrt(12)
    cum=(1+r).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    return ann, sr, dd, (1+r).prod()-1

idx = active.index
print(f"\n同窗口 {len(idx)} 个月 ({start.date()}~{end.date()})")
print("="*64)
print(f"{'组合/基准':<18}{'年化':>8}{'Sharpe':>8}{'最大回撤':>10}{'累计':>9}")
print("-"*64)
rows=[]
for nm, s in series.items():
    s2 = pd.Series(s).reindex(idx).dropna() if nm.startswith("🟢") else pd.Series(s).reindex(idx).dropna()
    rows.append((nm, *stats(s2)))
for nm, ann, sr, dd, tot in sorted(rows, key=lambda x:-(x[1] if x[1]==x[1] else -9)):
    print(f"{nm:<18}{ann*100:>7.1f}%{sr:>8.2f}{dd*100:>9.1f}%{tot*100:>8.0f}%")
print("="*64)

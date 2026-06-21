"""picker US(科技选股) vs 50% QQQ + 50% SPY, 过去一年。"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yfinance as yf
import picker, picker_us   # picker_us 改写 picker 全局为美股配置

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
res = picker.walk_forward(p, mf).set_index("date")
pk = res["port_ret"].tail(12); start = pk.index.min()

def m12(t):
    h = yf.Ticker(t).history(start=start - pd.Timedelta(days=40))["Close"]
    h.index = h.index.tz_localize(None)
    return h.resample("ME").last().pct_change().dropna().tail(12)

qqq, spy = m12("QQQ"), m12("SPY")
blend = pd.concat([qqq, spy], axis=1).mean(axis=1)
def tot(s): return ((1+s).prod()-1)*100
def shp(s): return s.mean()/s.std(ddof=1)*np.sqrt(12)

print(f"过去 ~12 个月({start.date()} 起):")
print(f"  🟢 Picker US(科技选股)      : {tot(pk):+.1f}%   Sharpe {shp(pk):.2f}")
print(f"  50% QQQ + 50% SPY(被动)    : {tot(blend):+.1f}%   Sharpe {shp(blend):.2f}")
print(f"  纯 QQQ(纳指)               : {tot(qqq):+.1f}%")
print(f"  纯 SPY(标普)               : {tot(spy):+.1f}%")
print(f"\nPicker US 相对 50/50 篮子 超额: {tot(pk)-tot(blend):+.1f}pp")

"""picker CA(模型选股) vs 行业 ETF 篮子(被动持有板块), 过去一年。"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yfinance as yf
import picker

# picker CA OOS 月收益
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
pk = res["port_ret"]
start = pk.index.min()

# 行业 ETF(picker 5 板块)+ XIU 基准, 同期月收益
etfs = {"金融XFN": "XFN.TO", "能源XEG": "XEG.TO", "工业ZIN": "ZIN.TO",
        "公用XUT": "XUT.TO", "材料金XGD": "XGD.TO", "大盘XIU": "XIU.TO"}
def m12(t):
    h = yf.Ticker(t).history(start=start - pd.Timedelta(days=40))["Close"]
    h.index = h.index.tz_localize(None)
    return h.resample("ME").last().pct_change().dropna()

mr = {k: m12(v) for k, v in etfs.items()}
basket = pd.concat([mr[k] for k in ["金融XFN","能源XEG","工业ZIN","公用XUT","材料金XGD"]], axis=1).mean(axis=1)

def tot(s): return ((1+s).prod()-1)*100
def shp(s): return s.mean()/s.std(ddof=1)*np.sqrt(12)

print(f"全 OOS(~5年)({start.date()} 起):")
print(f"  🟢 Picker CA(选股)  : {tot(pk):+.1f}%   Sharpe {shp(pk):.2f}")
print(f"  行业ETF等权篮子(5只): {tot(basket):+.1f}%   Sharpe {shp(basket):.2f}")
print(f"  大盘 XIU            : {tot(mr['大盘XIU']):+.1f}%")
print("\n各行业 ETF 单独(过去一年):")
for k in ["金融XFN","能源XEG","工业ZIN","公用XUT","材料金XGD"]:
    print(f"  {k:<10}{tot(mr[k]):+6.1f}%")
print(f"\nPicker 相对 行业ETF篮子 超额: {tot(pk)-tot(basket):+.1f}pp")

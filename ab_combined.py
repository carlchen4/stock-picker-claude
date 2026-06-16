"""
合并版:银行->ZEB.TO + 公用->ZUT.TO 一起 ETF 化。A/B + rigor。
顺带查 Materials/黄金 4 只在 panel 里的 sector_code 归属。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
import picker
from picker import (
    MACRO_TICKERS, FEATURE_COLS, USE_MOMENTUM_PCA, _RAW_MOMENTUM,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize, walk_forward,
    print_overfit_report, compute_cpcv, print_cpcv_report,
)

CODE2NAME = {v: k for k, v in picker.SECTOR_NAME_TO_CODE.items()}
BANKS = ["RY.TO","TD.TO","BMO.TO","CM.TO","BNS.TO","NA.TO","EQB.TO"]
UTILS = ["FTS.TO","H.TO","EMA.TO","AQN.TO"]
GOLD  = ["AEM.TO","ABX.TO","WPM.TO","FNV.TO"]


def build(universe):
    px = fetch_prices(universe + list(MACRO_TICKERS.values()), years=7)
    p = build_panel(px, px, universe)
    f = [c for c in FEATURE_COLS if c in p.columns]
    p = smart_impute(p, f); p = add_labels(p)
    if USE_MOMENTUM_PCA:
        p = apply_momentum_pca(p)
        f = [c for c in FEATURE_COLS if c in p.columns and c not in _RAW_MOMENTUM]
    else:
        f = [c for c in f if c not in ("mom_pc1","mom_pc2")]
    return cross_sectional_normalize(p, f)


def metrics(res):
    r = res["port_ret"].values; b = res["bench_ret"].values; ex = r-b
    sr = r.mean()/r.std(ddof=1)*np.sqrt(12)
    ann=(1+r).prod()**(12/len(r))-1; annb=(1+b).prod()**(12/len(b))-1
    te=ex.std(ddof=1)*np.sqrt(12); ir=(ann-annb)/te if te>0 else np.nan
    cum=(1+r).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    return sr, ir, ann, dd, float((ex>0).mean())


# 基线
print("基线 ...")
p0, mf0 = build(list(picker.TSX_UNIVERSE))
base = metrics(walk_forward(p0, mf0))

# 查 GOLD 归属
gp = p0[p0["ticker"].isin(GOLD)]
print("\n黄金 4 只在 panel 的 sector_code:")
if gp.empty:
    print("  (panel 里没有这 4 只!可能未进 universe 或被过滤)")
else:
    for t, g in gp.groupby("ticker"):
        codes = g["sector_code"].dropna().unique()
        print(f"  {t}: code={[ (int(c),CODE2NAME.get(int(c),c)) for c in codes ]}, 行数={len(g)}")

# 合并变体
print("\n合并变体 银行->ZEB + 公用->ZUT ...")
uni1 = [t for t in picker.TSX_UNIVERSE if t not in BANKS + UTILS]
for etf, sec in [("ZEB.TO","Financials"), ("ZUT.TO","Utilities")]:
    if etf not in uni1: uni1.append(etf)
    picker.STOCK_PROFILE[etf] = (sec, "core", "etf")
picker.TSX_UNIVERSE = uni1
p1, mf1 = build(uni1)
res1 = walk_forward(p1, mf1)
var = metrics(res1)

print("\n" + "="*72)
print(f"{'配置':<28}{'年化':>8}{'Sharpe':>8}{'IR':>7}{'回撤':>8}{'命中':>7}")
print("-"*72)
for lbl, m in [("基线(全选个股)", base), ("合并(银行ZEB+公用ZUT)", var)]:
    print(f"{lbl:<28}{m[2]*100:>7.1f}%{m[0]:>8.2f}{m[1]:>7.2f}{m[3]*100:>7.1f}%{m[4]*100:>6.0f}%")
print("="*72)

print("\n合并变体 rigor:")
ws = res1["port_ret"].mean()/res1["port_ret"].std(ddof=1)*np.sqrt(12)
print_overfit_report(res1)
print_cpcv_report(compute_cpcv(p1, mf1), wf_sharpe=ws)

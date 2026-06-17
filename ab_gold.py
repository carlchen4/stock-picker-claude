"""
Materials/黄金:① 诊断为何 IC 算不出(score NaN?)② A/B 黄金4只 -> ZGD.TO ETF。
XGD.TO 是 Materials 的 DML treatment,持仓用等权的 ZGD.TO。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import spearmanr
import picker
from picker import (
    MACRO_TICKERS, FEATURE_COLS, USE_MOMENTUM_PCA, _RAW_MOMENTUM,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize, walk_forward,
)

GOLD = ["AEM.TO", "ABX.TO", "WPM.TO", "FNV.TO"]
ZGD = "ZGD.TO"
MAT_CODE = picker.SECTOR_NAME_TO_CODE["Materials"]
ORIG = [t for t in picker.TSX_UNIVERSE if t != "ZUT.TO"] + ["FTS.TO","H.TO","EMA.TO","AQN.TO"]
picker.STOCK_PROFILE[ZGD] = ("Materials", "core", "etf")
picker.STOCK_PROFILE["ZUT.TO"] = ("Utilities","core","etf")  # 保持当前公用ETF


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
    sr=r.mean()/r.std(ddof=1)*np.sqrt(12)
    ann=(1+r).prod()**(12/len(r))-1; annb=(1+b).prod()**(12/len(b))-1
    te=ex.std(ddof=1)*np.sqrt(12); ir=(ann-annb)/te if te>0 else np.nan
    cum=(1+r).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    return sr, ir, ann, dd, float((ex>0).mean())


# ---- 诊断 + 基线 ----
print("基线 + Materials 诊断 ...")
picker.TSX_UNIVERSE = ORIG
p0, mf0 = build(ORIG)
res0, ps, _, _ = walk_forward(p0, mf0, return_perstock=True,
                              return_importance=True, return_raw_importance=True)
base = metrics(res0)
mat = ps[ps["sector_code"] == MAT_CODE]
print(f"\nMaterials 行数: {len(mat)} | score NaN: {mat['score'].isna().sum()} "
      f"| fwd_ret NaN: {mat['fwd_ret'].isna().sum()}")
m2 = mat.dropna(subset=["score", "fwd_ret"])
if len(m2):
    ics = [spearmanr(g["score"], g["fwd_ret"])[0]
           for _, g in m2.groupby("date") if g["ticker"].nunique() >= 2]
    ics = [x for x in ics if not np.isnan(x)]
    print(f"Materials 选股 IC(可算月数 {len(ics)}): "
          f"{np.mean(ics):.3f}" if ics else "仍无法计算")

# ---- A/B 黄金 -> ZGD ----
print("\n变体 黄金->ZGD.TO ...")
uni1 = [t for t in ORIG if t not in GOLD] + [ZGD]
picker.TSX_UNIVERSE = uni1
p1, mf1 = build(uni1)
var = metrics(walk_forward(p1, mf1))

print("\n" + "="*72)
print(f"{'配置':<26}{'年化':>8}{'Sharpe':>8}{'IR':>7}{'回撤':>8}{'命中':>7}")
print("-"*72)
for lbl, m in [("基线(选黄金股)", base), ("变体(黄金 ZGD.TO)", var)]:
    print(f"{lbl:<26}{m[2]*100:>7.1f}%{m[0]:>8.2f}{m[1]:>7.2f}{m[3]*100:>7.1f}%{m[4]*100:>6.0f}%")
print("="*72)

"""
Utilities -> ZUT.TO 的 A/B + rigor;顺带用门槛=2 重算各行业(含 Materials)IC。
基线与变体同一 session 同一份数据,可比。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import picker
from picker import (
    MACRO_TICKERS, FEATURE_COLS, USE_MOMENTUM_PCA, _RAW_MOMENTUM,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize, walk_forward,
    print_overfit_report, compute_cpcv, print_cpcv_report,
)

CODE2NAME = {v: k for k, v in picker.SECTOR_NAME_TO_CODE.items()}
UTILS = ["FTS.TO", "H.TO", "EMA.TO", "AQN.TO"]
ZUT = "ZUT.TO"


def build(universe):
    px = fetch_prices(universe + list(MACRO_TICKERS.values()), years=7)
    p = build_panel(px, px, universe)
    f = [c for c in FEATURE_COLS if c in p.columns]
    p = smart_impute(p, f); p = add_labels(p)
    if USE_MOMENTUM_PCA:
        p = apply_momentum_pca(p)
        f = [c for c in FEATURE_COLS if c in p.columns and c not in _RAW_MOMENTUM]
    else:
        f = [c for c in f if c not in ("mom_pc1", "mom_pc2")]
    return cross_sectional_normalize(p, f)


def metrics(results):
    r = results["port_ret"].values; b = results["bench_ret"].values
    ex = r - b
    sr = r.mean()/r.std(ddof=1)*np.sqrt(12)
    ann = (1+r).prod()**(12/len(r))-1
    annb = (1+b).prod()**(12/len(b))-1
    te = ex.std(ddof=1)*np.sqrt(12)
    ir = (ann-annb)/te if te>0 else np.nan
    cum=(1+r).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    return sr, ir, ann, dd, float((ex>0).mean())


# ---- 基线 + 各行业 IC(门槛=2)----
print("基线 ...")
p0, mf0 = build(list(picker.TSX_UNIVERSE))
res0, ps, _, _ = walk_forward(p0, mf0, return_perstock=True,
                              return_importance=True, return_raw_importance=True)
base = metrics(res0)
ps = ps.dropna(subset=["score", "fwd_ret", "sector_code"])
print("\n各行业选股 IC(门槛=2 名/月):")
ic_rows = []
for code, g in ps.groupby("sector_code"):
    ics = []
    for _, gm in g.groupby("date"):
        if gm["ticker"].nunique() >= 2:
            ic, _ = spearmanr(gm["score"], gm["fwd_ret"])
            if not np.isnan(ic): ics.append(ic)
    if ics:
        ic_rows.append((CODE2NAME.get(int(code), code), np.mean(ics), len(ics)))
for n, ic, m in sorted(ic_rows, key=lambda r: -r[1]):
    print(f"  {n:<14}{ic:>8.3f}  ({m} 月)")

# ---- 变体:公用 -> ZUT.TO ----
print("\n变体 公用->ZUT.TO ...")
uni1 = [t for t in picker.TSX_UNIVERSE if t not in UTILS]
if ZUT not in uni1: uni1.append(ZUT)
picker.STOCK_PROFILE[ZUT] = ("Utilities", "core", "util_etf")
picker.TSX_UNIVERSE = uni1
p1, mf1 = build(uni1)
res1 = walk_forward(p1, mf1)
var = metrics(res1)

print("\n" + "="*72)
print(f"{'配置':<26}{'年化':>8}{'Sharpe':>8}{'IR':>7}{'回撤':>8}{'命中':>7}")
print("-"*72)
for lbl, m in [("基线(选公用股)", base), ("变体(公用 ZUT.TO)", var)]:
    print(f"{lbl:<26}{m[2]*100:>7.1f}%{m[0]:>8.2f}{m[1]:>7.2f}{m[3]*100:>7.1f}%{m[4]*100:>6.0f}%")
print("="*72)

print("\n变体 rigor:")
ws = res1["port_ret"].mean()/res1["port_ret"].std(ddof=1)*np.sqrt(12)
print_overfit_report(res1)
print_cpcv_report(compute_cpcv(p1, mf1), wf_sharpe=ws)

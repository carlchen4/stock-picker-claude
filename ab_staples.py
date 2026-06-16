"""
防御位:公用 vs 消费必需品。
变体 = 移除 4 只公用股,加入 XST.TO(消费必需品 ETF),
       required_sectors 里 Utilities -> ConsumerStaples。
跟基线 + (已知的 ZUT 公用ETF) 对比 + rigor。
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

UTILS = ["FTS.TO","H.TO","EMA.TO","AQN.TO"]
XST = "XST.TO"


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


print("基线 ...")
p0, mf0 = build(list(picker.TSX_UNIVERSE))
base = metrics(walk_forward(p0, mf0))

print("\n变体 防御位->消费品 XST.TO ...")
uni1 = [t for t in picker.TSX_UNIVERSE if t not in UTILS]
if XST not in uni1: uni1.append(XST)
picker.STOCK_PROFILE[XST] = ("ConsumerStaples", "core", "staples_etf")
picker.CONSTRAINTS["required_sectors"] = [
    ("ConsumerStaples" if s == "Utilities" else s)
    for s in picker.CONSTRAINTS["required_sectors"]]
picker.TSX_UNIVERSE = uni1
p1, mf1 = build(uni1)
res1 = walk_forward(p1, mf1)
var = metrics(res1)

print("\n" + "="*74)
print(f"{'配置':<30}{'年化':>8}{'Sharpe':>8}{'IR':>7}{'回撤':>8}{'命中':>7}")
print("-"*74)
for lbl, m in [("基线(选公用股)", base), ("变体(消费品 XST.TO)", var)]:
    print(f"{lbl:<30}{m[2]*100:>7.1f}%{m[0]:>8.2f}{m[1]:>7.2f}{m[3]*100:>7.1f}%{m[4]*100:>6.0f}%")
print("(参考:公用ETF ZUT 同口径 = 33.1% / 2.20 / 2.10 / -8.2% / 73%)")
print("="*74)

print("\n消费品变体 rigor:")
ws = res1["port_ret"].mean()/res1["port_ret"].std(ddof=1)*np.sqrt(12)
print_overfit_report(res1)
print_cpcv_report(compute_cpcv(p1, mf1), wf_sharpe=ws)

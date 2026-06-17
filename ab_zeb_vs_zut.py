"""
同口径对比:基线 / 只换 ZEB(银行ETF,公用留个股) / 只换 ZUT(公用ETF,银行留个股)。
全部 train=28、同一份数据,可比。
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
)

BANKS = ["RY.TO","TD.TO","BMO.TO","CM.TO","BNS.TO","NA.TO","EQB.TO"]
UTILS = ["FTS.TO","H.TO","EMA.TO","AQN.TO"]
# 重建原始 universe(当前已含 ZUT,要拆回 4 只公用)
ORIG = [t for t in picker.TSX_UNIVERSE if t != "ZUT.TO"] + UTILS
picker.STOCK_PROFILE["ZEB.TO"] = ("Financials", "core", "etf")
picker.STOCK_PROFILE["ZUT.TO"] = ("Utilities", "core", "etf")


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


ORIG_REQ = list(picker.CONSTRAINTS["required_sectors"])
picker.STOCK_PROFILE["XST.TO"] = ("ConsumerStaples", "core", "etf")


def metrics(universe, req=None):
    picker.TSX_UNIVERSE = universe
    picker.CONSTRAINTS["required_sectors"] = req or ORIG_REQ
    p, mf = build(universe)
    res = walk_forward(p, mf)
    picker.CONSTRAINTS["required_sectors"] = ORIG_REQ
    r = res["port_ret"].values; b = res["bench_ret"].values; ex = r-b
    sr = r.mean()/r.std(ddof=1)*np.sqrt(12)
    ann=(1+r).prod()**(12/len(r))-1; annb=(1+b).prod()**(12/len(b))-1
    te=ex.std(ddof=1)*np.sqrt(12); ir=(ann-annb)/te if te>0 else np.nan
    cum=(1+r).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    return sr, ir, ann, dd, float((ex>0).mean())


STAPLES_REQ = [("ConsumerStaples" if s == "Utilities" else s) for s in ORIG_REQ]
configs = [
    ("基线(银行+公用都选个股)", ORIG, None),
    ("只换 ZEB(银行ETF)", [t for t in ORIG if t not in BANKS] + ["ZEB.TO"], None),
    ("只换 ZUT(公用ETF)", [t for t in ORIG if t not in UTILS] + ["ZUT.TO"], None),
    ("ZEB+ZUT(都换)", [t for t in ORIG if t not in BANKS + UTILS] + ["ZEB.TO", "ZUT.TO"], None),
    ("公用→消费品 XST", [t for t in ORIG if t not in UTILS] + ["XST.TO"], STAPLES_REQ),
]
rows = []
for lbl, uni, req in configs:
    print(f"跑 {lbl} ...")
    rows.append((lbl, *metrics(uni, req)))

print("\n" + "="*76)
print(f"{'配置':<30}{'年化':>8}{'Sharpe':>8}{'IR':>7}{'回撤':>8}{'命中':>7}")
print("-"*76)
for lbl, sr, ir, ann, dd, hit in rows:
    print(f"{lbl:<30}{ann*100:>7.1f}%{sr:>8.2f}{ir:>7.2f}{dd*100:>7.1f}%{hit*100:>6.0f}%")
print("="*76)

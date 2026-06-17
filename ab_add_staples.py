"""
是否值得把消费品(XST.TO)作为第6个必选板块加入(当前生产=5板块,已含ZUT)?
对比:当前生产(5板块) vs +XST第6板块(top_n=10 竞争 / top_n=12 扩容)。
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

PROD = list(picker.TSX_UNIVERSE)             # 当前生产态(已含 ZUT)
ORIG_REQ = list(picker.CONSTRAINTS["required_sectors"])
ORIG_TOPN = picker.CONSTRAINTS["top_n"]
picker.STOCK_PROFILE["XST.TO"] = ("ConsumerStaples", "core", "etf")


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


def run(universe, req, topn):
    picker.TSX_UNIVERSE = universe
    picker.CONSTRAINTS["required_sectors"] = req
    picker.CONSTRAINTS["top_n"] = topn
    p, mf = build(universe)
    res = walk_forward(p, mf)
    picker.CONSTRAINTS["required_sectors"] = ORIG_REQ
    picker.CONSTRAINTS["top_n"] = ORIG_TOPN
    r = res["port_ret"].values; b = res["bench_ret"].values; ex = r-b
    sr=r.mean()/r.std(ddof=1)*np.sqrt(12)
    ann=(1+r).prod()**(12/len(r))-1; annb=(1+b).prod()**(12/len(b))-1
    te=ex.std(ddof=1)*np.sqrt(12); ir=(ann-annb)/te if te>0 else np.nan
    cum=(1+r).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    return sr, ir, ann, dd, float((ex>0).mean())


REQ6 = ORIG_REQ + ["ConsumerStaples"]
configs = [
    ("当前生产(5板块,top10)", PROD, ORIG_REQ, 10),
    ("+XST 第6板块(top10 竞争)", PROD + ["XST.TO"], REQ6, 10),
    ("+XST 第6板块(top12 扩容)", PROD + ["XST.TO"], REQ6, 12),
]
rows = []
for lbl, uni, req, tn in configs:
    print(f"跑 {lbl} ...")
    rows.append((lbl, *run(uni, req, tn)))

print("\n" + "="*74)
print(f"{'配置':<28}{'年化':>8}{'Sharpe':>8}{'IR':>7}{'回撤':>8}{'命中':>7}")
print("-"*74)
for lbl, sr, ir, ann, dd, hit in rows:
    print(f"{lbl:<28}{ann*100:>7.1f}%{sr:>8.2f}{ir:>7.2f}{dd*100:>7.1f}%{hit*100:>6.0f}%")
print("="*74)

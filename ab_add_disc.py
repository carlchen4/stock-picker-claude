"""
把可选消费(Consumer Discretionary)作为第6板块加入(个股,模型可选股)。
对比当前生产(5板块);并算该板块选股 IC。top10/top12 两档。
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

DISC = ["DOL.TO", "QSR.TO", "MG.TO", "GIL.TO", "CTC-A.TO", "BYD.TO"]
for t, sub in zip(DISC, ["retail","restaurant","auto","apparel","retail","services"]):
    picker.STOCK_PROFILE[t] = ("ConsumerDisc", "core", sub)
DISC_CODE = picker.SECTOR_NAME_TO_CODE["ConsumerDisc"]

PROD = list(picker.TSX_UNIVERSE)
ORIG_REQ = list(picker.CONSTRAINTS["required_sectors"])
ORIG_TOPN = picker.CONSTRAINTS["top_n"]


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


def met(res):
    r = res["port_ret"].values; b = res["bench_ret"].values; ex = r-b
    sr=r.mean()/r.std(ddof=1)*np.sqrt(12)
    ann=(1+r).prod()**(12/len(r))-1; annb=(1+b).prod()**(12/len(b))-1
    te=ex.std(ddof=1)*np.sqrt(12); ir=(ann-annb)/te if te>0 else np.nan
    cum=(1+r).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    return sr, ir, ann, dd, float((ex>0).mean())


def run(universe, req, topn, perstock=False):
    picker.TSX_UNIVERSE = universe
    picker.CONSTRAINTS["required_sectors"] = req
    picker.CONSTRAINTS["top_n"] = topn
    p, mf = build(universe)
    out = walk_forward(p, mf, return_perstock=perstock,
                       return_importance=perstock, return_raw_importance=perstock)
    picker.CONSTRAINTS["required_sectors"] = ORIG_REQ
    picker.CONSTRAINTS["top_n"] = ORIG_TOPN
    return out


REQ6 = ORIG_REQ + ["ConsumerDisc"]
UNI6 = PROD + DISC

rows = [("当前生产(5板块,top10)", *met(run(PROD, ORIG_REQ, 10)))]
print("跑 +可选消费 top10(取IC) ...")
res, ps, _, _ = run(UNI6, REQ6, 10, perstock=True)
rows.append(("+可选消费(top10)", *met(res)))
print("跑 +可选消费 top12 ...")
rows.append(("+可选消费(top12)", *met(run(UNI6, REQ6, 12))))

# ConsumerDisc 选股 IC
d = ps[ps["sector_code"] == DISC_CODE].dropna(subset=["score","fwd_ret"])
ics = [spearmanr(g["score"], g["fwd_ret"])[0] for _, g in d.groupby("date")
       if g["ticker"].nunique() >= 2]
ics = [x for x in ics if not np.isnan(x)]
disc_ic = np.mean(ics) if ics else float("nan")

print("\n" + "="*74)
print(f"{'配置':<26}{'年化':>8}{'Sharpe':>8}{'IR':>7}{'回撤':>8}{'命中':>7}")
print("-"*74)
for lbl, sr, ir, ann, dd, hit in rows:
    print(f"{lbl:<26}{ann*100:>7.1f}%{sr:>8.2f}{ir:>7.2f}{dd*100:>7.1f}%{hit*100:>6.0f}%")
print("="*74)
print(f"可选消费 选股 IC = {disc_ic:.3f}  (>0.02 才算模型在此板块能选股)")

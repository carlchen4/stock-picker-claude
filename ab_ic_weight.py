"""按板块 IC 加权 vs 等权:超配模型真会选股的板块(工业/能源)。出图。"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import picker
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

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
res, ps, _, _ = picker.walk_forward(p, mf, return_perstock=True,
                                    return_importance=True, return_raw_importance=True)
res = res.set_index("date")
sel = ps[ps["is_selected"]].dropna(subset=["fwd_ret", "sector_code"])
bench = res["bench_ret"]

# 各板块 OOS IC(全样本;轻微look-ahead,仅可行性测试)
ic = {}
for c, g in sel.groupby("sector_code"):
    ics = [spearmanr(x["score"], x["fwd_ret"])[0] for _, x in g.groupby("date") if x["ticker"].nunique() >= 2]
    ics = [v for v in ics if v == v]
    ic[c] = np.mean(ics) if ics else 0.0
# 板块倾斜权重 ∝ max(IC, 0)+floor
tilt = {c: max(v, 0) + 0.02 for c, v in ic.items()}

def port(weighted):
    out = {}
    for dt, g in sel.groupby("date"):
        g = g.copy()
        if weighted:
            g["w"] = g["sector_code"].map(tilt)
        else:
            g["w"] = 1.0
        g["w"] /= g["w"].sum()
        out[dt] = float((g["w"] * g["fwd_ret"]).sum())
    return pd.Series(out)

def stats(r):
    r2 = r.reindex(bench.index).dropna(); b = bench.reindex(r2.index)
    sr = r2.mean()/r2.std(ddof=1)*np.sqrt(12)
    ann = (1+r2).prod()**(12/len(r2))-1; annb=(1+b).prod()**(12/len(b))-1
    te=(r2-b).std(ddof=1)*np.sqrt(12); ir=(ann-annb)/te if te>0 else np.nan
    cum=(1+r2).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    return ann, sr, ir, dd, cum

eq = port(False); icw = port(True)
print("各板块 OOS IC:", {picker.SECTOR_NAME_TO_CODE and [k for k,v in picker.SECTOR_NAME_TO_CODE.items() if v==c][0]: round(ic[c],3) for c in ic})
print(f"\n{'方案':<16}{'年化':>8}{'Sharpe':>8}{'IR':>7}{'回撤':>9}")
for nm, s in [("等权(现状)", eq), ("按IC加权", icw)]:
    a, sh, ir, d, cum = stats(s)
    print(f"{nm:<16}{a*100:>7.1f}%{sh:>8.2f}{ir:>7.2f}{d*100:>8.1f}%")

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(stats(eq)[4].index, stats(eq)[4].values, label="Equal weight", lw=2)
ax.plot(stats(icw)[4].index, stats(icw)[4].values, label="IC-tilted (overweight Industrials/Energy)", lw=2)
ax.set_title("Picker: Equal-weight vs Sector-IC-tilted", fontweight="bold"); ax.legend(); ax.grid(alpha=.3)
ax.set_ylabel("Growth of $1"); fig.tight_layout(); fig.savefig("ic_weight.png", dpi=130)
print("图: ic_weight.png")

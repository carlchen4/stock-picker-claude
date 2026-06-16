"""
每个行业内部的选股能力诊断:
  IC      = 行业内"分数 vs 真实未来收益"的月度秩相关均值(>0 才算在选股)
  分数离散 = 行业内每月分数的标准差均值(≈0 = 模型给同分,根本没在选,银行就是这样)
低 IC + 低离散 → 像银行一样,该换成行业 ETF。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import picker
from picker import (
    TSX_UNIVERSE, MACRO_TICKERS, FEATURE_COLS, USE_MOMENTUM_PCA, _RAW_MOMENTUM,
    fetch_prices, build_panel, smart_impute, add_labels,
    apply_momentum_pca, cross_sectional_normalize, walk_forward,
)

CODE2NAME = {v: k for k, v in picker.SECTOR_NAME_TO_CODE.items()}

price_df = fetch_prices(TSX_UNIVERSE + list(MACRO_TICKERS.values()), years=7)
panel = build_panel(price_df, price_df, TSX_UNIVERSE)
feats = [c for c in FEATURE_COLS if c in panel.columns]
panel = smart_impute(panel, feats)
panel = add_labels(panel)
if USE_MOMENTUM_PCA:
    panel = apply_momentum_pca(panel)
    feats = [c for c in FEATURE_COLS if c in panel.columns and c not in _RAW_MOMENTUM]
else:
    feats = [c for c in feats if c not in ("mom_pc1", "mom_pc2")]
panel, mf = cross_sectional_normalize(panel, feats)

print("跑 walk-forward 取逐股打分 ...")
_, ps, _, _ = walk_forward(panel, mf, return_perstock=True,
                           return_importance=True, return_raw_importance=True)
ps = ps.dropna(subset=["score", "fwd_ret", "sector_code"])

rows = []
for code, g in ps.groupby("sector_code"):
    name = CODE2NAME.get(int(code), str(code))
    ics, disp, ncount = [], [], []
    for dt, gm in g.groupby("date"):
        if gm["ticker"].nunique() >= 3:
            ic, _ = spearmanr(gm["score"], gm["fwd_ret"])
            if not np.isnan(ic):
                ics.append(ic)
            disp.append(gm["score"].std())
            ncount.append(gm["ticker"].nunique())
    if ics:
        rows.append((name, np.mean(ics), np.nanmean(disp),
                     np.mean(ncount), len(ics)))

rows.sort(key=lambda r: -r[1])
print("\n" + "=" * 70)
print(f"{'行业':<16}{'选股IC':>9}{'分数离散':>10}{'均名数':>8}{'月数':>7}")
print("-" * 70)
for name, ic, d, n, m in rows:
    flag = "  ← 像银行,可ETF化" if (ic < 0.02 and d < 0.05) else ""
    print(f"{name:<16}{ic:>9.3f}{d:>10.3f}{n:>8.1f}{m:>7}{flag}")
print("=" * 70)
print("IC>0.02 且离散>0.05 = 模型在真选股;反之 = 噪声,考虑换行业 ETF。")

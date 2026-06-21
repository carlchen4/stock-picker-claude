"""picker 最长窗口回测(价格能到的范围, ~2011-2026), vs XIU。含 2020/2022 崩盘。"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, picker

px = picker.fetch_prices(picker.TSX_UNIVERSE + list(picker.MACRO_TICKERS.values()), years=15)
panel = picker.build_panel(px, px, picker.TSX_UNIVERSE)
f = [c for c in picker.FEATURE_COLS if c in panel.columns]
panel = picker.smart_impute(panel, f); panel = picker.add_labels(panel)
if picker.USE_MOMENTUM_PCA:
    panel = picker.apply_momentum_pca(panel)
    f = [c for c in picker.FEATURE_COLS if c in panel.columns and c not in picker._RAW_MOMENTUM]
else:
    f = [c for c in f if c not in ("mom_pc1", "mom_pc2")]
panel, mf = picker.cross_sectional_normalize(panel, f)
res = picker.walk_forward(panel, mf).set_index("date")
print(f"OOS 窗口: {res.index.min().date()} → {res.index.max().date()} | {len(res)} 月")

def st(r, b=None):
    r = np.asarray(r); ann = (1+r).prod()**(12/len(r))-1
    sr = r.mean()/r.std(ddof=1)*np.sqrt(12)
    cum = (1+r).cumprod(); dd = float((cum/np.maximum.accumulate(cum)-1).min())
    out = [ann, sr, dd]
    if b is not None:
        b = np.asarray(b); annb=(1+b).prod()**(12/len(b))-1
        te=(r-b).std(ddof=1)*np.sqrt(12); out.append((ann-annb)/te if te>0 else np.nan)
    return out

p = res["port_ret"]; bm = res["bench_ret"]
ap, sp, dp, ir = st(p, bm); ab, sb, db = st(bm)
print("\n全窗口:")
print(f"  Picker : 年化 {ap*100:.1f}%  Sharpe {sp:.2f}  回撤 {dp*100:.0f}%  IR {ir:.2f}")
print(f"  XIU基准: 年化 {ab*100:.1f}%  Sharpe {sb:.2f}  回撤 {db*100:.0f}%")
print(f"  年化超额: {(ap-ab)*100:+.1f}pp")

# 分年/分崩盘段
print("\n分段(picker vs XIU 累计):")
for lbl, s, e in [("2020疫情(2-4月)","2020-02-01","2020-04-30"),
                  ("2022熊市","2022-01-01","2022-10-31"),
                  ("2020全年","2020-01-01","2020-12-31"),
                  ("2022全年","2022-01-01","2022-12-31")]:
    seg = res.loc[s:e]
    if len(seg) >= 1:
        pc=(1+seg["port_ret"]).prod()-1; bc=(1+seg["bench_ret"]).prod()-1
        print(f"  {lbl:<16} Picker {pc*100:+5.0f}%  vs XIU {bc*100:+5.0f}%")

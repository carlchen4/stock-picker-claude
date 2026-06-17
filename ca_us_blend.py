"""
CA picker vs US picker 的混合配置分析。
先跑 CA(当前生产),capture 月收益;再导入 picker_us(会改写 picker 全局),跑 US。
算相关性 + 不同 US 权重下的 年化/Sharpe/最大回撤,给配置建议。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd


def build_and_run():
    import picker
    px = picker.fetch_prices(picker.TSX_UNIVERSE + list(picker.MACRO_TICKERS.values()), years=7)
    p = picker.build_panel(px, px, picker.TSX_UNIVERSE)
    f = [c for c in picker.FEATURE_COLS if c in p.columns]
    p = picker.smart_impute(p, f); p = picker.add_labels(p)
    if picker.USE_MOMENTUM_PCA:
        p = picker.apply_momentum_pca(p)
        f = [c for c in picker.FEATURE_COLS if c in p.columns and c not in picker._RAW_MOMENTUM]
    else:
        f = [c for c in f if c not in ("mom_pc1","mom_pc2")]
    p, mf = picker.cross_sectional_normalize(p, f)
    res = picker.walk_forward(p, mf)
    return res.set_index("date")["port_ret"]


print("跑 CA picker(当前生产,含 ZUT) ...")
ca = build_and_run()

print("跑 US picker(科技) ...")
import picker_us            # 改写 picker 全局为 US 配置
us = build_and_run()

# 对齐
df = pd.concat([ca.rename("CA"), us.rename("US")], axis=1).dropna()
print(f"\n对齐后共同月数: {len(df)}")
corr = df["CA"].corr(df["US"])
print(f"CA vs US 月收益相关性: {corr:.2f}")


def stats(r):
    r = np.asarray(r)
    sr = r.mean()/r.std(ddof=1)*np.sqrt(12)
    ann = (1+r).prod()**(12/len(r))-1
    cum=(1+r).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    return ann, sr, dd


print("\n各成分单独:")
for nm, col in [("CA", df["CA"]), ("US", df["US"])]:
    ann, sr, dd = stats(col)
    print(f"  {nm}: 年化 {ann*100:5.1f}%  Sharpe {sr:4.2f}  最大回撤 {dd*100:6.1f}%")

print("\n不同 US 权重下的混合组合:")
print(f"{'US权重':>7}{'CA权重':>7}{'年化':>9}{'Sharpe':>8}{'最大回撤':>10}")
best = (None, -9)
for w in [0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0]:
    blend = (1-w)*df["CA"] + w*df["US"]
    ann, sr, dd = stats(blend)
    print(f"{w*100:6.0f}%{(1-w)*100:6.0f}%{ann*100:8.1f}%{sr:8.2f}{dd*100:9.1f}%")
    if sr > best[1]:
        best = (w, sr)
print(f"\n最大 Sharpe 的 US 权重 ≈ {best[0]*100:.0f}%")

"""
你的主动组合(80% CA-picker + 20% US-picker) vs 被动(80% XIU + 20% QQQ)。
同一段 walk-forward,捕获 port_ret(主动)和 bench_ret(基准ETF)。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd


def run():
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
    r = picker.walk_forward(p, mf).set_index("date")
    return r["port_ret"], r["bench_ret"]


print("CA picker + XIU ...")
ca_p, ca_b = run()
print("US picker + QQQ ...")
import picker_us
us_p, us_b = run()

df = pd.concat([ca_p.rename("ca_p"), ca_b.rename("ca_b"),
                us_p.rename("us_p"), us_b.rename("us_b")], axis=1).dropna()

def stats(r):
    r = np.asarray(r)
    sr = r.mean()/r.std(ddof=1)*np.sqrt(12)
    ann = (1+r).prod()**(12/len(r))-1
    cum=(1+r).cumprod(); dd=float((cum/np.maximum.accumulate(cum)-1).min())
    tot=(1+r).prod()-1
    return ann, sr, dd, tot

series = {
    "主动 80/20 (CA picker+US picker)": 0.8*df.ca_p + 0.2*df.us_p,
    "被动 80/20 (XIU+QQQ)":            0.8*df.ca_b + 0.2*df.us_b,
    "纯 XIU (加拿大大盘)":               df.ca_b,
    "纯 QQQ (美股科技大盘)":             df.us_b,
    "纯 CA picker":                    df.ca_p,
}
print(f"\n共同月数: {len(df)}\n" + "="*72)
print(f"{'组合':<34}{'年化':>8}{'Sharpe':>8}{'最大回撤':>9}{'累计':>9}")
print("-"*72)
for nm, s in series.items():
    ann, sr, dd, tot = stats(s)
    print(f"{nm:<34}{ann*100:>7.1f}%{sr:>8.2f}{dd*100:>8.1f}%{tot*100:>8.0f}%")
print("="*72)
# 主动相对被动的年化超额
a_ann = stats(series["主动 80/20 (CA picker+US picker)"])[0]
p_ann = stats(series["被动 80/20 (XIU+QQQ)"])[0]
print(f"主动 vs 被动 年化超额: {(a_ann-p_ann)*100:+.1f}pp")

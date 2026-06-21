"""回测逐月实例:展示 walk-forward 某几个月的 选股→实际收益→组合vs基准。"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,".")
import numpy as np, pandas as pd, picker
px=picker.fetch_prices(picker.TSX_UNIVERSE+list(picker.MACRO_TICKERS.values()),years=7)
p=picker.build_panel(px,px,picker.TSX_UNIVERSE)
f=[c for c in picker.FEATURE_COLS if c in p.columns]
p=picker.smart_impute(p,f); p=picker.add_labels(p)
if picker.USE_MOMENTUM_PCA:
    p=picker.apply_momentum_pca(p); f=[c for c in picker.FEATURE_COLS if c in p.columns and c not in picker._RAW_MOMENTUM]
else: f=[c for c in f if c not in ("mom_pc1","mom_pc2")]
p,mf=picker.cross_sectional_normalize(p,f)
res,ps,_,_=picker.walk_forward(p,mf,return_perstock=True,return_importance=True,return_raw_importance=True)
res=res.set_index("date")
# 取最近2个OOS月做实例
for dt in sorted(res.index)[-2:]:
    sel=ps[(ps["date"]==dt)&(ps["is_selected"])].copy()
    print(f"\n===== 测试月 {pd.Timestamp(dt).strftime('%Y-%m')} =====")
    print(f"训练窗口: 该月之前 ~28 个月(留1月embargo)→ 训练模型 → 给当月打分 → 选股")
    print(f"{'选中的票':<10}{'分数':>7}{'实际下月收益':>12}")
    for _,r in sel.sort_values("score",ascending=False).iterrows():
        print(f"  {r['ticker']:<8}{r['score']:>7.2f}{r['fwd_ret']*100:>11.1f}%")
    pr=res.loc[dt,"port_ret"]*100; br=res.loc[dt,"bench_ret"]*100
    print(f"  → 组合(等权)收益: {pr:+.1f}%   |   基准XIU: {br:+.1f}%   |   超额 {pr-br:+.1f}pp")

#!/usr/bin/env python3
"""
pairs_validate.py — 对配对候选做"真 edge vs 蒙中"验证
=====================================================
对 V-MA / MCD-YUM / ADP-PAYX 三对:
  ① 样本外切分:前一半训练、后一半检验,RV vs 50/50 Sharpe 在【两段】都赢才算稳;
  ② 参数稳健性:lookback×entry 网格,RV 跑赢 50/50 的比例(只在一个参数赢=过拟合)。
两关都过 = 真 edge;否则 = 多重检验蒙中。
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker          # noqa: E402
import pairs_rv as P   # noqa: E402

PAIRS = [("V", "MA"), ("MCD", "YUM"), ("ADP", "PAYX")]
LOOKBACKS = [40, 60, 90]
ENTRIES = [1.0, 1.5, 2.0]


def sharpe_pair(a, b, lb, en):
    (_, _, rs, _), (_, _, bs, _), _, _ = P.backtest_pair(a, b, lb, en)
    return rs, bs


def main():
    tk = sorted({t for p in PAIRS for t in p})
    # fetch_prices 只按"标的数+年数"缓存,会和同签名的别的请求(如6只银行)撞车 → 强制刷新
    import os
    cf = picker._cache_path(f"prices_{len(tk)}_8y.parquet")
    if os.path.exists(cf):
        os.remove(cf)
    px = picker.fetch_prices(tk, years=8)
    cl = {t: P.closes(px, t) for t in tk}
    for a, b in PAIRS:
        ca, cb = cl[a], cl[b]
        df = pd.concat([ca.rename("a"), cb.rename("b")], axis=1).dropna()
        mid = len(df) // 2
        first, second = df.iloc[:mid], df.iloc[mid:]
        print(f"\n===== {a}-{b}  ({df.index[0].date()} → {df.index[-1].date()}) =====")

        # ① 样本外切分(默认 60/1.5)
        r1, b1 = sharpe_pair(first["a"], first["b"], 60, 1.5)
        r2, b2 = sharpe_pair(second["a"], second["b"], 60, 1.5)
        print(f"① 样本外切分(60/1.5):")
        print(f"   前半(训练)  RV {r1:.2f} vs 50/50 {b1:.2f}  {'✅赢' if r1>b1 else '❌输'}")
        print(f"   后半(检验)  RV {r2:.2f} vs 50/50 {b2:.2f}  {'✅赢' if r2>b2 else '❌输'}")
        oos_ok = (r1 > b1) and (r2 > b2)

        # ② 参数稳健性(全样本网格)
        win = tot = 0
        deltas = []
        for lb in LOOKBACKS:
            for en in ENTRIES:
                rs, bs = sharpe_pair(df["a"], df["b"], lb, en)
                win += rs > bs; tot += 1; deltas.append(rs - bs)
        print(f"② 参数稳健性({tot} 组 lookback×entry):RV 跑赢 {win}/{tot},"
              f"平均 ΔSharpe {np.mean(deltas):+.2f}")
        robust = win >= tot * 0.7

        verdict = ("🟢 真 edge(两关都过)" if (oos_ok and robust) else
                   "🟡 半信半疑(过一关)" if (oos_ok or robust) else
                   "🔴 像蒙中(两关都没过)")
        print(f"   → {verdict}")
    print("\n判定:🟢 才值得碰;🟡/🔴 = 之前那个'赢'多半是多重检验的运气。非投资建议。")


if __name__ == "__main__":
    main()

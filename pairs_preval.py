#!/usr/bin/env python3
"""
pairs_preval.py — 配对的真·跨期验证(2018 年前 vs 2018 年后)
=============================================================
用 ~2008–2018(完全不同的十年,含 2008 危机)重测 V-MA / MCD-YUM / ADP-PAYX。
2018 后已验过赢;若 2018 前也赢 → 跨两个独立时代都成立 = 强证据;若垮 → 之前是这8年的运气。
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker          # noqa: E402
import pairs_rv as P   # noqa: E402

PAIRS = [("V", "MA"), ("MCD", "YUM"), ("ADP", "PAYX")]
SPLIT = pd.Timestamp("2018-01-01")
LOOKBACKS = [40, 60, 90]
ENTRIES = [1.0, 1.5, 2.0]


def sh(a, b, lb, en):
    (_, _, rs, _), (_, _, bs, _), _, hl = P.backtest_pair(a, b, lb, en)
    return rs, bs, hl


def main():
    tk = sorted({t for p in PAIRS for t in p})
    cf = picker._cache_path(f"prices_{len(tk)}_18y.parquet")
    if os.path.exists(cf):
        os.remove(cf)
    print("取价(近 18 年)…")
    px = picker.fetch_prices(tk, years=18)
    cl = {t: P.closes(px, t) for t in tk}

    for a, b in PAIRS:
        df = pd.concat([cl[a].rename("a"), cl[b].rename("b")], axis=1).dropna()
        pre = df[df.index < SPLIT]
        post = df[df.index >= SPLIT]
        print(f"\n===== {a}-{b} =====")
        print(f"  数据起点 {df.index[0].date()};pre-2018 段 {len(pre)} 日 "
              f"({pre.index[0].date() if len(pre) else '—'} → 2017),post 段 {len(post)} 日")
        if len(pre) < 60 + 250:
            print("  pre-2018 样本不足,跳过"); continue
        # pre-2018 默认参数
        r, bch, hl = sh(pre["a"], pre["b"], 60, 1.5)
        hl_s = f"{hl:.0f}日" if np.isfinite(hl) else "∞"
        print(f"  ① pre-2018(60/1.5):RV {r:.2f} vs 50/50 {bch:.2f}  "
              f"{'✅赢' if r > bch else '❌输'}  | 价差半衰期 {hl_s}")
        # pre-2018 参数网格
        win = tot = 0; ds = []
        for lb in LOOKBACKS:
            for en in ENTRIES:
                rs, bs, _ = sh(pre["a"], pre["b"], lb, en)
                win += rs > bs; tot += 1; ds.append(rs - bs)
        print(f"  ② pre-2018 参数稳健:RV 赢 {win}/{tot},平均 ΔSharpe {np.mean(ds):+.2f}")
        v = ("🟢 跨期成立(2018前也赢 + 参数稳)" if (r > bch and win >= tot * 0.7) else
             "🟡 跨期半成立" if (r > bch or win >= tot * 0.7) else
             "🔴 跨期垮了(2018后是运气)")
        print(f"  → {v}")
    print("\n两个独立十年都过 = 这个相对价值是真的;pre 垮掉 = 之前 8 年的运气。非投资建议。")


if __name__ == "__main__":
    main()

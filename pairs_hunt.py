#!/usr/bin/env python3
"""
pairs_hunt.py — 美股"双胞胎"配对大扫描(内置跨期验证)
======================================================
对一批业务高度相似的美股配对,直接算【2008–2018】和【2018–2026】两段的
多头相对价值(RV) vs 傻拿50/50 的 Sharpe。**两段都赢 = 🟢 跨期稳健**(真 edge);
只赢一段 = 🟡(像 MCD-YUM 那种近年运气)。一次性筛出所有真正稳健的对。
跑:./venv/bin/python pairs_hunt.py
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

PAIRS = [
    ("V", "MA", "支付双寡头"), ("ADP", "PAYX", "薪资处理"), ("MCD", "YUM", "快餐"),
    ("MCO", "SPGI", "评级机构"), ("AZO", "ORLY", "汽配零售"), ("TJX", "ROST", "折扣零售"),
    ("WM", "RSG", "废物管理"), ("TXN", "ADI", "模拟芯片"), ("SLB", "HAL", "油服"),
    ("LMT", "NOC", "国防"), ("UNP", "CSX", "铁路"), ("NSC", "CSX", "铁路2"),
    ("DHI", "LEN", "住宅建筑"), ("MO", "PM", "烟草"), ("CME", "ICE", "交易所"),
    ("NDAQ", "CBOE", "交易所2"), ("MS", "GS", "投行"), ("USB", "PNC", "区域银行"),
    ("CB", "TRV", "财险"), ("HD", "LOW", "家装"), ("KO", "PEP", "饮料"),
    ("PG", "CL", "日化"), ("COST", "WMT", "零售"), ("AMAT", "LRCX", "半导体设备"),
    ("DUK", "SO", "公用"), ("UNH", "ELV", "健康险"), ("CI", "HUM", "健康险2"),
    ("AON", "MMC", "保险经纪"), ("FAST", "GWW", "工业分销"), ("EMR", "HON", "工业"),
    ("XOM", "CVX", "石油"), ("JPM", "BAC", "大行"),
]
SPLIT = pd.Timestamp("2018-01-01")


def rv_win(seg):
    (_, _, rs, _), (_, _, bs, _), _, hl = P.backtest_pair(seg["a"], seg["b"], 60, 1.5)
    return rs, bs, hl


def main():
    tk = sorted({t for a, b, _ in PAIRS for t in (a, b)})
    cf = picker._cache_path(f"prices_{len(tk)}_18y.parquet")
    if os.path.exists(cf):
        os.remove(cf)
    print(f"取价 {len(tk)} 只(近 18 年)…")
    px = picker.fetch_prices(tk, years=18)
    cl = {t: P.closes(px, t) for t in tk}

    rows = []
    for a, b, desc in PAIRS:
        ca, cb = cl.get(a), cl.get(b)
        if ca is None or cb is None:
            rows.append((desc, f"{a}-{b}", None, None, None, "缺数据")); continue
        df = pd.concat([ca.rename("a"), cb.rename("b")], axis=1).dropna()
        pre, post = df[df.index < SPLIT], df[df.index >= SPLIT]
        if len(pre) < 320 or len(post) < 320:
            rows.append((desc, f"{a}-{b}", None, None, None, "样本不足")); continue
        pr, pb, _ = rv_win(pre)
        qr, qb, hl = rv_win(post)
        pre_w, post_w = pr > pb, qr > qb
        tag = "🟢稳健" if (pre_w and post_w) else ("🟡仅一段" if (pre_w or post_w) else "🔴都输")
        rows.append((desc, f"{a}-{b}", (pr - pb), (qr - qb), hl, tag))

    # 🟢 在前,按两段 ΔSharpe 之和排序
    def score(r):
        if r[2] is None:
            return -99
        return (2 if r[5].startswith("🟢") else 1 if r[5].startswith("🟡") else 0) * 10 + (r[2] + r[3])
    rows.sort(key=score, reverse=True)
    print(f"\n{'配对':<22}{'pre-2018 ΔSh':>13}{'post-2018 ΔSh':>14}{'半衰期':>8}{'  判定':>8}")
    print("-" * 70)
    for desc, pair, dpre, dpost, hl, tag in rows:
        if dpre is None:
            print(f"{(desc+' '+pair)[:21]:<22}  {tag}"); continue
        hl_s = f"{hl:.0f}日" if np.isfinite(hl) else "∞"
        print(f"{(desc+' '+pair)[:21]:<22}{dpre:>+12.2f}{dpost:>+13.2f}{hl_s:>8}{tag:>9}")
    g = [r for r in rows if r[5] and r[5].startswith("🟢")]
    print(f"\n🟢 跨期稳健(两段都赢):{len(g)} 对 —— "
          + "、".join(r[1] for r in g))
    print("⚠️ ΔSharpe 是 RV−50/50;两段都正才算真。仍需小仓位、非投资建议。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
pairs_scan.py — 加拿大"相似股票"配对扫描(多头相对价值,不做空)
================================================================
在 TSX 上挑业务高度相似的"双胞胎"配对,测:
  • 半衰期(均值回归速度,越短越适合配对;>~100日基本=不回归、做不了)
  • 多头相对价值(持便宜那只、价差回归就换)vs 傻拿 50/50,Sharpe 谁高
找出真正"价差快速回归、RV 又能赢"的对。复用 pairs_rv 的回测逻辑。
跑:./venv/bin/python pairs_scan.py [--lookback 60 --entry 1.5]
"""
import sys
import argparse
import numpy as np
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker          # noqa: E402
import pairs_rv as P   # 复用 closes/half_life/backtest_pair  # noqa: E402

# 业务高度相似的加拿大"双胞胎"
PAIRS = [
    ("CNR.TO", "CP.TO", "铁路双雄"),
    ("ENB.TO", "TRP.TO", "管道双雄"),
    ("BCE.TO", "T.TO", "电信"),
    ("L.TO", "MRU.TO", "超市(Loblaw/Metro)"),
    ("L.TO", "EMP-A.TO", "超市(Loblaw/Empire)"),
    ("AEM.TO", "ABX.TO", "金矿(Agnico/Barrick)"),
    ("FNV.TO", "WPM.TO", "黄金streaming"),
    ("MFC.TO", "SLF.TO", "寿险(Manulife/SunLife)"),
    ("GWO.TO", "IAG.TO", "寿险(GreatWest/iA)"),
    ("CNQ.TO", "SU.TO", "油砂(CNQ/Suncor)"),
    ("SU.TO", "CVE.TO", "油砂(Suncor/Cenovus)"),
    ("CNQ.TO", "IMO.TO", "油(CNQ/Imperial)"),
    ("FTS.TO", "EMA.TO", "公用(Fortis/Emera)"),
    ("FTS.TO", "H.TO", "公用(Fortis/Hydro One)"),
    ("RY.TO", "TD.TO", "银行(参照)"),
    ("WCN.TO", "GFL.TO", "废物管理"),
    ("QSR.TO", "ATD.TO", "消费(参照,业务不同)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--entry", type=float, default=1.5)
    ap.add_argument("--years", type=int, default=8)
    args = ap.parse_args()

    tickers = sorted({t for a, b, _ in PAIRS for t in (a, b)})
    print(f"取价 {len(tickers)} 只(近 {args.years} 年)…")
    px = picker.fetch_prices(tickers, years=args.years)
    cl = {t: P.closes(px, t) for t in tickers}

    rows = []
    for a, b, desc in PAIRS:
        ca, cb = cl.get(a), cl.get(b)
        if ca is None or cb is None:
            rows.append((desc, f"{a[:-3]}-{b[:-3]}", None, None, None, None, "缺数据"))
            continue
        import pandas as pd
        n = len(pd.concat([ca, cb], axis=1).dropna())
        if n < args.lookback + 120:
            rows.append((desc, f"{a[:-3]}-{b[:-3]}", None, None, None, None, "历史不足"))
            continue
        (rt, rc, rs, rmdd), (bt, bc, bs, bmdd), sw, hl = P.backtest_pair(ca, cb, args.lookback, args.entry)
        rows.append((desc, f"{a[:-3]}-{b[:-3]}", hl, rs, bs, sw / (n / P.TRADING), "ok"))

    # 按半衰期排序(短=回归快=好配对)
    ok = [r for r in rows if r[6] == "ok"]
    ok.sort(key=lambda r: r[2] if np.isfinite(r[2]) else 1e9)
    print(f"\n参数:lookback={args.lookback}日 entry=±{args.entry}z  (半衰期越短越适合配对)")
    print(f"{'配对':<26}{'半衰期':>9}{'RV Sharpe':>11}{'50/50 Sh':>10}{'RV赢?':>7}{'年换手':>8}")
    print("-" * 74)
    for desc, pair, hl, rs, bs, sw, st in ok:
        hl_s = f"{hl:.0f}日" if np.isfinite(hl) else "∞(不回归)"
        win = "✅" if rs > bs else "❌"
        star = " ⭐" if (np.isfinite(hl) and hl < 60 and rs > bs) else ""
        print(f"{(desc+' '+pair)[:25]:<26}{hl_s:>9}{rs:>10.2f}{bs:>9.2f}{win:>6}{sw:>7.1f}{star}")
    bad = [r for r in rows if r[6] != "ok"]
    for desc, pair, *_ , st in bad:
        print(f"{(desc+' '+pair)[:25]:<26}  {st}")
    n_short = sum(1 for r in ok if np.isfinite(r[2]) and r[2] < 60)
    n_win = sum(1 for r in ok if r[3] > r[4])
    print(f"\n半衰期<60日(可交易回归)的对:{n_short}/{len(ok)};RV 跑赢 50/50:{n_win}/{len(ok)}")
    print("⭐ = 半衰期<60日 且 RV 赢(真正值得做的对)。非投资建议;单对易过拟合,看整体。")


if __name__ == "__main__":
    main()

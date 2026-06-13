#!/usr/bin/env python3
"""
event_study.py — "VIX 飙升后买科技"事件研究
=============================================
检验对冲轮动策略的**核心前提**(不调参、不过拟合):
  历史上每次 VIX 收盘**上穿 28**(从 28 下方升到 28 上方),之后科技股未来 N 个交易日
  的反弹是否可靠?并和防守资产对比——"切去科技"相对"待在防守"到底有没有超额?

数据用长历史代理(样本多、稳):
  科技桶代理 = QQQ(纳指100)   防守桶代理 = XLP(必需消费)   大盘 = SPY
事件去重:两次上穿间隔 < COOLDOWN 交易日的只算一次(同一波 spike)。
入场=事件日收盘,出场=t+h 收盘(close-to-close,无前视)。

跑:  ./venv/bin/python event_study.py            # 默认 18 年, QQQ/XLP/SPY
     ./venv/bin/python event_study.py --years 25 --vix 30
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker  # noqa: E402

HORIZONS = [5, 10, 20, 40, 60]   # 交易日:~1周/2周/1月/2月/3月
COOLDOWN = 20                    # 事件去重:间隔 < 20 交易日算同一波


def close_of(price_df, t):
    c, _ = picker.get_ohlcv(price_df, t)
    return c.dropna() if c is not None else None


def fwd_ret(close, idx, h):
    if idx + h >= len(close):
        return None
    return (close.iloc[idx + h] / close.iloc[idx] - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=18)
    ap.add_argument("--vix", type=float, default=28)
    ap.add_argument("--tech", default="QQQ")
    ap.add_argument("--def", dest="defn", default="XLP")
    args = ap.parse_args()
    TECH, DEF, MKT, VIX = args.tech, args.defn, "SPY", "^VIX"

    print(f"取价 {[VIX, TECH, DEF, MKT]}(近 {args.years} 年)…")
    px = picker.fetch_prices([VIX, TECH, DEF, MKT], years=args.years)
    vix = close_of(px, VIX)
    tech = close_of(px, TECH)
    dfn = close_of(px, DEF)
    mkt = close_of(px, MKT)
    if any(x is None for x in (vix, tech, dfn, mkt)):
        print("❌ 数据不全"); return

    # 对齐到共同交易日
    common = vix.index.intersection(tech.index).intersection(dfn.index).intersection(mkt.index)
    vix, tech, dfn, mkt = vix[common], tech[common], dfn[common], mkt[common]
    v = vix.values

    # 找 VIX 上穿阈值的事件(收盘从 <vix 升到 >=vix),去重
    events = []
    last = -10**9
    for i in range(1, len(v)):
        if v[i - 1] < args.vix <= v[i] and (i - last) >= COOLDOWN:
            events.append(i); last = i
    print(f"\nVIX 上穿 {args.vix:.0f} 的独立事件:{len(events)} 次"
          f"(区间 {common[0].date()} → {common[-1].date()})\n")

    # 每个 horizon:科技/防守/大盘 的平均前向收益 + 科技胜率 + 科技-防守 价差
    print(f"{'持有期':<8}{'科技QQQ均值':>12}{'命中率':>8}{'防守XLP均值':>12}{'大盘SPY均值':>12}{'科技-防守':>11}{'科技>防守胜率':>13}")
    print("-" * 78)
    rows = []
    for h in HORIZONS:
        tr = [fwd_ret(tech, i, h) for i in events]
        dr = [fwd_ret(dfn, i, h) for i in events]
        mr = [fwd_ret(mkt, i, h) for i in events]
        pairs = [(a, b) for a, b in zip(tr, dr) if a is not None and b is not None]
        tr2 = [a for a, _ in pairs]; dr2 = [b for _, b in pairs]
        mr2 = [x for x in mr if x is not None]
        if not pairs:
            continue
        t_mean = np.mean(tr2); t_hit = np.mean([1 for x in tr2 if x > 0]) / 1 * 100 if tr2 else 0
        t_hit = sum(1 for x in tr2 if x > 0) / len(tr2) * 100
        d_mean = np.mean(dr2); m_mean = np.mean(mr2) if mr2 else float('nan')
        spread = np.mean([a - b for a, b in pairs])
        win = sum(1 for a, b in pairs if a > b) / len(pairs) * 100
        print(f"{h:>3}日{'':<3}{t_mean:>11.1f}%{t_hit:>7.0f}%{d_mean:>11.1f}%{m_mean:>11.1f}%{spread:>10.1f}%{win:>12.0f}%")
        rows.append((h, t_mean, t_hit, d_mean, spread, win))

    # 结论判定
    print("\n判定(策略前提 = '飙升后买科技 + 跑赢防守'):")
    good = [r for r in rows if r[1] > 0 and r[4] > 0 and r[5] >= 55]
    if not rows:
        print("  样本不足。")
    else:
        for h, tmean, thit, dmean, spread, win in rows:
            verdict = "✅ 支持" if (tmean > 0 and spread > 0 and win >= 55) else (
                      "⚠️ 一般" if (tmean > 0 and spread > 0) else "❌ 不支持")
            print(f"  {h:>2}日: 科技{tmean:+.1f}% 命中{thit:.0f}% / 比防守超{spread:+.1f}% 胜率{win:.0f}% → {verdict}")
        print(f"\n  样本仅 {len(events)} 次事件,横截面小;VIX 飙升本身罕见。结论谨慎看,非投资建议。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
trend_follow.py — 200 日均线趋势跟踪回测(Faber)
=================================================
规则:收盘 > 200日均线 → 持有;跌破 → 转现金(0 收益)。无前视:用昨日收盘/均线决定今日持仓。
检验:趋势跟踪 vs 买入持有,在多个标的上的 总收益/Sharpe/最大回撤/换手。
主价值预期 = 大幅降回撤(躲崩盘),代价 = 牛市可能跑输(滞后再入场 + 震荡假信号)。

跑:./venv/bin/python trend_follow.py [--ma 200 --years 20]
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker  # noqa: E402

ASSETS = ["QQQ", "SPY", "XIU.TO", "GLD"]   # 科技/美股大盘/TSX/黄金
COST = 0.001
TRADING = 252


def closes(px, t):
    c, _ = picker.get_ohlcv(px, t)
    return c.dropna() if c is not None else None


def metrics(daily):
    daily = daily.dropna()
    eq = (1 + daily).cumprod()
    n = len(daily)
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (TRADING / n) - 1
    vol = daily.std() * np.sqrt(TRADING)
    sharpe = daily.mean() * TRADING / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return total, cagr, sharpe, mdd


def backtest(close, ma):
    ret = close.pct_change()
    sma = close.rolling(ma).mean()
    inmkt = (close > sma).shift(1).fillna(False)        # 昨日信号定今日仓位,无前视
    strat = np.where(inmkt, ret, 0.0)
    strat = pd.Series(strat, index=close.index)
    switch = inmkt != inmkt.shift(1)
    strat = strat - switch.astype(float) * COST
    bh = ret.copy()
    # 对齐到均线可用之后
    strat, bh = strat.iloc[ma:], bh.iloc[ma:]
    sw_per_yr = switch.iloc[ma:].sum() / (len(strat) / TRADING)
    return metrics(strat), metrics(bh), sw_per_yr, inmkt.iloc[ma:].mean() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ma", type=int, default=200)
    ap.add_argument("--years", type=int, default=20)
    args = ap.parse_args()
    print(f"取价 {ASSETS}(近 {args.years} 年,{args.ma}日均线)…")
    px = picker.fetch_prices(ASSETS, years=args.years)
    print(f"\n{'标的':<9}{'':<2}{'总收益':>9}{'年化':>8}{'Sharpe':>8}{'最大回撤':>9}{'  在场%':>7}{'换手/年':>8}")
    print("-" * 70)
    for t in ASSETS:
        c = closes(px, t)
        if c is None or len(c) < args.ma + 252:
            print(f"{t:<11}数据不足"); continue
        (st, sc, ss, smdd), (bt, bc, bs, bmdd), sw, inpct = backtest(c, args.ma)
        yrs = len(c) / TRADING
        print(f"{t:<9}趋势{st*100:>9.0f}%{sc*100:>7.1f}%{ss:>8.2f}{smdd*100:>8.1f}%{inpct:>6.0f}%{sw:>7.1f}")
        print(f"{'':<9}持有{bt*100:>9.0f}%{bc*100:>7.1f}%{bs:>8.2f}{bmdd*100:>8.1f}%{'100%':>7}{'':>7}")
        better = []
        if ss > bs: better.append("Sharpe↑")
        if smdd > bmdd: better.append("回撤↓")
        if sc > bc: better.append("收益↑")
        print(f"{'':<9}趋势 vs 持有:{('、'.join(better) if better else '全面不如持有')}\n")
    print("看点:趋势跟踪通常 回撤明显改善、Sharpe 略升,但总收益可能输持有(牛市踏空+假信号)。")
    print("'出场转现金'若改成转债/短债会再好一点。非投资建议。")


if __name__ == "__main__":
    main()

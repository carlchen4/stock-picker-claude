#!/usr/bin/env python3
"""
pairs_rv.py — 多头相对价值(long-only pairs,不做空)
=====================================================
配对交易的"多头版":两只协整股票,持有**相对便宜**的那只,价差回归就换过去。
永远 100% 多头、注册账户可执行。检验:这套"换便宜的" vs "傻拿 50/50",到底有没有超额。

标的:加拿大六大行(RY/TD/BMO/BNS/CM/NA)——基本面同质、历史协整最强。
价差 = log(A/B);滚动 z-score(窗口 LOOKBACK)。
规则(带 hysteresis 防抖):z<-ENTRY → A 相对便宜 → 持 A;z>+ENTRY → 持 B;
        其间维持上次持仓;换仓扣 COST。
对照:50/50 等权持两只(每日再平衡)。
半衰期:用 Δspread ~ spread_{t-1} 回归估均值回归速度(越短=回归越快=越适合)。

跑:./venv/bin/python pairs_rv.py [--lookback 60 --entry 1.5]
"""
import sys
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker  # noqa: E402

BANKS = ["RY.TO", "TD.TO", "BMO.TO", "BNS.TO", "CM.TO", "NA.TO"]
COST = 0.001          # 换仓单边成本
TRADING = 252


def closes(px, t):
    c, _ = picker.get_ohlcv(px, t)
    return c.dropna() if c is not None else None


def half_life(spread):
    """OU 半衰期(交易日):Δs_t = a + b·s_{t-1};HL = -ln2 / ln(1+b)。"""
    s = spread.dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).dropna().loc[ds.index]
    b = np.polyfit(lag.values, ds.values, 1)[0]
    if b >= 0:
        return np.inf
    return -np.log(2) / np.log(1 + b)


def metrics(daily):
    eq = (1 + daily).cumprod()
    n = len(daily)
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (TRADING / n) - 1
    vol = daily.std() * np.sqrt(TRADING)
    sharpe = daily.mean() * TRADING / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return total, cagr, sharpe, mdd


def backtest_pair(a, b, lookback, entry):
    """a,b: 对齐的收盘 Series。返回 (RV结果, 50/50结果, 换仓次数, 半衰期)。"""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    ra, rb = df["a"].pct_change(), df["b"].pct_change()
    spread = np.log(df["a"] / df["b"])
    z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    hl = half_life(spread)

    pos = []                       # 当日持有谁(用昨日 z 决定,避免前视)
    cur = "a"
    switches = 0
    for t in range(len(df)):
        zt = z.iloc[t - 1] if t > 0 else np.nan
        new = cur
        if not np.isnan(zt):
            if zt < -entry:
                new = "a"          # A 相对便宜
            elif zt > entry:
                new = "b"
        if new != cur:
            switches += 1
        cur = new
        pos.append(cur)
    pos = pd.Series(pos, index=df.index)
    rv = np.where(pos == "a", ra, rb)
    rv = pd.Series(rv, index=df.index).fillna(0.0)
    # 扣换仓成本
    switch_days = pos != pos.shift(1)
    rv = rv - switch_days.astype(float) * COST
    blend = (ra.fillna(0) + rb.fillna(0)) / 2
    rv, blend = rv.iloc[lookback:], blend.iloc[lookback:]
    return metrics(rv), metrics(blend), switches, hl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--entry", type=float, default=1.5)
    ap.add_argument("--years", type=int, default=8)
    args = ap.parse_args()

    print(f"取价 六大行(近 {args.years} 年)…")
    px = picker.fetch_prices(BANKS, years=args.years)
    cl = {t: closes(px, t) for t in BANKS}
    cl = {t: c for t, c in cl.items() if c is not None}
    print(f"参数:lookback={args.lookback} 日,entry=±{args.entry} z,成本={COST*1e4:.0f}bps/换\n")
    print(f"{'配对':<16}{'半衰期':>8}{'RV年化':>9}{'RV Sharpe':>11}{'50/50年化':>11}{'50/50 Sh':>10}{'年换仓':>8}{'  RV赢?':>8}")
    print("-" * 80)
    wins = 0
    rows = []
    for x, y in combinations(cl.keys(), 2):
        a, b = cl[x], cl[y]
        n_common = len(pd.concat([a, b], axis=1).dropna())
        if n_common < args.lookback + 60:
            continue
        (rt, rc, rs, rmdd), (bt, bc, bs, bmdd), sw, hl = backtest_pair(a, b, args.lookback, args.entry)
        yrs = n_common / TRADING
        win = rs > bs
        wins += win
        hl_s = f"{hl:.0f}日" if np.isfinite(hl) else "∞"
        print(f"{x[:-3]}-{y[:-3]:<11}{hl_s:>8}{rc*100:>8.1f}%{rs:>10.2f}{bc*100:>10.1f}%{bs:>9.2f}{sw/yrs:>7.1f}{('✅' if win else '❌'):>7}")
        rows.append((f"{x}-{y}", rs, bs, win))
    print("-" * 80)
    print(f"\nRV 跑赢 50/50(按 Sharpe)的配对:{wins}/{len(rows)}")
    avg_rv = np.mean([r[1] for r in rows]); avg_bl = np.mean([r[2] for r in rows])
    print(f"平均 Sharpe:RV {avg_rv:.2f} vs 50/50 {avg_bl:.2f}  → "
          + ("RV 整体更优" if avg_rv > avg_bl + 0.05 else
             "50/50 整体更优" if avg_bl > avg_rv + 0.05 else "基本打平(差异在噪声内)"))
    print("\n注:单组样本/参数敏感,RV 赢的对可能是过拟合;关注'整体是否系统性赢',别挑赢的那对。非投资建议。")


if __name__ == "__main__":
    main()

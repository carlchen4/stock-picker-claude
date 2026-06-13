#!/usr/bin/env python3
"""
vol_rotation_backtest.py — 波动轮动策略回测
==========================================

⚠️ 结论(2026-06-10,--sweep 跑出来的,别再被高收益格子骗回去做轮动):
   1. 轮动那些诱人的数字(如 30/16 阈值 3 年 +249%)是 **过拟合**:同样 5 年,
      阈值 28→30→26 收益就在 95%/151%/177% 乱跳,对阈值如此敏感 = 不是真 edge。
   2. 轮动命门 = 赌"VIX 高位快速消退";2022 那种持续熊市一来就 -45%~-52% 回撤。
   3. 真正稳健:**一直持科技 9 只**长窗口收益最高(5 年 +220.8%)、不依赖任何可调阈值,
      代价是 -51.8% 回撤;受不了就 50-50(+192% / 仅 -23%)。
   → 线上 vol_rotation.py 已据此改成"持科技 9 只"(不轮动)。本脚本仅留作研究/复现。

用过去 N 年真实数据,逐日模拟 vol_rotation.py 的 VIX 双阈值状态机:
  • DEFENSIVE: VIX 收盘 > VIX_HIGH → 切 AGGRESSIVE
  • AGGRESSIVE: VIX 收盘 < VIX_LOW → 切 DEFENSIVE
  • 否则维持
无前视偏差:用**前一日** VIX 收盘决定今天持哪个桶,赚今天的桶收益。
桶内等权(每日再平衡)。对比基准:一直保守 / 一直进攻 / 50-50。

用法:
  ./venv/bin/python vol_rotation_backtest.py            # 默认 5 年,阈值 28/18
  ./venv/bin/python vol_rotation_backtest.py --years 3 --high 30 --low 16
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker  # noqa: E402

DEFENSIVE = ["KO", "XLE", "ZEO.TO", "FTS.TO"]
AGGRESSIVE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "SNOW"]
VIX_TICKER = "^VIX"
TRADING_DAYS = 252


def bucket_returns(price_df, tickers):
    """各标的日收益 → 横向等权均值(只对当日有数据的标的平均)。"""
    cols = {}
    for t in tickers:
        close, _ = picker.get_ohlcv(price_df, t)
        if close is not None and len(close) > 5:
            cols[t] = close.pct_change()
    df = pd.DataFrame(cols)
    return df.mean(axis=1, skipna=True), df


def metrics(equity, daily_ret):
    """equity: 净值序列(起点 1.0);daily_ret: 日收益序列。"""
    total = equity.iloc[-1] - 1
    n = len(equity)
    years = n / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (daily_ret.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    roll_max = equity.cummax()
    maxdd = (equity / roll_max - 1).min()
    return total, cagr, vol, sharpe, maxdd


def build_data(price_df):
    """对齐 VIX + 两桶日收益,返回 DataFrame(vix/def/agg) 或 None。"""
    vclose, _ = picker.get_ohlcv(price_df, VIX_TICKER)
    def_ret, _ = bucket_returns(price_df, DEFENSIVE)
    agg_ret, _ = bucket_returns(price_df, AGGRESSIVE)
    data = pd.DataFrame({"vix": vclose, "def": def_ret, "agg": agg_ret}).dropna()
    return data if len(data) >= 60 else None


def compute_curves(data, vix_high, vix_low):
    """返回 (curves{名:日收益Series}, switches, states, tb_meta{hold:(sw,st)})。"""
    vix, dret, aret = data["vix"].values, data["def"].values, data["agg"].values
    idx, n = data.index, len(data)

    def simulate(mode, hold=15):
        state, armed, cnt = "DEFENSIVE", True, 0
        rets = np.zeros(n)
        sw, states = [], []
        for t in range(n):
            states.append(state)
            rets[t] = dret[t] if state == "DEFENSIVE" else aret[t]
            v = vix[t]
            if state == "DEFENSIVE":
                if mode == "threshold":
                    if v > vix_high:
                        state = "AGGRESSIVE"; sw.append((t, idx[t].date(), "DEFENSIVE", "AGGRESSIVE", v))
                else:
                    if armed and v > vix_high:
                        state = "AGGRESSIVE"; cnt = hold; armed = False
                        sw.append((t, idx[t].date(), "DEFENSIVE", "AGGRESSIVE", v))
            else:
                if mode == "threshold":
                    if v < vix_low:
                        state = "DEFENSIVE"; sw.append((t, idx[t].date(), "AGGRESSIVE", "DEFENSIVE", v))
                else:
                    cnt -= 1
                    if cnt <= 0:
                        state = "DEFENSIVE"; sw.append((t, idx[t].date(), "AGGRESSIVE", "DEFENSIVE", v))
            if mode == "timebox" and state == "DEFENSIVE" and not armed and v < vix_high:
                armed = True
        return pd.Series(rets, index=idx), sw, states

    thr_ret, switches, states = simulate("threshold")
    curves = {
        "🔁 原版(VIX<低退)": thr_ret,
        "🛡️ 一直保守": data["def"],
        "🚀 一直进攻": data["agg"],
        "⚖️ 50-50":   (data["def"] + data["agg"]) / 2,
    }
    tb_meta = {}
    for hold in (5, 10, 15, 20):
        r, sw, st = simulate("timebox", hold)
        curves[f"⏱️ 限时进攻{hold}天"] = r
        tb_meta[hold] = (sw, st)
    return curves, switches, states, tb_meta


def run(years, vix_high, vix_low):
    tickers = [VIX_TICKER] + DEFENSIVE + AGGRESSIVE
    print(f"取价 {len(tickers)} 个标的(近 {years} 年)…")
    price_df = picker.fetch_prices(tickers, years=years)

    data = build_data(price_df)
    if data is None:
        print("❌ 有效数据太少,放弃。"); return
    start, end, n = data.index[0].date(), data.index[-1].date(), len(data)
    print(f"回测区间: {start} → {end}  共 {n} 个交易日 (~{n/TRADING_DAYS:.1f} 年)\n")

    curves, switches, states, tb_meta = compute_curves(data, vix_high, vix_low)

    # ---- 报告 ----
    print(f"阈值: VIX>{vix_high:.0f} 切进攻 / VIX<{vix_low:.0f} 切保守\n")
    hdr = f"{'策略':<14}{'总收益':>10}{'年化':>9}{'年化波动':>10}{'Sharpe':>9}{'最大回撤':>10}"
    print(hdr); print("-" * len(hdr.encode("gbk", "ignore")))
    for name, r in curves.items():
        eq = (1 + r).cumprod()
        total, cagr, vol, sharpe, maxdd = metrics(eq, r)
        print(f"{name:<12}{total*100:>9.1f}%{cagr*100:>8.1f}%{vol*100:>9.1f}%"
              f"{sharpe:>9.2f}{maxdd*100:>9.1f}%")

    # ---- 切换分析 ----
    yrs = len(data) / TRADING_DAYS
    print(f"\n[原版] 切换 {len(switches)} 次 (~{len(switches)/yrs:.1f}/年),"
          f"处于进攻 {sum(1 for s in states if s=='AGGRESSIVE')/n*100:.0f}% 的时间")
    for hold in (5, 10, 15, 20):
        sw, st = tb_meta[hold]
        print(f"[限时{hold}天] 切换 {len(sw)} 次 (~{len(sw)/yrs:.1f}/年),"
              f"处于进攻 {sum(1 for s in st if s=='AGGRESSIVE')/n*100:.0f}% 的时间")
    print(f"\n原版切换明细 (日期 / 由→至 / 当日VIX):")
    for pos, d, frm, to, v in switches:
        tag = "🚀" if to == "AGGRESSIVE" else "🛡️"
        print(f"  {d}  {frm[:3]:>3} → {to:<10} VIX={v:5.1f}  {tag}")


def _tot_dd(r):
    eq = (1 + r).cumprod()
    total, _, _, _, maxdd = metrics(eq, r)
    return total, maxdd


def sweep(years_list, threshold_pairs):
    """多周期 × 多阈值快捷对比:每个策略的总收益(括号内最大回撤)。"""
    names = ["🔁 原版(VIX<低退)", "⏱️ 限时进攻10天", "🛡️ 一直保守", "🚀 一直进攻", "⚖️ 50-50"]
    tickers = [VIX_TICKER] + DEFENSIVE + AGGRESSIVE
    for hi, lo in threshold_pairs:
        print(f"\n================  阈值 VIX>{hi:.0f}切进攻 / VIX<{lo:.0f}切保守  ================")
        header = f"{'策略':<16}" + "".join(f"{str(y)+'年':>16}" for y in years_list)
        print(header); print("-" * len(header.encode("gbk", "ignore")))
        col = {y: {} for y in years_list}
        for y in years_list:
            data = build_data(picker.fetch_prices(tickers, years=y))
            if data is None:
                continue
            curves, *_ = compute_curves(data, hi, lo)
            for nm in names:
                if nm in curves:
                    col[y][nm] = _tot_dd(curves[nm])
        for nm in names:
            cells = ""
            for y in years_list:
                if nm in col[y]:
                    t, dd = col[y][nm]
                    cells += f"{t*100:>8.0f}%({dd*100:>4.0f}%)"
                else:
                    cells += f"{'n/a':>16}"
            print(f"{nm:<14}{cells}")
    print("\n格子 = 总收益(最大回撤)。同一阈值看不同年份是否稳定;不同阈值看对设定是否敏感。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--high", type=float, default=28)
    ap.add_argument("--low", type=float, default=18)
    ap.add_argument("--sweep", action="store_true",
                    help="多周期(2/3/5年)× 多阈值(28/18,30/16,26/20)对比矩阵")
    args = ap.parse_args()
    if args.sweep:
        sweep([2, 3, 5], [(28, 18), (30, 16), (26, 20)])
    else:
        run(args.years, args.high, args.low)


if __name__ == "__main__":
    main()

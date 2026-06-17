"""
隔夜效应回测:收盘买入 -> 次日开盘卖出。
对比 隔夜(close->next open) / 盘中(open->close) / 买入持有(close->close)。
美股科技为主。注意:这是 GROSS(不含成本),每日往返成本另算。
"""
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf, pandas as pd, numpy as np

TECH = ["QQQ", "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "AMD", "TSLA", "AVGO"]
YEARS = 5
TD = 252


def legs(t):
    h = yf.Ticker(t).history(period=f"{YEARS}y")
    if len(h) < 50:
        return None
    o, c = h["Open"], h["Close"]
    overnight = (o.shift(-1) / c - 1).dropna()      # 收盘买, 次开卖
    intraday = (c / o - 1)                           # 开盘买, 收盘卖
    buyhold = (c / c.shift(1) - 1).dropna()
    return overnight, intraday, buyhold


def ann(r):
    r = np.asarray(r);
    return (1 + r).prod() ** (TD / len(r)) - 1


def stats(r):
    r = np.asarray(r)
    a = ann(r); vol = r.std() * np.sqrt(TD)
    shp = a / vol if vol > 0 else np.nan
    win = (r > 0).mean()
    return a, shp, win, len(r)


print(f"{'票':<7}{'隔夜年化':>10}{'盘中年化':>10}{'买持年化':>10}{'隔夜胜率':>9}{'隔夜日均':>9}")
print("-" * 60)
agg_on = []
for t in TECH:
    r = legs(t)
    if r is None:
        continue
    on, intra, bh = r
    a_on = ann(on); a_in = ann(intra); a_bh = ann(bh)
    win = (np.asarray(on) > 0).mean()
    davg = np.asarray(on).mean() * 100
    print(f"{t:<7}{a_on*100:>9.1f}%{a_in*100:>9.1f}%{a_bh*100:>9.1f}%{win*100:>8.0f}%{davg:>8.3f}%")
    agg_on.append(np.asarray(on).mean())

# 成本敏感性:每日往返成本 c%(买+卖),看隔夜日均收益扣掉后还剩多少
print("\n隔夜策略的成本现实(以 QQQ 为例):")
on_q, _, _ = legs("QQQ")
daily = np.asarray(on_q).mean()
print(f"  QQQ 隔夜 日均毛收益 = {daily*100:.3f}%")
for cost_bps in [1, 2, 5, 10]:        # 单边 bps -> 往返 *2
    net_daily = daily - 2 * cost_bps / 1e4
    net_ann = (1 + net_daily) ** TD - 1
    print(f"  扣往返成本 {cost_bps*2} bps/日: 日均 {net_daily*100:+.3f}% -> 年化 {net_ann*100:+.1f}%")

"""
Sell-Timing Analysis — "when / at what price to sell, this week"
================================================================
The live counterpart to entry_timing.py. For a ticker you want to SELL,
fold three layers into one place so you can place a smart limit order:

  (1) PRICE  — last ~week of 15m bars: VWAP, week hi/lo, resistance zone
               (where volume clusters near the highs), and the time-of-day
               buckets that have historically been strongest (best windows
               to let a sell limit fill on a bounce).
  (2) ANALYST— yfinance consensus: mean/median/high/low target, implied
               upside, rating distribution, valuation (PE).
  (3) NEWS   — yfinance recent headlines (titles + links). Shallow by design;
               for real event analysis do a web search.

This is DESCRIPTIVE, not predictive. It tells you where/when selling has
worked over the past week and what the consensus + tape look like NOW. It
does NOT forecast the future price.

Usage:
    python3 sell_timing.py TSLA              # one name
    python3 sell_timing.py RY.TO NVDA        # several
    python3 sell_timing.py legacy            # every LEGACY_HOLDINGS name
                                             #   (adds cost-basis P&L)
    python3 sell_timing.py TSLA --no-news    # skip the news layer
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# Reuse the legacy plumbing already in picker.py (cost basis, FX, live price).
sys.path.insert(0, ".")
try:
    from picker import LEGACY_HOLDINGS, legacy_unrealized, legacy_currency
except Exception:
    LEGACY_HOLDINGS = {}
    legacy_unrealized = None
    legacy_currency = None


def price_layer(tk):
    """Last ~5 sessions of 15m bars -> VWAP, range, resistance, time-of-day."""
    df = yf.Ticker(tk).history(period="5d", interval="15m", prepost=False)
    if df.empty:
        return None
    try:
        df = df.tz_convert("America/New_York")
    except Exception:
        pass
    df["ret"] = df["Close"].pct_change()
    last = df["Close"].iloc[-1]
    hi, lo = df["High"].max(), df["Low"].min()
    hi_t, lo_t = df["High"].idxmax(), df["Low"].idxmin()
    vwap = (df["Close"] * df["Volume"]).sum() / df["Volume"].sum()
    top = df[df["Close"] >= df["Close"].quantile(0.80)]
    resist = top["Close"].mean()
    df["tod"] = df.index.strftime("%H:%M")
    tod = df.groupby("tod")["ret"].mean().sort_values(ascending=False)
    return dict(
        last=last, hi=hi, lo=lo, hi_t=hi_t, lo_t=lo_t,
        vwap=vwap, resist=resist, tod=tod,
    )


def analyst_layer(tk):
    info = yf.Ticker(tk).info or {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    mean = info.get("targetMeanPrice")
    out = dict(
        price=price, mean=mean,
        high=info.get("targetHighPrice"), low=info.get("targetLowPrice"),
        median=info.get("targetMedianPrice"),
        rmean=info.get("recommendationMean"),
        rkey=info.get("recommendationKey"),
        n=info.get("numberOfAnalystOpinions"),
        pe=info.get("trailingPE"), fpe=info.get("forwardPE"),
    )
    out["upside"] = (mean / price - 1) * 100 if (price and mean) else None
    # rating distribution (most recent row)
    try:
        rec = yf.Ticker(tk).recommendations
        if rec is not None and not rec.empty:
            r = rec.iloc[0]
            out["dist"] = {k: int(r.get(k, 0)) for k in
                           ("strongBuy", "buy", "hold", "sell", "strongSell")}
    except Exception:
        pass
    return out


def news_layer(tk, n=5):
    try:
        items = yf.Ticker(tk).news or []
    except Exception:
        return []
    out = []
    for it in items[:n]:
        c = it.get("content", it)  # yfinance schema varies
        title = c.get("title") or it.get("title")
        link = (c.get("canonicalUrl") or {}).get("url") if isinstance(
            c.get("canonicalUrl"), dict) else it.get("link")
        if title:
            out.append((title, link or ""))
    return out


def report(tk, with_news=True):
    print(f"\n{'='*60}\n{tk}\n{'='*60}")

    # legacy cost-basis P&L, if held
    if tk in LEGACY_HOLDINGS and legacy_unrealized:
        try:
            u = legacy_unrealized(tk)
            cur = legacy_currency(tk) if legacy_currency else ""
            if u is not None:
                meta = LEGACY_HOLDINGS[tk]
                print(f"[你的持仓] {meta.get('shares','?')} 股 @ 成本 "
                      f"{meta.get('cost','?')} {cur}  |  浮动盈亏 {u*100:+.1f}%")
        except Exception:
            pass

    p = price_layer(tk)
    if not p:
        print("  (无价格数据)")
        return
    rel = (p["last"] / p["vwap"] - 1) * 100
    print(f"\n[① 价格 · 近5日 15m]")
    print(f"  现价 {p['last']:.2f}   VWAP {p['vwap']:.2f} (现价 {rel:+.1f}%)")
    print(f"  周高 {p['hi']:.2f} @ {p['hi_t']:%m-%d %H:%M}   "
          f"周低 {p['lo']:.2f} @ {p['lo_t']:%m-%d %H:%M}")
    print(f"  上方阻力区 ~{p['resist']:.2f}  ->  卖出限价参考: "
          f"{p['resist']:.2f}–{p['hi']:.2f}")
    strong = p["tod"].head(3)
    weak = p["tod"].tail(2)
    print(f"  偏强时段(挂卖等反弹): " +
          ", ".join(f"{t} {v*100:+.2f}%" for t, v in strong.items()))
    print(f"  偏弱时段(别市价砸): " +
          ", ".join(f"{t} {v*100:+.2f}%" for t, v in weak.items()))

    a = analyst_layer(tk)
    print(f"\n[② 分析师 · {a.get('n','?')} 家]")
    if a["mean"]:
        print(f"  目标 均{a['mean']:.0f} / 中{a.get('median') or '?'} / "
              f"高{a.get('high') or '?'} / 低{a.get('low') or '?'}   "
              f"隐含空间 {a['upside']:+.1f}%" if a["upside"] is not None else "")
    print(f"  评级 {a.get('rkey','?')} (均值 {a.get('rmean','?')}, 1强买-5强卖)")
    if a.get("dist"):
        d = a["dist"]
        print(f"  分布  强买{d['strongBuy']} 买{d['buy']} 持有{d['hold']} "
              f"卖{d['sell']} 强卖{d['strongSell']}")
    if a.get("fpe"):
        print(f"  估值  Forward PE {a['fpe']:.1f}" +
              (f"  TTM PE {a['pe']:.1f}" if a.get("pe") else ""))

    # one-line synthesized lean
    lean = []
    if a["upside"] is not None:
        lean.append("空间有限" if a["upside"] < 5 else
                    ("空间一般" if a["upside"] < 15 else "仍有空间"))
    if rel > 0:
        lean.append("现价在VWAP之上(卖方有利)")
    else:
        lean.append("现价在VWAP之下(本周弱)")
    print(f"\n[合议] {' · '.join(lean)}  ->  "
          f"反弹到 {p['resist']:.0f}–{p['hi']:.0f} 挂限价卖, "
          f"跌破 VWAP {p['vwap']:.0f} 视为强势结束")

    if with_news:
        nx = news_layer(tk)
        if nx:
            print(f"\n[③ 新闻 · yfinance 标题]")
            for title, link in nx:
                print(f"  • {title}")
                if link:
                    print(f"    {link}")
        print("  (深度事件/宏观请让 Claude 联网搜索 — yfinance 标题较浅)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_news = "--no-news" not in sys.argv

    if not args or args == ["legacy"]:
        tickers = list(LEGACY_HOLDINGS.keys())
        if not tickers:
            print("用法: python3 sell_timing.py TSLA [RY.TO ...]  |  legacy")
            return
    else:
        tickers = args

    print("⚠️  描述性分析(过去一周 + 当前共识),非预测。样本小,勿当铁律。")
    for tk in tickers:
        try:
            report(tk, with_news=with_news)
        except Exception as e:
            print(f"\n{tk}: 失败 {e}")


if __name__ == "__main__":
    main()

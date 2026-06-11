"""
rebalance.py — 月度换仓助手
=============================================================================
新一期 picks 出来后,对比当前持仓,明确【卖出 / 买入 / 保留】各几只、
目标权重,(给定组合市值时)每只股数。省去手动对照两份名单。

⚠️ 换仓【当日市价执行】——entry-timing 已实证日内择时负期望,别等别抄底。
本脚本只做账面对比,不下单、不改模型、不写日志。

用法:
    python3 rebalance.py            # CA
    python3 rebalance.py us         # US
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

US = len(sys.argv) > 1 and sys.argv[1].lower() == "us"
import picker
if US:
    import picker_us  # noqa
from picker import fetch_prices, get_ohlcv, COMPANY_NAMES

try:
    from portfolio_config import CURRENT_HOLDINGS
except Exception:
    CURRENT_HOLDINGS = []

LOG = "picks_log_us.csv" if US else "picks_log.csv"
MARKET = "US" if US else "CA"
RED, GREEN, YELLOW, CYAN, DIM, BOLD, RESET = (
    "\033[91m", "\033[92m", "\033[93m", "\033[96m", "\033[2m", "\033[1m", "\033[0m")


def _name(t):
    return COMPANY_NAMES.get(t, "")


def main():
    print(f"\n{'='*60}\n  月度换仓助手 — {MARKET}\n{'='*60}")
    if not os.path.exists(LOG):
        print(f"  没有 {LOG} — 先出 picks。\n")
        return
    log = pd.read_csv(LOG, parse_dates=["as_of"])
    months = sorted(log["as_of"].unique())
    latest = months[-1]
    tgt = log[(log["as_of"] == latest) & (log["weight"] > 0)]
    target = dict(zip(tgt["ticker"], tgt["weight"]))
    if not target:
        print("  最新一期没有有效 picks。\n")
        return

    # 当前持仓:portfolio_config 优先,否则取上一期 picks
    if CURRENT_HOLDINGS:
        current = list(CURRENT_HOLDINGS)
        src = "portfolio_config.CURRENT_HOLDINGS"
    elif len(months) >= 2:
        prev = months[-2]
        current = log[(log["as_of"] == prev) & (log["weight"] > 0)]["ticker"].tolist()
        src = f"上一期 picks（{pd.Timestamp(prev):%Y-%m}）"
    else:
        current = []
        src = "无（首期,全部视为买入）"

    tset, cset = set(target), set(current)
    sells = sorted(cset - tset)
    buys = sorted(tset - cset)
    holds = sorted(tset & cset)

    print(f"  目标期: {pd.Timestamp(latest):%Y-%m}（{len(target)} 只）")
    print(f"  当前持仓来源: {src}（{len(current)} 只）\n")

    # 可选:组合市值 → 股数
    pv = 0.0
    try:
        raw = input("  组合市值（直接回车=只看名单不算股数): ").strip()
        pv = float(raw.replace(",", "")) if raw else 0.0
    except (ValueError, EOFError):
        pv = 0.0

    prices = {}
    if pv > 0:
        pdf = fetch_prices(sorted(tset | cset), years=1)
        for t in (tset | cset):
            c, _ = get_ohlcv(pdf, t)
            if c is not None and len(c):
                prices[t] = float(c.iloc[-1])

    def shares(t):
        if pv > 0 and t in prices and prices[t] > 0:
            w = target.get(t, 0)
            return f"  →  ~{int(pv * w / prices[t])} 股 (${pv*w:,.0f} @ {prices[t]:.2f})"
        return ""

    print(f"\n  {RED}{BOLD}■ 卖出（持仓里、新一期已不在）{RESET}")
    if sells:
        for t in sells:
            print(f"    {RED}SELL{RESET} {t:<9} {DIM}{_name(t)}{RESET}")
    else:
        print(f"    {DIM}（无）{RESET}")

    print(f"\n  {GREEN}{BOLD}■ 买入（新进）{RESET}")
    if buys:
        for t in buys:
            print(f"    {GREEN}BUY {RESET} {t:<9} {target[t]*100:.1f}%  {DIM}{_name(t)}{RESET}{shares(t)}")
    else:
        print(f"    {DIM}（无）{RESET}")

    print(f"\n  {YELLOW}{BOLD}■ 保留（继续持有,权重可微调）{RESET}")
    if holds:
        for t in holds:
            print(f"    {YELLOW}HOLD{RESET} {t:<9} {target[t]*100:.1f}%  {DIM}{_name(t)}{RESET}{shares(t)}")
    else:
        print(f"    {DIM}（无）{RESET}")

    print(f"\n  {DIM}换仓当日市价执行;别等、别用限价抄底(见 entry-timing 结论)。{RESET}\n")


if __name__ == "__main__":
    main()

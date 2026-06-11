"""
oos_tracker.py — 样本外(实盘)业绩追踪
=============================================================================
回答整个项目的核心悬念:模型上线后到底灵不灵 / edge 是运气还是本事?
读 picks_log,补齐已实现收益,算实盘 Sharpe/IR/回撤/命中率,对比回测,
给一个保守的 kill-switch 信号。

⚠️ 这只是【记录与诊断】,不改模型、不影响选股。判断力受样本量限制——
头几个月只能"不下结论",这是特性不是 bug。证据靠时间累积。

用法:
    python3 oos_tracker.py          # CA(picks_log.csv vs XIU.TO)
    python3 oos_tracker.py us       # US(picks_log_us.csv vs QQQ)
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

US = len(sys.argv) > 1 and sys.argv[1].lower() == "us"
import picker
if US:
    import picker_us  # noqa: 设 QQQ + US 日志路径
from picker import fetch_prices, backfill_realized, compute_sharpe_ci, compute_ir_ci

LOG = "picks_log_us.csv" if US else "picks_log.csv"
MARKET = "US" if US else "CA"
# 回测参照(CA 有定论;US 早期回放跑赢 QQQ 但无干净 Sharpe 留存 → 不硬比)
BT_REF = None if US else {"sharpe": 2.13, "ir": 1.08}
KILL_MIN_MONTHS = 6     # 少于此月数:不下结论
KILL_IR_FLOOR = 0.0     # ≥6 个月且实盘 IR < 此值:警告

RED, GREEN, YELLOW, DIM, BOLD, RESET = (
    "\033[91m", "\033[92m", "\033[93m", "\033[2m", "\033[1m", "\033[0m")


def _ann(series):
    """年化(月频)Sharpe;<2 点返回 nan。"""
    r = np.asarray(series, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(12)


def build_series(path):
    """返回 (组合月收益, 基准月收益, 基准ticker);仅含已成熟(已实现)月份。"""
    log = pd.read_csv(path, parse_dates=["as_of"])
    bench_tkr = (log[log["weight"] == 0]["ticker"].iloc[0]
                 if (log["weight"] == 0).any() else None)
    filled = log.dropna(subset=["fwd_realized"])
    picks = filled[filled["weight"] > 0]
    if picks.empty:
        return None, None, bench_tkr
    port = picks.groupby("as_of").apply(
        lambda g: (g["weight"] * g["fwd_realized"]).sum() / g["weight"].sum())
    bench = None
    if bench_tkr is not None:
        bench = (filled[filled["ticker"] == bench_tkr]
                 .set_index("as_of")["fwd_realized"].reindex(port.index))
    return port, bench, bench_tkr


def update_ic_log(path, market):
    """每期横截面 IC(score vs 已实现)入库 ic_log.csv。
    比 2 个月的组合 Sharpe 更早能看出模型排序是否还有信息。
    返回 (本期IC, 累计均值IC, 期数) 或 None。"""
    log = pd.read_csv(path, parse_dates=["as_of"])
    df = log[log["weight"] > 0].dropna(subset=["score", "fwd_realized"])
    if df.empty:
        return None
    recs = []
    for asof, g in df.groupby("as_of"):
        if len(g) >= 3:
            recs.append({"as_of": asof.strftime("%Y-%m-%d"), "market": market, "n": len(g),
                         "spearman_ic": round(g["score"].corr(g["fwd_realized"], method="spearman"), 4),
                         "pearson_ic": round(g["score"].corr(g["fwd_realized"], method="pearson"), 4)})
    if not recs:
        return None
    icdf = pd.DataFrame(recs)
    icfile = "ic_log.csv"
    if os.path.exists(icfile):
        old = pd.read_csv(icfile)
        old = old[old["market"] != market]
        icdf = pd.concat([old, icdf], ignore_index=True)
    icdf.sort_values(["market", "as_of"]).to_csv(icfile, index=False)
    mine = icdf[icdf["market"] == market].sort_values("as_of")
    return mine["spearman_ic"].iloc[-1], mine["spearman_ic"].mean(), len(mine)


def main():
    print(f"\n{'='*60}\n  OOS 实盘业绩追踪 — {MARKET}\n{'='*60}")
    if not os.path.exists(LOG):
        print(f"  没有 {LOG} — 先跑 picker 出过 picks 才有记录。\n")
        return

    # 1) 补齐已实现收益(需要 as_of 之后一个月的价格)
    log = pd.read_csv(LOG, parse_dates=["as_of"])
    tickers = sorted(set(log["ticker"]))
    print(f"  补齐已实现收益({len(tickers)} 标的)…")
    price_df = fetch_prices(tickers, years=2)
    if price_df is not None and len(price_df):
        backfill_realized(price_df, path=LOG)

    port, bench, bench_tkr = build_series(LOG)
    if port is None:
        print(f"  已记录 picks,但还没有成熟的月份(下个月价格到位后才算)。\n"
              f"  {DIM}回测参照:{BT_REF}{RESET}\n")
        return

    n = len(port)
    cum = (1 + port).prod() - 1
    sharpe = _ann(port)
    hit = (port > 0).mean()
    cumcurve = (1 + port).cumprod()
    mdd = (cumcurve / cumcurve.cummax() - 1).min()
    down = port[port < 0]
    sortino = (port.mean() / down.std(ddof=1) * np.sqrt(12)
               if len(down) > 1 and down.std(ddof=1) > 0 else np.nan)

    excess = ir = beat = bcum = None
    if bench is not None and bench.notna().all():
        excess = (port - bench)
        bcum = (1 + bench).prod() - 1
        ir = _ann(excess)
        beat = (port.values > bench.values).mean()

    # ── 汇总 ──
    print(f"\n  成熟月份: {n}  ({port.index.min():%Y-%m} → {port.index.max():%Y-%m})")
    print(f"  {'组合累计':<12}{cum:+.1%}")
    if bcum is not None:
        print(f"  {bench_tkr+' 累计':<12}{bcum:+.1%}   超额 {cum-bcum:+.1%}")
    print(f"  {'月均':<12}{port.mean():+.2%}   命中率 {hit:.0%}"
          + (f"   跑赢{bench_tkr} {beat:.0%}" if beat is not None else ""))
    print(f"  {'实盘 Sharpe':<12}{sharpe:.2f}" if not np.isnan(sharpe) else f"  {'实盘 Sharpe':<12}n/a(月份太少)")
    if ir is not None and not np.isnan(ir):
        print(f"  {'实盘 IR':<12}{ir:.2f}")
    if not np.isnan(sortino):
        print(f"  {'Sortino':<12}{sortino:.2f}")
    print(f"  {'最大回撤':<12}{mdd:.1%}")
    # n<KILL_MIN_MONTHS 时 CI 退化(区间宽度≈0,会误导),不显示
    if n >= KILL_MIN_MONTHS:
        s_obs, s_lo, s_hi = compute_sharpe_ci(port.values)
        print(f"  {'Sharpe 95%CI':<12}[{s_lo:.2f}, {s_hi:.2f}]")
    else:
        print(f"  {DIM}{'Sharpe 95%CI':<12} N/A(n<{KILL_MIN_MONTHS},样本不足,CI 无意义){RESET}")

    # ── 对比回测 ──
    if BT_REF:
        print(f"\n  回测参照: Sharpe {BT_REF['sharpe']}, IR {BT_REF['ir']}")
        if not np.isnan(sharpe):
            gap = sharpe - BT_REF["sharpe"]
            print(f"  实盘 vs 回测 Sharpe: {sharpe:.2f} vs {BT_REF['sharpe']} ({gap:+.2f})")

    # ── 逐月 ──
    print(f"\n  {'月份':<10}{'组合':>9}" + (f"{bench_tkr:>10}{'超额':>9}" if bench is not None else ""))
    print("  " + "-" * (28 + (19 if bench is not None else 0)))
    for d in port.index:
        line = f"  {d:%Y-%m}{'':<3}{port[d]:>+8.2%}"
        if bench is not None and not np.isnan(bench[d]):
            line += f"{bench[d]:>+10.2%}{port[d]-bench[d]:>+9.2%}"
        print(line)

    # ── 横截面 IC(score 排序能力,比组合 Sharpe 更早有信号)──
    ic = update_ic_log(LOG, MARKET)
    if ic:
        last_ic, mean_ic, k = ic
        print(f"\n  {'横截面IC':<12}本期 {last_ic:+.2f}   累计均值 {mean_ic:+.2f}({k} 期, score vs 已实现)")
        print(f"  {DIM}IC>0=分数越高收益越高(排序有效);已存 ic_log.csv{RESET}")

    # ── kill-switch ──
    print(f"\n  {'─'*56}")
    if n < KILL_MIN_MONTHS:
        print(f"  {YELLOW}判定:样本太少({n}/{KILL_MIN_MONTHS} 月)→ 不下结论。继续累积。{RESET}")
    elif ir is not None and ir < KILL_IR_FLOOR:
        print(f"  {RED}⚠️ 判定:实盘 IR {ir:.2f} < {KILL_IR_FLOOR} → 模型疑似失效,复查/考虑停用。{RESET}")
    elif BT_REF and not np.isnan(sharpe) and sharpe < 0.5 * BT_REF["sharpe"]:
        print(f"  {RED}⚠️ 判定:实盘 Sharpe {sharpe:.2f} 远低于回测 {BT_REF['sharpe']} → 警惕过拟合衰减。{RESET}")
    else:
        print(f"  {GREEN}✓ 判定:实盘暂与预期一致({n} 月)。仍需更多样本才算稳。{RESET}")
    print(f"  {DIM}kill-switch 阈值在脚本顶部常量,可调。{RESET}\n")


if __name__ == "__main__":
    main()

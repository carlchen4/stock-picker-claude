"""
concentration.py — 集中度 / 相关性体检
=============================================================================
看当前 picks 是不是"表面 N 只、实则同一个 beta"。输出:
  · 日收益相关性矩阵 + 高相关对(>0.7)
  · 行业集中度(STOCK_PROFILE)
  · 有效独立赌注数(相关矩阵特征值的 participation ratio)

⚠️ 体检,不是信号。小宇宙 + 行业上限下天然有相关性;这只帮你看清
"分散程度",不改模型、不影响选股。

用法:
    python3 concentration.py        # CA
    python3 concentration.py us     # US
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
from picker import fetch_prices, get_ohlcv, last_logged_picks, STOCK_PROFILE, COMPANY_NAMES

LOG = "picks_log_us.csv" if US else "picks_log.csv"
MARKET = "US" if US else "CA"
HIGH_CORR = 0.70
RED, GREEN, YELLOW, CYAN, DIM, BOLD, RESET = (
    "\033[91m", "\033[92m", "\033[93m", "\033[96m", "\033[2m", "\033[1m", "\033[0m")


def main():
    print(f"\n{'='*60}\n  集中度 / 相关性体检 — {MARKET}\n{'='*60}")
    if not os.path.exists(LOG):
        print(f"  没有 {LOG} — 先出 picks。\n")
        return
    picks = last_logged_picks(LOG)
    if len(picks) < 2:
        print(f"  picks 太少({len(picks)})无法体检。\n")
        return

    pdf = fetch_prices(picks, years=1)
    rets = {}
    for t in picks:
        c, _ = get_ohlcv(pdf, t)
        if c is not None and len(c) > 60:
            rets[t] = c.pct_change()
    if len(rets) < 2:
        print("  价格数据不足。\n")
        return
    R = pd.DataFrame(rets).dropna()
    corr = R.corr()
    tickers = list(corr.columns)

    # ── 相关性矩阵 ──
    print(f"\n  日收益相关性矩阵({len(tickers)} 只,近 {len(R)} 日):\n")
    hdr = "        " + "".join(f"{t.replace('.TO',''):>7}" for t in tickers)
    print(DIM + hdr + RESET)
    for i, t in enumerate(tickers):
        row = f"  {t.replace('.TO',''):<6}"
        for j, u in enumerate(tickers):
            v = corr.iloc[i, j]
            if i == j:
                cell = f"{DIM}   1.0{RESET}"
            else:
                c = RED if v >= HIGH_CORR else YELLOW if v >= 0.4 else GREEN
                cell = f"{c}{v:>6.2f}{RESET}"
            row += cell + " "
        print(row)

    # ── 高相关对 ──
    pairs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            v = corr.iloc[i, j]
            if v >= HIGH_CORR:
                pairs.append((tickers[i], tickers[j], v))
    print(f"\n  {BOLD}高相关对(≥{HIGH_CORR:.0%}):{RESET}")
    if pairs:
        for a, b, v in sorted(pairs, key=lambda x: -x[2]):
            print(f"    {RED}{v:.2f}{RESET}  {a} ↔ {b}")
    else:
        print(f"    {GREEN}无 — 分散度尚可{RESET}")

    # ── 行业集中度 ──
    from collections import Counter
    sec = Counter(STOCK_PROFILE.get(t, ("Unknown",))[0] for t in tickers)
    print(f"\n  {BOLD}行业分布:{RESET}")
    for s, cnt in sec.most_common():
        bar = "█" * cnt
        c = RED if cnt > len(tickers) * 0.5 else YELLOW if cnt >= 3 else GREEN
        print(f"    {s:<16}{c}{bar} {cnt}{RESET}")

    # ── 有效独立赌注数 ──
    # 等权 HHI:N_eff = 1/Σw²(等权时 = N);相关性 participation ratio:
    # (Σλ)² / Σλ²,λ 为相关矩阵特征值 → "实际几个独立方向"
    eig = np.linalg.eigvalsh(corr.values)
    eig = eig[eig > 0]
    n_corr = (eig.sum() ** 2) / (eig ** 2).sum()
    print(f"\n  {BOLD}有效独立赌注数:{RESET}")
    print(f"    名义持仓 {len(tickers)} 只")
    print(f"    相关性调整后 ≈ {CYAN}{n_corr:.1f}{RESET} 个独立方向"
          f"  {DIM}(越接近名义数=越分散){RESET}")
    ratio = n_corr / len(tickers)
    verdict = (f"{GREEN}分散良好{RESET}" if ratio > 0.7 else
               f"{YELLOW}中等,部分重叠{RESET}" if ratio > 0.5 else
               f"{RED}集中,多只其实押同一方向{RESET}")
    print(f"    分散度 {ratio:.0%} → {verdict}")
    print(f"\n  {DIM}体检而非信号:小宇宙+行业上限下天然相关,模型已有 sector cap。{RESET}\n")


if __name__ == "__main__":
    main()

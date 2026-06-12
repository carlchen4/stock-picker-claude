#!/usr/bin/env python3
"""
vol_rotation.py — 波动轮动信号(VIX 双阈值:保守桶 ⇄ 科技9只)
================================================================
机械轮动信号(**非投资建议**):
  • 保守桶 DEFENSIVE = KO + XLE + ZEO.TO + FTS.TO(可乐/能源 + 加拿大能源/公用,等权)
  • 进攻桶 AGGRESSIVE = 七巨头 + AMD + SNOW(9 只科技,等权)

状态机(持久化 vol_rotation_state.json,初始 DEFENSIVE):
  • DEFENSIVE: VIX 收盘 > VIX_HIGH(28)→ 切 AGGRESSIVE(卖保守桶,等权买进攻桶)
  • AGGRESSIVE 的退出有两种模式(VOL_EXIT_MODE):
      - timebox(默认):进攻只持 HOLD_DAYS(10)个交易日就强制换回保守
        —— 回测最优:把原版 -45% 回撤压到 -18%(原版"死等 VIX<18"会抱着科技穿越整个熊市)
      - threshold:等 VIX 收盘 < VIX_LOW(18)才换回(原版,回测较差,仅备选)
  • 退出后需 VIX 回落到 < VIX_HIGH 才"重新武装",避免高位反复进出。

每工作日 09:10 跑,邮件给出:当前状态、今天是否切换、对冲检查、本段两桶收益,
并 call Claude 对两桶各给推荐组合。只发信号,手动下单。

⚠️ 回测提醒:即便限时变体,长期收益仍略逊"纯持有"(详见 vol_rotation_backtest.py 顶部结论)。
复用 picker(同目录):fetch_prices / get_ohlcv / send_report_email。
用法:
  ./venv/bin/python vol_rotation.py            # 跑 + 发邮件
  ./venv/bin/python vol_rotation.py --no-email
  ./venv/bin/python vol_rotation.py --force-switch --no-email   # 测试翻转
  ./venv/bin/python vol_rotation.py --reset     # 清状态回 DEFENSIVE
"""
import os
import sys
import json
import html as _html
import re
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker  # noqa: E402


def _env_list(name, default):
    v = os.environ.get(name)
    return [x.strip() for x in v.split(",") if x.strip()] if v else default


VIX_HIGH = float(os.environ.get("VOL_VIX_HIGH", "28"))
VIX_LOW = float(os.environ.get("VOL_VIX_LOW", "18"))
DEFENSIVE = _env_list("VOL_DEFENSIVE", ["KO", "XLE", "ZEO.TO", "FTS.TO"])
AGGRESSIVE = _env_list("VOL_AGGRESSIVE",
                       ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "SNOW", "SPCX"])
CORR_WINDOW = int(os.environ.get("VOL_CORR_WINDOW", "90"))
EXIT_MODE = os.environ.get("VOL_EXIT_MODE", "timebox")   # timebox | threshold
HOLD_DAYS = int(os.environ.get("VOL_HOLD_DAYS", "10"))   # timebox 模式进攻持有交易日数
VOL_AI = os.environ.get("VOL_AI", "1") == "1"
CLAUDE_CANDIDATES = [
    os.environ.get("VOL_CLAUDE_BIN", ""),
    str(Path.home() / ".petclaw/node/bin/claude"),
    "/opt/homebrew/bin/claude",
]
VIX_TICKER = "^VIX"
STATE_FILE = DIR / "vol_rotation_state.json"
HEARTBEAT = Path.home() / "Library/Application Support/vol-rotation-tracker/vol_run.log"
STATE_LABEL = {"DEFENSIVE": "🛡️ 保守", "AGGRESSIVE": "🚀 进攻"}


# ----------------------------------------------------------------- 状态持久化
def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"state": "DEFENSIVE", "since_date": None, "armed": True}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=1))


# ----------------------------------------------------------------- 取价/计算
def get_closes(price_df, tickers):
    out = {}
    for t in tickers:
        close, _ = picker.get_ohlcv(price_df, t)
        if close is not None and len(close) > 5:
            out[t] = close
    return out


def price_on(close, date_str):
    if date_str is None or close is None:
        return None
    sub = close[close.index <= pd.Timestamp(date_str)]
    return float(sub.iloc[-1]) if len(sub) else None


def pct(a, b):
    return None if (a is None or b is None or b == 0) else (a / b - 1) * 100


def bucket_return_since(closes, tickers, since_date):
    per = {}
    for t in tickers:
        c = closes.get(t)
        if c is None:
            continue
        r = pct(float(c.iloc[-1]), price_on(c, since_date))
        if r is not None:
            per[t] = r
    return (sum(per.values()) / len(per) if per else None), per


def bucket_change_1m(closes, tickers):
    per = {}
    for t in tickers:
        c = closes.get(t)
        if c is None or len(c) < 22:
            continue
        per[t] = pct(float(c.iloc[-1]), float(c.iloc[-22]))
    return per


def bucket_daily_returns(closes, tickers):
    cols = {t: closes[t].pct_change() for t in tickers if t in closes}
    if not cols:
        return None
    return pd.DataFrame(cols).dropna(how="all").mean(axis=1)


def hedge_corr(closes):
    d = bucket_daily_returns(closes, DEFENSIVE)
    a = bucket_daily_returns(closes, AGGRESSIVE)
    if d is None or a is None:
        return None
    j = pd.concat([d.rename("d"), a.rename("a")], axis=1).dropna().tail(CORR_WINDOW)
    return float(j["d"].corr(j["a"])) if len(j) >= 20 else None


def trading_days_since(closes, since_date):
    """自 since_date 起已收盘的交易日数(用 VIX 的日期索引)。"""
    c = closes.get(VIX_TICKER)
    if c is None or since_date is None:
        return 0
    return int((c.index > pd.Timestamp(since_date)).sum())


# ----------------------------------------------------------------- Claude 两桶推荐
def _run_claude(prompt):
    import time
    for _ in range(2):
        cb = next((c for c in CLAUDE_CANDIDATES if c and Path(c).exists()), None)
        if not cb:
            time.sleep(3); continue
        env = dict(os.environ); env["PATH"] = str(Path(cb).parent) + ":" + env.get("PATH", "/usr/bin:/bin")
        try:
            r = subprocess.run([cb, "-p"], input=prompt, capture_output=True, text=True, timeout=240, env=env)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            print("Claude 失败:", (r.stderr or "")[:160]); return None
        except FileNotFoundError:
            time.sleep(3)
        except Exception as e:
            print("Claude 异常:", e); return None
    return None


def claude_recommend(vix, state, def_1m, agg_1m, corr):
    if not VOL_AI:
        return None

    def fmt(per):
        return "\n".join(f"  {t}: 近一月 {v:+.1f}%" if v is not None else f"  {t}: n/a"
                         for t, v in per.items())

    prompt = (
        "你是我的资产配置助手。我有一个 VIX 双阈值波动轮动策略,两个固定篮子:\n"
        f"  • 保守桶:{', '.join(DEFENSIVE)}\n  • 进攻桶(科技):{', '.join(AGGRESSIVE)}\n"
        f"当前 VIX={vix:.1f}(高 {VIX_HIGH:.0f}/低 {VIX_LOW:.0f}),当前状态={state}。"
        f"两桶近 {CORR_WINDOW} 天日收益相关系数={corr:.2f}(越低/负越好)。\n\n"
        f"保守桶近一月:\n{fmt(def_1m)}\n\n进攻桶近一月:\n{fmt(agg_1m)}\n\n"
        "请用**中文**,只在给定标的池内:\n"
        "1. **保守桶推荐**:4 只权重(合 100%)+ 一句理由;\n"
        "2. **进攻桶推荐**:9 只权重/取舍(合约 100%)+ 一句理由;\n"
        "3. 一句话:当前波动环境更该偏哪个桶。\n简洁分两段,可直接放邮件。**不是投资建议**。"
    )
    out = _run_claude(prompt)
    if out:
        print("🤖 Claude 两桶推荐完成")
    return out


# ----------------------------------------------------------------- markdown→HTML
def _inline(s):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _html.escape(s))


def md_to_html(md):
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        st = lines[i].strip()
        if "|" in st and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]):
            header = [c.strip() for c in st.strip("|").split("|")]; i += 2; rows = []
            while i < len(lines) and "|" in lines[i]:
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            th = "".join(f'<th style="border:1px solid #ddd;padding:5px;background:#f5f5f5">{_inline(c)}</th>' for c in header)
            trs = "".join("<tr>" + "".join(f'<td style="border:1px solid #ddd;padding:5px">{_inline(c)}</td>' for c in r) + "</tr>" for r in rows)
            out.append(f'<table style="border-collapse:collapse;margin:8px 0;font-size:13px"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'); continue
        if re.match(r"^#{1,6}\s", st):
            out.append("<h4 style='margin:10px 0 4px'>" + _inline(st.lstrip("# ").strip()) + "</h4>")
        elif re.match(r"^(-|\*|\d+[.)])\s+", st):
            items = []
            while i < len(lines) and re.match(r"^\s*(-|\*|\d+[.)])\s+", lines[i]):
                items.append("<li>" + _inline(re.sub(r"^\s*(-|\*|\d+[.)])\s+", "", lines[i].strip())) + "</li>"); i += 1
            out.append("<ul style='margin:4px 0;padding-left:18px'>" + "".join(items) + "</ul>"); continue
        elif st:
            out.append("<p style='margin:6px 0'>" + _inline(st) + "</p>")
        i += 1
    return "\n".join(out)


# ----------------------------------------------------------------- 邮件
def _per_html(per):
    def c(v):
        return "#2e7d32" if (v is not None and v >= 0) else "#c62828"
    return " &nbsp; ".join(f'{t} <span style="color:{c(v)}">{v:+.1f}%</span>' if v is not None else f"{t} n/a"
                           for t, v in per.items())


def build_email(vix, state, action, switched, corr, def_ret, agg_ret,
                def_1m, agg_1m, since_date, hold_info, claude_txt):
    label = STATE_LABEL.get(state, state)
    if corr is None:
        hedge = "对冲检查:数据不足"
    else:
        verdict = ("对冲成立 ✅(低/负相关)" if corr < 0.3 else
                   "对冲一般 ⚠️" if corr < 0.6 else "对冲较弱 ❌(高度同向)")
        hedge = f"保守桶 vs 进攻桶 近 {CORR_WINDOW} 天相关 <b>{corr:+.2f}</b> — {verdict}"

    def rf(r):
        if r is None:
            return "n/a"
        return f'<span style="color:{"#2e7d32" if r>=0 else "#c62828"}">{r:+.1f}%</span>'

    claude_html = (f'<div style="background:#f0f6ff;border:1px solid #cfe0ff;border-radius:8px;padding:10px 14px;margin:12px 0">'
                   f'<b>🤖 Claude 两桶组合推荐</b><div style="font-size:13.5px">{md_to_html(claude_txt)}</div></div>'
                   if claude_txt else
                   '<div style="background:#fff8e1;border:1px solid #ffe0a3;border-radius:8px;padding:10px 14px;margin:12px 0">'
                   '<b>🤖 Claude 两桶推荐</b><div style="font-size:13.5px">等权(Claude 未返回)</div></div>')
    ac = "#c62828" if switched else "#555"
    html = (
        f'<div style="font-family:-apple-system,Arial;font-size:14px;max-width:680px">'
        f'<p style="font-size:16px"><b>📊 波动轮动信号</b> &nbsp; {datetime.now():%Y-%m-%d}</p>'
        f'<p>VIX 现值 <b>{vix:.1f}</b> &nbsp;(>{VIX_HIGH:.0f} 切进攻 / 退出: {("限时 "+str(HOLD_DAYS)+" 交易日") if EXIT_MODE=="timebox" else ("VIX<"+str(int(VIX_LOW)))})</p>'
        f'<p>当前状态:<b style="font-size:15px">{label}</b> &nbsp; '
        f'{"持 "+"/".join(DEFENSIVE) if state=="DEFENSIVE" else "持科技9只 "+"/".join(AGGRESSIVE)}'
        f'{hold_info}</p>'
        f'<p style="background:#f5f5f5;border-radius:6px;padding:8px 12px">今日动作:<b style="color:{ac}">{action}</b></p>'
        f'<p>🔗 {hedge}</p>'
        f'<p>本段持有以来(自 {since_date or "(未记录)"}):保守桶 {rf(def_ret)} &nbsp;|&nbsp; 进攻桶 {rf(agg_ret)}</p>'
        f'<p style="color:#666;font-size:13px">保守桶近一月:{_per_html(def_1m)}</p>'
        f'<p style="color:#666;font-size:13px">进攻桶近一月:{_per_html(agg_1m)}</p>'
        f'{claude_html}'
        f'<p style="color:#999;font-size:12px">⚠️ 机械信号(VIX 高买高 beta 科技、限时后回防),逆向/均值回归思路;'
        f'回测显示长期略逊纯持有,科技在 VIX 飙升时可能继续跌。<b>非投资建议</b>,自行判断手动下单。</p></div>')
    text = (f"波动轮动 {datetime.now():%Y-%m-%d}\nVIX {vix:.1f} 状态 {label} | 动作: {action}\n"
            f"对冲相关: {corr if corr is None else round(corr,2)} | 本段 保守{def_ret}/进攻{agg_ret}\n\n"
            + (claude_txt or "(Claude 未返回)"))
    return text, html


# ----------------------------------------------------------------- 主流程
def heartbeat(msg):
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        with open(HEARTBEAT, "a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--force-switch", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if args.reset:
        save_state({"state": "DEFENSIVE", "since_date": None, "armed": True})
        print("已重置 → DEFENSIVE"); return

    st = load_state()
    state = st["state"]
    today = datetime.now().strftime("%Y-%m-%d")
    since_date = st.get("since_date") or today   # 首次运行从今天起记录本段
    armed = st.get("armed", True)

    tickers = [VIX_TICKER] + DEFENSIVE + AGGRESSIVE
    # 每天信号必须用最新 VIX:清掉 picker 的 ~20h 缓存,强制拉最新
    try:
        cf = picker._cache_path(f"prices_{len(tickers)}_2y.parquet")
        if cf.exists():
            cf.unlink()
    except Exception as e:
        print("清缓存失败(不致命):", e)
    print(f"取价 {len(tickers)} 个标的(强制最新)…")
    price_df = picker.fetch_prices(tickers, years=2)
    closes = get_closes(price_df, tickers)

    vclose = closes.get(VIX_TICKER)
    if vclose is None:
        print("❌ 拿不到 VIX"); heartbeat("FAIL no-vix"); return
    vix = float(vclose.iloc[-1])

    # ---- 状态机 ----
    new_state, switched = state, False
    held_days = trading_days_since(closes, since_date) if state == "AGGRESSIVE" else 0
    if args.force_switch:
        new_state = "AGGRESSIVE" if state == "DEFENSIVE" else "DEFENSIVE"; switched = True
    elif state == "DEFENSIVE":
        if armed and vix > VIX_HIGH:
            new_state, switched = "AGGRESSIVE", True
    else:  # AGGRESSIVE
        if EXIT_MODE == "timebox":
            if held_days >= HOLD_DAYS:
                new_state, switched = "DEFENSIVE", True
        else:  # threshold
            if vix < VIX_LOW:
                new_state, switched = "DEFENSIVE", True

    # 重新武装:回到保守且 VIX 已回落到高阈值下
    if new_state == "DEFENSIVE" and vix < VIX_HIGH:
        armed = True
    elif new_state == "AGGRESSIVE":
        armed = False

    if switched:
        action = ("🔔 切换:卖保守桶 → 等权买进攻桶 9 只" if new_state == "AGGRESSIVE"
                  else "🔔 切换:卖进攻桶 9 只 → 买回保守桶")
    else:
        action = f"维持(不动)— 继续持 {'保守桶' if state=='DEFENSIVE' else '进攻桶 9 只'}"

    # 进攻进度提示(timebox)
    hold_info = ""
    cur_state_for_disp = new_state
    if cur_state_for_disp == "AGGRESSIVE" and EXIT_MODE == "timebox":
        d = 0 if switched else held_days
        hold_info = f' <span style="color:#888;font-size:12px">(进攻第 {d}/{HOLD_DAYS} 交易日,到点回防)</span>'

    corr = hedge_corr(closes)
    new_since = today if switched else since_date
    def_ret, _ = bucket_return_since(closes, DEFENSIVE, new_since)
    agg_ret, _ = bucket_return_since(closes, AGGRESSIVE, new_since)
    def_1m = bucket_change_1m(closes, DEFENSIVE)
    agg_1m = bucket_change_1m(closes, AGGRESSIVE)

    print(f"VIX={vix:.1f} 状态 {state}→{new_state} switched={switched} held_days={held_days} corr={corr}")
    print(f"动作: {action}")

    claude_txt = claude_recommend(vix, new_state, def_1m, agg_1m, corr if corr is not None else 0.0)
    text, html = build_email(vix, new_state, action, switched, corr, def_ret, agg_ret,
                             def_1m, agg_1m, new_since, hold_info, claude_txt)

    if args.no_email:
        print("\n--- 邮件预览 ---\n" + text)
    else:
        picker.send_report_email(text, html, f"📊 波动轮动 — {STATE_LABEL.get(new_state,new_state)} — VIX {vix:.1f}")

    if not (args.no_email and args.force_switch):
        save_state({"state": new_state, "since_date": new_since, "armed": armed})
    heartbeat(f"OK vix={vix:.1f} state={new_state} switched={switched}")


if __name__ == "__main__":
    main()

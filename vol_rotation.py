#!/usr/bin/env python3
"""
vol_rotation.py — 科技 9 只持仓日报(等权,Claude 给权重建议)
================================================================
背景:原本设计成"VIX 双阈值轮动(保守桶 ⇄ 科技桶)",但 5 年回测显示
**轮动反而拖累收益**(原版 +94.9% / 最大回撤 -44.9%),而**一直持有科技 9 只
等权收益最高(+220.8%)**。所以改成:不轮动,每天就跟踪这 9 只科技,
Claude 在固定池内给权重建议,显示持有以来收益,VIX 仅作市场情绪提示。
  科技 9 只 = 七巨头 + AMD + SNOW(AAPL/MSFT/NVDA/AMZN/META/GOOGL/TSLA/AMD/SNOW)

⚠️ 只发信号、不自动交易。**这是高 beta 篮子,历史上有过 -51.8% 回撤,非投资建议。**
回测脚本见 vol_rotation_backtest.py。

复用 picker(同目录):fetch_prices / get_ohlcv / send_report_email。
用法:
  ./venv/bin/python vol_rotation.py            # 跑 + 发邮件
  ./venv/bin/python vol_rotation.py --no-email # 只打印
  ./venv/bin/python vol_rotation.py --reset     # 重置"持有起始日"为今天
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


TECH = _env_list("VOL_TECH",
                 ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "SNOW"])
VOL_AI = os.environ.get("VOL_AI", "1") == "1"
# 候选 claude 二进制(取第一个存在的;petclaw 偶尔在自更新时短暂消失,故多备一个)
CLAUDE_CANDIDATES = [
    os.environ.get("VOL_CLAUDE_BIN", ""),
    str(Path.home() / ".petclaw/node/bin/claude"),
    "/opt/homebrew/bin/claude",
]
VIX_TICKER = "^VIX"

STATE_FILE = DIR / "vol_rotation_state.json"
HEARTBEAT = Path.home() / "Library/Application Support/vol-rotation-tracker/vol_run.log"


# ----------------------------------------------------------------- 起始日持久化
def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"since_date": None}


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


def change_since(closes, tickers, since_date):
    """各标的自 since_date 起 %;返回 (等权均值, {t:%})。"""
    per = {}
    for t in tickers:
        c = closes.get(t)
        if c is None:
            continue
        r = pct(float(c.iloc[-1]), price_on(c, since_date))
        if r is not None:
            per[t] = r
    avg = sum(per.values()) / len(per) if per else None
    return avg, per


def change_1m(closes, tickers):
    per = {}
    for t in tickers:
        c = closes.get(t)
        if c is None or len(c) < 22:
            continue
        per[t] = pct(float(c.iloc[-1]), float(c.iloc[-22]))
    return per


# ----------------------------------------------------------------- Claude 权重
def _claude_bin():
    for c in CLAUDE_CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


def _run_claude(prompt):
    """跑 claude -p,自更新窗口可能短暂找不到二进制 → 重试一次。"""
    import time
    for attempt in range(2):
        cb = _claude_bin()
        if not cb:
            time.sleep(3); continue
        env = dict(os.environ)
        env["PATH"] = str(Path(cb).parent) + ":" + env.get("PATH", "/usr/bin:/bin")
        try:
            r = subprocess.run([cb, "-p"], input=prompt, capture_output=True,
                               text=True, timeout=240, env=env)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            print("Claude 失败:", (r.stderr or "")[:200])
            return None
        except FileNotFoundError:
            print("Claude 二进制在自更新中,3s 后重试…"); time.sleep(3)
        except Exception as e:
            print("Claude 异常:", e); return None
    return None


def claude_weights(vix, since_avg, m1, since_per):
    if not VOL_AI:
        return None

    def fmt(t):
        a = f"{m1.get(t):+.1f}%" if m1.get(t) is not None else "n/a"
        b = f"{since_per.get(t):+.1f}%" if since_per.get(t) is not None else "n/a"
        return f"  {t}: 近一月 {a},持有以来 {b}"

    prompt = (
        "你是我的持仓配置助手。我**等权持有这 9 只科技股**(固定池,不增删):\n"
        + "\n".join(fmt(t) for t in TECH) + "\n\n"
        f"当前 VIX={vix:.1f}(市场情绪;高=恐慌)。\n\n"
        "请用**中文**,**只在这 9 只内**给出建议:\n"
        "1. 一份推荐权重(9 只加起来 100%,可对动量强/估值合理的超配,对最弱的低配但别清零);\n"
        "2. 每只一句极简理由(就一行);\n"
        "3. 结合当前 VIX,一句话说该整体加仓、减仓还是持平;\n"
        "4. 一句风险提示(这是高 beta 篮子)。\n"
        "简洁、可直接放邮件。**不是投资建议**。"
    )
    out = _run_claude(prompt)
    if out:
        print("🤖 Claude 权重建议完成")
    return out


# ----------------------------------------------------------------- markdown→HTML
def _inline(s):
    s = _html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return s


def md_to_html(md):
    """markdown → HTML:标题/管道表格/列表/加粗/段落。"""
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        st = lines[i].strip()
        # 表格:本行有 | 且下一行是分隔行 |---|---|
        if "|" in st and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]):
            header = [c.strip() for c in st.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i]:
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            th = "".join(f'<th style="border:1px solid #ddd;padding:5px;background:#f5f5f5">{_inline(c)}</th>' for c in header)
            trs = "".join("<tr>" + "".join(f'<td style="border:1px solid #ddd;padding:5px">{_inline(c)}</td>' for c in r) + "</tr>" for r in rows)
            out.append(f'<table style="border-collapse:collapse;margin:8px 0;font-size:13px"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>')
            continue
        if re.match(r"^#{1,6}\s", st):
            out.append("<h4 style='margin:10px 0 4px'>" + _inline(st.lstrip("# ").strip()) + "</h4>")
        elif re.match(r"^(-|\*|\d+[.)])\s+", st):
            items = []
            while i < len(lines) and re.match(r"^\s*(-|\*|\d+[.)])\s+", lines[i]):
                items.append("<li>" + _inline(re.sub(r"^\s*(-|\*|\d+[.)])\s+", "", lines[i].strip())) + "</li>"); i += 1
            out.append("<ul style='margin:4px 0;padding-left:18px'>" + "".join(items) + "</ul>"); continue
        elif st == "":
            pass
        else:
            out.append("<p style='margin:6px 0'>" + _inline(st) + "</p>")
        i += 1
    return "\n".join(out)


# ----------------------------------------------------------------- 邮件渲染
def _color(v):
    return "#2e7d32" if (v is not None and v >= 0) else "#c62828"


def _per_table(m1, since_per):
    rows = ""
    for t in TECH:
        a, b = m1.get(t), since_per.get(t)
        rows += (f'<tr><td style="padding:3px 10px">{t}</td>'
                 f'<td style="padding:3px 10px;text-align:right;color:{_color(a)}">'
                 f'{a:+.1f}%</td>' if a is not None else
                 f'<tr><td style="padding:3px 10px">{t}</td><td style="padding:3px 10px;text-align:right">n/a</td>')
        rows += (f'<td style="padding:3px 10px;text-align:right;color:{_color(b)}">'
                 f'{b:+.1f}%</td></tr>' if b is not None else
                 '<td style="padding:3px 10px;text-align:right">n/a</td></tr>')
    return (f'<table style="border-collapse:collapse;font-size:13px;border:1px solid #eee">'
            f'<tr style="background:#f5f5f5"><th style="padding:3px 10px;text-align:left">标的</th>'
            f'<th style="padding:3px 10px">近一月</th><th style="padding:3px 10px">持有以来</th></tr>'
            f'{rows}</table>')


def build_email(vix, since_avg, since_date, m1, since_per, claude_txt):
    avg_txt = (f'<span style="color:{_color(since_avg)}">{since_avg:+.1f}%</span>'
               if since_avg is not None else "n/a")
    vix_note = ("⚠️ VIX 偏高,波动剧烈,注意控制仓位" if vix >= 25
                else "市场情绪平稳" if vix < 18 else "市场情绪中性")
    claude_html = (
        f'<div style="background:#f0f6ff;border:1px solid #cfe0ff;border-radius:8px;padding:10px 14px;margin:12px 0">'
        f'<b>🤖 Claude 权重建议(9 只科技)</b><div style="font-size:13.5px">{md_to_html(claude_txt)}</div></div>'
        if claude_txt else
        '<div style="background:#fff8e1;border:1px solid #ffe0a3;border-radius:8px;padding:10px 14px;margin:12px 0">'
        '<b>🤖 Claude 权重建议</b><div style="font-size:13.5px">等权(Claude 未返回)</div></div>')

    html = (
        f'<div style="font-family:-apple-system,Arial;font-size:14px;max-width:680px">'
        f'<p style="font-size:16px"><b>📈 科技 9 只持仓日报</b> &nbsp; {datetime.now():%Y-%m-%d}</p>'
        f'<p>持有(等权):{"/".join(TECH)}</p>'
        f'<p>VIX <b>{vix:.1f}</b> — {vix_note}</p>'
        f'<p style="background:#f5f5f5;border-radius:6px;padding:8px 12px">'
        f'持有以来(自 {since_date or "(未记录)"}):篮子等权收益 <b>{avg_txt}</b></p>'
        f'{_per_table(m1, since_per)}'
        f'{claude_html}'
        f'<p style="color:#999;font-size:12px">⚠️ 高 beta 科技篮子,5 年回测最大回撤 -51.8%;'
        f'纯持有比 VIX 轮动收益更高(220.8% vs 94.9%),但波动大。<b>非投资建议</b>,自行判断。</p>'
        f'</div>')

    text = (f"科技 9 只持仓日报 {datetime.now():%Y-%m-%d}\n"
            f"持有(等权): {', '.join(TECH)}\n"
            f"VIX {vix:.1f} — {vix_note}\n"
            f"持有以来等权收益: {since_avg if since_avg is None else round(since_avg,1)}%\n\n"
            + (claude_txt or "(Claude 未返回,等权)"))
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
    ap.add_argument("--reset", action="store_true", help="重置持有起始日为今天")
    args = ap.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    if args.reset:
        save_state({"since_date": today})
        print(f"已重置持有起始日 → {today}"); return

    state = load_state()
    since_date = state.get("since_date") or today

    tickers = [VIX_TICKER] + TECH
    # 每天的信号必须用最新 VIX:清掉 picker 的 ~20h 价格缓存,强制拉最新
    # (否则早上 09:10 跑会吃到昨天的缓存 → VIX 卡住不更新)
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
    vix = float(vclose.iloc[-1]) if vclose is not None else float("nan")

    since_avg, since_per = change_since(closes, TECH, since_date)
    m1 = change_1m(closes, TECH)

    print(f"VIX={vix:.1f}  持有以来等权收益={since_avg}")
    claude_txt = claude_weights(vix, since_avg, m1, since_per)

    text, html = build_email(vix, since_avg, since_date, m1, since_per, claude_txt)

    if args.no_email:
        print("\n--- 邮件文本预览 ---\n" + text)
    else:
        subject = f"📈 科技9只持仓 — VIX {vix:.1f} — 持有 {since_avg:+.1f}%" \
            if since_avg is not None else f"📈 科技9只持仓 — VIX {vix:.1f}"
        picker.send_report_email(text, html, subject)

    # 落起始日(首次运行时记下今天)
    save_state({"since_date": since_date})
    heartbeat(f"OK vix={vix:.1f} since_avg={since_avg}")


if __name__ == "__main__":
    main()

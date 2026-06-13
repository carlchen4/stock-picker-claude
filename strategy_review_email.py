#!/usr/bin/env python3
"""
strategy_review_email.py — 波动轮动策略评审邮件(基于真实代码输出 + Claude 评价)
==============================================================================
**实跑** vol_rotation_backtest.py / event_study.py / oos_tracker.py,把**真实 stdout**
原样喂给 Claude 评价(不喂手写总结),发邮件。这样 Claude 的评价扎根在真数据上。
跑:./venv/bin/python strategy_review_email.py
"""
import os
import re
import sys
import html as _html
import subprocess
from pathlib import Path
from datetime import datetime

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker  # noqa: E402

PY = str(DIR / "venv/bin/python")
CLAUDE_CANDIDATES = [str(Path.home() / ".petclaw/node/bin/claude"), "/opt/homebrew/bin/claude"]


def run(args, timeout=300):
    """跑脚本,返回清洗后的真实 stdout。"""
    try:
        r = subprocess.run([PY] + args, cwd=str(DIR), capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.returncode != 0 else "")
    except Exception as e:
        out = f"(运行失败: {e})"
    # 去噪
    out = "\n".join(l for l in out.splitlines()
                    if "NotOpenSSL" not in l and "warnings.warn" not in l and l.strip())
    return out.strip()


def claude_eval(digest):
    prompt = (
        "你是资深量化策略评审,立场中立、敢说真话。下面是我一套 VIX 波动轮动对冲策略的"
        "**真实代码输出**(回测、事件研究、实盘 OOS 追踪,原样粘贴,数字都是跑出来的)。\n"
        "请**只基于这些真实数字**批判性评价(中文,简洁分点,可放邮件),务必**引用具体数字**佐证:\n"
        "1. 这套择时策略 vs 里面的简单基准(一直保守 / 50-50 / 一直进攻),到底有没有创造价值?用数字说。\n"
        "2. 实盘 OOS 和回测的差距说明什么?\n"
        "3. 事件研究的数字支持还是不支持'飙升后买科技跑赢防守'?\n"
        "4. 最大硬伤(结合数字)。\n"
        "5. 一句话诚实结论:跑、不跑、还是怎么跑。\n"
        "别泛泛而谈、别复述定义,每个判断后面跟它依据的数字。不是投资建议。\n\n"
        f"=== 真实代码输出 ===\n{digest}"
    )
    import time
    for _ in range(2):
        cb = next((c for c in CLAUDE_CANDIDATES if c and Path(c).exists()), None)
        if not cb:
            time.sleep(3); continue
        env = dict(os.environ); env["PATH"] = str(Path(cb).parent) + ":" + env.get("PATH", "/usr/bin:/bin")
        try:
            r = subprocess.run([cb, "-p"], input=prompt, capture_output=True, text=True, timeout=300, env=env)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            return None
        except FileNotFoundError:
            time.sleep(3)
        except Exception:
            return None
    return None


def _inline(s):
    s = _html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code style='background:#eee;padding:0 3px;border-radius:3px'>\1</code>", s)
    return s


def md(text):
    """markdown → HTML:管道表格 / 标题 / 列表 / 加粗 / 分割线 / 段落。"""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        st = lines[i].strip()
        # 管道表格:本行含 | 且下一行是分隔行 |---|---|
        if "|" in st and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]):
            header = [c.strip() for c in st.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            th = "".join(f'<th style="border:1px solid #ddd;padding:6px 10px;background:#f5f5f5;text-align:left">{_inline(c)}</th>' for c in header)
            trs = ""
            for r in rows:
                trs += "<tr>" + "".join(f'<td style="border:1px solid #ddd;padding:6px 10px">{_inline(c)}</td>' for c in r) + "</tr>"
            out.append(f'<table style="border-collapse:collapse;margin:10px 0;font-size:13.5px"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>')
            continue
        if re.match(r"^#{1,6}\s", st):
            lvl = len(st) - len(st.lstrip("#"))
            out.append(f"<h{min(lvl+2,5)} style='margin:12px 0 4px'>" + _inline(st.lstrip("# ").strip()) + f"</h{min(lvl+2,5)}>")
        elif re.match(r"^-{3,}$", st) or re.match(r"^={3,}$", st):
            out.append("<hr style='border:none;border-top:1px solid #eee;margin:10px 0'>")
        elif re.match(r"^(-|\*|\d+[.、)])\s+", st):
            items = []
            while i < len(lines) and re.match(r"^\s*(-|\*|\d+[.、)])\s+", lines[i]):
                items.append("<li style='margin:2px 0'>" + _inline(re.sub(r"^\s*(-|\*|\d+[.、)])\s+", "", lines[i].strip())) + "</li>"); i += 1
            out.append("<ul style='margin:4px 0;padding-left:20px'>" + "".join(items) + "</ul>"); continue
        elif st == "":
            pass
        else:
            out.append("<p style='margin:6px 0'>" + _inline(st) + "</p>")
        i += 1
    return "\n".join(out)


def main():
    print("实跑回测…"); bt = run(["vol_rotation_backtest.py", "--years", "5"])
    print("实跑事件研究…"); es = run(["event_study.py", "--years", "18"])
    print("实跑 OOS 追踪…"); oos = run(["oos_tracker.py"])
    digest = (f"【1. 5年回测:择时 vs 简单基准】\n{bt}\n\n"
              f"【2. 事件研究:VIX>28 后科技 vs 防守(18年)】\n{es}\n\n"
              f"【3. 实盘 OOS 追踪】\n{oos}")
    print("调 Claude 基于真实输出评价…")
    ev = claude_eval(digest)
    print("Claude:", "OK" if ev else "无返回")

    ev_html = (f'<div style="background:#f0f6ff;border:1px solid #cfe0ff;border-radius:8px;padding:12px 16px;margin:12px 0">'
               f'<b>🤖 Claude 评价(基于本机实跑回测/事件研究/OOS 真实数字)</b><div style="font-size:14px">{md(ev)}</div></div>'
               if ev else '<p>(Claude 未返回)</p>')
    html = (f'<div style="font-family:-apple-system,Arial;font-size:14px;max-width:760px">'
            f'<p style="font-size:17px"><b>📊 波动轮动策略评审 — Claude 基于真实回测的评价</b></p>'
            f'<p>{datetime.now():%Y-%m-%d %H:%M}</p>'
            f'{ev_html}'
            f'<p style="color:#999;font-size:12px">数字均为本机实跑 vol_rotation_backtest / event_study / oos_tracker 的输出(原始输出已喂给 Claude,不在此展开)。非投资建议。</p></div>')
    text = f"波动轮动策略评审 {datetime.now():%Y-%m-%d}\n\n=== Claude 评价 ===\n{ev or '(无)'}"
    picker.send_report_email(text, html, f"📊 波动轮动评审(Claude×真实回测)— {datetime.now():%m-%d}")
    print("已发邮件。")


if __name__ == "__main__":
    main()

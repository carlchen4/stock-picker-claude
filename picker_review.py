#!/usr/bin/env python3
"""
picker_review.py — 选股复盘(每月 picker 跑完自动执行)
========================================================
① 模型复盘(每次都跑):喂给 Claude「方法论文档 + 最新选股(因子分/已实现) +
   样本外实盘业绩(oos_tracker)」,让它评模型优缺点/结果/是否需优化/不足/
   可考虑的代码改动方向。
② 我的选择 vs 模型(my_picks.txt 填了才跑):对比我实际选的 vs 模型推荐,
   分析偏离的优点和风险。
两部分合成一封邮件(复用 picker 的 email_config)。仅分析,不改任何代码。
"""
import os
import re
import csv
import html as _html
import subprocess
from pathlib import Path
from datetime import datetime

import smtplib
import ssl
from email.message import EmailMessage

DIR = Path(__file__).resolve().parent
MY_PICKS = DIR / "my_picks.txt"
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", str(Path.home() / ".petclaw/node/bin/claude"))
PY = str(DIR / "venv/bin/python")
BENCH = {"XIU.TO", "QQQ", "SPY"}
DOC_FILES = ["WORKFLOW.md", "PLAN.md", "README_run.md"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------- 数据收集
def latest_recos(csv_path, with_realized=False):
    if not Path(csv_path).exists():
        return None, []
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r.get("as_of")]
    if not rows:
        return None, []
    as_of = max(r["as_of"] for r in rows)
    out = []
    for r in rows:
        if r["as_of"] != as_of:
            continue
        t = r["ticker"]
        try:
            w = float(r["weight"] or 0)
        except ValueError:
            w = 0
        if t in BENCH or w <= 0:
            continue
        out.append((t, w, r.get("score", ""), r.get("fwd_realized", "")))
    return as_of, out


def run_capture(args):
    try:
        r = subprocess.run([PY] + args, cwd=str(DIR), capture_output=True, text=True, timeout=180)
        return ANSI.sub("", r.stdout or "")
    except Exception as e:
        return f"(运行失败: {e})"


def read_docs(limit_each=1800):
    parts = []
    for f in DOC_FILES:
        p = DIR / f
        if p.exists():
            parts.append(f"### {f}\n" + p.read_text(encoding="utf-8")[:limit_each])
    return "\n\n".join(parts)[:5500]


def claude(prompt):
    env = dict(os.environ)
    env["PATH"] = str(Path(CLAUDE_BIN).parent) + ":" + env.get("PATH", "/usr/bin:/bin")
    try:
        r = subprocess.run([CLAUDE_BIN, "-p"], input=prompt, capture_output=True,
                           text=True, timeout=300, env=env)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        print("Claude 失败:", r.stderr[:200])
    except Exception as e:
        print("Claude 异常:", e)
    return None


# ---------------------------------------------------------------- ① 模型复盘
def model_critique():
    ca_asof, ca = latest_recos(DIR / "picks_log.csv")
    us_asof, us = latest_recos(DIR / "picks_log_us.csv")

    def fmt(lst):
        return "\n".join(f"  - {t} 权重{w*100:.1f}% 分数{s} 本期已实现{fr}"
                         for t, w, s, fr in lst) or "  (无)"
    oos_ca = run_capture(["oos_tracker.py"])
    oos_us = run_capture(["oos_tracker.py", "us"])
    docs = read_docs()
    prompt = (
        "你是资深量化策略评审。下面是一个多因子选股模型(加拿大 TSX + 美股科技)的"
        "方法论文档、最新一期选股(含因子分数与已实现收益)、以及样本外实盘业绩追踪。\n"
        "请**基于这些实际数据**(不要臆造)用**中文**给出:\n"
        "1. 模型**优点**(做对了什么);\n"
        "2. 模型**缺点/不足**(数据暴露的问题:实盘 vs 回测衰减、IR、命中率、集中度、样本量等);\n"
        "3. **本期结果**评价;\n"
        "4. **是否需要优化、往哪优化**(因子/权重/再平衡/风控 kill-switch/universe 等);\n"
        "5. 具体**可考虑的代码改动方向**(描述即可,不写完整代码;指出可能涉及哪些文件/逻辑);\n"
        "6. 一句话结论。\n"
        "样本量小,结论务必保守、标明不确定性。不是投资建议。\n\n"
        f"== 方法论文档(节选) ==\n{docs}\n\n"
        f"== 最新选股(CA,截至 {ca_asof}) ==\n{fmt(ca)}\n\n"
        f"== 最新选股(US,截至 {us_asof}) ==\n{fmt(us)}\n\n"
        f"== 样本外实盘业绩(CA) ==\n{oos_ca[-1800:]}\n\n"
        f"== 样本外实盘业绩(US) ==\n{oos_us[-1800:]}"
    )
    return claude(prompt)


# ---------------------------------------------------------------- ② 我的选择 vs 模型
def my_picks():
    if not MY_PICKS.exists():
        return []
    out = []
    for ln in MY_PICKS.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln.split()[0].upper())
    return out


def picks_review(mine):
    ca_asof, ca = latest_recos(DIR / "picks_log.csv")
    us_asof, us = latest_recos(DIR / "picks_log_us.csv")
    rec_all = {t for t, *_ in ca} | {t for t, *_ in us}
    ms = set(mine)

    def fmt(lst):
        return "\n".join(f"  - {t} 权重{w*100:.1f}% 分数{s}" for t, w, s, _ in lst) or "  (无)"
    prompt = (
        "你是我的投资助理。下面是本月**选股模型推荐**和我**最后实际选/买的票**。\n"
        "用**中文**分析(非投资建议):\n"
        "1. 我**保留/跳过/自己加(off-model)**了哪些;\n"
        "2. 每个『和模型不一样』的决定的**优点和风险**;\n"
        "3. 整体偏离点评(激进/保守、集中度、行业暴露);\n4. 一句话总结。\n\n"
        f"== 模型推荐(CA,{ca_asof}) ==\n{fmt(ca)}\n\n"
        f"== 模型推荐(US,{us_asof}) ==\n{fmt(us)}\n\n"
        f"== 我最后选的 ==\n  {', '.join(mine)}\n"
        f"(保留 {sorted(rec_all & ms) or '无'};跳过 {sorted(rec_all - ms) or '无'};自己加 {sorted(ms - rec_all) or '无'})"
    )
    return claude(prompt)


# ---------------------------------------------------------------- markdown → HTML
def _inline(s):
    s = _html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def md_to_html(md):
    """把 Claude 的 markdown(标题/加粗/列表/表格/分割线)渲染成邮件 HTML。"""
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i].rstrip()
        st = ln.strip()
        # 表格:连续含 | 的行,第二行是 |---|
        if "|" in st and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]):
            header = [c.strip() for c in st.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i]:
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th = "".join(f'<th style="border:1px solid #ddd;padding:5px;background:#f5f5f5">{_inline(c)}</th>' for c in header)
            trs = ""
            for r in rows:
                trs += "<tr>" + "".join(f'<td style="border:1px solid #ddd;padding:5px">{_inline(c)}</td>' for c in r) + "</tr>"
            out.append(f'<table style="border-collapse:collapse;margin:8px 0"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>')
            continue
        if re.match(r"^#{1,6}\s", st):
            lvl = len(st) - len(st.lstrip("#"))
            out.append(f'<h{min(lvl+1,5)} style="margin:12px 0 4px">{_inline(st.lstrip("# ").strip())}</h{min(lvl+1,5)}>')
        elif re.match(r"^(-{3,}|\*{3,}|_{3,})$", st):
            out.append("<hr style='border:none;border-top:1px solid #eee;margin:10px 0'>")
        elif re.match(r"^[-*]\s+", st):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                txt = re.sub(r"^\s*[-*]\s+", "", lines[i])
                items.append("<li>" + _inline(txt) + "</li>")
                i += 1
            out.append("<ul style='margin:4px 0;padding-left:20px'>" + "".join(items) + "</ul>")
            continue
        elif re.match(r"^\d+\.\s+", st):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                txt = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                items.append("<li>" + _inline(txt) + "</li>")
                i += 1
            out.append("<ol style='margin:4px 0;padding-left:20px'>" + "".join(items) + "</ol>")
            continue
        elif st == "":
            out.append("")
        else:
            out.append(f"<p style='margin:6px 0'>{_inline(st)}</p>")
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------- 邮件
def send(sections, n_my):
    import email_config as ec
    body = "\n\n".join(f"{'='*48}\n{title}\n{'='*48}\n{txt}" for title, txt in sections)
    html_parts = []
    for title, txt in sections:
        html_parts.append(f'<h2 style="border-bottom:2px solid #333;margin-top:18px">{title}</h2>'
                          f'<div>{md_to_html(txt)}</div>')
    html = ('<div style="font-family:-apple-system,Arial;font-size:14px;line-height:1.5">'
            f'<p>🧠 <b>选股复盘</b> {datetime.now():%Y-%m-%d}</p>' + "".join(html_parts) + '</div>')
    msg = EmailMessage()
    extra = f" + 我选{n_my}只" if n_my else ""
    msg["Subject"] = f"🧠 模型复盘{extra} — {datetime.now():%m-%d}"
    msg["From"] = ec.EMAIL_FROM; msg["To"] = ec.EMAIL_TO
    msg.set_content(body); msg.add_alternative(html, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(ec.EMAIL_FROM, ec.EMAIL_APP_PASSWORD); s.send_message(msg)
    print(f"📧 复盘已发送到 {ec.EMAIL_TO}")


def main():
    sections = []
    print("① 模型复盘(Claude)…")
    crit = model_critique()
    if crit:
        sections.append(("① 模型复盘与优化建议", crit))
        print("\n" + crit + "\n")
    mine = my_picks()
    if mine:
        print("② 我的选择 vs 模型(Claude)…")
        rev = picks_review(mine)
        if rev:
            sections.append(("② 我的选择 vs 模型", rev))
            print("\n" + rev + "\n")
    else:
        print("my_picks.txt 未填,跳过②。")
    if not sections:
        print("无内容,不发信。"); return
    try:
        send(sections, len(mine))
    except Exception as e:
        print("发信失败:", e)


if __name__ == "__main__":
    main()

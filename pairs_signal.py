#!/usr/bin/env python3
"""
pairs_signal.py — 相对价值信号面板(多头版,不做空)
========================================================
对验证过的 4 对(V-MA / ADP-PAYX / AZO-ORLY / MCO-SPGI;CB-TRV 已剔除):
每天算 价差 z-score(log(A/B),60日窗),给出"该持哪只 + 偏离程度",记录 CSV 供 paper-track。
规则:z<-ENTRY → A 相对便宜 → 持 A;z>+ENTRY → 持 B;|z|<EXIT → 接近均衡(维持/各半)。
⚠️ 已验证(样本外+跨期+参数),但 edge 温和(+0.1 Sharpe)。先纸上跟踪 2-3 月,别急投真钱。非投资建议。

跑:./venv/bin/python pairs_signal.py            # 打印 + 记 CSV
   ./venv/bin/python pairs_signal.py --email     # 同时发邮件
"""
import os
import sys
import csv
import io
import ssl
import smtplib
import html as _html
from datetime import datetime
from pathlib import Path
from email.message import EmailMessage

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import picker  # noqa: E402


def chart_pair_png(sa, sb, la, lb, title):
    """两只股票价格走势(各自归一到100)画在一起,返回 PNG bytes。"""
    sa, sb = sa.dropna(), sb.dropna()
    idx = sa.index.intersection(sb.index)
    sa, sb = sa[idx], sb[idx]
    na, nb = sa / sa.iloc[0] * 100, sb / sb.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(5.4, 2.1))
    ax.plot(range(len(na)), na.values, color="#1565c0", lw=1.4, label=f"{la} {sa.iloc[-1]:.0f}")
    ax.plot(range(len(nb)), nb.values, color="#e65100", lw=1.4, label=f"{lb} {sb.iloc[-1]:.0f}")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_ylabel("归一=100", fontsize=8); ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return buf.getvalue()

# CB-TRV(财险)2026-06-13 剔除:保险巨灾/准备金噪声大、基本面最不同质、统计最弱(post仅+0.02)
PAIRS = [("V", "MA", "支付双寡头"), ("ADP", "PAYX", "薪资处理"),
         ("AZO", "ORLY", "汽配零售"), ("MCO", "SPGI", "评级机构")]
LOOKBACK = 60
ENTRY = 1.5
EXIT = 0.5
LOG = DIR / "pairs_signal_log.csv"


def closes(px, t):
    c, _ = picker.get_ohlcv(px, t)
    return c.dropna() if c is not None else None


def main():
    email = "--email" in sys.argv
    tk = sorted({t for a, b, _ in PAIRS for t in (a, b)})
    cf = picker._cache_path(f"prices_{len(tk)}_2y.parquet")
    if os.path.exists(cf):
        os.remove(cf)
    px = picker.fetch_prices(tk, years=2)
    cl = {t: closes(px, t) for t in tk}

    rows = []
    prices = {}       # "A-B" -> (近252日 A价, B价),画走势图用
    for a, b, desc in PAIRS:
        ca, cb = cl.get(a), cl.get(b)
        if ca is None or cb is None:
            rows.append((desc, a, b, None, None, None, None, "缺数据")); continue
        df = pd.concat([ca.rename("a"), cb.rename("b")], axis=1).dropna()
        spread = np.log(df["a"] / df["b"])
        m, s = spread.rolling(LOOKBACK).mean(), spread.rolling(LOOKBACK).std()
        zs = ((spread - m) / s).dropna()
        z = float(zs.iloc[-1])
        prices[f"{a}-{b}"] = (df["a"].tail(252), df["b"].tail(252))
        pa, pb = float(df["a"].iloc[-1]), float(df["b"].iloc[-1])
        cheap = a if z < 0 else b               # z<0: A 相对便宜
        if abs(z) < EXIT:
            action = "≈ 接近均衡 → 各半/维持"
        elif abs(z) >= ENTRY:
            action = f"🔔 持 {cheap}(显著便宜)"
        else:
            action = f"持 {cheap}(偏离中)"
        rows.append((desc, a, b, z, pa, pb, cheap, action))

    # 打印
    print(f"\n📐 相对价值信号  {datetime.now():%Y-%m-%d}  (z=log(A/B)偏离;|z|≥{ENTRY}显著)")
    print(f"{'配对':<18}{'z':>7}{'  该持':>8}   动作")
    print("-" * 60)
    for desc, a, b, z, pa, pb, cheap, action in rows:
        if z is None:
            print(f"{desc+' '+a+'-'+b:<18}  缺数据"); continue
        print(f"{(desc+' '+a+'-'+b)[:17]:<18}{z:>+7.2f}{cheap:>8}   {action}")

    # 记 CSV(paper-track)
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        if new:
            w.writerow(["date", "pair", "z", "px_a", "px_b", "hold"])
        today = datetime.now().strftime("%Y-%m-%d")
        for desc, a, b, z, pa, pb, cheap, action in rows:
            if z is not None:
                w.writerow([today, f"{a}-{b}", round(z, 3), round(pa, 2), round(pb, 2), cheap])
    print(f"\n✅ 已记录到 {LOG.name}(paper-track)")

    actionable = [r for r in rows if r[3] is not None and abs(r[3]) >= ENTRY]
    if "--email-on-signal" in sys.argv:
        email = bool(actionable)            # 只在有显著信号时发
        if not actionable:
            print("无 actionable 信号(|z|<1.5),不发邮件。")
    if not email:
        return

    # 表格 + 5 张走势图(CID 内嵌,Hotmail/Outlook 可显示)
    imgs = []          # (cid, png_bytes)
    trs = charts = ""
    for i, (desc, a, b, z, pa, pb, cheap, action) in enumerate(rows):
        if z is None:
            continue
        col = "#c62828" if abs(z) >= ENTRY else ("#888" if abs(z) < EXIT else "#1565c0")
        trs += (f'<tr><td style="padding:4px 10px">{_html.escape(desc)} <b>{a}-{b}</b></td>'
                f'<td style="padding:4px 10px;text-align:right;color:{col}"><b>{z:+.2f}</b></td>'
                f'<td style="padding:4px 10px"><b>{cheap}</b></td>'
                f'<td style="padding:4px 10px">{_html.escape(action)}</td></tr>')
        sa, sb = prices[f"{a}-{b}"]
        cid = f"chart{i}"
        imgs.append((cid, chart_pair_png(sa, sb, a, b, f"{desc} {a}-{b}  (z={z:+.2f} → 持{cheap})")))
        charts += f'<img src="cid:{cid}" style="width:100%;max-width:560px;display:block;margin:8px 0"><br>'

    html = (f'<div style="font-family:-apple-system,Arial;font-size:14px;max-width:600px">'
            f'<p style="font-size:16px"><b>📐 相对价值信号({len(PAIRS)}对)</b> &nbsp; {datetime.now():%Y-%m-%d}</p>'
            f'<p style="color:#666;font-size:13px">z=log(A/B)的60日z-score;z&lt;0→A相对便宜→持A。|z|≥{ENTRY}显著。</p>'
            f'<table style="border-collapse:collapse;border:1px solid #eee">'
            f'<tr style="background:#f5f5f5"><th style="padding:4px 10px;text-align:left">配对</th>'
            f'<th style="padding:4px 10px">z</th><th style="padding:4px 10px">持</th>'
            f'<th style="padding:4px 10px;text-align:left">动作</th></tr>{trs}</table>'
            f'<p style="font-weight:bold;margin-top:12px">📈 {len(PAIRS)}对价格走势(各自归一到100)</p>{charts}'
            f'<p style="color:#999;font-size:12px">已验证(样本外+跨期+参数稳)但edge温和(+0.1Sh);'
            f'paper-track中,先别投真钱。多头相对价值、注册账户可执行。非投资建议。</p></div>')
    text = "相对价值信号\n" + "\n".join(
        f"{a}-{b}: z={z:+.2f} 持{cheap} {action}" for desc, a, b, z, pa, pb, cheap, action in rows if z is not None)

    try:
        import email_config as cfg
        msg = EmailMessage()
        msg["Subject"] = f"📐 相对价值信号({len(PAIRS)}对)— {datetime.now():%m-%d}"
        msg["From"], msg["To"] = cfg.EMAIL_FROM, cfg.EMAIL_TO
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
        html_part = msg.get_payload()[1]
        for cid, png in imgs:
            html_part.add_related(png, maintype="image", subtype="png", cid=f"<{cid}>")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
            s.login(cfg.EMAIL_FROM, cfg.EMAIL_APP_PASSWORD); s.send_message(msg)
        print(f"📧 已发邮件(含 {len(imgs)} 张走势图)")
    except Exception as e:
        print("发邮件失败:", e)


if __name__ == "__main__":
    main()

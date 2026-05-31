"""
intraday_monitor.py — 当日盯盘态势感知(NOT a timing signal)
=============================================================================
全天实时追踪当天选中股票(CA + US picks,如 CIBC=CM.TO)的分钟级盘面:
现价 / 当日区间位置 / VWAP 偏离 / 成交量 / 支撑阻力 / 日内拐点 / 日内 RSI。

⚠️ 这是【盯盘辅助】,不是择时工具,不出任何"买/卖"信号。
用户自己的 entry-timing 研究(entry_timing.py / limit_buy_test.py,CIBC 例子)
已实证:日内抄底 / 等确认 / 限价挂单都是负期望(净 +0.8~1.1% TSX / +3% US
比直接市价买更差)。执行铁律 = pick 当日市价买入。本工具只帮你看清盘面、
挑机下单、心理踏实,买不买、何时买由你定。

用法:
    python3 intraday_monitor.py            # 当天全部 picks(CA+US),全天刷新
    python3 intraday_monitor.py CM.TO      # 只盯一只
    python3 intraday_monitor.py CM.TO 5m   # 指定 interval(默认 1m,空则回退 5m)
Ctrl-C 退出。
"""
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from scipy.signal import argrelextrema
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# picker 依赖可降级:本地正常 import;云端(GitHub Actions 只装轻量依赖,
# 没有 sklearn 等)import 失败时用内置最小实现 + company_names.json。
try:
    from picker import last_logged_picks, COMPANY_NAMES, compute_rsi
except Exception:
    import json as _json
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "company_names.json"), encoding="utf-8") as _f:
            COMPANY_NAMES = _json.load(_f)
    except Exception:
        COMPANY_NAMES = {}

    def last_logged_picks(path=None):
        if not path or not os.path.exists(path):
            return []
        try:
            log = pd.read_csv(path, parse_dates=["as_of"])
        except Exception:
            return []
        if log.empty:
            return []
        last = log[log["as_of"] == log["as_of"].max()]
        return last[last["weight"] > 0]["ticker"].tolist()

    def compute_rsi(close, window=14):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


def _to_et(ts):
    """把(可能是 UTC 的)时间戳转成 ET 显示字符串 HH:MM。"""
    try:
        if ts.tzinfo is not None:
            return ts.tz_convert("America/New_York").strftime("%H:%M")
    except Exception:
        pass
    return ts.strftime("%H:%M")

# ── ANSI 配色(沿用 run_monitor 风格)──────────────────────────────
RED, GREEN, YELLOW = "\033[91m", "\033[92m", "\033[93m"
CYAN, BOLD, DIM, RESET = "\033[96m", "\033[1m", "\033[2m", "\033[0m"
CLEAR = "\033[2J\033[H"

REFRESH_SEC = 60          # 终端刷新间隔
DEFAULT_INTERVAL = "1m"

# ── HTML 手机版(GitHub Pages,iPhone 适配)────────────────────────
HTML_FILE = "docs/intraday.html"
HTML_URL = "https://carlchen4.github.io/stock-picker-claude/intraday.html"
HTML_META_REFRESH = 60    # 页面自动重载秒数
PUSH_EVERY_SEC = 300      # 推送 GitHub 节奏(慢,避免 commit 刷屏)


# ── 时区 / 交易时段(9:30–16:00 ET)────────────────────────────────
def _now_et():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()  # 退化:用本地时间(仅影响开/收盘判断的提示)


def market_phase(now_et):
    """returns 'pre' | 'open' | 'closed'."""
    if now_et.weekday() >= 5:
        return "closed"
    t = now_et.time()
    if t < dtime(9, 30):
        return "pre"
    if t <= dtime(16, 0):
        return "open"
    return "closed"


# ── picks 加载 ─────────────────────────────────────────────────────
def _picks_from_dashboard():
    """从已公开的 docs/data*.json 读当天 picks(云端用:pick log 是私有 gitignored)。"""
    import json
    root = os.path.dirname(os.path.abspath(__file__))
    out = []
    for fn in ("docs/data.json", "docs/data_us.json"):
        p = os.path.join(root, fn)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                picks = json.load(f).get("picks", [])
            for it in picks:
                t = it.get("ticker")
                if t and it.get("weight", 1) and t not in out:
                    out.append(t)
        except Exception:
            continue
    return out


def load_picks(cli_ticker):
    if cli_ticker:
        return [cli_ticker.upper() if "." not in cli_ticker else cli_ticker]
    out, seen = [], set()
    # 本地优先:私有 pick log(完整历史)
    for path in ("picks_log.csv", "picks_log_us.csv"):
        for t in last_logged_picks(path):
            if t not in seen:
                seen.add(t)
                out.append(t)
    # 回退(云端:仓库里没有 pick log)→ 用公开的 dashboard json
    if not out:
        out = _picks_from_dashboard()
    return out


# ── 数据抓取 ───────────────────────────────────────────────────────
def fetch_intraday(ticker, interval):
    df = yf.download(ticker, period="1d", interval=interval,
                     progress=False, auto_adjust=False)
    if (df is None or df.empty) and interval == "1m":
        df = yf.download(ticker, period="1d", interval="5m",
                         progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


def prev_close(ticker):
    """上一交易日收盘(用于当日涨跌基准)。"""
    try:
        h = yf.Ticker(ticker).history(period="7d", auto_adjust=False)
        if h.empty:
            return None
        today = _now_et().date()
        prior = h[[d.date() < today for d in h.index]]
        if not prior.empty:
            return float(prior["Close"].iloc[-1])
        return float(h["Close"].iloc[-2]) if len(h) >= 2 else float(h["Close"].iloc[-1])
    except Exception:
        return None


# ── 指标计算(全部描述性)──────────────────────────────────────────
def turning_points(close, order):
    """返回最近的局部极小/极大点列表 [(idx, price, 'low'/'high')]。"""
    pts = []
    vals = close.values
    if _HAVE_SCIPY and len(vals) > 2 * order + 1:
        lows = argrelextrema(vals, np.less_equal, order=order)[0]
        highs = argrelextrema(vals, np.greater_equal, order=order)[0]
        for i in lows:
            pts.append((i, vals[i], "low"))
        for i in highs:
            pts.append((i, vals[i], "high"))
    pts.sort(key=lambda x: x[0])
    # 去掉相邻同类挤在一起的(保留每段一个)
    dedup = []
    for p in pts:
        if dedup and dedup[-1][2] == p[2] and p[0] - dedup[-1][0] <= order:
            continue
        dedup.append(p)
    return dedup


def analyze(ticker, interval):
    df = fetch_intraday(ticker, interval)
    if df is None or len(df) < 3:
        return {"ticker": ticker, "error": "无日内数据"}
    close = df["Close"]
    last = float(close.iloc[-1])
    day_high, day_low = float(df["High"].max()), float(df["Low"].min())
    rng = day_high - day_low
    range_pos = (last - day_low) / rng * 100 if rng > 0 else 50.0

    # VWAP
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    vwap_ser = (typical * df["Volume"]).cumsum() / cum_vol.replace(0, np.nan)
    vwap = float(vwap_ser.iloc[-1]) if not np.isnan(vwap_ser.iloc[-1]) else last
    vwap_dev = (last / vwap - 1) * 100 if vwap else 0.0

    # 成交量
    last_vol = float(df["Volume"].iloc[-1])
    avg_vol = float(df["Volume"].replace(0, np.nan).mean() or 0)
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else np.nan

    # 开盘区间(前 30 分钟)
    bars_30m = 30 if interval == "1m" else 6
    opening = df.iloc[:bars_30m]
    or_high, or_low = float(opening["High"].max()), float(opening["Low"].min())

    # 前收
    pc = prev_close(ticker)
    day_chg = (last / pc - 1) * 100 if pc else np.nan

    # 日内拐点
    order = 5 if interval == "1m" else 2
    tps = turning_points(close, order)
    recent_tps = tps[-3:]
    tp_desc = []
    for i, price, kind in recent_tps:
        since = (last / price - 1) * 100
        label = "低点" if kind == "low" else "高点"
        tp_desc.append((_to_et(close.index[i]), label, price, since))

    # 日内 RSI + EMA 状态
    rsi = compute_rsi(close)
    rsi_val = float(rsi.iloc[-1]) if not rsi.dropna().empty else np.nan
    ema9 = close.ewm(span=9).mean().iloc[-1]
    ema21 = close.ewm(span=21).mean().iloc[-1]
    ema_state = "EMA9>EMA21" if ema9 > ema21 else "EMA9<EMA21"

    # 折线图用:价格序列(下采样)+ 拐点位置(按全长比例,便于在图上标)
    ser = close.values.astype(float).tolist()
    n = len(ser)
    step = max(1, n // 140)
    series_ds = ser[::step]
    tp_pts = [(i / (n - 1) if n > 1 else 0.0, float(price), kind)
              for (i, price, kind) in tps]

    return {
        "ticker": ticker, "last": last, "day_chg": day_chg, "prev_close": pc,
        "day_high": day_high, "day_low": day_low, "range_pos": range_pos,
        "vwap": vwap, "vwap_dev": vwap_dev,
        "last_vol": last_vol, "vol_ratio": vol_ratio,
        "or_high": or_high, "or_low": or_low,
        "tps": tp_desc, "rsi": rsi_val, "ema_state": ema_state,
        "last_bar_time": _to_et(close.index[-1]),
        "interval": interval if not df.empty else interval,
        "series": series_ds, "tp_pts": tp_pts,
    }


# ── 展示 ───────────────────────────────────────────────────────────
def _col(v, fmt="{:+.2f}%", pos=GREEN, neg=RED, zero=DIM):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{DIM}N/A{RESET}"
    c = pos if v > 0 else neg if v < 0 else zero
    return f"{c}{fmt.format(v)}{RESET}"


def range_zone(p):
    if p >= 66:
        return f"{RED}上沿{RESET}"
    if p <= 33:
        return f"{GREEN}下沿{RESET}"
    return f"{YELLOW}中部{RESET}"


def render(picks_data, interval):
    now = _now_et()
    phase = market_phase(now)
    phase_txt = {"pre": f"{YELLOW}盘前(未开盘){RESET}",
                 "open": f"{GREEN}交易中{RESET}",
                 "closed": f"{DIM}已收盘{RESET}"}[phase]
    out = [CLEAR]
    out.append(f"{BOLD}{CYAN}══ 日内盯盘监控 ══{RESET}  {now.strftime('%Y-%m-%d %H:%M:%S ET')}  |  {phase_txt}  |  interval={interval}")
    out.append(f"{DIM}盯盘辅助 · 非择时 · 执行=pick当日市价买入(见 entry-timing 结论) · 数据可能延迟~15min{RESET}")
    out.append("")
    for d in picks_data:
        t = d["ticker"]
        name = COMPANY_NAMES.get(t, "")
        if "error" in d:
            out.append(f"  {BOLD}{t:<9}{RESET} {DIM}{name}{RESET}  {RED}{d['error']}{RESET}")
            out.append("")
            continue
        out.append(f"  {BOLD}{t:<9}{RESET} {DIM}{name}{RESET}   现价 {BOLD}{d['last']:.2f}{RESET}  "
                   f"当日 {_col(d['day_chg'])}  {DIM}(更新 {d['last_bar_time']}){RESET}")
        out.append(f"     区间  低 {d['day_low']:.2f} ─[{range_zone(d['range_pos'])} {d['range_pos']:.0f}%]─ 高 {d['day_high']:.2f}"
                   f"   开盘区间 {d['or_low']:.2f}–{d['or_high']:.2f}")
        vol_txt = (f"{d['vol_ratio']:.1f}×均量" if not np.isnan(d['vol_ratio']) else "N/A")
        out.append(f"     VWAP {d['vwap']:.2f} (现价 {_col(d['vwap_dev'])})   "
                   f"量 {vol_txt}   RSI {d['rsi']:.0f}   {DIM}{d['ema_state']}{RESET}"
                   if not np.isnan(d['rsi']) else
                   f"     VWAP {d['vwap']:.2f} (现价 {_col(d['vwap_dev'])})   量 {vol_txt}   {DIM}{d['ema_state']}{RESET}")
        if d["tps"]:
            parts = []
            for tm, lbl, pr, since in d["tps"]:
                c = GREEN if lbl == "低点" else RED
                parts.append(f"{tm} {c}{lbl}{RESET} {pr:.2f}{DIM}(距今{since:+.1f}%){RESET}")
            out.append(f"     {DIM}近期拐点:{RESET} " + "  ".join(parts))
        out.append("")
    out.append(f"{DIM}Ctrl-C 退出 · 每 {REFRESH_SEC}s 刷新{RESET}")
    print("\n".join(out), flush=True)


def _h(v, fmt="{:+.2f}%"):
    """HTML 涨跌着色 span。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="na">N/A</span>'
    cls = "up" if v > 0 else "dn" if v < 0 else "fl"
    return f'<span class="{cls}">{fmt.format(v)}</span>'


def _zone_html(p):
    if p >= 66:
        return '<b class="dn">上沿</b>'
    if p <= 33:
        return '<b class="up">下沿</b>'
    return '<b class="fl">中部</b>'


def svg_chart(d, W=404, H=132):
    """内联 SVG 分时折线图:价格线 + VWAP 虚线 + 前收虚线 + 拐点标记。"""
    ser = d.get("series") or []
    if len(ser) < 3:
        return ""
    PADX, PADT, PADB = 4, 10, 14
    lo, hi = min(ser), max(ser)
    # 让 VWAP/前收也纳入纵轴范围,标记不会跑出框
    for extra in (d.get("vwap"), d.get("prev_close")):
        if extra and not np.isnan(extra):
            lo, hi = min(lo, extra), max(hi, extra)
    rng = (hi - lo) or 1.0
    def px(frac): return PADX + frac * (W - 2 * PADX)
    def py(v):    return PADT + (1 - (v - lo) / rng) * (H - PADT - PADB)

    n = len(ser)
    pts = " ".join(f"{px(j/(n-1)):.1f},{py(v):.1f}" for j, v in enumerate(ser))
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" preserveAspectRatio="none">']
    # 前收参考线
    pc = d.get("prev_close")
    if pc and lo <= pc <= hi:
        y = py(pc)
        parts.append(f'<line x1="{PADX}" y1="{y:.1f}" x2="{W-PADX}" y2="{y:.1f}" '
                     f'stroke="#475569" stroke-width="1" stroke-dasharray="2 3"/>')
    # VWAP 线
    vw = d.get("vwap")
    if vw and lo <= vw <= hi:
        y = py(vw)
        parts.append(f'<line x1="{PADX}" y1="{y:.1f}" x2="{W-PADX}" y2="{y:.1f}" '
                     f'stroke="#d97706" stroke-width="1" stroke-dasharray="4 3"/>')
    # 价格线
    up = d["day_chg"] is not None and not np.isnan(d["day_chg"]) and d["day_chg"] >= 0
    line_c = "#34d399" if up else "#f87171"
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{line_c}" '
                 f'stroke-width="1.6" stroke-linejoin="round"/>')
    # 拐点标记
    for frac, price, kind in d.get("tp_pts", []):
        if not (lo <= price <= hi):
            continue
        cx, cy = px(frac), py(price)
        c = "#22c55e" if kind == "low" else "#ef4444"
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.6" fill="{c}" '
                     f'stroke="#0f172a" stroke-width="0.8"/>')
    # 最新点
    parts.append(f'<circle cx="{px(1.0):.1f}" cy="{py(ser[-1]):.1f}" r="3.2" '
                 f'fill="#fff" stroke="{line_c}" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(picks_data, interval):
    """写 docs/intraday.html(单列、深色、393px 适配)。"""
    now = _now_et()
    phase = market_phase(now)
    phase_txt = {"pre": "盘前(未开盘)", "open": "交易中", "closed": "已收盘"}[phase]
    cards = []
    for d in picks_data:
        t = d["ticker"]
        name = COMPANY_NAMES.get(t, "")
        if "error" in d:
            cards.append(f'<div class="card"><div class="tk">{t}'
                         f'<span class="nm">{name}</span></div>'
                         f'<div class="err">{d["error"]}</div></div>')
            continue
        rsi_txt = f'{d["rsi"]:.0f}' if not np.isnan(d["rsi"]) else "N/A"
        vol_txt = f'{d["vol_ratio"]:.1f}×' if not np.isnan(d["vol_ratio"]) else "N/A"
        tps = ""
        if d["tps"]:
            items = []
            for tm, lbl, pr, since in d["tps"]:
                cls = "up" if lbl == "低点" else "dn"
                items.append(f'<span class="tp"><i>{tm}</i> '
                             f'<b class="{cls}">{lbl}</b> {pr:.2f}'
                             f'<i>({since:+.1f}%)</i></span>')
            tps = '<div class="tps">近期拐点 ' + " ".join(items) + "</div>"
        cards.append(f"""
        <div class="card">
          <div class="row1">
            <div class="tk">{t}<span class="nm">{name}</span></div>
            <div class="px">{d['last']:.2f} <span class="chg">{_h(d['day_chg'])}</span></div>
          </div>
          {svg_chart(d)}
          <div class="rng"><span>低 {d['day_low']:.2f}</span>
            <span class="lg"><i class="d up"></i>低点 <i class="d dn"></i>高点 <i class="l vw"></i>VWAP <i class="l pc"></i>前收</span>
            <span>高 {d['day_high']:.2f}</span></div>
          <div class="metrics">
            <div><label>VWAP</label>{d['vwap']:.2f} {_h(d['vwap_dev'])}</div>
            <div><label>量</label>{vol_txt}均量</div>
            <div><label>RSI</label>{rsi_txt}</div>
            <div><label>EMA</label>{d['ema_state']}</div>
          </div>
          {tps}
          <div class="upd">更新 {d['last_bar_time']} ET</div>
        </div>""")

    html = f"""<!DOCTYPE html><html lang="zh"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="{HTML_META_REFRESH}">
<title>日内盯盘</title>
<style>
  :root {{ --up:#16a34a; --dn:#dc2626; --fl:#d97706; --bg:#0f172a; --card:#1e293b; --bd:#334155; --mut:#94a3b8; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:#e2e8f0; font-size:14px; -webkit-text-size-adjust:100%; }}
  .wrap {{ max-width:430px; margin:0 auto; padding:12px; }}
  header {{ background:linear-gradient(135deg,#0f172a,#1e3a5f 55%,#1d4ed8); border-radius:12px; padding:14px 16px; margin-bottom:12px; }}
  header h1 {{ font-size:18px; font-weight:800; }}
  .sub {{ font-size:11px; color:var(--mut); margin-top:4px; line-height:1.5; }}
  .ph {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700; background:#334155; }}
  .ph.open {{ background:#14532d; color:#86efac; }} .ph.closed {{ background:#3f3f46; color:#a1a1aa; }} .ph.pre {{ background:#713f12; color:#fde68a; }}
  .card {{ background:var(--card); border:1px solid var(--bd); border-radius:12px; padding:12px 14px; margin-bottom:10px; }}
  .row1 {{ display:flex; justify-content:space-between; align-items:baseline; }}
  .tk {{ font-size:16px; font-weight:800; }}
  .nm {{ font-size:11px; color:var(--mut); font-weight:400; margin-left:6px; }}
  .px {{ font-size:18px; font-weight:800; }} .chg {{ font-size:13px; }}
  .chart {{ width:100%; height:120px; display:block; margin:8px 0 4px; background:#0f172a; border-radius:8px; }}
  .rng {{ display:flex; justify-content:space-between; align-items:center; font-size:10px; color:var(--mut); gap:6px; }}
  .rng b {{ margin:0 3px; }}
  .lg {{ display:flex; align-items:center; gap:6px; flex-wrap:wrap; justify-content:center; }}
  .lg .d {{ display:inline-block; width:7px; height:7px; border-radius:50%; vertical-align:middle; margin-right:2px; }}
  .lg .l {{ display:inline-block; width:10px; height:0; border-top:2px dashed; vertical-align:middle; margin-right:2px; }}
  .lg .d.up {{ background:#22c55e; }} .lg .d.dn {{ background:#ef4444; }}
  .lg .l.vw {{ border-color:#d97706; }} .lg .l.pc {{ border-color:#475569; }}
  .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:4px 12px; margin-top:8px; font-size:13px; }}
  .metrics label {{ color:var(--mut); font-size:11px; margin-right:6px; }}
  .tps {{ margin-top:8px; font-size:11px; color:var(--mut); line-height:1.7; }}
  .tp i {{ font-style:normal; color:#64748b; }}
  .upd {{ margin-top:6px; font-size:10px; color:#64748b; text-align:right; }}
  .up {{ color:var(--up); }} .dn {{ color:var(--dn); }} .fl {{ color:var(--fl); }}
  .na {{ color:var(--mut); }} .err {{ color:var(--dn); font-size:12px; margin-top:6px; }}
  .foot {{ font-size:10px; color:#64748b; text-align:center; padding:8px 0 16px; line-height:1.6; }}
</style></head><body><div class="wrap">
<header>
  <h1>日内盯盘 <span class="ph {phase}">{phase_txt}</span></h1>
  <div class="sub">{now.strftime('%Y-%m-%d %H:%M ET')} · interval={interval} · 每{HTML_META_REFRESH}s自动刷新<br>
  盯盘辅助 · 非择时 · 执行=pick当日市价买入 · 数据可能延迟~15min</div>
</header>
{''.join(cards)}
<div class="foot">本页仅态势感知,不构成买卖信号。<br>entry-timing 已实证日内抄底负期望 → 当日市价买入,别等、别抄底。</div>
</div></body></html>"""

    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, HTML_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def push_html():
    """提交并推送 docs/intraday.html(仅在文件有变化时,沿用 _push_dashboard 模式)。"""
    import subprocess
    root = os.path.dirname(os.path.abspath(__file__))
    # 无变化则跳过(避免空 commit)
    diff = subprocess.run(["git", "diff", "--quiet", "--", HTML_FILE],
                          cwd=root, capture_output=True)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", HTML_FILE],
                            cwd=root, capture_output=True)
    if diff.returncode == 0 and staged.returncode == 0:
        return None  # 无变化
    for cmd in (["git", "add", HTML_FILE],
                ["git", "commit", "-m", "intraday: snapshot"],
                ["git", "push"]):
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            return f"push 失败({cmd[1]}):{r.stderr.strip()[:80]}"
    return "pushed"


def main():
    args = [a for a in sys.argv[1:]]
    cli_ticker = None
    interval = DEFAULT_INTERVAL
    html_mode = False
    once = False
    for a in args:
        if a in ("1m", "2m", "5m", "15m"):
            interval = a
        elif a in ("--html", "html", "--phone"):
            html_mode = True
        elif a in ("--once", "once"):
            once = True
            html_mode = True
        else:
            cli_ticker = a

    picks = load_picks(cli_ticker)
    if not picks:
        print("没有可监控的标的 — 先跑 picker.py pick / picker_us.py pick,或显式传 ticker。")
        return

    # --once:生成一次 HTML 即退出(GitHub Actions 用,提交由 workflow 负责)
    if once:
        data = [analyze(t, interval) for t in picks]
        path = render_html(data, interval)
        print(f"wrote {path}")
        return

    print(f"加载标的:{', '.join(picks)}  (interval={interval})  "
          f"{'HTML手机版 → '+HTML_URL if html_mode else '终端模式'}  首次抓取中…")
    last_push = 0.0
    try:
        while True:
            data = [analyze(t, interval) for t in picks]
            phase = market_phase(_now_et())
            if html_mode:
                render_html(data, interval)
                now = time.time()
                if now - last_push >= PUSH_EVERY_SEC:
                    res = push_html()
                    if res:
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] docs/intraday.html {res}")
                    last_push = now
            else:
                render(data, interval)
            # 盘后/周末:刷一次后降频(避免空转烧请求 / commit 刷屏)
            sleep_s = REFRESH_SEC if phase == "open" else max(REFRESH_SEC, 300)
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        msg = "已退出盯盘。记住:执行 = 当日市价买入,别等、别抄底。"
        print(f"\n{DIM}{msg}{RESET}" if not html_mode else f"\n{msg}")


if __name__ == "__main__":
    main()

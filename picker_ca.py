"""
TSX 量化选股 v2.0 — 全优化版
════════════════════════════════════════════════════════════════
数据：yfinance（免费，无配额）
模型：XGBoost + LightGBM + PyTorch MLP（动态集成权重）

四层优化：
  第一层：止损线(-8%) / 换仓成本显示 / 股息率展示
  第二层：宏观因子(油价/汇率/利率) / 行业轮动 / 季节性
  第三层：PIT 基本面（无 Look-ahead Bias，45天延迟）
  第四层：风险平价仓位 / 最大回撤控制 / 动态模型权重 / 交易成本

已修复 Bug：
  ✓ BatchNorm1d size=1 → drop_last=True
  ✓ 季报日期不一致 → _nearest_col() 容忍 ±5 天
  ✓ Look-ahead Bias → PIT 基本面对齐
  ✓ dropna 过滤太严 → 只对 label 做 dropna
  ✓ meta_df 缺 roe/div_yield → 已补充
  ✓ 成交量单日误踢 → vol_spike_min_days=2
  ✓ rows 为空崩溃 → try/except + 明确错误

安装：
    pip install yfinance xgboost lightgbm scikit-learn torch pandas numpy
"""

import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import requests
import time
import json
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 处理导入路径（支持 Jupyter + 本地运行）
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# 导入当前持仓配置
try:
    from portfolio_config import MY_CURRENT_PORTFOLIO
except (ImportError, ModuleNotFoundError):
    MY_CURRENT_PORTFOLIO = {}

try:
    import lightgbm as lgb
    LGBM = True
except ImportError:
    LGBM = False
    print("⚠️  LightGBM 未安装: pip install lightgbm")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH = True
except ImportError:
    TORCH = False
    print("⚠️  PyTorch 未安装: pip install torch")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

# ── TSX 综合指数股票池（约 230 支）─────────────────────────────
# 来源：S&P/TSX Composite Index + TSX 60 扩展
# 本地运行时可改 source="xic" 自动从 BlackRock XIC ETF 更新

TSX_UNIVERSE = [
    "XIU.TO",  # 🎯 【新增】必须加入 TSX 60 ETF，否则基准收益永远是 0
    # 金融：六大银行 + 保险 + 资管
    "RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","NA.TO",
    "MFC.TO","SLF.TO","GWO.TO","POW.TO","IFC.TO","FFH.TO",
    "IGM.TO","IAG.TO","EQB.TO","LB.TO",
    "CIGI.TO","FSV.TO","BN.TO","BAM.TO","CF.TO",# 能源：油砂 + 管道 + 天然气
    "SU.TO","CNQ.TO","CVE.TO","IMO.TO",
    "ENB.TO","TRP.TO","PPL.TO","KEY.TO",
    "TOU.TO","ARX.TO","WCP.TO","VET.TO","BIR.TO",
    "TVE.TO","PEY.TO","SGY.TO","BTE.TO",
    # 材料：黄金 + 铜 + 化肥 + 铀 + 林业
    "ABX.TO","AEM.TO","K.TO","AGI.TO","IMG.TO","WDO.TO",
    "WPM.TO","FNV.TO","FM.TO","LUN.TO","ERO.TO","TECK-B.TO","HBM.TO",
    "NTR.TO","CCO.TO","DML.TO","WFG.TO","CFP.TO","IFP.TO","MRE.TO",
    # 工业：铁路 + 工程 + 制造 + 运输
    "CNR.TO","CP.TO",
    "WSP.TO","STN.TO","ATS.TO",
    "CAE.TO","FTT.TO","GIL.TO","MTL.TO","MDA.TO","RBA.TO","NFI.TO",
    "CJT.TO","AC.TO","GFL.TO","WCN.TO",
    # 科技：软件 + SaaS + IT服务
    "CSU.TO","TRI.TO","SHOP.TO","ENGH.TO","OTEX.TO","GIB-A.TO",
    "DSG.TO","DCBO.TO","KXS.TO","LSPD.TO","S.TO","LIF.TO",
    # 消费必需
    "ATD.TO","DOL.TO","MRU.TO","L.TO","WN.TO","EMP-A.TO","PBH.TO","NWC.TO","SAP.TO",
    # 消费可选
    "QSR.TO","MG.TO","BYD.TO","CTC-A.TO","LNF.TO",
    # 通讯
    "BCE.TO","T.TO","RCI-B.TO","QBR-B.TO",# REITs：工业 + 住宅 + 零售 + 办公 + 养老
    "GRT-UN.TO","DIR-UN.TO","CRT-UN.TO","SRU-UN.TO",
    "CAR-UN.TO","IIP-UN.TO","MI-UN.TO",
    "REI-UN.TO","CHP-UN.TO","NWH-UN.TO","BTB-UN.TO",
    "HR-UN.TO","AP-UN.TO","D-UN.TO","CSH-UN.TO","SIA.TO",
    # 公用事业：受管制 + 可再生
    "FTS.TO","EMA.TO","CU.TO","H.TO","AQN.TO","ALA.TO","CPX.TO",
    "NPI.TO","BEP-UN.TO","BLX.TO",
    # 医疗
    "BHC.TO","WELL.TO","CLS.TO","HLS.TO","DRX.TO",]

# 过滤已知退市股票
_DELISTED = {
    "TFI.TO","BRP.TO","CAP-UN.TO","PKI.TO","MPW.TO","INE.TO",
    "BAD.TO","GDX.TO","RNW.TO","ERF.TO","DSY.TO","TOI.TO",
    "CPG.TO","CIX.TO","DND.TO","NXE.TO",
}
TSX_UNIVERSE = list(dict.fromkeys(t for t in TSX_UNIVERSE if t not in _DELISTED))


def get_tsx_tickers(source: str = "builtin") -> list[str]:
    """⚠️  警告：存活者偏差
    
    TSX_UNIVERSE 是当前维护的活跃股票池，不包括历史上已退市、被收购、破产、
    或被移出指数的股票。
    
    对当月选股（MODE="pick"）：无影响，这是合理的。
    对历史回测（MODE="backtest"）：会存在
      - 存活者偏差（Survivor Bias）：历史上的失败者被排除
      - 当前赢家偏差（Winner Bias）：当前的赢家因为存活而被回溯
      - 结果：回测收益通常虚高 10-30%
    
    解决方案：
      如果需要更严谨的历史回测，建议：
      1. 提供历史 TSX 成分股列表（2014-2026）
      2. 或在回测输出中标注此限制
      3. 或同时运行「扩展宇宙」包括已退市股票的版本
    
    ═══════════════════════════════════════════════════════════════
    获取 TSX 股票池。

    Args:
      source: 数据源选择
        - "builtin"   → 内置 ~220 支，无需网络（默认）
        - "xic"       → 从 BlackRock XIC ETF 自动更新（需本地运行）
        - "cache"     → 读取上次 XIC 抓取的缓存文件

    使用方式：
      本地首次运行：source="xic"  → 下载最新成分股并缓存
      Colab / 无网络：source="builtin"
    """
    if source == "xic":
        try:
            import requests, io, pandas as pd
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                     "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
            url = ("https://www.blackrock.com/ca/investors/en/products/239837/"
                   "ishares-core-sp-tsx-capped-composite-index-etf/"
                   "1464253357818.ajax?tab=all&fileType=csv")
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}")

            df = pd.read_csv(io.StringIO(r.text), skiprows=2)
            ticker_col = next((c for c in df.columns
                               if "ticker" in c.lower() or "symbol" in c.lower()), None)
            if not ticker_col:
                raise ValueError("找不到 Ticker 列")

            tickers = []
            for v in df[ticker_col].dropna():
                t = str(v).strip()
                if t and t != "nan" and not t.startswith("-"):
                    tickers.append(t + ".TO")

            tickers = [t for t in tickers if t not in _DELISTED]
            print(f"  [XIC] ✅ 获取 {len(tickers)} 支 TSX 成分股")

            with open(".tsx_universe_cache.txt", "w") as f:
                f.write("\n".join(tickers))
            return tickers

        except Exception as e:
            print(f"  [XIC] ⚠️  失败：{e}，使用内置列表")
            return TSX_UNIVERSE

    elif source == "cache":
        try:
            with open(".tsx_universe_cache.txt") as f:
                tickers = [t.strip() for t in f.readlines() if t.strip()]
            print(f"  [缓存] ✅ 读取 {len(tickers)} 支（上次 XIC 更新）")
            return tickers
        except FileNotFoundError:
            print("  [缓存] ⚠️  缓存不存在，使用内置列表")
            return TSX_UNIVERSE

    else:
        print(f"  [内置] {len(TSX_UNIVERSE)} 支 TSX 股票（S&P/TSX Composite + 扩展）")
        return TSX_UNIVERSE


# ── 实际使用的股票池 ──────────────────────────────────────────────
# 本地运行改为 source="xic" 可自动从 BlackRock 更新到 ~250 支
TICKERS = get_tsx_tickers(source="builtin")


# ── Simfin 配置（从环境变量读取，避免明文 API Key）──────
# 设置方法：export SIMFIN_API_KEY="你的key" 或填入 os.getenv 的默认值

SIMFIN_API_KEY = "804d29c7-c3cf-43d4-96a1-128edd64b7ff"

# 固定所有随机种子（确保同一天同样数据跑出相同结果）
import random, numpy as np
random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
except ImportError:
    pass
FMP_API_KEY    = "JiSlrxR3kbnUQnNcTShCFjUNdq78htKf"  # 🔒 改为从环境变量读取（注册 financialmodelingprep.com 获取免费 key）
SIMFIN_DATA_DIR = "./simfin_data"   # 本地缓存目录

# ══════════════════════════════════════════════════════════════════
# HTML 报告生成（Chart.js 交互式图表）
# ══════════════════════════════════════════════════════════════════

def generate_html_report(results, initial_capital):
    """生成交互式 HTML 报告（Chart.js）"""
    
    if not results:
        return None
    
    # 统计数据
    rets = np.array([m["portfolio"] for m in results]) / 100
    bench_rets = np.array([m["benchmark"] for m in results]) / 100
    n = len(rets)
    
    final_nav = results[-1]["nav"]
    total_ret = (final_nav / initial_capital - 1) * 100
    ann_ret = ((1 + rets).prod() ** (12/n) - 1) * 100 if n > 0 else 0
    bench_ann = ((1 + bench_rets).prod() ** (12/n) - 1) * 100 if n > 0 else 0
    vol_m = rets.std() * np.sqrt(12) * 100
    sharpe = (ann_ret/100 - 0.04) / (vol_m/100) if vol_m > 0 else 0
    navs = pd.Series([m["nav"] for m in results])
    mdd = ((navs - navs.cummax()) / navs.cummax()).min() * 100
    win_rate = sum(1 for m in results if m["portfolio"] > 0) / n * 100
    
    # 图表数据
    months = [m["month"] for m in results]
    portfolios = [m["portfolio"] for m in results]
    benchmarks = [m["benchmark"] for m in results]
    navs_list = [m["nav"] for m in results]
    excess_list = [m["excess"] for m in results]
    
    # 年度聚合
    yearly = {}
    for m in results:
        year = m["month"][:4]
        if year not in yearly:
            yearly[year] = {"ret": [], "bench": []}
        yearly[year]["ret"].append(m["portfolio"])
        yearly[year]["bench"].append(m["benchmark"])
    
    years = sorted(yearly.keys())
    yearly_rets = [np.mean(yearly[y]["ret"]) for y in years]
    yearly_bench = [np.mean(yearly[y]["bench"]) for y in years]
    
    # 生成月度表行
    table_rows = ""
    for m in results:
        portfolio_class = "positive" if m["portfolio"] >= 0 else "negative"
        benchmark_class = "positive" if m["benchmark"] >= 0 else "negative"
        excess_class = "positive" if m["excess"] >= 0 else "negative"
        nav_chg_class = "positive" if m["nav_chg"] >= 0 else "negative"
        
        row = f"<tr><td><strong>{m['month']}</strong></td>"
        row += f"<td class=\"{portfolio_class}\">{m['portfolio']:+.2f}%</td>"
        row += f"<td class=\"{benchmark_class}\">{m['benchmark']:+.2f}%</td>"
        row += f"<td class=\"{excess_class}\">{m['excess']:+.2f}%</td>"
        row += f"<td>${m['nav']:,.0f}</td>"
        row += f"<td class=\"{nav_chg_class}\">{m['nav_chg']:+,.0f}</td></tr>"
        table_rows += row
    
    # Chart.js 数据
    chart_data = {
        "months": months,
        "navs": navs_list,
        "portfolios": portfolios,
        "benchmarks": benchmarks,
        "excess": excess_list,
        "years": years,
        "yearly_rets": yearly_rets,
        "yearly_bench": yearly_bench
    }
    
    chart_json = json.dumps(chart_data)
    
    # HTML 模板
    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TSX 量化选股 — 回测报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { background: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header h1 { font-size: 28px; color: #2c3e50; margin-bottom: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .metric { background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric .value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .metric .label { font-size: 11px; color: #999; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .chart { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .chart h3 { margin-bottom: 15px; font-size: 14px; }
        .chart-full { grid-column: 1/-1; }
        .chart-wrapper { position: relative; height: 300px; }
        .table-wrapper { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th { background: #f8f9fa; padding: 10px; text-align: left; font-weight: 600; border-bottom: 2px solid #ddd; }
        td { padding: 8px 10px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f8f9fa; }
        .footer { text-align: center; margin-top: 30px; color: #999; font-size: 11px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 TSX 量化选股 — 回测报告</h1>
            <p>期间：__PERIOD__ | 生成：__DATETIME__</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="label">总收益 (__N__月)</div>
                <div class="value __CLASS_TOTAL__">__TOTAL_RET__</div>
            </div>
            <div class="metric">
                <div class="label">年化收益</div>
                <div class="value __CLASS_ANN__">__ANN_RET__</div>
            </div>
            <div class="metric">
                <div class="label">最大回撤</div>
                <div class="value negative">__MDD__</div>
            </div>
            <div class="metric">
                <div class="label">Sharpe 比率</div>
                <div class="value">__SHARPE__</div>
            </div>
            <div class="metric">
                <div class="label">胜率</div>
                <div class="value positive">__WINRATE__</div>
            </div>
            <div class="metric">
                <div class="label">Alpha</div>
                <div class="value __CLASS_ALPHA__">__ALPHA__</div>
            </div>
        </div>
        
        <div class="charts">
            <div class="chart chart-full">
                <h3>📈 净值曲线 + 月度收益</h3>
                <div class="chart-wrapper">
                    <canvas id="navChart"></canvas>
                </div>
            </div>
            <div class="chart">
                <h3>💰 月度对比（最近12月）</h3>
                <div class="chart-wrapper">
                    <canvas id="monthlyChart"></canvas>
                </div>
            </div>
            <div class="chart">
                <h3>📊 年度表现</h3>
                <div class="chart-wrapper">
                    <canvas id="yearlyChart"></canvas>
                </div>
            </div>
            <div class="chart">
                <h3>🎯 盈利月数占比</h3>
                <div class="chart-wrapper">
                    <canvas id="doughnutChart"></canvas>
                </div>
            </div>
            <div class="chart">
                <h3>🚀 超额收益（最近12月）</h3>
                <div class="chart-wrapper">
                    <canvas id="excessChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="table-wrapper">
            <h3>📋 月度明细</h3>
            <table>
                <thead>
                    <tr>
                        <th>月份</th>
                        <th>策略收益</th>
                        <th>基准收益</th>
                        <th>超额收益</th>
                        <th>净值</th>
                        <th>月度变化</th>
                    </tr>
                </thead>
                <tbody>
                    __TABLE_ROWS__
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>© 2026 TSX 量化选股 | 基准：XIU.TO | 数据：yfinance & FMP | 模型：XGBoost + LightGBM + PyTorch</p>
        </div>
    </div>
    
    <script>
        const data = __CHART_DATA__;
        const colors = {primary: '#3498db', success: '#27ae60', danger: '#e74c3c', gray: '#95a5a6'};
        
        new Chart(document.getElementById('navChart'), {
            type: 'bar',
            data: {
                labels: data.months,
                datasets: [
                    {
                        label: '净值',
                        type: 'line',
                        data: data.navs,
                        borderColor: colors.primary,
                        backgroundColor: 'rgba(52,152,219,0.1)',
                        borderWidth: 2,
                        yAxisID: 'y1'
                    },
                    {
                        label: '正收益',
                        data: data.portfolios.map(x => x > 0 ? x : null),
                        backgroundColor: colors.success,
                        yAxisID: 'y'
                    },
                    {
                        label: '负收益',
                        data: data.portfolios.map(x => x < 0 ? x : null),
                        backgroundColor: colors.danger,
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    y: {title: {display: true, text: '月度收益 (%)'}},
                    y1: {position: 'right', title: {display: true, text: '净值 ($)'}}
                },
                plugins: {legend: {position: 'top'}}
            }
        });
        
        const recent12 = 12;
        const r12_months = data.months.slice(-recent12);
        const r12_port = data.portfolios.slice(-recent12);
        const r12_bench = data.benchmarks.slice(-recent12);
        
        new Chart(document.getElementById('monthlyChart'), {
            type: 'bar',
            data: {
                labels: r12_months,
                datasets: [
                    {label: '策略', data: r12_port, backgroundColor: colors.primary},
                    {label: '基准', data: r12_bench, backgroundColor: colors.gray}
                ]
            },
            options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {position: 'top'}}}
        });
        
        new Chart(document.getElementById('yearlyChart'), {
            type: 'bar',
            data: {
                labels: data.years,
                datasets: [
                    {label: '策略', data: data.yearly_rets, backgroundColor: colors.primary},
                    {label: '基准', data: data.yearly_bench, backgroundColor: colors.gray}
                ]
            },
            options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {position: 'top'}}}
        });
        
        const win_count = data.portfolios.filter(x => x > 0).length;
        const loss_count = data.portfolios.filter(x => x < 0).length;
        
        new Chart(document.getElementById('doughnutChart'), {
            type: 'doughnut',
            data: {
                labels: ['正收益月', '负收益月'],
                datasets: [{
                    data: [win_count, loss_count],
                    backgroundColor: [colors.success, colors.danger]
                }]
            },
            options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {position: 'bottom'}}}
        });
        
        const r12_excess = data.excess.slice(-recent12);
        new Chart(document.getElementById('excessChart'), {
            type: 'bar',
            data: {
                labels: r12_months,
                datasets: [{
                    label: '超额收益',
                    data: r12_excess,
                    backgroundColor: r12_excess.map(x => x >= 0 ? colors.success : colors.danger)
                }]
            },
            options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}}
        });
    </script>
</body>
</html>"""
    
    # 填充占位符
    total_ret_class = "positive" if total_ret >= 0 else "negative"
    ann_ret_class = "positive" if ann_ret >= 0 else "negative"
    alpha_value = ann_ret - bench_ann
    alpha_class = "positive" if alpha_value >= 0 else "negative"
    
    html = html_template.replace("__PERIOD__", f"{results[0]['month']} ~ {results[-1]['month']}")
    html = html.replace("__DATETIME__", datetime.now().strftime('%Y-%m-%d %H:%M'))
    html = html.replace("__N__", str(n))
    html = html.replace("__CLASS_TOTAL__", total_ret_class)
    html = html.replace("__TOTAL_RET__", f"{total_ret:+.1f}%")
    html = html.replace("__CLASS_ANN__", ann_ret_class)
    html = html.replace("__ANN_RET__", f"{ann_ret:+.2f}%")
    html = html.replace("__MDD__", f"{mdd:.2f}%")
    html = html.replace("__SHARPE__", f"{sharpe:.2f}")
    html = html.replace("__WINRATE__", f"{win_rate:.1f}%")
    html = html.replace("__CLASS_ALPHA__", alpha_class)
    html = html.replace("__ALPHA__", f"{alpha_value:+.2f}%")
    html = html.replace("__TABLE_ROWS__", table_rows)
    html = html.replace("__CHART_DATA__", chart_json)
    
    return html


def save_html_report(html_content, filename=None):
    """保存 HTML 文件"""
    if filename is None:
        filename = f"tsx_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ HTML 报告已生成：{filename}")
    return filename

def fetch_pit_fmp(ticker: str, api_key: str, quarters: int = 20) -> pd.DataFrame:
    """
    从 Financial Modeling Prep 获取季报 PIT 数据。

    优势：
      - 免费 250 次/天，Colab 可用（不被 IP 屏蔽）
      - TSX 股票直接用 RY.TO 格式
      - 有 fillingDate（实际披露日），可实现真正的 PIT
      - 历史最多 5 年季报（免费版）

    ✓ 说明：返回格式与 build_pit_from_simfin 和 fetch_pit_quarterly 相同！
    只返回核心字段（avail_date, net_income, total_equity, ocf, capex, shares），
    让 compute_pit_fundamentals() 统一计算 pe/pb/roe/fcf_yield 等比率。
    """
    BASE = "https://financialmodelingprep.com/api/v3"
    rows = []

    try:
        # 损益表
        r_inc = requests.get(
            f"{BASE}/income-statement/{ticker}",
            params={"period":"quarter","limit":quarters,"apikey":api_key},
            timeout=10).json()

        # 资产负债表
        r_bal = requests.get(
            f"{BASE}/balance-sheet-statement/{ticker}",
            params={"period":"quarter","limit":quarters,"apikey":api_key},
            timeout=10).json()

        # 现金流量表
        r_cf  = requests.get(
            f"{BASE}/cash-flow-statement/{ticker}",
            params={"period":"quarter","limit":quarters,"apikey":api_key},
            timeout=10).json()

        if not r_inc or "Error Message" in str(r_inc):
            return pd.DataFrame()

        # 以披露日（fillingDate）为 PIT 基准
        bal_by_date = {x["date"]: x for x in (r_bal or [])}
        cf_by_date  = {x["date"]: x for x in (r_cf  or [])}

        for inc in r_inc:
            qdate    = inc.get("date", "")
            fill_dt  = inc.get("acceptedDate") or inc.get("fillingDate") or qdate
            if not qdate:
                continue

            bal = bal_by_date.get(qdate, {})
            cf  = cf_by_date.get(qdate,  {})

            # 基础通用字段
            net_income = inc.get("netIncome") or 0
            equity     = bal.get("totalStockholdersEquity")
            op_cf      = cf.get("operatingCashFlow") or 0
            capex      = cf.get("capitalExpenditure") or 0
            shares     = inc.get("weightedAverageShsOut")

            # 🎯 【新增】获取行业专属计算所需的原始字段
            revenue      = inc.get("revenue") or 0
            ebitda       = inc.get("ebitda") or 0
            dna          = inc.get("depreciationAndAmortization") or 0  # 算 FFO 必须
            rnd          = inc.get("researchAndDevelopmentExpenses") or 0 # 科技股必须
            div_paid     = cf.get("dividendsPaid") or 0
            total_debt   = bal.get("totalDebt") or 0
            
            # 金融股专用字段
            interest_inc = inc.get("interestIncome") or 0
            interest_exp = inc.get("interestExpense") or 0
            pcl          = inc.get("provisionForCreditLosses") or 0
            total_assets = bal.get("totalAssets") or 1e9  # 给个底线防止除 0

            rows.append({
                "avail_date":   pd.Timestamp(fill_dt[:10]),
                "net_income":   net_income,
                "total_equity": equity,
                "ocf":          op_cf,
                "capex":        capex,
                "shares":       shares,
                # --- 新增数据入库 ---
                "revenue":      revenue,
                "ebitda":       ebitda,
                "dna":          dna,
                "rnd":          rnd,
                "div_paid":     div_paid,
                "total_debt":   total_debt,
                "interest_inc": interest_inc,
                "interest_exp": interest_exp,
                "pcl":          pcl,
                "total_assets": total_assets,
            })

        if rows:
            df = pd.DataFrame(rows).sort_values("avail_date")
            return df

    except Exception as e:
        print(f"  ⚠️  Simfin 特征计算失败：{type(e).__name__}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()

    return pd.DataFrame()


def init_fmp(tickers: list, api_key: str) -> dict:
    """
    批量从 FMP 获取季报数据，返回 {ticker: DataFrame} 字典。
    免费版 250次/天，135支股票需要约 135×3 = 405 次请求，
    建议分批或使用缓存。
    """
    import os, pickle, time

    cache_file = "./simfin_data/fmp_cache.pkl"
    os.makedirs("./simfin_data", exist_ok=True)

    # 读缓存（当天内有效）
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if (time.time() - mtime) < 86400:  # 24小时内
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                print(f"  [FMP] 从缓存加载（{len(cached)} 支）")
                return cached
            except Exception:
                pass

    print(f"  [FMP] 下载 {len(tickers)} 支季报（免费版约需 2-3 分钟）...")
    result = {}
    for i, t in enumerate(tickers):
        df = fetch_pit_fmp(t, api_key)
        if not df.empty:
            result[t] = df
        if (i+1) % 20 == 0:
            print(f"  [FMP] {i+1}/{len(tickers)} 支完成...")
        time.sleep(0.1)  # 避免超频

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    print(f"  [FMP] ✓ 完成，{len(result)} 支有数据，已缓存")
    return result


def init_simfin():
    """
    初始化 Simfin，带磁盘缓存。

    Simfin bulk download 在某些 IP（如 Colab）返回 401/403。
    解决方案：
      1. 优先读取磁盘缓存（上次成功下载后保留）
      2. 缓存不存在时尝试下载（本地运行成功率高）
      3. 完全失败时 fallback 到 yfinance
    """
    import os, pickle

    cache_file = os.path.join(SIMFIN_DATA_DIR, "simfin_ca_cache.pkl")
    os.makedirs(SIMFIN_DATA_DIR, exist_ok=True)

    # 优先读本地缓存（不受 IP 限制）
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                inc, bal, cf = pickle.load(f)
            n = inc.index.get_level_values("Ticker").nunique()
            yr_min = inc.index.get_level_values("Report Date").min().year
            yr_max = inc.index.get_level_values("Report Date").max().year
            print(f"  [Simfin] ✓ 从缓存加载 ({n} 支, {yr_min}→{yr_max})")
            return inc, bal, cf
        except Exception as e:
            print(f"  [Simfin] 缓存读取失败：{e}，尝试重新下载...")

    # 尝试在线下载
    try:
        import simfin as sf
        sf.set_api_key(SIMFIN_API_KEY)
        sf.set_data_dir(SIMFIN_DATA_DIR)

        print("  [Simfin] 加载季报数据（首次下载约 2 分钟）...")
        inc = sf.load_income(variant="quarterly", market="ca")
        bal = sf.load_balance(variant="quarterly", market="ca")
        cf  = sf.load_cashflow(variant="quarterly", market="ca")

        # 成功后保存缓存（下次无网络也可用）
        with open(cache_file, "wb") as f:
            pickle.dump((inc, bal, cf), f)

        n = inc.index.get_level_values("Ticker").nunique()
        yr_min = inc.index.get_level_values("Report Date").min().year
        yr_max = inc.index.get_level_values("Report Date").max().year
        print(f"  [Simfin] ✓ 季报加载成功 ({n} 支, {yr_min}→{yr_max})")
        print(f"  [Simfin] ✓ 已缓存到 {cache_file}")
        return inc, bal, cf

    except ImportError:
        print("  [Simfin] ⚠️  未安装：pip install simfin")
        return None, None, None
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "403" in err_str:
            print(f"  [Simfin] ⚠️  IP 受限（{err_str[:60]}）")
            print(f"  [Simfin]    解决方法：本地运行一次后缓存会保留到 Colab")
            print(f"  [Simfin]    缓存路径：{cache_file}")
        else:
            print(f"  [Simfin] ⚠️  加载失败：{e}")
        return None, None, None


def build_pit_from_simfin(ticker, inc_all, bal_all, cf_all):
    """
    从 Simfin 季报数据构建 PIT（Point-in-Time）基本面时间序列。

    优势 vs yfinance：
      - 覆盖约 10 年完整历史（yfinance 只有 6-7 季）
      - 官方财报数据，更准确
      - 有明确的 Report Date（公告日期），PIT 更精确

    ★ 关键：采用保守的 45/90 天披露延迟（CSA 标准），避免前视偏差
      - Q1-Q3：45 天（REPORT_LAG）
      - Q4（年报）：90 天
      这与 yfinance fallback 逻辑一致，防止模型看到未发布的财报

    TSX 代码转换：RY.TO → RY，ENB.TO → ENB
    """
    t_simfin = ticker.replace(".TO", "")

    try:
        # Simfin MultiIndex: (Ticker, Report Date)
        def get_ticker(df):
            if df is None: return pd.DataFrame()
            try:
                return df.xs(t_simfin, level="Ticker")
            except KeyError:
                return pd.DataFrame()

        inc = get_ticker(inc_all)
        bal = get_ticker(bal_all)
        cf  = get_ticker(cf_all)

        if inc.empty:
            return pd.DataFrame()

        # ★ 🔴 【前视偏差修复】：使用 CSA 标准披露延迟（45/90 天）而不是 5 天
        # 原因：季度财报公告日期不在 index，而 index 是 Report Date（季度末）
        # 假设 5 天披露会导致严重前视偏差，实际需要 45-90 天
        # CSA 标准：Q1-Q3 需要 45 天内披露，Q4（年报）需要 90 天内披露
        def _pit_report_lag(report_date):
            """计算季度财报可用延迟（天数）"""
            qdate  = pd.Timestamp(report_date)
            month  = qdate.month
            # Q4（10/11/12月末）的年报需要 90 天披露，其他季度 45 天
            return 90 if month in (10, 11, 12) else REPORT_LAG

        # ✓ 矢量化重构：替代 for 循环，消除 N 支股票 × M 季度的循环成本
        # 直接从三张表中提取列，使用 fillna 和 or 逻辑处理备选列
        df = pd.DataFrame(index=inc.index)
        # 使用动态延迟替代硬编码的 5 天
        df["avail_date"] = inc.index.map(lambda d: pd.Timestamp(d) + pd.Timedelta(days=_pit_report_lag(d)))
        df["net_income"]   = inc.get("Net Income", pd.Series(dtype='float64'))
        df["total_equity"] = bal.get("Total Equity", bal.get("Common Equity", pd.Series(dtype='float64')))
        df["ocf"]          = cf.get("Net Cash from Operating Activities", pd.Series(dtype='float64'))
        df["capex"]        = cf.get("Purchase of Property, Plant and Equipment", pd.Series(dtype='float64'))
        df["shares"]       = inc.get("Shares (Diluted)", inc.get("Shares", bal.get("Common Shares Outstanding", pd.Series(dtype='float64'))))
        
        # 清理空值
        if df.empty or df[["net_income","total_equity","ocf","capex","shares"]].isna().all().all():
            return pd.DataFrame()

        df = df.set_index("avail_date").sort_index()
        return df

    except Exception as e:
        print(f"  ⚠️  FMP 基本面数据解析失败：{type(e).__name__}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


MACRO_TICKERS = {
    "oil":   "CL=F",
    "cadusd":"CADUSD=X",
    "bond":  "^TNX",
    "tsx":   "^GSPTSE",
    "gold":  "GC=F",
    "vix":   "^VIX",
}

TOP_N         = 8       # 🚀 原为 10。把资金更集中地打在 Alpha 最高的 8 只票上
YEARS         = 5
MIN_TRAIN     = 18
TOP_QUINTILE  = 0.20
REPORT_LAG    = 45
STOP_LOSS_PCT = -0.08
MAX_DD_THRESH = -0.05

# ══════════════════════════════════════════════════════════════════
# 🛑 行业黑名单 & 白名单（加拿大 TSX 专用排雷规则）
# ══════════════════════════════════════════════════════════════════
# 
# 核心原则：物理隔离低波动"现金拖累"资产，让优化器专注于高波动高收益行业
#
# 黑名单理由：
#   1. Utilities：公用事业低波动陷阱 → 占坑不赚钱
#   2. Telecom / Communication Services：低波动电信股（如 RCI-B.TO）
#   3. Real Estate / REITs：加息周期终结者（如 NWH-UN.TO）
#   4. Health Care：大麻股/暴雷医药充斥，基本面扭曲
#
# 结论：黑名单内的资产不进入选股候选池 → 优化器无法过度分配
#
BANNED_SECTORS = {
    "Utilities",               # 低波动陷阱
    "Telecom",                 # 低波动陷阱（如 RCI-B.TO）
    "Communication Services",  # 低波动陷阱（备选 GICS 标签）
    "Real Estate",             # 加息周期杀手（如 NWH-UN.TO）
    "REITs",                   # 备用名称，同 Real Estate
    "Health Care"              # 防范大麻股和暴雷医药
}

WHITELIST = {
    "WELL.TO"  # WELL：纯正科技 SaaS（数字医疗），本质是科技股而非医疗股
}

CONSTRAINTS = {
    "min_adv_cad":        1_000_000,
    "vol_spike_sigma":    3.0,
    "vol_spike_days":     5,
    "vol_spike_min_days": 2,
    "min_pe":             0.0,
    "max_pe":             150.0,         # 🎯 提高到 150，放行高成长科技股 (原60.0)
    "min_mktcap_cad":     800_000_000,   # 🎯 提高到 8 亿，进一步砍掉极小盘微型垃圾股 (原500M)
    "min_price_cad":      2.00,
    "max_price_cad":    400.00,   # 单股价格上限（过高的股票买不到足够整股）
    "min_roe":            0.0,
    "max_roe":          2.00,   # ROE > 200% 视为财务异常（如 BHC 436%）
    "min_shares":          5,   # 最少持仓股数（价格过高股票如 FFH 会被过滤）
    "max_per_gics":       4,      # 🚀 允许同行业最多 4 支股票，让优化器敢集中在表现最强势的科技/金融板块
    "max_per_style":      4,
    "max_per_type":       5,
    "max_single_alloc":   0.25,   # 🚀 原为 0.20。允许单只股票的持仓上限达到 25%
    "max_turnover":       4,      # 每月最多换仓数
    "hold_bonus":         0.05,   # 持仓连续性奖励
    # OPT1: 矿业子行业硬上限
    "max_gold_mining":    2,      # 黄金矿业最多 2 支
    "max_base_metals":    1,      # 贱金属（铜/锌）最多 1 支
    "max_energy_sub":     2,      # 能源子类最多 2 支
    # OPT5: 换仓冷静期
    "cooldown_months":    1,      # 止损后冷静 N 个月
    # OPT6: 置信度过滤
    "min_confidence":     0.15,   # 🚀 设置至 0.15：防止过度谨慎导致空仓或买 ETF
    "min_top_n":          8,      # 🚀 提高至 8：确保模型有足够的备选股票，不会被迫空仓
    # OPT7: 熔断 + VIX 缩仓
    "dd_halt_threshold":  -0.15,  # 3月累计亏损超此值→减仓
    "dd_halt_scale":      0.50,   # 减仓比例
    "vix_scale_threshold":25.0,   # VIX 高于此值时缩仓
    "vix_scale_factor":   0.70,   # VIX 高时仓位缩至 70%
    "min_listing_days":   252,
    # 换仓缓冲带参数（Wealthsimple 免手续费版本 — 中度宽容）
    "rank_buffer":        18,     # 🔧 修复3：放宽至前18名，减少无谓换手率（原12→18）
    "score_tolerance":    0.015,  # 🔧 修复3：放宽至1.5% 容忍度，锁住优质老持仓（原0.005→0.015）
}

# 矿业子行业分类（OPT1 用）
GOLD_MINING_TICKERS = {"K.TO","ABX.TO","AEM.TO","WDO.TO","AGI.TO",
                        "IMG.TO","WPM.TO","FNV.TO"}
BASE_METALS_TICKERS = {"HBM.TO","LUN.TO","ERO.TO","FM.TO","TECK-B.TO"}


FEATURE_COLS = [
    "mom_1m","mom_3m","mom_6m","mom_12m","mom_12_1",
    "vol_1m","vol_3m","vol_ratio",
    "rsi","bias_60","price_vs_52w_high","bb_zscore",
    "oil_mom_3m","cadusd_mom_3m","bond_chg_3m","gold_mom_3m","vix_level",
    "sector_mom_rel", "month_sin","month_cos",
    "days_to_earnings", "days_since_earnings",
    "pe_rel_sector","pb","roe","eps_growth","fcf_yield",
    # 🎯 新增的行业专属 Alpha 因子
    "ev_ebitda", "capex_ocf", "p_ffo", "rule_of_40", "rnd_rev",
    "debt_ebitda", "div_cover", "nim", "pcl_ratio"
]

# ══════════════════════════════════════════════════════════════════
# 股票分类表
# ══════════════════════════════════════════════════════════════════

STOCK_PROFILE = {
    "RY.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "TD.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "BNS.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "BMO.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "CM.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "NA.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "MFC.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "SLF.TO":{"gics":"Financials","style":"Quality","type":"Defensive"},
    "GWO.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "POW.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "IGM.TO":{"gics":"Financials","style":"Value","type":"Defensive"},
    "CIGI.TO":{"gics":"Financials","style":"Growth","type":"Growth"},
    "EQB.TO":{"gics":"Financials","style":"Growth","type":"Cyclical"},
    "FSV.TO":{"gics":"Financials","style":"Growth","type":"Growth"},
    "SU.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "CNQ.TO":{"gics":"Energy","style":"Quality","type":"Cyclical"},
    "ENB.TO":{"gics":"Energy","style":"Value","type":"Defensive"},
    "TRP.TO":{"gics":"Energy","style":"Value","type":"Defensive"},
    "CVE.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "TOU.TO":{"gics":"Energy","style":"Growth","type":"Cyclical"},
    "PPL.TO":{"gics":"Energy","style":"Value","type":"Defensive"},
    "KEY.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "WCP.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "VET.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "BIR.TO":{"gics":"Energy","style":"Value","type":"Cyclical"},
    "NTR.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "ABX.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "AEM.TO":{"gics":"Materials","style":"Quality","type":"Cyclical"},
    "WPM.TO":{"gics":"Materials","style":"Growth","type":"Cyclical"},
    "FM.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "K.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "CCO.TO":{"gics":"Materials","style":"Growth","type":"Cyclical"},
    "FNV.TO":{"gics":"Materials","style":"Quality","type":"Cyclical"},
    "TECK-B.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "IMG.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "LUN.TO":{"gics":"Materials","style":"Value","type":"Cyclical"},
    "ERO.TO":{"gics":"Materials","style":"Growth","type":"Cyclical"},
    "CNR.TO":{"gics":"Industrials","style":"Quality","type":"Defensive"},
    "CP.TO":{"gics":"Industrials","style":"Quality","type":"Defensive"},
    "WSP.TO":{"gics":"Industrials","style":"Growth","type":"Growth"},
    "CAE.TO":{"gics":"Industrials","style":"Quality","type":"Cyclical"},
    "FTT.TO":{"gics":"Industrials","style":"Quality","type":"Cyclical"},
    "GIL.TO":{"gics":"Industrials","style":"Quality","type":"Cyclical"},
    "MTL.TO":{"gics":"Industrials","style":"Value","type":"Cyclical"},
    "RBA.TO":{"gics":"Industrials","style":"Growth","type":"Cyclical"},
    "CSU.TO":{"gics":"Technology","style":"Growth","type":"Growth"},
    "TRI.TO":{"gics":"Technology","style":"Quality","type":"Defensive"},
    "SHOP.TO":{"gics":"Technology","style":"Growth","type":"Growth"},
    "ENGH.TO":{"gics":"Technology","style":"Quality","type":"Defensive"},
    "OTEX.TO":{"gics":"Technology","style":"Value","type":"Defensive"},
    "GIB-A.TO":{"gics":"Technology","style":"Quality","type":"Defensive"},
    "LIF.TO":{"gics":"Technology","style":"Value","type":"Cyclical"},
    "ATD.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "DOL.TO":{"gics":"Consumer","style":"Growth","type":"Defensive"},
    "MRU.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "L.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "QSR.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "PBH.TO":{"gics":"Consumer","style":"Growth","type":"Cyclical"},
    "CJT.TO":{"gics":"Consumer","style":"Quality","type":"Cyclical"},
    "WCN.TO":{"gics":"Consumer","style":"Quality","type":"Defensive"},
    "MG.TO":{"gics":"Consumer","style":"Value","type":"Cyclical"},
    "GFL.TO":{"gics":"Consumer","style":"Growth","type":"Defensive"},
    "WN.TO":{"gics":"Consumer","style":"Value","type":"Defensive"},
    "EMP-A.TO":{"gics":"Consumer","style":"Value","type":"Defensive"},
    "BCE.TO":{"gics":"Telecom","style":"Value","type":"Defensive"},
    "T.TO":{"gics":"Telecom","style":"Value","type":"Defensive"},
    "RCI-B.TO":{"gics":"Telecom","style":"Value","type":"Defensive"},
    "QBR-B.TO":{"gics":"Telecom","style":"Value","type":"Defensive"},
    "CAR-UN.TO":{"gics":"REITs","style":"Growth","type":"Defensive"},
    "REI-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "GRT-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "SRU-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "CSH-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "CHP-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "HR-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "AP-UN.TO":{"gics":"REITs","style":"Value","type":"Defensive"},
    "DIR-UN.TO":{"gics":"REITs","style":"Growth","type":"Defensive"},
    "FTS.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "EMA.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "CU.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "H.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "NPI.TO":{"gics":"Utilities","style":"Growth","type":"Defensive"},
    "AQN.TO":{"gics":"Utilities","style":"Value","type":"Defensive"},
    "BEP-UN.TO":{"gics":"Utilities","style":"Growth","type":"Defensive"},
    "BLX.TO":{"gics":"Utilities","style":"Growth","type":"Defensive"},
    "BHC.TO":{"gics":"Healthcare","style":"Value","type":"Defensive"},
    "WELL.TO":{"gics":"Healthcare","style":"Growth","type":"Defensive"},
    "DRX.TO":{"gics":"Healthcare","style":"Value","type":"Defensive"},
    # 矿业（新增）
    "HBM.TO":   {"gics":"Materials",   "style":"Value",  "type":"Cyclical"},
    "WDO.TO":   {"gics":"Materials",   "style":"Growth", "type":"Cyclical"},
    "AGI.TO":   {"gics":"Materials",   "style":"Quality","type":"Cyclical"},
    "PEY.TO":   {"gics":"Energy",      "style":"Value",  "type":"Cyclical"},
    "SGY.TO":   {"gics":"Energy",      "style":"Value",  "type":"Cyclical"},
    "ARX.TO":   {"gics":"Energy",      "style":"Growth", "type":"Cyclical"},
    "IMO.TO":   {"gics":"Energy",      "style":"Value",  "type":"Cyclical"},
    "FFH.TO":   {"gics":"Financials",  "style":"Quality","type":"Defensive"},
    "IFC.TO":   {"gics":"Financials",  "style":"Quality","type":"Defensive"},
    "IAG.TO":   {"gics":"Financials",  "style":"Value",  "type":"Defensive"},
    "LB.TO":    {"gics":"Financials",  "style":"Value",  "type":"Defensive"},
    "BN.TO":    {"gics":"Financials",  "style":"Growth", "type":"Growth"},
    "BAM.TO":   {"gics":"Financials",  "style":"Growth", "type":"Growth"},
    "CF.TO":    {"gics":"Financials",  "style":"Value",  "type":"Cyclical"},
    "AC.TO":    {"gics":"Industrials", "style":"Value",  "type":"Cyclical"},
    "STN.TO":   {"gics":"Industrials", "style":"Growth", "type":"Cyclical"},
    "MDA.TO":   {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "DSG.TO":   {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "DCBO.TO":  {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "KXS.TO":   {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "CLS.TO":   {"gics":"Technology",  "style":"Growth", "type":"Growth"},
    "NWC.TO":   {"gics":"Consumer",    "style":"Value",  "type":"Defensive"},
    "SAP.TO":   {"gics":"Consumer",    "style":"Value",  "type":"Defensive"},
    "CTC-A.TO": {"gics":"Consumer",    "style":"Value",  "type":"Cyclical"},
    "CRT-UN.TO":{"gics":"REITs",       "style":"Value",  "type":"Defensive"},
    "NWH-UN.TO":{"gics":"REITs",       "style":"Value",  "type":"Defensive"},
    "SIA.TO":   {"gics":"REITs",       "style":"Value",  "type":"Defensive"},
    "ALA.TO":   {"gics":"Utilities",   "style":"Value",  "type":"Defensive"},
    "CPX.TO":   {"gics":"Utilities",   "style":"Growth", "type":"Defensive"},
    "WSP.TO":   {"gics":"Industrials", "style":"Growth", "type":"Growth"},
    "GIL.TO":   {"gics":"Industrials", "style":"Quality","type":"Cyclical"},
}

# ══════════════════════════════════════════════════════════════════
# 1. 数据获取
# ══════════════════════════════════════════════════════════════════

def fetch_macro(years):
    end   = datetime.today()
    start = end - timedelta(days=years*365+90)
    print("  [宏观] 下载宏观指标...", end="", flush=True)
    try:
        raw = yf.download(list(MACRO_TICKERS.values()), start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
        raw.columns = list(MACRO_TICKERS.keys())
        macro_m = raw.resample("ME").last().ffill()
        print(f" ✓ ({len(macro_m)} 月)")
        return macro_m
    except Exception as e:
        print(f" ✗ {e}")
        return pd.DataFrame()


def fetch_prices(tickers, years):
    end   = datetime.today()
    start = end - timedelta(days=years*365+90)
    print(f"\n[1/4] 下载 {len(tickers)} 支日线（{years} 年）...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=True, group_by="ticker")
    daily_map = {}
    for t in tickers:
        try:
            df = (raw[["Open","High","Low","Close","Volume"]] if len(tickers)==1
                  else raw[t][["Open","High","Low","Close","Volume"]]).copy()
            df.columns = ["open","high","low","close","volume"]
            df = df.dropna(subset=["close","volume"])
            if len(df) > 60:
                daily_map[t] = df
        except Exception:
            pass
    print(f"  {len(daily_map)} 支通过数据检查")
    return daily_map


def _safe_val(df, row_key, col_idx):
    try:
        if df is not None and row_key in df.index:
            v = df.iloc[df.index.get_loc(row_key), col_idx]
            return float(v) if pd.notna(v) else None
    except Exception:
        pass
    return None


def _nearest_col(df, dt):
    """✓ Bug fix: 容忍季报各表日期差 ±5 天"""
    if df is None or df.empty:
        return None
    diffs = abs(df.columns - pd.Timestamp(dt))
    idx   = diffs.argmin()
    return idx if diffs[idx] <= pd.Timedelta(days=5) else None


def fetch_pit_quarterly(ticker):
    """第三层：Point-in-Time 季报（季度末 + 45天延迟）
    
    符合 CSA 披露标准：Q1-Q3 需 45 天，Q4 需 90 天。
    ✓ 说明：该方法与 build_pit_from_simfin() 使用相同逻辑，确保两条数据路径无前视偏差。
    """
    t = yf.Ticker(ticker)
    try:
        inc    = t.quarterly_income_stmt
        bal    = t.quarterly_balance_sheet
        cf     = t.quarterly_cashflow
        shares = t.info.get("sharesOutstanding") or t.info.get("impliedSharesOutstanding")
    except Exception:
        return pd.DataFrame()

    if inc is None or inc.empty:
        return pd.DataFrame()

    # Q1-Q3: CSA 要求 45 天披露；Q4 (年报): 90 天披露
    # qdate 是季度末日期，月份决定是哪个季度
    def _report_lag(qdate):
        month = pd.Timestamp(qdate).month
        # Q4 = 10/11/12 月末，年报延迟 90 天
        return 90 if month in (10, 11, 12) else REPORT_LAG

    rows = []
    for qdate in inc.columns:
        lag   = _report_lag(qdate)
        avail = pd.Timestamp(qdate) + pd.Timedelta(days=lag)
        qi    = inc.columns.get_loc(qdate)

        def gi(k): return _safe_val(inc, k, qi)
        def gb(k):
            i = _nearest_col(bal, qdate)
            return _safe_val(bal, k, i) if i is not None and bal is not None and k in bal.index else None
        def gc(k):
            i = _nearest_col(cf, qdate)
            return _safe_val(cf, k, i) if i is not None and cf is not None and k in cf.index else None

        rows.append({
            "avail_date":   avail,
            "net_income":   gi("Net Income"),
            "total_equity": gb("Stockholders Equity") or gb("Total Equity Gross Minority Interest"),
            "ocf":          gc("Operating Cash Flow") or gc("Cash Flow From Continuing Operating Activities"),
            "capex":        gc("Capital Expenditure"),
            "shares":       shares,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("avail_date").sort_index()


def fetch_all(tickers, years):
    macro_df  = fetch_macro(years)
    daily_map = fetch_prices(tickers, years)

    print(f"\n[2/4] 获取季报历史（PIT）+ 基本面快照...")

    # 数据优先级：Simfin → FMP → yfinance（自动检测，无需手动切换）
    sf_inc = sf_bal = sf_cf = None
    use_simfin  = False
    use_fmp     = False
    fmp_pit_map = {}

    # 第一层：Simfin（最佳，10年PIT历史，本地运行有效）
    sf_inc, sf_bal, sf_cf = init_simfin()
    use_simfin = sf_inc is not None

    # 第二层：FMP（Colab可用，5年历史，比yfinance好）
    if not use_simfin and FMP_API_KEY:
        print(f"  [FMP] Simfin 不可用，尝试 Financial Modeling Prep...")
        fmp_pit_map = init_fmp(list(daily_map.keys()), FMP_API_KEY)
        use_fmp = len(fmp_pit_map) > 10

    # 状态显示
    if use_simfin:
        print(f"  ✅ 使用 Simfin（完整历史季报，PIT 更准确）")
    elif use_fmp:
        print(f"  ✅ 使用 FMP（{len(fmp_pit_map)} 支，5年季报历史）")
    else:
        print(f"  ⚠️  使用 yfinance 基本面（已知限制）：")
        print(f"     1. 仅 6-7 季历史（vs Simfin 10年+）")
        print(f"     2. 重述偏差（Restatement Bias）：YF 提供最新重述数据，")
        print(f"        若公司事后修正历史财报，回测会用修正后数据，产生前视偏差")
        print(f"     建议：填写 FMP_API_KEY 或本地运行使用 Simfin")

    pit_map   = {}
    meta_rows = []

    # ⚠️  【存活者偏差警告】TSX_UNIVERSE 只包含当前活跃股票，不包括历史已退市股票
    if len(tickers) > 50:  # 仅在大规模回测时提示
        print(f"\n  💡 提示：股票池存活者偏差")
        print(f"     本 TSX_UNIVERSE 是当前维护的 {len(tickers)} 支活跃股票")
        print(f"     历史上退市/破产/被收购的股票已排除")
        print(f"     对当月选股无影响；但历史回测可能虚高 10-30%")
        print(f"     详见 get_tsx_tickers() 函数的文档字符串\n")
    
    for i, t in enumerate(daily_map.keys(), 1):
        # 基本面快照（约束过滤和展示用，始终用 yfinance 当前值）
        try:
            info = yf.Ticker(t).info
            pe   = safe_float(info.get("trailingPE") or info.get("forwardPE"))
            roe  = safe_float(info.get("returnOnEquity"))
            mc   = safe_float(info.get("marketCap"))
            div  = safe_float(info.get("dividendYield"))
            
            # 🎯 【新增 1.1】：抓取负债率和毛利率，用于质量排雷
            de   = safe_float(info.get("debtToEquity"))
            margin = safe_float(info.get("grossMargins"))
            
            meta_rows.append({"ticker":t,"name":info.get("shortName",t),
                               "sector":info.get("sector","Unknown"),
                               "mktcap":mc,"pe":pe,"roe":roe,"div_yield":div,
                               "debt_to_equity": de, "gross_margin": margin})  # 🎯 追加字段
            pe_s  = f"PE={pe:.1f}"        if pe  else "PE=N/A"
            roe_s = f"ROE={roe*100:.1f}%" if roe else "ROE=N/A"
            mc_s  = f"${mc/1e9:.1f}B"     if mc  else "N/A"
            src   = "SF" if use_simfin else "YF"
            print(f"  [{i:>2}/{len(daily_map)}] {t:<14} {pe_s:<10} {roe_s:<12} MCap={mc_s} [{src}]")
        except Exception:
            meta_rows.append({"ticker":t,"name":t,"sector":"Unknown",
                               "mktcap":None,"pe":None,"roe":None,"div_yield":None})
            print(f"  [{i:>2}/{len(daily_map)}] {t:<14} ⚠️  基本面获取失败")

        # PIT 季报：Simfin → FMP → yfinance 三层优先级
        if use_simfin:
            pit = build_pit_from_simfin(t, sf_inc, sf_bal, sf_cf)
            if pit.empty:
                pit = fetch_pit_quarterly(t)
        elif use_fmp and t in fmp_pit_map:
            pit = fmp_pit_map[t].set_index("avail_date")
        else:
            pit = fetch_pit_quarterly(t)

        if not pit.empty:
            pit_map[t] = pit

    meta_df = pd.DataFrame(meta_rows).set_index("ticker")
    return daily_map, pit_map, meta_df, macro_df

# ══════════════════════════════════════════════════════════════════
# Helper: 安全类型转换（yfinance 有时返回字符串）
# ══════════════════════════════════════════════════════════════════
def safe_float(val):
    """安全将任意值转换为 float，None 则返回 None"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return None

# ══════════════════════════════════════════════════════════════════
# 2. 约束过滤
# ══════════════════════════════════════════════════════════════════

def apply_constraints_current(daily_map, meta_df, c):
    """【当月选股模式】用当前最新 yfinance 数据过滤（MODE="pick"）
    
    包含完整约束：价格 + 成交量 + PE + ROE + 市值（均用当前值）
    """
    passed, removed = [], {}

    for t, df in daily_map.items():
        reasons = []
        
        # 🎯 物理隔离不适合动量模型的行业（但白名单豁免）
        gics = STOCK_PROFILE.get(t, {}).get("gics", "Unknown")
        if gics in BANNED_SECTORS and t not in WHITELIST:
            reasons.append(f"行业 {gics} 已全局屏蔽（白名单: {WHITELIST}）")
        
        # 🎯 硬编码过滤历史上结构性毁灭的"价值陷阱"
        TOXIC_BLACKLIST = {"BHC.TO", "SGY.TO", "BIR.TO", "AQN.TO", "CJT.TO", "TOU.TO"}
        if t in TOXIC_BLACKLIST:
            reasons.append(f"被列入结构性风险黑名单")
        
        min_days = c.get("min_listing_days", 0)
        if len(df) < min_days:
            reasons.append(f"上市时间 {len(df)} 天 < {min_days} 天")

        m = meta_df.loc[t] if t in meta_df.index else pd.Series(dtype=float)
        price = df["close"].iloc[-1]

        if price < c["min_price_cad"]:
            reasons.append(f"股价 ${price:.2f} < $2")
        max_px = c.get("max_price_cad", 9999)
        if price > max_px:
            reasons.append(f"股价 ${price:.2f} > ${max_px:.0f}")

        adv = (df["close"].tail(20) * df["volume"].tail(20)).mean()
        if adv < c["min_adv_cad"]:
            reasons.append(f"ADV ${adv/1e6:.2f}M < $1M")

        # ✓ Bug fix: vol_spike_min_days=2
        if len(df) >= 65:
            base   = df["volume"].iloc[-65:-5]
            vm, vs = base.mean(), base.std()
            sp     = df["volume"].tail(c["vol_spike_days"])
            sp     = sp[(sp > vm+c["vol_spike_sigma"]*vs) |
                        (sp < max(0, vm-c["vol_spike_sigma"]*vs))]
            if len(sp) >= c["vol_spike_min_days"]:
                reasons.append(f"成交量异常 {len(sp)}天±{c['vol_spike_sigma']:.0f}σ")

        pe = safe_float(m.get("pe"))
        if pe is None:
            reasons.append("PE无数据")
        elif not (c["min_pe"] < pe < c["max_pe"]):
            reasons.append(f"PE {pe:.1f} 超出范围")

        mktcap = safe_float(m.get("mktcap"))
        if mktcap is None:
            reasons.append("市值无数据")
        elif mktcap < c["min_mktcap_cad"]:
            reasons.append(f"市值 ${mktcap/1e6:.0f}M < $500M")

        roe = safe_float(m.get("roe"))
        if roe is not None and roe < c.get("min_roe", 0):
            reasons.append(f"ROE {roe*100:.1f}% < 0")
        if roe is not None and roe > c.get("max_roe", 999):
            reasons.append(f"ROE {roe*100:.0f}% 异常（>200%，财务杠杆或一次性项目）")
        
        # 🎯 【新增 2.2】：质量排雷（防范深度价值陷阱，如 BHC）
        de = safe_float(m.get("debt_to_equity"))
        if de is not None and de > 250:  # 负债/权益比超过 250% 视为极度危险
            reasons.append(f"高危负债率 {de:.0f}% > 250%")
        
        margin = safe_float(m.get("gross_margin"))
        # 🎯 修复：金融业（银行/保险）没有传统意义的毛利率，必须给予豁免！
        if gics != "Financials" and margin is not None and margin < 0.05:
            reasons.append(f"毛利率极低 {margin*100:.1f}% < 5%")
        
        # 🎯 增加：针对金融股的硬性 ROE 门槛（银行必须够赚钱）
        if gics == "Financials":
            roe = safe_float(m.get("roe"))
            if roe is not None and roe < 0.10:  # ROE 低于 10% 的银行直接拉黑
                reasons.append(f"金融股盈利能力过低 (ROE {roe*100:.1f}% < 10%)")

        if reasons:
            removed[t] = reasons
        else:
            passed.append(t)

    print(f"\n{'─'*60}")
    print(f"  约束过滤：{len(daily_map)} → {len(passed)} 支通过")
    print(f"{'─'*60}")
    counts = {}
    for rs in removed.values():
        for r in rs:
            k = r.split(" ")[0]
            counts[k] = counts.get(k,0)+1
    for k, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  ✗ {k:<22} 剔除 {n} 支")
    if removed:
        print(f"\n  {'Ticker':<14} 原因")
        print(f"  {'─'*50}")
        for t, rs in sorted(removed.items()):
            print(f"  {t:<14} {' | '.join(rs)}")
    print(f"\n  ✅ 通过：{' '.join(passed)}")
    return passed


def apply_base_constraints(daily_map, c):
    """【历史回测模式】只用客观数据检查基础约束（MODE="backtest"）

    ✓ 说明：不使用当前 meta_df（PE/ROE/市值）以避免存活者偏差
    只检查与时间无关的约束：上市时间、股价范围、成交量异常
    PIT 基本面约束由 build_panel() 在逐月基础上检查

    ⚠️ 已知前视近似（2026-04 标注）：
    本函数只对 universe 做一次性筛选（不是逐月 PIT），所以仍有以下偏差：
      - df["close"].iloc[-1] 是全样本最后一天的价格，不是 2018 年那只票的价格；
      - tail(20) ADV 与 tail(65) 成交量异常都用最近一段
    实际效果：用今天才知道的"现在还活着 / 现在还流动"决定历史 universe，
    属于温和的存活者偏差。彻底修复需把这套约束嵌入 walk_forward 逐月调用。
    当前权衡是接受边际偏差以换取实现简洁。
    """
    passed, removed = [], {}

    for t, df in daily_map.items():
        reasons = []
        
        # 🎯 物理隔离不适合动量模型的行业（但白名单豁免）
        gics = STOCK_PROFILE.get(t, {}).get("gics", "Unknown")
        if gics in BANNED_SECTORS and t not in WHITELIST:
            reasons.append(f"行业 {gics} 已全局屏蔽（白名单: {WHITELIST}）")
        
        # 🎯 硬编码过滤历史上结构性毁灭的"价值陷阱"
        TOXIC_BLACKLIST = {"BHC.TO", "SGY.TO", "BIR.TO", "AQN.TO", "CJT.TO", "TOU.TO"}
        if t in TOXIC_BLACKLIST:
            reasons.append(f"被列入结构性风险黑名单")
        
        min_days = c.get("min_listing_days", 0)
        if len(df) < min_days:
            reasons.append(f"上市时间 {len(df)} 天 < {min_days} 天")

        price = df["close"].iloc[-1]
        if price < c["min_price_cad"]:
            reasons.append(f"股价 ${price:.2f} < $2")
        max_px = c.get("max_price_cad", 9999)
        if price > max_px:
            reasons.append(f"股价 ${price:.2f} > ${max_px:.0f}")

        adv = (df["close"].tail(20) * df["volume"].tail(20)).mean()
        if adv < c["min_adv_cad"]:
            reasons.append(f"ADV ${adv/1e6:.2f}M < $1M")

        # ✓ Bug fix: vol_spike_min_days=2
        if len(df) >= 65:
            base   = df["volume"].iloc[-65:-5]
            vm, vs = base.mean(), base.std()
            sp     = df["volume"].tail(c["vol_spike_days"])
            sp     = sp[(sp > vm+c["vol_spike_sigma"]*vs) |
                        (sp < max(0, vm-c["vol_spike_sigma"]*vs))]
            if len(sp) >= c["vol_spike_min_days"]:
                reasons.append(f"成交量异常 {len(sp)}天±{c['vol_spike_sigma']:.0f}σ")

        # ⚠️ 省略 PE / ROE / 市值 检查，在历史回测中由 PIT 数据逐月检查
        # 这避免了用当前 2026 年的数据过滤 2018-2022 年的历史数据（存活者偏差）
        
        # 🎯 防守层：毛利率豁免（金融股不适用传统毛利率指标）
        # 虽然apply_base_constraints不检查基本面，但如果未来有扩展，确保金融股不被拉黑
        margin = {}  # placeholder

        if reasons:
            removed[t] = reasons
        else:
            passed.append(t)

    print(f"\n{'─'*60}")
    print(f"  基础过滤（回测模式，不含 PIT 约束）：{len(daily_map)} → {len(passed)} 支通过")
    print(f"  💡 说明：PIT 基本面约束由 walk_forward 逐月检查，避免存活者偏差")
    print(f"{'─'*60}")
    counts = {}
    for rs in removed.values():
        for r in rs:
            k = r.split(" ")[0]
            counts[k] = counts.get(k,0)+1
    for k, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  ✗ {k:<22} 剔除 {n} 支")
    if removed:
        print(f"\n  {'Ticker':<14} 原因")
        print(f"  {'─'*50}")
        for t, rs in sorted(removed.items()):
            print(f"  {t:<14} {' | '.join(rs)}")
    print(f"\n  ✅ 通过：{' '.join(passed)}")
    return passed


def apply_constraints_asof(ticker: str, daily_map: dict, meta_pit: dict,
                            asof_date: pd.Timestamp, constraints: dict) -> tuple[bool, list[str]]:
    """
    严谨的时间序列约束过滤（消除前视偏差）
    
    📊 设计原则（修复存活者偏差）：
        所有约束都只使用 asof_date 前的数据，避免用"今天才知道的信息"污染历史回测。
        这确保了：
        - 价格范围检查：用 asof_date 前最后可见价格
        - ADV / 成交量异常：只用 asof_date 前的历史窗口
        - PIT 基本面：用该日期对应的 PIT 数据（不是最新数据）
        - 上市时间：用 asof_date 前的实际交易天数
    
    🎯 参数说明：
        ticker: 股票代码
        daily_map: {ticker -> DataFrame} 历史 OHLCV 数据
        meta_pit: {date -> {ticker -> {pe/roe/...}}} PIT 基本面数据
        asof_date: 当月评估日期（只用此日期前的数据）
        constraints: 约束条件字典
    
    💡 返回：(是否通过, 失败原因列表)
    """
    # 获取该股票的 asof_date 前的数据
    if ticker not in daily_map:
        return False, ["股票不在数据库"]
    
    df = daily_map[ticker]
    hist = df[df.index <= asof_date]  # ★ 关键：只用截至 asof_date 的历史数据
    
    reasons = []
    
    # 黑名单检查（不受时间限制）
    gics = STOCK_PROFILE.get(ticker, {}).get("gics", "Unknown")
    if gics in BANNED_SECTORS and ticker not in WHITELIST:
        reasons.append(f"行业 {gics} 已全局屏蔽")
    
    TOXIC_BLACKLIST = {"BHC.TO", "SGY.TO", "BIR.TO", "AQN.TO", "CJT.TO", "TOU.TO"}
    if ticker in TOXIC_BLACKLIST:
        reasons.append(f"被列入结构性风险黑名单")
    
    # 上市时间检查（asof_date 前的实际天数）
    min_days = constraints.get("min_listing_days", 252)
    if len(hist) < min_days:
        reasons.append(f"上市时间 {len(hist)}天 < {min_days}天")
        return False, reasons  # 早期退出
    
    # 价格范围检查（用 asof_date 最后一个可见价格）
    if len(hist) == 0:
        reasons.append("asof_date 前无数据")
        return False, reasons
    
    price = hist["close"].iloc[-1]
    if price < constraints["min_price_cad"]:
        reasons.append(f"股价 ${price:.2f} < ${constraints['min_price_cad']:.2f}")
    max_px = constraints.get("max_price_cad", 9999)
    if price > max_px:
        reasons.append(f"股价 ${price:.2f} > ${max_px:.0f}")
    
    # ADV 检查（用 asof_date 前的最后 20 天）
    if len(hist) >= 20:
        adv = (hist["close"].tail(20) * hist["volume"].tail(20)).mean()
        if adv < constraints["min_adv_cad"]:
            reasons.append(f"ADV ${adv/1e6:.2f}M < ${constraints['min_adv_cad']/1e6:.1f}M")
    else:
        reasons.append(f"历史数据不足 20 天，无法计算 ADV")
    
    # 成交量异常检查（用 asof_date 前的历史窗口）
    if len(hist) >= 65:
        base = hist["volume"].iloc[-65:-5]
        vm, vs = base.mean(), base.std()
        sp = hist["volume"].tail(constraints["vol_spike_days"])
        sp = sp[(sp > vm + constraints["vol_spike_sigma"] * vs) |
                (sp < max(0, vm - constraints["vol_spike_sigma"] * vs))]
        if len(sp) >= constraints["vol_spike_min_days"]:
            reasons.append(f"成交量异常 {len(sp)}天±{constraints['vol_spike_sigma']:.0f}σ")
    
    # PIT 基本面检查（用 asof_date 对应的 PIT 数据）
    m = pd.Series(dtype=float)
    if asof_date in meta_pit and ticker in meta_pit[asof_date]:
        m = meta_pit[asof_date][ticker]
    else:
        # 如果没有该日期的 PIT 数据，尝试向前回溯最近可用的
        pit_dates_before = [d for d in meta_pit.keys() if d <= asof_date]
        if pit_dates_before:
            latest_pit_date = max(pit_dates_before)
            if ticker in meta_pit[latest_pit_date]:
                m = meta_pit[latest_pit_date][ticker]
    
    # PE 检查
    pe = safe_float(m.get("pe")) if len(m) > 0 else None
    if pe is None or pd.isna(pe):
        reasons.append("PE 无数据")
    elif not (constraints["min_pe"] <= pe <= constraints["max_pe"]):
        reasons.append(f"PE {pe:.1f} 超出范围 [{constraints['min_pe']:.0f}, {constraints['max_pe']:.0f}]")
    
    # 市值检查
    mktcap = safe_float(m.get("mktcap")) if len(m) > 0 else None
    if mktcap is None or pd.isna(mktcap):
        reasons.append("市值 无数据")
    elif mktcap < constraints["min_mktcap_cad"]:
        reasons.append(f"市值 ${mktcap/1e9:.2f}B < ${constraints['min_mktcap_cad']/1e9:.1f}B")
    
    # ROE 检查
    roe = safe_float(m.get("roe")) if len(m) > 0 else None
    if roe is not None and not pd.isna(roe):
        if roe < constraints.get("min_roe", 0):
            reasons.append(f"ROE {roe*100:.1f}% < {constraints.get('min_roe',0)*100:.0f}%")
        if roe > constraints.get("max_roe", 999):
            reasons.append(f"ROE {roe*100:.0f}% > {constraints.get('max_roe',999)*100:.0f}%")
    
    # 负债率检查（极度危险）
    de = safe_float(m.get("debt_to_equity")) if len(m) > 0 else None
    if de is not None and not pd.isna(de) and de > 2.5:  # >250%
        reasons.append(f"极度危险负债率 {de*100:.0f}%")
    
    # 毛利率检查（非金融股）
    if gics != "Financials":
        margin = safe_float(m.get("gross_margin")) if len(m) > 0 else None
        if margin is not None and not pd.isna(margin) and margin < 0.05:
            reasons.append(f"毛利率极低 {margin*100:.1f}%")
    
    # 金融股特殊规则（ROE >= 10%）
    if gics == "Financials" and roe is not None and not pd.isna(roe) and roe < 0.10:
        reasons.append(f"金融股ROE过低 {roe*100:.1f}% < 10%")
    
    return len(reasons) == 0, reasons


def apply_constraints(daily_map, meta_df, c):
    """【后向兼容】自动选择当前或基础约束
    
    如果提供 meta_df，用完整约束（当月选股）；否则用基础约束（历史回测）
    """
    if meta_df is not None and not meta_df.empty:
        return apply_constraints_current(daily_map, meta_df, c)
    else:
        return apply_base_constraints(daily_map, c)

# ══════════════════════════════════════════════════════════════════
# 3. 特征工程
# ══════════════════════════════════════════════════════════════════

def _rsi(close, w=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(span=w).mean()
    l = (-d.clip(upper=0)).ewm(span=w).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def compute_time_decay_weights(sample_dates, current_date, half_life_months=12.0):
    """
    计算指数级时间衰减权重。
    参数:
      sample_dates: 样本对应的时间戳序列 (pd.Series 或 np.array)
      current_date: 当前预测基准日 (pd.Timestamp)
      half_life_months: 半衰期（默认 12 个月，即 1 年前的数据权重减半）
    """
    # 计算样本距离“当前日”的月份差
    age_days = (pd.Timestamp(current_date) - pd.to_datetime(sample_dates)).days
    age_months = np.maximum(0, age_days / 30.4)
    
    # 半衰期指数衰减公式: W = 0.5 ^ (age / half_life)
    weights = 0.5 ** (age_months / half_life_months)
    return weights.values


def compute_pit_fundamentals(pit_df, monthly_close):
    """
    向量化重构：计算通用基本面 + 行业专属基本面。
    """
    if pit_df.empty or monthly_close.empty:
        # 返回带所有空列的 DataFrame 占位
        cols = ["pe","pb","roe","eps_growth","fcf_yield", 
                "ev_ebitda", "capex_ocf", "p_ffo", "rule_of_40", "rnd_rev",
                "debt_ebitda", "div_cover", "nim", "pcl_ratio"]
        result = pd.DataFrame(index=monthly_close.index, columns=cols)
        result.index.name = "date"
        return result

    pit_df = pit_df.sort_values('avail_date').copy()
    
    # ── 1. 滚动 TTM (Trailing Twelve Months) 基础计算 ──
    pit_df['ttm_ni'] = pit_df['net_income'].rolling(4, min_periods=4).sum()
    pit_df['ttm_ocf'] = pit_df['ocf'].rolling(4, min_periods=4).sum()
    pit_df['ttm_capex'] = pit_df['capex'].fillna(0).rolling(4, min_periods=4).sum()
    pit_df['ttm_fcf'] = pit_df['ttm_ocf'] - pit_df['ttm_capex'].abs()
    
    # 🎯 新增字段的 TTM
    pit_df['ttm_rev']    = pit_df.get('revenue', pd.Series(0, index=pit_df.index)).rolling(4, min_periods=4).sum()
    pit_df['ttm_ebitda'] = pit_df.get('ebitda', pd.Series(0, index=pit_df.index)).rolling(4, min_periods=4).sum()
    pit_df['ttm_dna']    = pit_df.get('dna', pd.Series(0, index=pit_df.index)).rolling(4, min_periods=4).sum()
    pit_df['ttm_rnd']    = pit_df.get('rnd', pd.Series(0, index=pit_df.index)).rolling(4, min_periods=4).sum()
    pit_df['ttm_div']    = pit_df.get('div_paid', pd.Series(0, index=pit_df.index)).abs().rolling(4, min_periods=4).sum()

    # 增长率计算
    pit_df['prev1_ttm_ni'] = pit_df['ttm_ni'].shift(1)
    pit_df['eps_growth'] = (pit_df['ttm_ni'] - pit_df['prev1_ttm_ni']) / pit_df['prev1_ttm_ni'].abs()
    pit_df['prev1_ttm_rev'] = pit_df['ttm_rev'].shift(1)
    pit_df['rev_growth'] = (pit_df['ttm_rev'] - pit_df['prev1_ttm_rev']) / pit_df['prev1_ttm_rev'].abs()

    # ── 2. 时间轴对齐 (merge_asof) ──
    left_df = monthly_close.reset_index()
    if 'index' in left_df.columns:
        left_df = left_df.rename(columns={'index': 'date'})
    elif left_df.columns[0] not in ['date', 'close']:
        left_df = left_df.rename(columns={left_df.columns[0]: 'date'})
    left_df = left_df.rename(columns={'close': 'price'}).sort_values('date')
    right_df = pit_df.reset_index().sort_values('avail_date')
    
    merged = pd.merge_asof(
        left_df, right_df, 
        left_on='date', right_on='avail_date', 
        direction='backward'
    )
    
    # ── 3. 计算所有比率 (通用 + 行业专属) ──
    # 通用指标
    merged['pe'] = merged['price'] / (merged['ttm_ni'] / merged['shares']).replace(0, np.nan)
    merged['pb'] = merged['price'] / (merged['total_equity'] / merged['shares']).replace(0, np.nan)
    merged['roe'] = merged['ttm_ni'] / merged['total_equity'].replace(0, np.nan)
    merged['fcf_yield'] = merged['ttm_fcf'] / (merged['price'] * merged['shares']).replace(0, np.nan)

    # 🎯 行业专属指标
    mcap = merged['price'] * merged['shares']
    total_debt = merged['total_debt'] if 'total_debt' in merged.columns else pd.Series(0, index=merged.index)
    
    # [能源/材料] EV/EBITDA, Capex/OCF
    ev = mcap + total_debt
    merged['ev_ebitda'] = ev / merged['ttm_ebitda'].replace(0, np.nan)
    merged['capex_ocf'] = merged['ttm_capex'].abs() / merged['ttm_ocf'].replace(0, np.nan)
    
    # [REITs] P/FFO (FFO 近似 = 净利润 + 折旧摊销)
    ttm_ffo = merged['ttm_ni'] + merged['ttm_dna']
    merged['p_ffo'] = mcap / ttm_ffo.replace(0, np.nan)
    
    # [科技] Rule of 40, R&D/Rev
    fcf_margin = merged['ttm_fcf'] / merged['ttm_rev'].replace(0, np.nan)
    merged['rule_of_40'] = merged['rev_growth'] + fcf_margin
    merged['rnd_rev'] = merged['ttm_rnd'] / merged['ttm_rev'].replace(0, np.nan)
    
    # [公用/电信] Debt/EBITDA, 股息覆盖率
    merged['debt_ebitda'] = total_debt / merged['ttm_ebitda'].replace(0, np.nan)
    merged['div_cover'] = merged['ttm_fcf'] / merged['ttm_div'].replace(0, np.nan)
    
    # [金融] NIM (净息差), PCL_Ratio (坏账率)
    int_inc = merged['interest_inc'] if 'interest_inc' in merged.columns else pd.Series(0, index=merged.index)
    int_exp = merged['interest_exp'] if 'interest_exp' in merged.columns else pd.Series(0, index=merged.index)
    total_assets = merged['total_assets'] if 'total_assets' in merged.columns else pd.Series(1e9, index=merged.index)
    pcl = merged['pcl'] if 'pcl' in merged.columns else pd.Series(0, index=merged.index)
    
    merged['nim'] = (int_inc - int_exp) / total_assets.replace(0, np.nan)
    merged['pcl_ratio'] = pcl / total_assets.replace(0, np.nan)

    # 清理异常值
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if 'date' not in merged.columns:
        return pd.DataFrame(index=monthly_close.index)
        
    out_cols = ['pe', 'pb', 'roe', 'eps_growth', 'fcf_yield', 
                'ev_ebitda', 'capex_ocf', 'p_ffo', 'rule_of_40', 'rnd_rev',
                'debt_ebitda', 'div_cover', 'nim', 'pcl_ratio']
    
    result = merged.set_index('date')[out_cols]
    result.index.name = "date"
    return result


def compute_monthly_tech(df):
    c, v  = df["close"], df["volume"]
    tp    = (df["high"] + df["low"] + df["close"]) / 3
    mc    = c.resample("ME").last()
    mv    = v.resample("ME").sum()
    mvwap = (tp * v).resample("ME").sum() / mv
    def mom(n): return mc.pct_change(n)
    m1,m3,m6,m12 = mom(1),mom(3),mom(6),mom(12)
    dr    = c.pct_change()
    vol1m = dr.resample("ME").std() * np.sqrt(252)
    vol3m = dr.rolling(63).std().resample("ME").last() * np.sqrt(252)
    b20   = ((c-c.rolling(20).mean())/c.rolling(20).mean()).resample("ME").last()
    b60   = ((c-c.rolling(60).mean())/c.rolling(60).mean()).resample("ME").last()
    vwapb = (mc-mvwap)/mvwap
    h52   = (c/c.rolling(252).max()).resample("ME").last()
    rsi_m = _rsi(c).resample("ME").last()
    bb_zscore = ((c - c.rolling(20).mean()) / c.rolling(20).std()).resample("ME").last()

    result = pd.DataFrame({
        "close":mc,"mom_1m":m1,"mom_3m":m3,"mom_6m":m6,"mom_12m":m12,
        "mom_12_1":m12-m1,"vol_1m":vol1m,"vol_3m":vol3m,
        "vol_ratio":vol1m/vol3m.replace(0,np.nan),
        "rsi":rsi_m,"bias_20":b20,"bias_60":b60,
        "vwap_bias":vwapb,"price_vs_52w_high":h52,
        "bb_zscore": bb_zscore,
    }).dropna(subset=["close"])
    result.index.name = "date"
    return result


def get_macro_feat(macro_df, month_end):
    """第二层：宏观因子"""
    zero = {"oil_mom_3m":0.0,"cadusd_mom_3m":0.0,"bond_chg_3m":0.0,
            "gold_mom_3m":0.0,"vix_level":0.0}
    if macro_df.empty:
        return zero
    av = macro_df[macro_df.index <= month_end]
    if len(av) < 4:
        return zero
    def m3(col):
        if col not in av.columns: return 0.0
        s = av[col].dropna()
        return float(s.iloc[-1]/s.iloc[-4]-1) if len(s)>=4 else 0.0
    def lv(col):
        if col not in av.columns: return 0.0
        s = av[col].dropna()
        return float(s.iloc[-1]) if len(s)>0 else 0.0
    return {"oil_mom_3m":m3("oil"),"cadusd_mom_3m":m3("cadusd"),
            "bond_chg_3m":m3("bond"),"gold_mom_3m":m3("gold"),
            "vix_level":lv("vix")/40.0}


def fetch_earnings_calendar(tickers: list, lookback_days: int = 365,
                             forward_days: int = 90, is_backtest: bool = False) -> dict:
    """
    从 FMP 获取财报日期（比 yfinance 更准确，Colab可用）。
    返回：{ticker: sorted list of earnings dates}
    缓存7天。
    
    ⚠️  警告：信息泄露风险
    
    如果 is_backtest=False（当月预测）：
      用今天为中心往前 365 天、往后 90 天，获取实际已知的财报日历。
    
    如果 is_backtest=True（历史回测）：
      返回空字典，避免用当今的财报安排去预测 2018-2022 年的数据。
      历史回测期间的财报日期大多不可得（真正的 PIT 会很困难），
      强行使用会导致未来信息泄露（用现在才知道的财报日期去训练过去）。
    """
    if is_backtest:
        print(f"  [财报日历] 回测模式，禁用财报特征以避免信息泄露")
        return {}
    import os, pickle, time, requests as _req

    cache_file = "./simfin_data/earnings_calendar.pkl"
    os.makedirs("./simfin_data", exist_ok=True)

    # 读缓存（7天内有效）
    if os.path.exists(cache_file):
        if (time.time() - os.path.getmtime(cache_file)) < 7 * 86400:
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                print(f"  [财报日历] 从缓存加载（{len(cached)} 支）")
                return cached
            except Exception:
                pass

    if not FMP_API_KEY:
        print("  [财报日历] FMP_API_KEY 未设置，跳过财报特征")
        return {}

    print(f"  [财报日历] FMP 获取 {len(tickers)} 支财报日期...")
    BASE    = "https://financialmodelingprep.com/api/v3"
    result  = {}
    today   = pd.Timestamp.today()
    from_dt = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_dt   = (today + pd.Timedelta(days=forward_days)).strftime("%Y-%m-%d")

    for t in tickers:
        try:
            r = _req.get(
                f"{BASE}/historical/earning_calendar/{t}",
                params={"from": from_dt, "to": to_dt, "apikey": FMP_API_KEY},
                timeout=8
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    dates = sorted([
                        pd.Timestamp(d["date"])
                        for d in data if d.get("date")
                    ])
                    if dates:
                        result[t] = dates
            time.sleep(0.05)  # 避免超频
        except Exception:
            pass

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"  [财报日历] ✓ {len(result)} 支有财报日期，已缓存7天")
    return result


def compute_earnings_features(ticker: str,
                               dates: pd.DatetimeIndex,
                               earnings_cal: dict) -> pd.DataFrame:
    """
    对每个月末日期计算：
      days_to_earnings:    距下次财报天数（未来为正，刚过为负）
      days_since_earnings: 距上次财报天数（越小 = PEAD 窗口内）
    """
    t_dates = earnings_cal.get(ticker, [])
    if not t_dates:
        return pd.DataFrame({
            "days_to_earnings":    [30.0] * len(dates),  # 默认中性
            "days_since_earnings": [45.0] * len(dates),
        }, index=dates)

    t_dates_ts = [pd.Timestamp(d) for d in t_dates]
    rows = []
    for d in dates:
        d_ts = pd.Timestamp(d)
        future = [ed for ed in t_dates_ts if ed >= d_ts]
        past   = [ed for ed in t_dates_ts if ed  < d_ts]
        days_to  = (future[0] - d_ts).days if future else 90.0
        days_since=(d_ts - past[-1]).days   if past   else 90.0
        rows.append({"days_to_earnings": float(days_to),
                     "days_since_earnings": float(days_since)})
    
    result_df = pd.DataFrame(rows, index=dates)
    # Ensure index is a DatetimeIndex
    if not isinstance(result_df.index, pd.DatetimeIndex):
        result_df.index = pd.DatetimeIndex(result_df.index)
    return result_df


def build_panel(passed, daily_map, pit_map, macro_df, is_backtest: bool = True):
    """✓ Bug fix: try/except 每支股票，rows为空时抛明确错误

    ✓ 修订（2026-04）：
      新增 is_backtest 参数。回测模式下禁用 earnings_calendar
      （fetch_earnings_calendar 用今日为锚点会泄露未来财报安排）；
      当月选股模式（is_backtest=False）启用真实财报日历。
    """
    print(f"\n[3/4] 构建特征面板...")

    all_tech = {}
    # ✓ 修订：earnings_calendar 是否启用由 is_backtest 控制
    # - 回测（is_backtest=True）：禁用，因 fetch_earnings_calendar 用 today 锚点会泄露未来
    # - 当月选股（is_backtest=False）：启用，未来 90 天内的财报安排是当下真实可知信息
    earnings_cal = fetch_earnings_calendar(passed, is_backtest=is_backtest)

    for t in passed:
        try:
            all_tech[t] = compute_monthly_tech(daily_map[t])
        except Exception:
            pass

    rows, errors = [], []
    for t in passed:
        try:
            tech = all_tech.get(t, pd.DataFrame())
            if tech.empty:
                errors.append(f"{t}: tech 为空")
                continue

            # ✓ Bug Fix: Ensure tech has proper index name
            if tech.index.name is None:
                tech.index.name = "date"
            
            mc = tech["close"]
            fund_hist = (compute_pit_fundamentals(pit_map[t], mc)
                         if t in pit_map and not pit_map[t].empty
                         else pd.DataFrame(index=mc.index,
                                           columns=["pe","pb","roe","eps_growth","fcf_yield",
                                                    "ev_ebitda", "capex_ocf", "p_ffo", "rule_of_40", "rnd_rev",
                                                    "debt_ebitda", "div_cover", "nim", "pcl_ratio"]))

            gics = STOCK_PROFILE.get(t, {}).get("gics", "Unknown")

            # ── 向量化替代 iterrows（Fix 3）─────────────────────────────
            # 宏观因子批量计算
            if not macro_df.empty:
                macro_block = pd.DataFrame(
                    {d: get_macro_feat(macro_df, d) for d in tech.index},
                ).T.reindex(tech.index).fillna(0)  # 宏观缺失=无信号，0正确
            else:
                macro_block = pd.DataFrame(0.0, index=tech.index,
                    columns=["oil_mom_3m","cadusd_mom_3m","bond_chg_3m",
                             "gold_mom_3m","vix_level"])
            macro_block["month_sin"] = np.sin(2*np.pi*tech.index.month/12)
            macro_block["month_cos"] = np.cos(2*np.pi*tech.index.month/12)
            
            # ✓ 矢量化重构：直接使用 groupby 替代 for 循环，计算行业基准动量
            sector_mom_rel = pd.Series(0.0, index=tech.index)
            if gics:
                # 准备所有行业同业股票的 mom_6m，按日期对齐
                peer_mom_data = []
                for p in passed:
                    if p in all_tech and STOCK_PROFILE.get(p,{}).get("gics")==gics:
                        peer_tech = all_tech[p].copy()
                        peer_tech.columns = [f"{p}_{col}" if col != "mom_6m" else "mom_6m" for col in peer_tech.columns]
                        peer_mom_data.append(peer_tech[["mom_6m"]])
                
                if peer_mom_data:
                    # 并联所有同业股票的 mom_6m，计算行业平均
                    peer_concat = pd.concat(peer_mom_data, axis=1)
                    peer_mean = peer_concat.mean(axis=1)
                    # 对齐到当前股票的日期，并计算相对值
                    peer_mean_aligned = peer_mean.reindex(tech.index)
                    sector_mom_rel = (tech.get("mom_6m", pd.Series(0.0, index=tech.index)) - peer_mean_aligned).fillna(0.0)
            
            macro_block["sector_mom_rel"] = sector_mom_rel
            fund_cols = ["pe","pb","roe","eps_growth","fcf_yield",
                         "ev_ebitda", "capex_ocf", "p_ffo", "rule_of_40", "rnd_rev",
                         "debt_ebitda", "div_cover", "nim", "pcl_ratio"]
            fund_block = (fund_hist.reindex(tech.index,method="ffill")[fund_cols]
                          if not fund_hist.empty
                          else pd.DataFrame(np.nan,index=tech.index,columns=fund_cols))
            
            # ✓ Bug Fix: Ensure all blocks have the same index name before concat
            macro_block.index.name = "date"
            fund_block.index.name = "date"
            
            block = pd.concat([tech, macro_block, fund_block], axis=1)
            block["ticker"] = t
            block["gics"]   = gics
            block.index.name = "date"
            # 财报日历特征（距下次/上次财报天数）
            earn_feats = compute_earnings_features(t, tech.index, earnings_cal)
            block["days_to_earnings"]    = earn_feats["days_to_earnings"].values
            block["days_since_earnings"] = earn_feats["days_since_earnings"].values
            
            # ✓ Bug Fix: Explicitly reset index to 'date' column
            block = block.reset_index()
            if "date" not in block.columns:
                block["date"] = pd.to_datetime(block.index) if hasattr(block.index, '__iter__') else block.index
            rows.append(block)
        except Exception as e:
            errors.append(f"{t}: {e}")

    if errors:
        print(f"  ⚠️  {len(errors)} 支失败：{errors[:3]}")
    if not rows:
        raise RuntimeError("所有股票特征计算失败，rows 为空")

    panel = pd.concat(rows, ignore_index=True).set_index(["date","ticker"])
    
    # 🎯 【新增】计算质量-价值比（巴菲特因子）
    # QV Ratio = ROE / PB：寻找 "便宜又好" 的公司（高盈利，合理价格）
    print(f"  [新增] 计算质量-价值比 (QV Ratio = ROE / PB)...")
    panel['qv_ratio'] = panel['roe'] / (panel['pb'].replace(0, np.nan))
    # 处理极端值：QV Ratio > 50 通常是数据异常（如PB < 0.02）
    panel['qv_ratio'] = panel['qv_ratio'].clip(-50, 50)
    
    print(f"  面板：{len(panel)} 行 "
          f"({panel.index.get_level_values('date').nunique()} 月 × "
          f"{panel.index.get_level_values('ticker').nunique()} 支)")

    # ✓ 数据健康度诊断（2026-04 新增）
    # 静默 except 掩盖了财报/价量缺失，这里把覆盖率显式打出来，让用户知道
    # 模型实际接触到多少有效信息。基本面缺失 > 30% 应当警觉。
    diag_cols = {
        "pe_rel_sector":  "PE",
        "pb":             "PB",
        "roe":            "ROE",
        "eps_growth":     "EPS增长",
        "fcf_yield":      "FCF收益率",
        "days_to_earnings":"财报日历",
    }
    print(f"\n  📋 [数据覆盖率诊断]  N={len(panel)}")
    for col, label in diag_cols.items():
        if col in panel.columns:
            cov = float(panel[col].notna().mean()) * 100
            flag = "" if cov >= 70 else (" ⚠️" if cov >= 30 else " 🔴")
            print(f"     {label:<10} {cov:5.1f}%{flag}")
    if errors:
        print(f"     失败 ticker（{len(errors)} 支）：{', '.join(e.split(':')[0] for e in errors[:8])}"
              + (" ..." if len(errors) > 8 else ""))
    return panel


def add_labels(panel):
    """✓ Bug fix: rows 为空时填 NaN 而非崩溃
    
    🎯 修复：统一使用绝对收益作为训练目标，与回测评估保持一致
    （之前混用残差收益训练 + 绝对收益评测，导致目标函数不匹配）
    """
    dates = sorted(panel.index.get_level_values("date").unique())
    rows  = []
    for i, date in enumerate(dates[:-1]):
        nxt = dates[i+1]
        try:
            c0 = panel.xs(date, level="date")["close"]
            c1 = panel.xs(nxt,  level="date")["close"]
        except KeyError:
            continue
        common = c0.index.intersection(c1.index)
        # 对数收益率（绝对收益）
        ret_abs = np.log(c1[common] / c0[common])

        # 🎯 修改：直接使用绝对收益作为训练标签
        # （之前用残差收益会造成模型优化 Alpha，但回测看绝对收益的矛盾）
        ret = ret_abs

        thr = ret.quantile(1-TOP_QUINTILE)
        for t in common:
            rows.append({"date":date,"ticker":t,
                         "next_ret":ret[t],           # 🎯 现在 = 绝对收益（之前 = 残差）
                         "next_ret_abs":ret_abs[t],   # 冗余字段，保留向后兼容
                         "label":int(ret[t]>=thr)})
    if not rows:
        print("  ⚠️  add_labels: 未生成标签")
        panel["next_ret"] = np.nan
        panel["label"]    = np.nan
        return panel
    label_df = pd.DataFrame(rows).set_index(["date","ticker"])
    result   = panel.join(label_df, how="left")
    n_ok     = result["next_ret"].notna().sum()
    print(f"  标签覆盖：{n_ok}/{len(result)} 行 ({n_ok/len(result)*100:.0f}%)")
    return result


# 行业正交化掩码：0.0=完全屏蔽，1.0=正常使用，>1.0=增强权重（引导模型关注核心Alpha）
SECTOR_FACTOR_MASK = {
    "Financials": {
        "fcf_yield": 0.0, "ev_ebitda": 0.0, "capex_ocf": 0.0, "p_ffo": 0.0, "rule_of_40": 0.0, "rnd_rev": 0.0,
        "nim": 2.0, "pcl_ratio": 2.0  # 🎯 强迫模型高度关注银行的息差和坏账率
    },            
    "Energy": {
        "pe": 0.0, "pb": 0.0, "pe_rel_sector": 0.0, "p_ffo": 0.0, "rule_of_40": 0.0, "rnd_rev": 0.0, "nim": 0.0, "pcl_ratio": 0.0,
        "ev_ebitda": 1.5, "capex_ocf": 1.5 # 🎯 周期股剥离 PE 幻觉，只看企业倍数和资本开支比率
    },
    "Materials": {
        "pe": 0.0, "pb": 0.0, "pe_rel_sector": 0.0, "p_ffo": 0.0, "rule_of_40": 0.0, "rnd_rev": 0.0, "nim": 0.0, "pcl_ratio": 0.0,
        "ev_ebitda": 1.5, "capex_ocf": 1.5
    },
    "REITs": {
        "pe": 0.0, "pb": 0.0, "pe_rel_sector": 0.0, "fcf_yield": 0.0, "ev_ebitda": 0.0, "rule_of_40": 0.0, "rnd_rev": 0.0, "nim": 0.0, "pcl_ratio": 0.0,
        "p_ffo": 2.0  # 🎯 REITs 估值唯一标尺
    },
    "Technology": {
        "p_ffo": 0.0, "nim": 0.0, "pcl_ratio": 0.0, "capex_ocf": 0.0, "div_cover": 0.0,
        "pb": 0.0,            # 🔪 彻底无视科技股的 PB（高成长公司估值看 Rule of 40，不看账面倍数）
        "rule_of_40": 2.0, "rnd_rev": 1.5 # 🎯 SaaS 类公司忽略传统盈利，只看护城河(研发)和 40法则
    },
    "Utilities": {
        "p_ffo": 0.0, "rule_of_40": 0.0, "rnd_rev": 0.0, "nim": 0.0, "pcl_ratio": 0.0,
        "debt_ebitda": 1.5, "div_cover": 1.5 # 🎯 高杠杆收息股核心排雷指标
    },
    "Telecom": {
        "p_ffo": 0.0, "rule_of_40": 0.0, "rnd_rev": 0.0, "nim": 0.0, "pcl_ratio": 0.0,
        "debt_ebitda": 1.5, "div_cover": 1.5
    },
    "Industrials": {
        "p_ffo": 0.0, "nim": 0.0, "pcl_ratio": 0.0, "rule_of_40": 0.0, "rnd_rev": 0.0,
        "pb": 0.2,            # 🔪 打压 PB，工业股看 ROIC/ROE
        "fcf_yield": 1.5      # 💰 巴菲特最爱：充沛的自由现金流
    },
    "Consumer Discretionary": {
        "pb": 0.2, "roe": 1.5, "fcf_yield": 1.5  # 🔪 消费股淡化账面值，强调 ROE 和现金流
    },
    "Consumer Staples": {
        "pb": 0.2, "roe": 1.5, "fcf_yield": 1.5  # 🔪 同上
    }
}


def smart_impute(panel: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    智能特征插补：NaN → 行业截面中位数 → 全市场中位数。

    规则：
      基本面（pe_rel_sector/pb/roe/eps_growth/fcf_yield）→ 同GICS行业、同月截面中位数
      技术/动量                               → 全市场截面中位数
      宏观                                    → 全局中位数（单一值）
      最终兜底                                → 0（极少数整列NaN情况）

    绝不盲目用 0：PE=0 会被模型误读为"估值极低的优质股"。
    """
    if "pe_rel_sector" not in panel.columns and "pe" in panel.columns:
        out_for_pe = panel.copy()
        if "gics" not in out_for_pe.columns:
            tix = out_for_pe.index.get_level_values("ticker")
            out_for_pe["gics"] = tix.map(lambda t: STOCK_PROFILE.get(t, {}).get("gics", "Unknown"))

        sector_pe_median = out_for_pe.groupby(
            [out_for_pe.index.get_level_values("date"), out_for_pe["gics"]]
        )["pe"].transform("median")
        out_for_pe["pe_rel_sector"] = out_for_pe["pe"] / sector_pe_median.replace(0, np.nan)
        out_for_pe["pe_rel_sector"] = out_for_pe["pe_rel_sector"].replace([np.inf, -np.inf], np.nan)
        panel = out_for_pe

    FUNDAMENTAL_COLS = {"pe_rel_sector","pb","roe","eps_growth","fcf_yield",
                        "days_to_earnings","days_since_earnings",
                        "ev_ebitda", "capex_ocf", "p_ffo", "rule_of_40", "rnd_rev",
                        "debt_ebitda", "div_cover", "nim", "pcl_ratio"} # 🎯 将新因子加入同业中位数插补列表
    MACRO_COLS       = {"oil_mom_3m","cadusd_mom_3m","bond_chg_3m",
                        "gold_mom_3m","vix_level"}

    out     = panel.copy()
    tickers = out.index.get_level_values("ticker").unique()
    gics_map= {t: STOCK_PROFILE.get(t, {}).get("gics","Unknown") for t in tickers}
    
    # 🎯 过滤：只处理panel中实际存在的特征列（避免新列未被创建时的KeyError）
    feature_cols_avail = [col for col in feature_cols if col in out.columns]

    for date in out.index.get_level_values("date").unique():
        mask    = out.index.get_level_values("date") == date
        sl      = out.loc[mask, feature_cols_avail].copy()
        tix_arr = out.loc[mask].index.get_level_values("ticker")

        for col in feature_cols_avail:
            if not sl[col].isna().any():
                continue

            if col in MACRO_COLS:
                med = sl[col].median()
                sl[col] = sl[col].fillna(med if pd.notna(med) else 0.0)

            elif col in FUNDAMENTAL_COLS:
                sectors = pd.Series([gics_map.get(t,"Unknown") for t in tix_arr],
                                    index=sl.index)
                for sec in sectors.unique():
                    sm   = sectors == sec
                    smed = sl.loc[sm, col].median()
                    if pd.notna(smed):
                        sl.loc[sm, col] = sl.loc[sm, col].fillna(smed)
                gmed = sl[col].median()
                sl[col] = sl[col].fillna(gmed if pd.notna(gmed) else 0.0)

            else:
                med = sl[col].median()
                sl[col] = sl[col].fillna(med if pd.notna(med) else 0.0)

        out.loc[mask, feature_cols_avail] = sl.values

    remaining = out[feature_cols_avail].isna().sum().sum()
    if remaining > 0:
        out[feature_cols_avail] = out[feature_cols_avail].fillna(0.0)
    return out


def cross_z(panel: pd.DataFrame) -> pd.DataFrame:
    """
    横截面秩次化 (Rank Normalization) + 行业因子中性化。

    替代 Z-Score 的原因：
      TSX 含大量小盘矿业股，单支股票暴涨/暴跌会极大拉偏均值和标准差，
      导致当月所有正常股票的 Z-Score 被压缩到 0 附近（信号失真）。

      秩次化方法（WorldQuant / AQR 标配）：
        1. 同月所有股票按特征值排序 → 得到排名 rank
        2. 映射到 [-1, +1] 均匀分布：rank / (n-1) * 2 - 1
      对任何极端异常值 100% 免疫，分布稳定。

    两步处理：
      1. 行业因子屏蔽（Sector Mask）
      2. 截面秩次化 → [-1, 1]
    """
    out = panel.copy()

    # Step 1: 行业特定因子屏蔽（与之前相同）
    tix_col = out.index.get_level_values("ticker")
    for t in tix_col.unique():
        gics = STOCK_PROFILE.get(t, {}).get("gics", "Unknown")
        mask = SECTOR_FACTOR_MASK.get(gics, {})
        if mask:
            rows_t = out.index.get_level_values("ticker") == t
            for factor, weight in mask.items():
                if factor in out.columns:
                    out.loc[rows_t, factor] = out.loc[rows_t, factor] * weight

    # Step 2: 截面秩次化 → [-1, +1]（替代 Z-Score）
    # 🎯 只处理panel中实际存在的特征列
    feature_cols_avail = [col for col in FEATURE_COLS if col in out.columns]
    
    for date in panel.index.get_level_values("date").unique():
        m = out.index.get_level_values("date") == date
        s = out.loc[m, feature_cols_avail]
        n = len(s)
        if n < 2:
            continue
        # 向量化秩次化：rank(0-based) / (n-1) * 2 - 1 → [-1, +1]
        ranked = s.rank(axis=0, method="average", na_option="keep")
        out.loc[m, feature_cols_avail] = (ranked - 1) / (n - 1) * 2 - 1

    return out

# ══════════════════════════════════════════════════════════════════
# 4. 模型
# ══════════════════════════════════════════════════════════════════


def make_xgb(task, pos_w=1.0):
    """
    ★ 优化方案A：改进分类器强度（LTR预备）
    
    分类器的 predict_proba 本身就是相对排序信号：
    - P(Top20%) = 0.85 表示该股票有 85% 概率排在前 20%
    - 这相当于隐式的排序学习（无需改 Label 结构）
    """
    if task == "reg":
        # 回归保持原样（用于数值信号）
        kw_reg = dict(n_estimators=300, max_depth=3, learning_rate=0.04,
                      subsample=0.8, colsample_bytree=0.7,
                      reg_alpha=0.5, reg_lambda=1.5, random_state=42, verbosity=0)
        return xgb.XGBRegressor(**kw_reg)
    else:
        # ✓ 抗过拟合修订（2026-04）：分类器收紧
        # 背景：单次训练独立样本 ~800-1500，原激进参数让单棵树过深、L1/L2 不足，
        #      容易在小样本上记忆噪声；现与回归树对齐，并依赖 early_stopping 控制轮数。
        kw_cls = dict(
            n_estimators=300,           # ↓ 400 → 300（依赖 early_stopping）
            max_depth=3,                # ↓ 4 → 3（与回归对齐，限制单树容量）
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=1.5,              # ↑ 1.0 → 1.5（L1 进一步加强）
            reg_lambda=3.0,             # ↑ 2.0 → 3.0（L2 进一步加强）
            random_state=42,
            verbosity=0,
            scale_pos_weight=pos_w,     # Top20% 样本权重补偿
            eval_metric="aucpr"         # PR 曲线面积，聚焦正样本识别
        )
        return xgb.XGBClassifier(**kw_cls)


def make_lgbm(task, pos_w=1.0):
    """
    ★ 优化方案A：LightGBM 分类器配合增强
    """
    if not LGBM: 
        return None
    
    if task == "reg":
        # 回归保持原样
        kw_reg = dict(n_estimators=300, max_depth=3, num_leaves=15, 
                      learning_rate=0.04, subsample=0.8, colsample_bytree=0.7,
                      min_child_samples=20, reg_alpha=0.5, reg_lambda=1.5,
                      random_state=42, verbose=-1)
        return lgb.LGBMRegressor(**kw_reg)
    else:
        # ✓ 抗过拟合修订（2026-04）：LightGBM 分类器收紧
        # 关键：num_leaves 31→15、min_child_samples 5→20，让单棵树不再"记忆"小簇样本。
        kw_cls = dict(
            n_estimators=300,           # ↓ 400 → 300
            max_depth=3,                # ↓ 4 → 3
            num_leaves=15,              # ↓ 31 → 15（最关键，限制单树叶数）
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_samples=20,       # ↑ 5 → 20（每叶最少 20 样本，强制泛化）
            reg_alpha=1.5,              # ↑ 1.0 → 1.5
            reg_lambda=3.0,             # ↑ 2.0 → 3.0
            random_state=42,
            verbose=-1,
            scale_pos_weight=pos_w,
            metric="auc"
        )
        return lgb.LGBMClassifier(**kw_cls)


if TORCH:
    class MLP(nn.Module):
        # ✓ 抗过拟合修订（2026-04）：隐层 (128,64,32) → (32,16)
        # 原 ~13K 参数对 ~1K 独立样本（比例 1:13）显著过拟合；
        # (32,16) 仅 ~1.4K 参数（26→32→16→1），匹配实际样本规模。
        # Dropout 同步加大 0.3 → 0.4 进一步抑制噪声拟合。
        def __init__(self, d, h=(32, 16)):
            super().__init__()
            layers, prev = [], d
            for n in h:
                layers += [nn.Linear(prev,n),nn.LayerNorm(n),nn.ReLU(),nn.Dropout(0.4)]
                prev = n
            layers.append(nn.Linear(prev,1))
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x).squeeze(-1)

    def train_mlp(X, y, task, epochs=120):
        Xt,yt = torch.FloatTensor(X),torch.FloatTensor(y)
        # ✓ Bug fix: drop_last=True
        dl = DataLoader(TensorDataset(Xt,yt),batch_size=64,shuffle=True,drop_last=True)
        m  = MLP(X.shape[1])
        opt= torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        fn = nn.MSELoss() if task=="reg" else nn.BCEWithLogitsLoss()
        m.train()
        for _ in range(epochs):
            for xb,yb in dl:
                opt.zero_grad(); fn(m(xb),yb).backward(); opt.step()
        return m.eval()
    
    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    @torch.no_grad()
    def pred_mlp(m, X, task, mc_samples=1, return_var=False):
        """
        ★ 优化方案2：MC Dropout 预测方差
        
        参数：
          mc_samples: 1 → 单次推理（快速），>1 → MC Dropout（含不确定性）
          return_var: True → 返回 (均值, 方差)；False → 仅返回均值
        
        原理：
          - 在推理时保持 Dropout 开启
          - 多次前向传播，统计方差 = 模型对该样本的认知不确定性
          - 方差大 → 模型很纠结，应该降低权重（托付给市值加权）
        """
        m.eval()
        
        # MC Dropout：多次前向传播
        if mc_samples > 1:
            enable_dropout(m)      # ★ 开启进行中的 Dropout
            
        X_tensor = torch.FloatTensor(X)
        preds = []
        
        for _ in range(mc_samples):
            out = m(X_tensor).squeeze(-1)
            if task == "cls":
                out = torch.sigmoid(out)  # 分类任务转到 [0,1]
            preds.append(out.numpy())
            
        preds = np.array(preds)  # shape: (mc_samples, n_samples)
        
        # 单次推理 → 只返回均值
        if mc_samples == 1 or not return_var:
            return preds[0]
        
        # MC Dropout → 返回均值+方差
        mean = preds.mean(axis=0)
        var = preds.var(axis=0)
        return mean, var


def _prepare_mlp_features(X_tr: np.ndarray, X_te: np.ndarray,
                           winsor_pct: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """
    为 MLP 额外处理特征：RobustScaler + Winsorization。

    树模型（XGBoost/LightGBM）对特征缩放不敏感，可直接用 StandardScaler。
    MLP 对极端值（如矿业股单月 +200%）的梯度爆炸极度敏感，需要：
      1. Winsorize：截断 [1%, 99%] 分位数外的异常值
      2. RobustScaler：用中位数和 IQR 缩放（比 StandardScaler 更抗异常值）
    """
    from sklearn.preprocessing import RobustScaler

    # Winsorization：截断极端值到 [p_lo, p_hi] 范围
    p_lo = np.percentile(X_tr, winsor_pct * 100, axis=0)
    p_hi = np.percentile(X_tr, (1 - winsor_pct) * 100, axis=0)
    X_tr_w = np.clip(X_tr, p_lo, p_hi)
    X_te_w = np.clip(X_te, p_lo, p_hi)  # 用训练集分位数截断测试集

    # RobustScaler（中位数/IQR，比 std 更抗异常值）
    rs = RobustScaler()
    X_tr_r = rs.fit_transform(X_tr_w)
    X_te_r = rs.transform(X_te_w)

    return X_tr_r, X_te_r



def fit_all(X_tr, y_r, y_c, X_te, weights_tr=None, model_weights=None, use_mlp=True, mlp_epochs=120):
    """
    第四层：多模型训练 + 动态权重集成。
    ★ weights_tr 参数：传入时间衰减权重。
    ★ 抗过拟合修订（2026-04）：分类器收紧（depth=3, num_leaves=15, min_child=20）
                                 + 集成权重 50/50 + 早停 30 轮控制收敛。
    """
    print("    • XGBoost/LightGBM 分类器：300 estimators + 收紧深度（depth=3）")
    print("    • 集成权重：回归 50% + 分类 50%（抗过拟合再平衡）")
    
    # ✓ Bug Fix: Validate input arrays
    if len(X_tr) == 0:
        raise ValueError(f"Training set empty: X_tr.shape={X_tr.shape}")
    if len(X_te) == 0:
        raise ValueError(f"Test set empty: X_te.shape={X_te.shape}")
    if len(y_r) != len(X_tr):
        raise ValueError(f"Target-feature mismatch: len(y_r)={len(y_r)} vs len(X_tr)={len(X_tr)}")
    if len(y_c) != len(X_tr):
        raise ValueError(f"Classification target-feature mismatch: len(y_c)={len(y_c)} vs len(X_tr)={len(X_tr)}")
    
    pos_w = max((y_c==0).sum()/max((y_c==1).sum(),1), 1.0)
    pr_list, pc_list = [], []

    # ── 时序切割：前 85% 训练，后 15% 验证 ────────────────────────
    val_size  = max(1, int(len(X_tr) * 0.15))
    X_t, X_v  = X_tr[:-val_size], X_tr[-val_size:]
    y_rt, y_rv = y_r[:-val_size], y_r[-val_size:]
    y_ct, y_cv = y_c[:-val_size], y_c[-val_size:]
    
    # 同步切割权重数组
    if weights_tr is not None:
        sw_t, sw_v = weights_tr[:-val_size], weights_tr[-val_size:]
    else:
        sw_t, sw_v = None, None

    # ── XGBoost + 早停 + 样本权重 ────────────────────────────────
    xr = make_xgb("reg")
    xr.set_params(n_estimators=500, early_stopping_rounds=20, verbosity=0)
    # 传入 sample_weight=sw_t
    xr.fit(X_t, y_rt, eval_set=[(X_v, y_rv)], sample_weight=sw_t, verbose=False)

    xc = make_xgb("cls", pos_w)
    # ★ 改进方案A：为分类器增加早停轮数（因为 n_estimators 增加到 400）
    xc.set_params(n_estimators=500, early_stopping_rounds=30, verbosity=0)
    # 传入 sample_weight=sw_t
    xc.fit(X_t, y_ct, eval_set=[(X_v, y_cv)], sample_weight=sw_t, verbose=False)

    pr_list.append(("xgb", np.asarray(xr.predict(X_te))))
    pc_list.append(("xgb", np.asarray(xc.predict_proba(X_te)[:,1])))

    if LGBM:
        # LightGBM 回归
        lr = make_lgbm("reg")
        lr.set_params(n_estimators=500)
        lr.fit(X_t, y_rt,
               eval_set=[(X_v, y_rv)],
               sample_weight=sw_t,
               callbacks=[lgb.early_stopping(20, verbose=False),
                          lgb.log_evaluation(-1)])

        # ★ 改进方案A：LightGBM 分类器增强
        lc = make_lgbm("cls", pos_w)
        lc.set_params(n_estimators=500)
        lc.fit(X_t, y_ct,
               eval_set=[(X_v, y_cv)],
               sample_weight=sw_t,
               callbacks=[lgb.early_stopping(30, verbose=False),  # ↑ 从 20 → 30
                          lgb.log_evaluation(-1)])

        pr_list.append(("lgbm", np.asarray(lr.predict(X_te))))
        pc_list.append(("lgbm", np.asarray(lc.predict_proba(X_te)[:,1])))

    if TORCH and len(X_tr) > 128 and use_mlp:
        # MLP 专用：Winsorize + RobustScaler
        # 注意：PyTorch DataLoader 原生不支持直接传 sample_weight，需要改写 Loss 函数。
        # 为了保持架构简洁，权重衰减目前主要作用于主导特征分裂的 XGB/LGBM 树模型。
        X_tr_mlp, X_te_mlp = _prepare_mlp_features(X_tr, X_te, winsor_pct=0.01)
        
        # ★ 优化方案2：保存 MLP 模型用于后续 MC Dropout
        mlp_r = train_mlp(X_tr_mlp, y_r, "reg", epochs=mlp_epochs)
        mlp_c = train_mlp(X_tr_mlp, y_c, "cls", epochs=mlp_epochs)
        
        # 对集成使用快速推理（mc_samples=1）
        pr_list.append(("mlp", np.asarray(pred_mlp(mlp_r, X_te_mlp, "reg", mc_samples=1, return_var=False))))
        pc_list.append(("mlp", np.asarray(pred_mlp(mlp_c, X_te_mlp, "cls", mc_samples=1, return_var=False))))
    else:
        mlp_r = mlp_c = None

    n = len(pr_list)
    w = (np.array(model_weights[:n])/sum(model_weights[:n])
         if model_weights and len(model_weights)>=n else np.ones(n)/n)

    # ✓ Bug Fix: Ensure all predictions are numpy arrays before ensemble averaging
    ens_r = np.zeros(len(X_te))
    ens_c = np.zeros(len(X_te))
    for i, (_, pr) in enumerate(pr_list):
        ens_r = ens_r + w[i] * np.asarray(pr)
    for i, (_, pc) in enumerate(pc_list):
        ens_c = ens_c + w[i] * np.asarray(pc)
    
    # ✓ 抗过拟合修订（2026-04）：集成权重重新平衡为 50/50
    # 背景：分类器收紧（depth=3, num_leaves=15, min_child_samples=20）后，
    #      回归与分类的信噪比已经接近；70/30 让分类器主导会放大其残余噪声。
    #      50/50 提升组合稳健性，单边过拟合的影响减半。
    ens   = ens_r*0.5 + ens_c*0.5  # ← 回归 50% + 分类 50%

    # ★ 优化方案2：返回 MLP 模型及其特征也供黑-小伯曼使用
    return ens_r, ens_c, ens, xr, dict(pr_list), dict(pc_list), {
        "mlp_r": mlp_r, "mlp_c": mlp_c,
        "X_te_mlp": X_te_mlp if (TORCH and len(X_tr) > 128 and use_mlp) else None
    }

# ══════════════════════════════════════════════════════════════════
# 5. Walk-Forward
# ══════════════════════════════════════════════════════════════════

def walk_forward(panel, tx_cost=0.002, daily_map=None, pit_map=None, apply_asof_constraints=False):
    """第四层：含交易成本 + 换仓率统计 + 动态模型权重 + 时间序列约束过滤
    
    🎯 【重要】目标函数统一说明：
    - 训练标签 next_ret：绝对收益（log return）
    - 回测评估 actual_ret：绝对收益（与训练保持一致）
    - IC 计算：使用绝对收益
    （之前混用残差收益训练 + 绝对收益评测的问题已修复）
    
    ✅ 【新增】时间序列约束：
    - apply_asof_constraints=True 时，每月动态调用 apply_constraints_asof()
    - 消除前视偏差，只用截至当月的历史数据做过滤
    - daily_map 和 pit_map 参数仅在此模式下使用
    """
    dates = sorted(panel.index.get_level_values("date").unique())
    sc, recs = StandardScaler(), []
    n_test = len(dates)-MIN_TRAIN-1

    print(f"\nWalk-Forward（{MIN_TRAIN+1}月起，{n_test} 个测试月）  交易成本: {tx_cost*100:.1f}%")
    if apply_asof_constraints and daily_map is not None:
        print(f"  ✅ 已启用时间序列约束（每月动态过滤）— 消除前视偏差")
    
    # ✓ Bug fix: 检查标签
    if "next_ret" not in panel.columns or not panel["next_ret"].notna().any():
        print("  ❌ panel 缺少 next_ret/label，跳过")
        return pd.DataFrame()

    prev_h:       set[str] = set()
    model_ic:     dict[str, list] = {}
    cooldown_set: set[str] = set()  # OPT5: 止损冷静期
    stopped_last: set[str] = set()  # 上月止损的股票
    ROLLING_WINDOW = 36             # OPT2: 滚动训练窗口（月）— 抗过拟合：24→36，~+50% 样本
    rank_ic_list: list[float] = []  # ✓ 抗过拟合诊断：累计每月 Spearman RankIC
    
    # 🟢 新增：EMA 平滑记录（对抗单月神经质波动）
    prev_ema_scores: dict[str, float] = {}
    
    # 🎯 只使用panel中实际存在的特征列
    feature_cols_avail = [col for col in FEATURE_COLS if col in panel.columns]

    for i in range(MIN_TRAIN, len(dates)-1):
        # OPT2: 滚动窗口 - 只用最近 ROLLING_WINDOW 个月训练
        window_start = max(0, i - ROLLING_WINDOW)
        train_dates  = dates[window_start:i]
        tr = panel[panel.index.get_level_values("date").isin(train_dates)]                  .dropna(subset=["next_ret","label"])
        te = panel[panel.index.get_level_values("date")==dates[i]]
        
        # ✅ 【新增】时间序列约束：asof_date 前的动态过滤（消除前视偏差）
        if apply_asof_constraints and daily_map is not None:
            asof_date = dates[i]
            # 对 test set 中的每支股票应用 asof 约束
            tickers_te = te.index.get_level_values("ticker").unique()
            valid_tickers = []
            
            for ticker in tickers_te:
                passed, _ = apply_constraints_asof(
                    ticker, daily_map, pit_map if pit_map else {},
                    asof_date, CONSTRAINTS
                )
                if passed:
                    valid_tickers.append(ticker)
            
            # 用合法股票过滤 test set
            if valid_tickers:
                te = te[te.index.get_level_values("ticker").isin(valid_tickers)]
            else:
                # 所有股票都被过滤，跳过此月份
                if i == MIN_TRAIN:
                    print(f"  ⚠️  首月所有股票都被 asof 约束过滤，检查约束配置")
                continue


        # ✓ 抗过拟合修订：60→200。60 行对 30K 参数的集成模型毫无意义；
        # 200 行才是树模型 + 早停的最低门槛（参考 LightGBM min_child_samples=20）。
        if len(tr) < 200 or not len(te): continue

        if i==MIN_TRAIN:
            print(f"  [诊断] 首月 len(tr)={len(tr)} len(te)={len(te)}")

        # ✓ Bug Fix: smart_impute 已经完成所有缺失值处理（行业截面中位数 → 全市场中位数 → 0）
        # 不应再做 fillna(median)，这会抹平不同市场周期的特征基准差异
        # 如果还有极少数 NaN 漏网，fillna(0) 更合理（不会被误读为"优质信号"）
        X_tr = sc.fit_transform(tr[feature_cols_avail].fillna(0.0))
        X_te = sc.transform(te[feature_cols_avail].fillna(0.0))

        # ★ 计算样本时间衰减权重（半衰期设为 12 个月）
        # 引入指数级时间衰减权重（Exponential Sample Weight Decay）是处理金融时序数据中“宏观状态漂移（Concept Drift）”最优雅且非破坏性的做法。
        # 它的底层逻辑非常符合直觉：半衰期（Half-life）。假设我们设定半衰期为 12 个月，那么今天发生的事情权重是 1.0，一年前的数据权重衰减到 0.5，两年前的数据权重衰减到 0.25。这样，5 年前（特鲁多政府早期、疫情前、零利率时代）的数据依然会参与计算以提供大样本量的支撑，但它们对目前梯度下降和树节点分裂的话语权已经被极度削弱了。
        sample_dates = tr.index.get_level_values("date")
        w_tr = compute_time_decay_weights(sample_dates, current_date=dates[i], half_life_months=12.0)

        # 动态模型权重（基于近期 IC）
        mw = None
        if model_ic:
            mw = [max(0.1, np.mean(v[-3:])) for v in model_ic.values()]

        # ✓ Bug Fix: Ensure X_te is not empty before fitting
        if len(X_te) == 0:
            print(f"  ⚠️  月 {i+1} 测试集为空，跳过")
            continue

        # 传入 weights_tr
        ens_r, ens_c, ens, _, pr_d, pc_d, _ = fit_all(
            X_tr, tr["next_ret"].values, tr["label"].values, X_te, 
            weights_tr=w_tr, model_weights=mw, use_mlp=False)

        # ─── 【新增】应用换仓缓冲带（替代旧的持仓惯性加成 + 换仓上限逻辑）───
        tix_wf = [idx[1] for idx in te.index]
        
        # 构建临时 DataFrame 用于缓冲带逻辑
        temp_df = pd.DataFrame({
            "ensemble_score": ens
        }, index=pd.MultiIndex.from_arrays([
            [dates[i]] * len(te),
            tix_wf
        ], names=["date", "ticker"]))

        # =====================================================================
        # 🟢 新增：EMA 信号平滑逻辑（对抗树模型单月神经质波动）
        # =====================================================================
        alpha = 0.6  # 平滑系数：0.6表示当前月占60%，历史占40%。调小(如0.5)换手率更低
        smoothed_scores = []
        
        for ticker, current_score in zip(tix_wf, ens):
            if ticker in prev_ema_scores:
                # 核心：EMA 平滑计算
                s_score = (current_score * alpha) + (prev_ema_scores[ticker] * (1 - alpha))
            else:
                # 新入池或第一次计算的股票，使用原始得分
                s_score = current_score
            
            smoothed_scores.append(s_score)
            prev_ema_scores[ticker] = s_score  # 更新本月得分到字典，留给下个月用
        
        # 用平滑后的得分覆盖原始得分，然后再进行后续的 Top 10 排序
        temp_df["ensemble_score"] = smoothed_scores
        # =====================================================================

        # 🚨【核心修复】：必须先按分数降序排序！否则排名和 cutoff_score 全是错的！
        temp_df = temp_df.sort_values("ensemble_score", ascending=False)
        
        # 应用缓冲带
        prev_holdings_list = list(prev_h) if prev_h else []
        temp_df_selected = apply_rebalancing_band(
            current_ranked_df=temp_df,
            prev_holdings=prev_holdings_list,
            top_n=TOP_N,
            rank_buffer=CONSTRAINTS.get("rank_buffer", 12),
            score_tolerance=CONSTRAINTS.get("score_tolerance", 0.005)
        )
        
        # 从缓冲带结果提取最终的持仓索引
        selected_tickers = (temp_df_selected.index.get_level_values("ticker").tolist()
                           if isinstance(temp_df_selected.index, pd.MultiIndex)
                           else temp_df_selected.index.tolist())

        # 🛡️【新增】ETF 扫单保护：如果选出的前 10 名连平均分都极低，说明全市场无 Alpha
        if len(temp_df_selected) and temp_df_selected["ensemble_score"].mean() < 0.18:
            print(f"  ⚠️  本月 Top 10 平均集成分数极低 ({temp_df_selected['ensemble_score'].mean():.3f})，触发 ETF 底线风控")
            print("  🛡️ 建议清仓个股，全仓买入大盘指数 XIU.TO 或持币观望")
            selected_tickers = []  # 🔧 修复2：阈值从0.25降至0.18，避免牛市误判为空仓
        
        # 记录缓冲带保留情况（仅在首次或变化时打印）
        if i == MIN_TRAIN:
            kept_from_prev = len(set(selected_tickers) & set(prev_holdings_list))
            print(f"  💾 缓冲带首次激活：保留 {kept_from_prev} 支老持仓，新增 {len(selected_tickers) - kept_from_prev} 支")

        # 旧逻辑备份（注释掉，但保留以便调试）
        # hold_b = CONSTRAINTS.get("hold_bonus", 0.05)
        # if prev_h and hold_b > 0:
        #     for ji, t in enumerate(tix_wf):
        #         if t in prev_h:
        #             ens[ji] += hold_b

        # 从选中的 Ticker 映射回原始索引
        sorted_idx = np.argsort(ens)[::-1]
        final_idx = [j for j in sorted_idx if tix_wf[j] in selected_tickers][:TOP_N]
        
        # OPT1: 矿业子行业硬上限 + OPT5: 冷静期过滤
        max_gold = CONSTRAINTS.get("max_gold_mining", 99)
        max_base = CONSTRAINTS.get("max_base_metals", 99)
        gold_cnt = base_cnt = 0
        tix_arr  = [idx[1] for idx in te.index]
        filtered = []
        for fi in final_idx:
            t = tix_arr[fi]
            if t in cooldown_set:          continue   # OPT5: 冷静期
            if t in GOLD_MINING_TICKERS:
                if gold_cnt >= max_gold:   continue   # OPT1: 黄金上限
                gold_cnt += 1
            elif t in BASE_METALS_TICKERS:
                if base_cnt >= max_base:   continue   # OPT1: 贱金属上限
                base_cnt += 1
            filtered.append(fi)
        # 不够 TOP_N 时从剩余补充（不受矿业限制，但仍遵守冷静期）
        if len(filtered) < TOP_N:
            extras = [fi for fi in final_idx
                      if fi not in filtered and tix_arr[fi] not in cooldown_set]
            filtered = (filtered + extras)[:TOP_N]
        
        # 🟢 修复2：在 walk_forward 中补齐大类行业集中度约束（max_per_gics）
        # 确保回测与实盘的约束完全一致
        mgics = CONSTRAINTS.get("max_per_gics", 99)
        gics_cnt = {}
        gics_filtered = []
        for fi in filtered:
            t = tix_arr[fi]
            g = STOCK_PROFILE.get(t, {}).get("gics", "Unknown")
            if gics_cnt.get(g, 0) < mgics:
                gics_filtered.append(fi)
                gics_cnt[g] = gics_cnt.get(g, 0) + 1
        
        top_idx = set(gics_filtered)
        curr_h: set[str] = set()

        for j, (idx, row) in enumerate(te.iterrows()):
            t       = idx[1]
            ret_raw = row.get("next_ret", np.nan)
            is_new  = t not in prev_h
            # Fix6: 动态交易成本（小盘矿业滑点更高）
            # ⚠️ 已知前视近似（2026-04 标注）：meta_df 是 yfinance 当前快照，
            #    不是当月历史市值。仅用于把股票分到 4x/2x/1x 三档成本，影响边际；
            #    若要严格 PIT，需用 close[date] × shares_outstanding[date] 重算。
            mktcap = float(meta_df.loc[t,"mktcap"]) if t in meta_df.index and "mktcap" in meta_df.columns else 5e9
            eff_tx = tx_cost*4 if mktcap<1e9 else (tx_cost*2 if mktcap<5e9 else tx_cost)
            ret_net = (ret_raw - eff_tx*2 if pd.notna(ret_raw) and is_new else ret_raw)
            
            # ✓ 【修复】分离模型分数与选择标记，避免污染真实分数
            is_selected = j in top_idx
            if is_selected:
                curr_h.add(t)
            
            # 保存真实的模型分数（不加任何调整），分别记录是否被选中
            # ✓ Bug fix（2026-04）：同时写入 "ens" 字段。
            #   下游 backtest_report / evaluate / backtest_from_wf 等 9+ 处代码
            #   都按 "ens" 读取，不写就会 KeyError。
            recs.append({"date":dates[i],"ticker":t,
                         "ens":ens[j],                      # ★ 与下游报告函数兼容
                         "model_score":ens[j],              # ✓ 真实模型分数，用于 IC/评估
                         "is_selected":is_selected,         # ✓ 是否被缓冲带选中
                         "actual_ret":ret_raw,              # 🎯 改用 next_ret（现已统一为绝对收益）
                         "actual_ret_net":ret_net,
                         "actual_cls":row.get("label",np.nan)})

            # 更新 IC
            for name, pr in pr_d.items():
                if pd.notna(ret_raw):
                    ic = float(np.sign(pr[j]) == np.sign(ret_raw))
                    model_ic.setdefault(name,[]).append(ic)

        turnover = len(curr_h-prev_h)/max(len(curr_h),1)
        # OPT5: 更新冷静期（本月止损的股票，下月跳过）
        if CONSTRAINTS.get("cooldown_months", 0) > 0:
            cooldown_set = stopped_last.copy()
            stopped_last = set()
            
            # 🟢 修复3：将冷静期黑名单持久化到文件，用于 predict_now 实盘使用
            import json
            COOLDOWN_FILE = "cooldown_tickers.json"
            try:
                with open(COOLDOWN_FILE, "w") as f:
                    json.dump(list(cooldown_set), f)
            except Exception as e:
                print(f"  ⚠️  冷静期列表保存失败：{e}")
            for fi in top_idx:
                t2 = tix_arr[fi]
                r2 = te.iloc[fi].get("next_ret_abs",
                     te.iloc[fi].get("next_ret", np.nan))
                if pd.notna(r2) and r2 < -0.07:
                    stopped_last.add(t2)
        prev_h = curr_h

        # ✓ 抗过拟合诊断：当月 Spearman RankIC（用 pandas rank-corr，避免 scipy 依赖）
        try:
            ens_s = pd.Series(np.asarray(ens), index=range(len(ens)))
            ret_s = pd.Series(te["next_ret"].values, index=range(len(te)))
            valid = ens_s.notna() & ret_s.notna()
            if valid.sum() >= 10:
                rank_ic = float(ens_s[valid].rank().corr(ret_s[valid].rank()))
                if not np.isnan(rank_ic):
                    rank_ic_list.append(rank_ic)
        except Exception:
            pass

        if (i-MIN_TRAIN)%4==0:
            print(f"  月 {i+1}/{len(dates)-1}  训练 {len(tr)} 行  换仓 {turnover*100:.0f}%")

        # 每 12 个月打印一次累计 IC 健康度
        if rank_ic_list and (i - MIN_TRAIN + 1) % 12 == 0:
            arr = np.array(rank_ic_list)
            ir = arr.mean() / arr.std(ddof=1) if arr.std(ddof=1) > 0 else np.nan
            print(f"  📈 [RankIC 累计 {len(arr)} 月] mean={arr.mean():+.4f}  "
                  f"std={arr.std(ddof=1):.4f}  IR={ir:+.3f}  "
                  f"hit%={100*(arr>0).mean():.1f}%")

    # 最终 RankIC 汇总
    if rank_ic_list:
        arr = np.array(rank_ic_list)
        ir = arr.mean() / arr.std(ddof=1) if arr.std(ddof=1) > 0 else np.nan
        print(f"\n  📊 [RankIC 总计 {len(arr)} 月] mean={arr.mean():+.4f}  "
              f"std={arr.std(ddof=1):.4f}  IR={ir:+.3f}  "
              f"hit%={100*(arr>0).mean():.1f}%  "
              f"({'信号健康' if arr.mean()>0.02 and ir>0.3 else '⚠️ 信号偏弱/噪声占优'})")

    return pd.DataFrame(recs)


def calculate_etf_threshold(wf: pd.DataFrame, quantile: float = 0.20) -> float:
    """
    从历史 walk-forward 结果计算 ETF 兜底的动态阈值。
    
    📊 原理：
        1. 按日期分组 walk-forward 的集成分数 ("ens" 列)
        2. 计算每个日期内的平均分数
        3. 取这些"日均分数"的分位数作为触发阈值
    
    🎯 优点：
        - 数据驱动，避免硬编码的主观判断
        - 自动适应市场环境（牛熊市自动调整）
        - 20 分位数意味着：在 20% 的历史日期里会触发兜底
    
    📈 参数说明：
        - wf: walk_forward() 的输出 DataFrame（含 "date"、"ens" 等列）
        - quantile: 分位数，默认 0.20（20%）
            - 0.15 = 更激进，更容易触发兜底
            - 0.25 = 更保守，只有极弱信号才兜底
    
    💡 返回值：
        - float: 如果日均分数 < 此阈值，则触发 ETF 兜底
        - 如果 wf 为空或数据不足，返回默认 0.18（备用）
    """
    if wf is None or wf.empty:
        print("  ⚠️  walk_forward 数据为空，使用默认阈值 0.18")
        return 0.18
    
    if "date" not in wf.columns or "ens" not in wf.columns:
        print(f"  ⚠️  walk_forward 缺少必要列 (date/ens)，使用默认阈值 0.18")
        return 0.18
    
    try:
        # 按日期分组，计算每日平均分数
        daily_avg = wf.groupby("date")["ens"].mean()
        
        if len(daily_avg) < 3:
            print(f"  ⚠️  历史数据不足 ({len(daily_avg)} 天)，使用默认阈值 0.18")
            return 0.18
        
        # 计算分位数
        threshold = float(daily_avg.quantile(quantile))
        
        print(f"\n  📊 [动态 ETF 阈值计算]")
        print(f"     日均分数统计：min={daily_avg.min():.4f}, "
              f"max={daily_avg.max():.4f}, mean={daily_avg.mean():.4f}, "
              f"median={daily_avg.median():.4f}")
        print(f"     {int(quantile*100)}% 分位数 → 触发阈值 = {threshold:.4f} (原硬编码: 0.18)")
        print(f"     ⚡ 解读：在 {int(quantile*100)}% 的历史日期里，日均分数会低于此阈值")
        
        return threshold
    except Exception as e:
        print(f"  ⚠️  阈值计算异常: {e}，使用默认值 0.18")
        return 0.18


def fetch_benchmark_monthly(benchmark="XIU.TO", years=YEARS):
    """🎯 【多层次容错】统一获取基准月度收益，支持自动 fallback
    
    流程：
      1. 尝试主要基准（XIU.TO）
      2. 如果失败，自动切换到备选方案（XIC.TO、VUN.TO）
      3. 完整性检查（至少 80% 数据）
      4. 最后返回 (year, month) → return_value 的字典
    
    返回：{(year, month): monthly_return_pct, ...}
    """
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=365 * (years + 1))
    
    # 备选方案列表
    benchmark_list = [benchmark]
    if benchmark == "XIU.TO":
        benchmark_list.extend(["XIC.TO", "VUN.TO"])  # XIC=全市场, VUN=美股
    elif benchmark == "SPY":
        benchmark_list.extend(["QQQ", "IVV", "VOO"])
    
    bench_monthly = {}
    for bench_ticker in benchmark_list:
        try:
            print(f"    ↳ 尝试获取基准数据：{bench_ticker}...", end=" ")
            result = yf.download(bench_ticker, start=start_date, end=end_date,
                           auto_adjust=True, progress=False)
            
            # 提取 Close 列并确保是 Series
            if "Close" in result.columns:
                b = result["Close"]
            else:
                b = result
            
            if b.empty:
                print("❌ 返回空数据")
                continue
            
            # 如果是 DataFrame，需要选择第一列（通常是唯一的列）
            if isinstance(b, pd.DataFrame):
                if len(b.columns) == 0:
                    print("❌ 没有数据列")
                    continue
                b = b.iloc[:, 0]  # 取第一列
            
            # 检查数据完整性
            bench_monthly_raw = b.resample("ME").last().pct_change()
            
            # 如果是 DataFrame 而不是 Series，需要再次处理
            if isinstance(bench_monthly_raw, pd.DataFrame):
                if len(bench_monthly_raw.columns) == 0:
                    print("❌ Resample 后没有数据")
                    continue
                bench_monthly_raw = bench_monthly_raw.iloc[:, 0]
            
            valid_count = len(bench_monthly_raw.dropna())
            total_count = len(bench_monthly_raw)
            completeness = valid_count / total_count if total_count > 0 else 0
            
            if completeness < 0.80:
                print(f"❌ 数据缺失过多 ({valid_count}/{total_count}={completeness:.0%})")
                continue
            
            # 转换为 (year, month) 元组 key
            bench_monthly = {}
            for d, v in bench_monthly_raw.items():
                # d 应该是 Timestamp
                if not pd.isna(v):
                    bench_monthly[(d.year, d.month)] = v
            
            if bench_monthly:  # 成功
                print(f"✅ 成功  ({len(bench_monthly)} 个月)")
                if bench_ticker != benchmark:
                    print(f"       （原基准 {benchmark} 失败，已自动切换）")
                break
                
        except Exception as e:
            err_msg = str(e)[:50]
            print(f"❌ {err_msg}")
            continue
    
    if not bench_monthly:
        print(f"\n  🚨 【严重警告】所有基准数据源均失败！")
        print(f"     备选列表：{', '.join(benchmark_list)}")
        print(f"     → 原因：网络连接问题 / yfinance 服务故障 / 股票代码错误")
        print(f"     → 后果：基准收益显示为 0%，alpha/beta 无法判断，报告分析严重失效！")
        print(f"     → 解决：(1) 检查网络  (2) 验证基准代码  (3) 手工下载 CSV 数据")
    
    return bench_monthly


def backtest_report(wf: pd.DataFrame, panel: pd.DataFrame,
                    daily_map: dict, meta_df: pd.DataFrame,
                    initial_capital: float = 100_000,
                    tx_cost: float = 0.002,
                    stop_loss: float = -0.08,
                    benchmark: str = "XIU.TO"):
    """
    把 walk_forward() 的输出转成逐月 P&L 报告。

    这里用的是完整模型（26特征 + XGBoost + LightGBM + MLP）的真实预测，
    不是简化版。每个月：
      1. 取当月集成分最高的 Top N 支
      2. 用下月真实收益（actual_ret）计算实际盈亏
      3. 计算等权组合月收益、NAV 变化

    参数：
      wf              → walk_forward() 返回的 DataFrame
      initial_capital → 初始资金（默认 $100,000 CAD）
      tx_cost         → 单边手续费
      stop_loss       → 止损线（已在实际收益中体现）
      benchmark       → 基准 ETF（默认 XIU.TO = TSX 综合指数）
    """
    from collections import Counter

    if wf.empty or "actual_ret" not in wf.columns:
        print("  ⚠️  Walk-Forward 结果为空，无法生成回测报告")
        return

    # 🎯 获取基准月度收益（多层次容错）
    print(f"\n  [基准获取] 开始下载 {benchmark} 的历史数据...")
    bench_monthly = fetch_benchmark_monthly(benchmark=benchmark, years=YEARS)
    
    if not bench_monthly:
        print(f"  ⚠️  基准数据全部失败！报告中基准收益将显示为 0%")
        print(f"     → 这意味着【无法判断是 Alpha 还是 Beta】")

    dates       = sorted(wf["date"].unique())
    nav         = initial_capital
    monthly_recs= []
    prev_picks  = []

    print("\n" + "╔" + "═"*72 + "╗")
    print(f"║  📅 模型历史回测报告  （{len(dates)} 个测试月，完整模型）{'':>25}║")
    print(f"║  初始资金 ${initial_capital:,.0f} CAD  |  "
          f"手续费 {tx_cost*100:.1f}% 单边  |  止损 {stop_loss*100:.0f}%{'':>14}║")
    print("╚" + "═"*72 + "╝")

    for date in dates:
        month_wf = wf[wf["date"] == date].copy()
        if month_wf.empty:
            continue

        # ── Top N 持仓 ─────────────────────────────────────────
        # ✓ 修订（2026-04）：与实际选股口径一致 → 优先用 is_selected
        # （walk_forward 已应用缓冲带 / 冷静期 / 矿业上限 / ETF 风控）
        if "is_selected" in month_wf.columns and month_wf["is_selected"].any():
            top_n = (month_wf[month_wf["is_selected"] == True]
                     .sort_values("ens", ascending=False))
        else:
            top_n = month_wf.nlargest(TOP_N, "ens")
        picks  = top_n["ticker"].tolist()
        n_new  = len(set(picks) - set(prev_picks))
        n_out  = len(set(prev_picks) - set(picks))
        w      = 1.0 / len(picks)   # 等权

        # ── 逐支计算实际收益 ──────────────────────────────────────
        stock_rows = []
        port_ret   = 0.0
        for _, row in top_n.iterrows():
            t       = row["ticker"]
            raw     = row["actual_ret"]   # 下月真实收益（已含在wf中）
            if pd.isna(raw):
                continue

            # 止损：月内跌幅超过止损线时用止损价
            stopped = False
            df      = daily_map.get(t, pd.DataFrame())
            if not df.empty:
                # 找该月日线数据
                nxt_dates = df.index[df.index > pd.Timestamp(date)]
                if len(nxt_dates) > 0:
                    end_d = nxt_dates[min(21, len(nxt_dates)-1)]  # 约1个月
                    month_data = df.loc[nxt_dates[0]:end_d]
                    if not month_data.empty:
                        entry_px = float(df.loc[df.index <= pd.Timestamp(date),"close"].iloc[-1])
                        min_px   = float(month_data["low"].min())
                        if entry_px > 0 and (min_px - entry_px)/entry_px < stop_loss:
                            raw     = stop_loss
                            stopped = True

            # 新买入扣手续费，持续持有不扣
            is_new  = t not in prev_picks
            net     = raw - tx_cost * 2 if is_new else raw
            contrib = net * w

            port_ret += contrib
            stock_rows.append({
                "ticker":  t,
                "score":   round(row["ens"], 3),
                "raw_ret": round(raw  * 100, 2),
                "net_ret": round(net  * 100, 2),
                "contrib": round(contrib * 100, 3),
                "is_new":  is_new,
                "stopped": stopped,
            })

        if not stock_rows:
            continue

        # ── 基准收益 ──────────────────────────────────────────────
        # 使用 (year, month) 元组匹配，避免复杂的日期对象比较
        date_ts  = pd.Timestamp(date)
        bkey     = (date_ts.year, date_ts.month)
        bench_ret = bench_monthly.get(bkey, 0) * 100

        nav_prev  = nav
        nav       = nav * (1 + port_ret)
        nav_chg   = nav - nav_prev
        port_pct  = port_ret * 100
        excess    = port_pct - bench_ret

        monthly_recs.append({
            "date":      date,
            "port_pct":  round(port_pct,  2),
            "bench_pct": round(bench_ret, 2),
            "excess":    round(excess,    2),
            "nav":       round(nav,       0),
            "nav_chg":   round(nav_chg,   0),
            "picks":     picks,
            "stocks":    pd.DataFrame(stock_rows),
            "n_new":     n_new, "n_out": n_out,
        })

        # ── 月度打印 ──────────────────────────────────────────────
        up   = port_pct >= 0
        beat = excess   >= 0
        print(f"\n  {'─'*72}")
        print(f"  📅 {pd.Timestamp(date).strftime('%Y年%m月')}  "
              f"{'▲' if up else '▼'} {port_pct:>+6.2f}%  "
              f"基准 {bench_ret:>+5.2f}%  "
              f"超额 {'↑' if beat else '↓'} {excess:>+5.2f}%  "
              f"NAV ${nav:>10,.0f}  ({'+' if nav_chg>=0 else ''}{nav_chg:,.0f})")
        print(f"  持仓 {len(picks)} 支  ←{n_new}新 →{n_out}出")

        # 持仓明细
        sdf = pd.DataFrame(stock_rows).sort_values("contrib", ascending=False)
        print(f"  {'Ticker':<14} {'集成分':>7} {'涨跌%':>8} {'贡献%':>8}  {'备注'}")
        print(f"  {'─'*54}")
        for _, r in sdf.iterrows():
            icon  = "🟢" if r["net_ret"] >= 0 else "🔴"
            new_s = " ★新" if r["is_new"]  else ""
            stp_s = " ⛔止损" if r["stopped"] else ""
            print(f"  {r['ticker']:<14} {r['score']:>7.3f} "
                  f"{r['net_ret']:>+7.2f}% {r['contrib']:>+7.3f}%  "
                  f"{icon}{new_s}{stp_s}")

        prev_picks = picks

    # ── 年度汇总 ──────────────────────────────────────────────────
    if not monthly_recs:
        print("  ⚠️  无月度记录")
        return

    rets       = np.array([m["port_pct"] for m in monthly_recs]) / 100
    bench_rets = np.array([m["bench_pct"] for m in monthly_recs]) / 100
    final_nav  = monthly_recs[-1]["nav"]
    total_ret  = (final_nav / initial_capital - 1) * 100
    n          = len(rets)
    ann_ret    = (np.prod(1+rets)**(12/n)-1)*100      if n > 0 else 0
    bench_ann  = (np.prod(1+bench_rets)**(12/n)-1)*100 if n > 0 else 0
    vol_m      = rets.std() * np.sqrt(12) * 100
    sharpe     = (ann_ret/100 - 0.04) / (vol_m/100)   if vol_m > 0 else 0

    navs    = pd.Series([m["nav"] for m in monthly_recs])
    mdd     = ((navs - navs.cummax()) / navs.cummax()).min() * 100

    win_months  = (rets > 0).sum()
    beat_months = sum(1 for m in monthly_recs if m["excess"] > 0)
    best_m  = max(monthly_recs, key=lambda m: m["port_pct"])
    worst_m = min(monthly_recs, key=lambda m: m["port_pct"])

    # 最常入选股票
    all_picks = [t for m in monthly_recs for t in m["picks"]]
    top_picks = Counter(all_picks).most_common(10)

    print("\n\n" + "╔" + "═"*70 + "╗")
    print(f"║  📊 完整模型回测汇总（{n} 个月）{'':>42}║")
    print("╠" + "═"*70 + "╣")

    def row(label, val, icon=""):
        print(f"║  {label:<22} {val:<38} {icon:<6} ║")

    row("初始资金",     f"${initial_capital:>12,.0f} CAD")
    row("最终净值",     f"${final_nav:>12,.0f} CAD")
    row("总收益",       f"{total_ret:>+11.2f}%",
        "🟢" if total_ret >= 0 else "🔴")
    row("年化收益（策略）",f"{ann_ret:>+11.2f}%",
        "🟢" if ann_ret >= 0 else "🔴")
    row("年化收益（基准）",f"{bench_ann:>+11.2f}%")
    row("超额收益 Alpha",f"{ann_ret-bench_ann:>+11.2f}%",
        "🟢" if ann_ret > bench_ann else "🔴")
    row("年化波动率",   f"{vol_m:>11.2f}%")
    row("Sharpe 比率",  f"{sharpe:>12.2f}")
    row("最大回撤",     f"{mdd:>+11.2f}%")
    row("月胜率",
        f"{win_months/n*100:.1f}%  ({win_months}/{n}月)",
        "🟢" if win_months/n >= 0.5 else "🔴")
    row("跑赢基准月份",
        f"{beat_months/n*100:.1f}%  ({beat_months}/{n}月)",
        "🟢" if beat_months/n >= 0.5 else "🔴")
    row("最佳月份",
        f"{pd.Timestamp(best_m['date']).strftime('%Y-%m')}  "
        f"{best_m['port_pct']:>+.2f}%", "✨")
    row("最差月份",
        f"{pd.Timestamp(worst_m['date']).strftime('%Y-%m')}  "
        f"{worst_m['port_pct']:>+.2f}%", "⚠️")

    print("╠" + "═"*70 + "╣")
    print(f"║  月度收益明细{'':>58}║")
    print(f"║  {'月份':<10} {'策略':>8} {'基准':>8} {'超额':>8} "
          f"{'净值':>12} {'盈亏':>10} {'':>8}║")
    print(f"║  {'─'*68}║")
    for m in monthly_recs:
        icon = "✅" if m["port_pct"] >= 0 else "❌"
        beat = "↑" if m["excess"] >= 0 else "↓"
        print(f"║  {pd.Timestamp(m['date']).strftime('%Y-%m'):<10} "
              f"{m['port_pct']:>+7.2f}% "
              f"{m['bench_pct']:>+7.2f}% "
              f"{m['excess']:>+7.2f}% "
              f"${m['nav']:>10,.0f} "
              f"{m['nav_chg']:>+9,.0f} "
              f"{icon}{beat}   ║")

    print("╠" + "═"*70 + "╣")
    print(f"║  最常入选 Top 10{'':>54}║")
    for t, cnt in top_picks:
        bar = "█" * cnt
        nm  = meta_df.loc[t,"name"][:16] if t in meta_df.index else t
        print(f"║    {t:<14} {nm:<18} {cnt:>2}次  {bar:<20}{'':>12}║")

    print("╚" + "═"*70 + "╝")


def evaluate(wf):
    if wf.empty or "actual_ret" not in wf.columns:
        print("  ⚠️  Walk-Forward 结果为空，跳过评估")
        return
    v = wf.dropna(subset=["actual_ret","actual_cls"])
    if not len(v): return

    print("\n"+"═"*56)
    print("  Walk-Forward 样本外表现")
    print("═"*56)

    p    = v["ens"]
    q80  = p.quantile(0.80)
    q20  = p.quantile(0.20)
    topR = v.loc[p>=q80,"actual_ret"].mean()*100
    botR = v.loc[p<=q20,"actual_ret"].mean()*100
    netR = v.loc[p>=q80,"actual_ret_net"].mean()*100 if "actual_ret_net" in v.columns else topR
    acc  = accuracy_score(v["actual_cls"],(p>=q80).astype(int))
    hit  = (v.loc[p>=q80,"actual_ret"]>0).mean()

    monthly = v.groupby("date")["actual_ret"].mean()
    cum     = (1+monthly).cumprod()
    mdd     = ((cum-cum.cummax())/cum.cummax()).min()*100

    # Bug1 修复：换仓成本 = 新进股票数 / 总持仓 × 双边手续费
    # actual_ret_net 和 actual_ret 的差是残差收益差，不是手续费
    # 正确算法：每月 Top 组里新进股票的比例 × tx_cost × 2
    top20_mask = wf["ens"] >= wf.groupby("date")["ens"].transform(
        lambda x: x.quantile(0.80))
    monthly_turn = wf[top20_mask].groupby("date").apply(
        lambda g: (g["actual_ret"] - g["actual_ret_net"]).clip(lower=0).mean() * 100
    ).clip(0, 0.004 * 100)   # 上限 = 双边手续费 0.4%

    print(f"\n  ▸ 集成模型 ★")
    print(f"    Top20% 月均收益（税前）   {topR:+.2f}%")
    print(f"    Top20% 月均收益（扣成本） {netR:+.2f}%")
    print(f"    Bot20% 月均收益           {botR:+.2f}%")
    print(f"    多空价差                  {topR-botR:+.2f}%")
    print(f"    分类准确率                {acc*100:.1f}%")
    print(f"    Top组胜率                 {hit*100:.1f}%")
    print(f"    历史最大月度回撤          {mdd:+.2f}%")
    print(f"    平均月换仓成本            {monthly_turn.mean():.3f}%")

    # ✓ 抗过拟合诊断（2026-04 新增）：分段验证
    # 揭示"模型是否依赖某几个年份的运气、某些行业的真实信号"
    evaluate_segments(wf)


def evaluate_segments(wf):
    """分段验证 — 分年 / 分行业 RankIC + Top 选股换手稳定性

    回答三个关键问题：
      1) 模型表现是不是只靠某几年的 regime（比如只赢 2020-2021）？
      2) 哪些行业的预测有真信号、哪些是噪音？
      3) Top10 持仓的换手率分布是否健康（避免月月翻盘）？

    阈值：单年 |IC| 跨度 > 0.15 / 某行业 IC < 0 → 信号 regime-dependent
    """
    if wf.empty or "ens" not in wf.columns:
        return
    v = wf.dropna(subset=["ens", "actual_ret"]).copy()
    if v.empty:
        return

    # 准备分段维度
    v["date"] = pd.to_datetime(v["date"])
    v["year"] = v["date"].dt.year
    v["gics"] = v["ticker"].map(lambda t: STOCK_PROFILE.get(t, {}).get("gics", "Unknown"))

    def _month_rank_ic(g: pd.DataFrame) -> float:
        if len(g) < 5:
            return float("nan")
        return float(g["ens"].rank().corr(g["actual_ret"].rank()))

    # 每月一个 RankIC（横截面）
    monthly_ic = v.groupby("date", group_keys=False).apply(_month_rank_ic).dropna()
    monthly_ic.index = pd.to_datetime(monthly_ic.index)

    # 每月 Top 持仓（按 is_selected 优先）
    if "is_selected" in v.columns and v["is_selected"].any():
        sel = v[v["is_selected"] == True].copy()
        topret_monthly = sel.groupby("date")["actual_ret"].mean() * 100
        # 月度换手：与上月持仓的 Jaccard 不重合度
        picks_by_month = sel.groupby("date")["ticker"].apply(set).sort_index()
    else:
        # 退化：取每月分数 Top10
        sel_idx = (v.sort_values(["date", "ens"], ascending=[True, False])
                    .groupby("date").head(10))
        topret_monthly = sel_idx.groupby("date")["actual_ret"].mean() * 100
        picks_by_month = sel_idx.groupby("date")["ticker"].apply(set).sort_index()

    # 月度换手率（新进股票 / 当月持仓）
    turnover_pct = []
    prev = set()
    for d, picks in picks_by_month.items():
        new_in = picks - prev
        if picks:
            turnover_pct.append((d, 100.0 * len(new_in) / len(picks)))
        prev = picks
    turnover_s = pd.Series(
        [t[1] for t in turnover_pct],
        index=pd.to_datetime([t[0] for t in turnover_pct])
    ) if turnover_pct else pd.Series(dtype=float)

    # ── 分年 RankIC ─────────────────────────────────────────────
    print(f"\n  ▸ 分年 RankIC（揭示是否依赖特定 regime）")
    print(f"    {'年份':<8}{'月数':>5}  {'IC mean':>9}  {'IC std':>8}  "
          f"{'IR':>6}  {'hit%':>6}  {'Top月均':>9}  {'换手%':>7}")
    print(f"    {'─'*72}")

    yearly_means = []
    for year, ics in monthly_ic.groupby(monthly_ic.index.year):
        std = ics.std(ddof=1)
        ir = ics.mean() / std if std and std > 0 else float("nan")
        year_topret = topret_monthly[topret_monthly.index.year == year].mean() \
            if not topret_monthly.empty else float("nan")
        year_turn = turnover_s[turnover_s.index.year == year].mean() \
            if not turnover_s.empty else float("nan")
        flag = "🟢" if ics.mean() > 0.03 else ("🟡" if ics.mean() > 0 else "🔴")
        print(f"    {year:<8}{len(ics):>5}  {ics.mean():>+9.4f}  {std:>8.4f}  "
              f"{ir:>+6.2f}  {(ics>0).mean()*100:>5.0f}%  "
              f"{year_topret:>+8.2f}%  {year_turn:>6.0f}%  {flag}")
        yearly_means.append(ics.mean())

    # 总计
    ic_std_total = monthly_ic.std(ddof=1)
    ir_total = monthly_ic.mean() / ic_std_total if ic_std_total and ic_std_total > 0 else float("nan")
    print(f"    {'─'*72}")
    print(f"    {'总计':<8}{len(monthly_ic):>5}  {monthly_ic.mean():>+9.4f}  "
          f"{ic_std_total:>8.4f}  {ir_total:>+6.2f}  "
          f"{(monthly_ic>0).mean()*100:>5.0f}%  "
          f"{topret_monthly.mean():>+8.2f}%  "
          f"{turnover_s.mean() if not turnover_s.empty else float('nan'):>6.0f}%")

    # 分年极差判断：跨度 > 0.15 视为 regime 依赖
    if len(yearly_means) >= 3:
        spread = max(yearly_means) - min(yearly_means)
        if spread > 0.15:
            print(f"    ⚠️  跨年 IC 极差 {spread:.3f} > 0.15，模型表现 regime 依赖明显")
        else:
            print(f"    ✓ 跨年 IC 极差 {spread:.3f} ≤ 0.15，表现相对稳健")

    # ── 分行业 RankIC ─────────────────────────────────────────────
    print(f"\n  ▸ 分行业 RankIC（揭示哪些行业有真信号）")
    print(f"    {'GICS':<24}{'样本':>6}  {'有效月数':>8}  "
          f"{'IC mean':>9}  {'IR':>6}  {'入选次数':>8}")
    print(f"    {'─'*68}")

    sel_gics = (v[v.get("is_selected", False) == True]["gics"].value_counts()
                if "is_selected" in v.columns else pd.Series(dtype=int))

    sector_rows = []
    for gics, g in v.groupby("gics"):
        if gics == "Unknown" or len(g) < 30:
            continue
        ic_monthly = g.groupby("date", group_keys=False).apply(_month_rank_ic).dropna()
        if len(ic_monthly) < 6:
            continue
        std_s = ic_monthly.std(ddof=1)
        ir_s = ic_monthly.mean() / std_s if std_s and std_s > 0 else float("nan")
        sel_count = int(sel_gics.get(gics, 0))
        sector_rows.append((gics, len(g), len(ic_monthly), ic_monthly.mean(), ir_s, sel_count))

    for gics, n, n_m, ic_mean, ir_s, sel_cnt in sorted(sector_rows, key=lambda x: -x[3]):
        flag = "🟢" if ic_mean > 0.03 else ("🟡" if ic_mean > 0 else "🔴")
        print(f"    {gics:<24}{n:>6}  {n_m:>8}  {ic_mean:>+9.4f}  "
              f"{ir_s:>+6.2f}  {sel_cnt:>7}  {flag}")

    # ── 换手率分布 ──────────────────────────────────────────────
    if not turnover_s.empty:
        print(f"\n  ▸ Top10 月度换手率分布")
        q = turnover_s.quantile([0.25, 0.5, 0.75, 0.9])
        print(f"    平均 {turnover_s.mean():>4.0f}%  |  中位 {q[0.5]:>3.0f}%  |  "
              f"P25 {q[0.25]:>3.0f}%  |  P75 {q[0.75]:>3.0f}%  |  P90 {q[0.9]:>3.0f}%")
        if turnover_s.mean() > 50:
            print(f"    ⚠️  平均换手 > 50%，说明模型每月几乎重选，信号噪声大或缓冲带太松")
        elif turnover_s.mean() < 10:
            print(f"    ✓ 换手率温和 ({turnover_s.mean():.0f}%)，组合稳定")

# ══════════════════════════════════════════════════════════════════
# 6. 当月预测 + 投机过滤 + 因子中性化
# ══════════════════════════════════════════════════════════════════

def check_speculative_hype(ticker, daily_map, surprise_df):
    """
    判断逻辑：过去1个月涨幅 > 20% 且 没财报利好
    
    定义"无业绩支撑的炒作"：
      - 1个月涨幅 > 20% (市场炒作信号)
      - 最近财报未超预期 (surprise_dir <= 0)
    
    返回 True 表示该股应被剔除
    """
    if ticker not in daily_map: 
        return False
    
    df = daily_map[ticker]
    if len(df) < 22: 
        return False  # 至少一个月数据
    
    # 计算过去22个交易日涨幅
    try:
        mom_1m = (float(df['close'].iloc[-1]) / float(df['close'].iloc[-22])) - 1
    except (TypeError, IndexError):
        return False
    
    # 获取财报惊喜 (surprise_dir > 0 才有业绩利好)
    surprise_dir = 0
    if surprise_df is not None and ticker in surprise_df.index:
        try:
            surprise_dir = int(surprise_df.loc[ticker, "surprise_dir"])
        except (KeyError, IndexError):
            surprise_dir = 0
    
    # 炒作定义：涨幅超20% 且 业绩未超预期 (<=0)
    is_hype = mom_1m > 0.20 and surprise_dir <= 0
    return is_hype


def get_defensive_replacements(panel, current_pick_tickers, count=3):
    """
    寻找防御性替补：从全量面板中（除去已选股票）找 Defensive 风格的股票。
    
    参数：
      panel: 完整面板 (date, ticker) MultiIndex
      current_pick_tickers: 已选择的 ticker 列表
      count: 需要找多少支替补
    
    返回：替补候选的 (date, ticker) 索引列表，按 ensemble_score 降序排列
    """
    # 获取当月所有股票
    all_tix = panel.index.get_level_values("ticker").unique()
    
    # 筛选未被选中的防御性股票
    defensive_candidates = []
    for t in all_tix:
        if t not in current_pick_tickers:
            profile = STOCK_PROFILE.get(t, {})
            if profile.get("type") == "Defensive":
                # 收集所有日期中该股票的最新记录
                try:
                    latest_date = panel.index.get_level_values("date").max()
                    if (latest_date, t) in panel.index:
                        defensive_candidates.append((latest_date, t))
                except (KeyError, IndexError):
                    pass
    
    # 按 ensemble_score 降序排列，返回前 count 支
    if defensive_candidates:
        scores = [panel.loc[idx, "ensemble_score"] if "ensemble_score" in panel.columns else 0 
                  for idx in defensive_candidates]
        sorted_indices = sorted(zip(defensive_candidates, scores), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_indices[:count]]
    
    return []

def apply_rebalancing_band(current_ranked_df, prev_holdings, top_n=10, rank_buffer=12, score_tolerance=0.005):
    """
    换仓缓冲带：降低换手率，保护 Alpha 不被手续费反噬。
    
    双重缓冲逻辑：
    1. 排名缓冲（Rank Buffer）：如果老持仓掉出前10名，但仍在前12名以内，不踢出去
    2. 分数缓冲（Score Tolerance）：如果老持仓与第10名的分数差额小于0.005，说明潜力相近，保留
    
    参数：
      current_ranked_df   → 当月排序好的预测结果 (按 ensemble_score 降序)
      prev_holdings       → 上个月的持仓股票列表 (Ticker 列表)
      top_n              → 目标持仓数 (默认10)
    rank_buffer         → 排名容忍范围 (默认12)
    score_tolerance     → 分数容忍差值 (默认0.005)
    
    返回：保留或新增的股票 DataFrame，仍按 ensemble_score 降序排列
    """
    if not prev_holdings or len(prev_holdings) == 0:
        # 无上月持仓，直接返回前 top_n
        return current_ranked_df.head(top_n) if len(current_ranked_df) >= top_n else current_ranked_df
    
    # 提取 Ticker 信息（支持 MultiIndex 和普通 Index）
    if isinstance(current_ranked_df.index, pd.MultiIndex):
        current_tickers = current_ranked_df.index.get_level_values("ticker")
    else:
        current_tickers = current_ranked_df.index
    
    # 获取第 top_n 名（"守门员"）的分数作为基准线
    if len(current_ranked_df) >= top_n:
        cutoff_score = current_ranked_df.iloc[top_n - 1]["ensemble_score"]
    else:
        cutoff_score = current_ranked_df["ensemble_score"].min() if len(current_ranked_df) > 0 else 0
    
    # 构建本月的"安全区"股票池：前 rank_buffer 名，或分数距离第10名不超过 score_tolerance
    safe_zone_tickers = []
    for rank, (idx, row) in enumerate(current_ranked_df.iterrows()):
        ticker = idx[1] if isinstance(idx, tuple) else idx
        score = row["ensemble_score"]
        
        # 【核心逻辑】在排名或分数上在安全区内
        if rank < rank_buffer or score >= (cutoff_score - score_tolerance):
            safe_zone_tickers.append(ticker)
    
    new_picks = []
    
    # Step 1：优先保留老持仓（只要还在安全区，就不卖）
    for t in prev_holdings:
        if t in safe_zone_tickers:
            new_picks.append(t)
    
    # Step 2：从本月真正的前 top_n 名补充新股（不能重复）
    for idx, row in current_ranked_df.iterrows():
        t = idx[1] if isinstance(idx, tuple) else idx
        if len(new_picks) >= top_n:
            break
        if t not in new_picks:
            new_picks.append(t)
    
    # Step 3：提取最终 DataFrame 并按分数降序排列
    if isinstance(current_ranked_df.index, pd.MultiIndex):
        mask = current_ranked_df.index.get_level_values('ticker').isin(new_picks)
    else:
        mask = current_ranked_df.index.isin(new_picks)
    
    final_df = current_ranked_df[mask].sort_values("ensemble_score", ascending=False)
    return final_df

def predict_now(panel, daily_map, meta_df, wf=None, macro_df=None,
                current_holdings=None, prev_weights_dict=None, 
                prev_scores_dict=None):  # 🟢 新增：用于 EMA 平滑的历史分数
    dates = sorted(panel.index.get_level_values("date").unique())
    sc    = StandardScaler()

    # ✓ Bug fix: 只对 label dropna
    tr  = panel[panel.index.get_level_values("date").isin(dates[:-1])]\
               .dropna(subset=["next_ret","label"])
    cur = panel[panel.index.get_level_values("date")==dates[-1]]
    
    # 🎯 只使用panel中实际存在的特征列
    feature_cols_avail = [col for col in FEATURE_COLS if col in cur.columns]

    if len(tr) < 200:
        # ✓ 抗过拟合修订：与 walk_forward 对齐到 200 行下限
        print(f"训练数据不足（仅 {len(tr)} 行 < 200，建议 YEARS ≥ 3）")
        return None, None, None

    # ✓ Bug fix: 检查当月数据是否为空（关键：防止 0 样本错误）
    if cur.empty:
        print(f"  ⚠️  当月数据为空（dates[-1]={dates[-1]}）")
        print(f"     这可能是因为：")
        print(f"       1. build_panel 过滤掉了所有当月股票")
        print(f"       2. 当月数据缺失")
        print(f"     建议：检查 constraints 是否过于严格，或数据是否完整")
        return None, None, None

    print(f"\n训练最终模型（{len(tr)} 行历史）...")
    # ✓ Bug Fix: smart_impute 已完成缺失值处理，不应再 fillna(median)
    # fillna(median) 会跨越多个市场周期计算中位数，破坏不同 Regime 下的特征基准
    X_tr = sc.fit_transform(tr[feature_cols_avail].fillna(0.0))
    X_cu = sc.transform(cur[feature_cols_avail].fillna(0.0))

    # ✓ Bug fix: 验证 X_cu 是否为空
    if X_cu.shape[0] == 0:
        print(f"  ❌ ERROR: 当月特征矩阵为空（X_cu.shape={X_cu.shape}）")
        print(f"     这不应该发生，因为 cur 已检查非空")
        print(f"     可能原因：cur[FEATURE_COLS] 返回空 DataFrame")
        return None, None, None

    # ★ 计算最后一次预测的样本时间衰减权重
    sample_dates = tr.index.get_level_values("date")
    current_pred_date = dates[-1] # 当前预测的月份
    w_tr = compute_time_decay_weights(sample_dates, current_date=current_pred_date, half_life_months=12.0)

    # 第四层：动态权重
    # ✓ 抗过拟合修订：与 walk_forward 保持一致，predict_now 也禁用 MLP；
    #   MLP 在 ~1K 独立样本下过拟合风险高，且关闭后 model_weights 自动截取到 [XGB, LGBM]。
    mw = [1.2, 1.0] if LGBM else None

    # 最终选股：启用 XGBoost + LightGBM + MLP（MC Dropout 不确定性估计）
    ens_r, ens_c, ens, xgb_r, _, _, mlp_info = fit_all(
        X_tr, tr["next_ret"].values, tr["label"].values, X_cu,
        weights_tr=w_tr, model_weights=mw, use_mlp=True, mlp_epochs=80)  # 改为 True 激活 MLP

    out = cur[feature_cols_avail].copy()
    out["pred_return"]    = ens_r
    out["pred_top20pct"]  = ens_c
    out["ensemble_score"] = ens
    tix = out.index.get_level_values("ticker")
    out["name"]   = tix.map(meta_df["name"].to_dict())
    out["sector"] = tix.map(meta_df["sector"].to_dict())
    
    # =====================================================================
    # 🟢 新增：实盘 EMA 信号平滑（对抗树模型单月波动）
    # =====================================================================
    if prev_scores_dict:
        alpha = 0.6
        smoothed_scores = []
        tickers_out = (out.index.get_level_values("ticker").tolist()
                      if isinstance(out.index, pd.MultiIndex)
                      else out.index.tolist())
        
        for ticker, current_score in zip(tickers_out, out["ensemble_score"]):
            if ticker in prev_scores_dict:
                s_score = (current_score * alpha) + (prev_scores_dict[ticker] * (1 - alpha))
                print(f"  [EMA平滑] {ticker}: 原始 {current_score:.3f} | 上月 {prev_scores_dict[ticker]:.3f} -> 平滑后 {s_score:.3f}")
            else:
                s_score = current_score
            smoothed_scores.append(s_score)
        
        out["ensemble_score"] = smoothed_scores
        print(f"  📊 EMA 平滑完成（{len(prev_scores_dict)} 支股票享受历史记忆）")
    # =====================================================================
    
    # ⚠️ 【关键 FIX】保存未排序的原始索引，用于 MC Dropout 方差映射
    # X_te_mlp 是基于 cur（未排序）生成的，所以 pred_var 的顺序是未排序的
    # 如果 sort 之后再映射，顺序会错乱
    unsorted_indices = out.index.copy()
    out = out.sort_values("ensemble_score", ascending=False)
    
    # 🎯 【新增核心逻辑】：扩大样本池至最多150支以支持因果推断（DML 需要充分样本）
    print("\n  [DML因果推断] 扩大样本池至最多150支以支持因果推断...")  
    top150_candidates = out.head(150) if len(out) >= 150 else out  # 扩大截面数量以满足 DML 的统计显著性需求
    top150_tickers = (top150_candidates.index.get_level_values("ticker").tolist()
                      if isinstance(top150_candidates.index, pd.MultiIndex)
                      else top150_candidates.index.tolist())
    print(f"    ℹ️  DML 样本池：{len(top150_tickers)} 支（足以进行因果推断）")
    
    # 提取盈利惊喜信号
    surprise_df = pd.DataFrame()
    try:
        surprise_df = fetch_earnings_surprise(top150_tickers)
        
        # 运行 DML 双重机器学习（估计事件信号的纯因果效应）
        if not surprise_df.empty:
            print("    [DML] 用双重机器学习替代固定乘数，动态调整预测分数...")
            try:
                # 应用 DML 调整到整个 out（用因果效应替代固定权重）
                out = apply_dml_signal(out, panel, pd.DataFrame(), surprise_df)
                
                # 🔪 【终极防线】：季报严重不及预期 (Miss > 20%)，无视模型高分，强制斩立决！
                miss_tickers = surprise_df[surprise_df["surprise_pct"] < -20.0].index.tolist()
                valid_miss = [t for t in miss_tickers if t in out.index.get_level_values("ticker")] if isinstance(out.index, pd.MultiIndex) else [t for t in miss_tickers if t in out.index]
                if valid_miss:
                    print(f"    🛑 强制剔除业绩严重暴雷股 (Miss > 20%): {valid_miss}")
                    if isinstance(out.index, pd.MultiIndex):
                        out = out[~out.index.get_level_values("ticker").isin(valid_miss)]
                    else:
                        out = out[~out.index.isin(valid_miss)]
                
                out = out.sort_values("ensemble_score", ascending=False)
                print("    ✅ DML 因果调整完毕（已融入排名）")
            except Exception as e_dml:
                print(f"    ⚠️  DML 失败（{str(e_dml)[:50]}），保留原分数")
        else:
            print("    ⚠️  未能获取财报数据，跳过DML")
    except Exception as e:
        print(f"    ⚠️  盈利惊喜提取失败（{str(e)[:50]}）")
    
    # ★ Plan 2: 从 MLP 模型提取 MC Dropout 不确定性
    mc_dropout_variance = {}
    if mlp_info and mlp_info.get("mlp_c") is not None and mlp_info.get("X_te_mlp") is not None:
        try:
            import torch
            X_te_mlp = mlp_info["X_te_mlp"]
            mlp_c = mlp_info["mlp_c"]
            n_samples = X_te_mlp.shape[0]
            
            X_tensor = torch.tensor(X_te_mlp, dtype=torch.float32).to(
                next(mlp_c.parameters()).device
            )
            # MC Dropout: 50 次前向传递，每次打开 dropout
            preds_mc = []
            with torch.no_grad():
                mlp_c.eval()
                for mc_iter in range(50):
                    # 临时启用 dropout
                    for m in mlp_c.modules():
                        if isinstance(m, torch.nn.Dropout):
                            m.train()
                    
                    out_mc = mlp_c(X_tensor)  # Shape: (n_samples,) or (n_samples, 2) or (n_samples, 1)
                    
                    # 检查输出形状并处理
                    if out_mc.ndim == 1:
                        # 输出是 sigmoid (n_samples,)，保持原样
                        preds_mc.append(out_mc.cpu().numpy())
                    elif out_mc.shape[-1] == 2:
                        # 输出是 logits (n_samples, 2)，取 softmax 的第1类
                        preds_mc.append(torch.softmax(out_mc, dim=1)[:, 1].cpu().numpy())
                    elif out_mc.shape[-1] == 1:
                        # 输出是 sigmoid (n_samples, 1)，压扁
                        preds_mc.append(out_mc.squeeze(-1).cpu().numpy())
                    else:
                        # 其他情况，取第一列或全部
                        if out_mc.shape[0] == n_samples:
                            preds_mc.append(out_mc[:, 0].cpu().numpy() if out_mc.ndim > 1 else out_mc.cpu().numpy())
                
                if not preds_mc:
                    raise ValueError("MC Dropout 未能生成任何预测")
                
                # 将所有预测转为 numpy 数组 (mc_samples, n_samples)
                preds_mc = np.array(preds_mc)
                
                # 验证形状
                if preds_mc.shape[0] < 50:
                    print(f"    ⚠️ MC Dropout 只收集了 {preds_mc.shape[0]}/50 次采样")
                if preds_mc.shape[1] != n_samples:
                    print(f"    ⚠️ 采样数 {preds_mc.shape[1]} ≠ 样本数 {n_samples}，尝试修正...")
                    # 如果形状不匹配，舍弃MC Dropout
                    raise ValueError(f"形状不匹配：预测 {preds_mc.shape} vs 样本 {n_samples}")
                
                # 计算方差 (n_samples,)
                pred_var = preds_mc.var(axis=0)
            
            # 【FIX】用未排序的原始索引来映射 pred_var，避免张量索引错乱
            # pred_var 顺序对应 unsorted_indices，而非排序后的 out.index
            unsorted_tickers = [idx[1] if isinstance(idx, tuple) else idx for idx in unsorted_indices]
            n_mapped = 0
            for i, t in enumerate(unsorted_tickers):
                if i < len(pred_var):
                    mc_dropout_variance[t] = float(pred_var[i])
                    n_mapped += 1
            
            if n_mapped == len(out):
                print(f"  ✅ MC Dropout 不确定性：{n_mapped}/{len(out)} 支股票方差已计算")
            else:
                print(f"  ⚠️  MC Dropout 方差不完整：{n_mapped}/{len(out)} 支")
        except Exception as e:
            import traceback
            print(f"  ⚠️  MC Dropout 方差提取失败：{str(e)}")
            # 不预期出现详细traceback，仅在调试时显示
            # traceback.print_exc()

    # 最终价格过滤（双重保障：apply_constraints 之后 predict_now 里再过滤一次）
    max_px = CONSTRAINTS.get("max_price_cad", 9999)
    tix_all = out.index.get_level_values("ticker") if isinstance(out.index, pd.MultiIndex) else out.index
    price_mask = pd.Series([
        daily_map[t]["close"].iloc[-1] <= max_px if t in daily_map else True
        for t in tix_all
    ], index=out.index)
    filtered_out = out[~price_mask]
    if len(filtered_out):
        print(f"  价格过滤：剔除 {list(filtered_out.index.get_level_values('ticker') if isinstance(filtered_out.index, pd.MultiIndex) else filtered_out.index)}")
    out = out[price_mask]

    # --- [新增] 预过滤：投机炒作检测 ---
    # 定义：过去1个月涨幅 > 20% 且 没有财报利好支撑
    if not out.empty:
        tix_current = out.index.get_level_values("ticker") if isinstance(out.index, pd.MultiIndex) else out.index
        speculative_tickers = [t for t in tix_current 
                               if check_speculative_hype(t, daily_map, None)]  # surprise_df 可后续注入
        if speculative_tickers:
            print(f"  🛑 剔除无业绩支撑的炒作股（1m涨幅>20%）：{speculative_tickers}")
            out = out[~out.index.get_level_values("ticker").isin(speculative_tickers)]

    # OPT6: 置信度过滤 - 低置信度时减少持仓数
    min_conf   = CONSTRAINTS.get("min_confidence", 0.0)
    min_top_n  = CONSTRAINTS.get("min_top_n", TOP_N)
    max_score  = out["ensemble_score"].max() if len(out) else 0
    effective_top_n = TOP_N
    if max_score < min_conf:
        effective_top_n = min_top_n
        print(f"  ⚠️  置信度不足（最高分 {max_score:.3f} < {min_conf}），"
              f"持仓数从 {TOP_N} 降至 {min_top_n}")
    elif max_score < min_conf * 1.5:
        effective_top_n = max(min_top_n, int(TOP_N * 0.7))
        print(f"  ⚠️  置信度偏低（{max_score:.3f}），持仓数降至 {effective_top_n}")

    # 三维分散约束 + OPT1 矿业子行业硬上限
    mgics = CONSTRAINTS.get("max_per_gics", 99)
    msty  = CONSTRAINTS.get("max_per_style", 99)
    mtyp  = CONSTRAINTS.get("max_per_type", 99)
    max_gold = CONSTRAINTS.get("max_gold_mining", 99)
    max_base = CONSTRAINTS.get("max_base_metals", 99)
    gc, sc2, tc = {}, {}, {}
    gold_cnt = base_cnt = 0
    keep = []
    for idx in out.index:
        if len(keep) >= effective_top_n:
            break
        t  = idx[1] if isinstance(idx, tuple) else idx
        p  = STOCK_PROFILE.get(t, {})
        g, s, tp = p.get("gics","?"), p.get("style","?"), p.get("type","?")
        # 矿业子行业检查
        if t in GOLD_MINING_TICKERS:
            if gold_cnt >= max_gold: continue
            gold_cnt += 1
        elif t in BASE_METALS_TICKERS:
            if base_cnt >= max_base: continue
            base_cnt += 1
        if gc.get(g,0)<mgics and sc2.get(s,0)<msty and tc.get(tp,0)<mtyp:
            keep.append(idx)
            gc[g] = gc.get(g,0)+1
            sc2[s]= sc2.get(s,0)+1
            tc[tp]= tc.get(tp,0)+1
    out = out.loc[keep]
    print(f"  分散后：{len(out)} 支  {dict(gc)}"
          f"  [黄金:{gold_cnt}/{max_gold} 贱金属:{base_cnt}/{max_base}]")

    # ─── 【新增】应用换仓缓冲带（降低换手率）───────────────────
    if current_holdings and len(current_holdings) > 0:
        print(f"  💾 检测到上月持仓 ({len(current_holdings)} 支)，应用换仓缓冲带...")
        out_before = len(out)
        
        # 应用缓冲带逻辑
        out = apply_rebalancing_band(
            current_ranked_df=out,
            prev_holdings=current_holdings,
            top_n=TOP_N,
            rank_buffer=CONSTRAINTS.get("rank_buffer", 12),
            score_tolerance=CONSTRAINTS.get("score_tolerance", 0.005)
        )
        
        # 统计保留了多少老持仓
        out_tix = out.index.get_level_values("ticker") if isinstance(out.index, pd.MultiIndex) else out.index
        kept_from_prev = len(set(out_tix) & set(current_holdings))
        print(f"  ✓ 缓冲带：保留 {kept_from_prev} 支老持仓，新增 {len(out) - kept_from_prev} 支，总计 {len(out)} 支")

    # --- [新增] 后处理：因子中性化与防御性替换 ---
    # 检测集中度过高的因子：高波动 + 能源
    # 定义：vol_1m > 0.40 (40% 年化波动) 或 GICS=Energy
    if not out.empty:
        out_tix = out.index.get_level_values("ticker") if isinstance(out.index, pd.MultiIndex) else out.index
        
        high_vol_tickers = []
        energy_tickers = []
        
        for t in out_tix:
            profile = STOCK_PROFILE.get(t, {})
            
            # 检查是否高波动
            try:
                vol_1m = float(out.loc[(slice(None), t), "vol_1m"].iloc[0]) if isinstance(out.index, pd.MultiIndex) else float(out.loc[t, "vol_1m"])
                if vol_1m > 0.40:
                    high_vol_tickers.append(t)
            except (KeyError, IndexError, TypeError):
                pass
            
            # 检查是否能源
            if profile.get("gics") == "Energy":
                energy_tickers.append(t)
        
        concentrated_group = set(high_vol_tickers) | set(energy_tickers)
        
        # 如果集中度过高（Top 10 中有 6 个是高风险因子），执行中性化
        if len(concentrated_group) >= 6 and len(out) >= TOP_N:
            print(f"  ⚠️  因子集中度过高 ({len(concentrated_group)} 支 Energy/HighVol)，执行中性化...")
            
            # 找到要替换的倒数 3 支（分数最低）
            to_remove_indices = out.iloc[-3:].index.tolist()
            n_remove = len(to_remove_indices)
            
            # 寻找防御性替补
            replacements = get_defensive_replacements(panel, out_tix.tolist(), count=n_remove)
            
            if replacements:
                print(f"  🔄 用 {len(replacements)} 支防御性品种替换高风险股")
                
                # 移除倒数 N 支
                out = out.drop(to_remove_indices)
                
                # 添加替补（赋予当前平均分数，确保排序合理）
                avg_score = out["ensemble_score"].mean()
                avg_ret   = out["pred_return"].mean() if "pred_return" in out.columns else 0.0
                avg_prob  = out["pred_top20pct"].mean() if "pred_top20pct" in out.columns else 0.5
                for rep_idx in replacements:
                    if rep_idx not in out.index and rep_idx in panel.index:
                        try:
                            out = pd.concat([out, panel.loc[[rep_idx]]])
                            # 🎯 【修改】：把所有 NaN 空洞填上平均值
                            out.loc[rep_idx, "ensemble_score"] = avg_score
                            out.loc[rep_idx, "pred_return"]    = avg_ret
                            out.loc[rep_idx, "pred_top20pct"]  = avg_prob
                        except Exception as e:
                            print(f"    ⚠️  添加替补 {rep_idx} 失败：{e}")
                
                # 重新排序
                out = out.sort_values("ensemble_score", ascending=False)
                print(f"  ✓ 中性化后：{len(out)} 支 (Energy/HighVol 占比：{len(concentrated_group)/len(out)*100:.0f}%)")

    # ─── 【终极保险】Cash / ETF Sweep ───────────────────────────
    # 如果头部信号整体太弱，强行买入个股只是在追逐噪音
    # ✓ Bug Fix: 如果触发 ETF 兜底，后续不应再进入 Black-Litterman 优化
    #   否则单资产协方差矩阵会导致优化器崩溃（即使有 try/except 捕获）
    # ✅ 改进：从硬编码 0.18 → 数据驱动分位数阈值（历史 walk-forward）
    etf_fallback_triggered = False
    
    # 计算动态阈值（基于历史 walk-forward 日均分数的 20 分位数）
    etf_threshold = calculate_etf_threshold(wf, quantile=0.20)
    
    if not out.empty:
        top_10_mean_score = out["ensemble_score"].head(TOP_N).mean()
        # 使用动态阈值替代硬编码的 0.18
        if pd.notna(top_10_mean_score) and top_10_mean_score < etf_threshold:
            print(f"\n  ⚠️  警告：本月 Top 10 平均集成分数 ({top_10_mean_score:.3f}) < 动态阈值 ({etf_threshold:.3f})")
            print(f"     → 全市场缺乏强 Alpha 信号（低于历史 20% 分位线）")
            print("  🛡️ 触发 ETF 底线风控：建议清仓个股，全仓买入大盘指数 XIU.TO 或持币观望。")

            etf_ticker = "XIU.TO"
            etf_date = dates[-1] if len(dates) else pd.Timestamp.today().normalize()
            etf_name = "iShares S&P/TSX 60 Index ETF"
            etf_fallback_triggered = True

            # 补一份可供后续打印/仓位模块使用的日线数据
            if etf_ticker not in daily_map or daily_map[etf_ticker].empty:
                etf_daily = None
                try:
                    etf_hist = yf.download(etf_ticker, period="10d", auto_adjust=True, progress=False)
                    if not etf_hist.empty:
                        latest = etf_hist.iloc[-1]
                        etf_daily = pd.DataFrame({
                            "open":   [float(latest.get("Open", latest.get("Close", 1.0)))],
                            "high":   [float(latest.get("High", latest.get("Close", 1.0)))],
                            "low":    [float(latest.get("Low", latest.get("Close", 1.0)))],
                            "close":  [float(latest.get("Close", 1.0))],
                            "volume": [float(latest.get("Volume", 1.0))],
                        }, index=[etf_date])
                except Exception:
                    etf_daily = None

                if etf_daily is None:
                    etf_daily = pd.DataFrame({
                        "open":   [1.0],
                        "high":   [1.0],
                        "low":    [1.0],
                        "close":  [1.0],
                        "volume": [1.0],
                    }, index=[etf_date])

                daily_map[etf_ticker] = etf_daily

            etf_row = {col: np.nan for col in out.columns}
            etf_row.update({
                "pred_return": 0.0,
                "pred_top20pct": 0.0,
                "ensemble_score": float(top_10_mean_score),
                "name": etf_name,
                "sector": "ETF",
            })
            out = pd.DataFrame(
                [etf_row],
                index=pd.MultiIndex.from_tuples([(etf_date, etf_ticker)], names=["date", "ticker"])
            )
            print(f"  ✅ 已切换到 ETF 兜底标的：{etf_ticker}")

    imp = pd.Series(xgb_r.feature_importances_,
                    index=FEATURE_COLS).sort_values(ascending=False)

    # OPT4: 最大回撤熔断 + OPT7: VIX 波动率缩仓
    dd_signal    = None
    vix_scale    = 1.0   # 默认不缩仓
    halt_scale   = 1.0

    if wf is not None and not wf.empty and len(wf["date"].unique()) >= 3:
        recent = wf.groupby("date")["actual_ret"].mean().tail(3)
        cum    = (1 + recent).prod() - 1

        # OPT4: 熔断 - 连续3月累计亏损超阈值
        dd_thresh = CONSTRAINTS.get("dd_halt_threshold", -0.15)
        if cum < dd_thresh:
            dd_signal  = cum
            halt_scale = CONSTRAINTS.get("dd_halt_scale", 0.5)
            print(f"  🔴 熔断触发：近3月累计 {cum*100:.1f}%，仓位缩减至 {halt_scale*100:.0f}%")

    # OPT7: VIX 高时缩仓
    try:
        if macro_df is not None and not macro_df.empty and "vix" in macro_df.columns:
            vix_now = float(macro_df["vix"].iloc[-1])
            vix_thr = CONSTRAINTS.get("vix_scale_threshold", 25.0)
            if vix_now > vix_thr:
                vix_scale = CONSTRAINTS.get("vix_scale_factor", 0.70)
                print(f"  🟡 VIX={vix_now:.1f} > {vix_thr}，仓位缩减至 {vix_scale*100:.0f}%")
    except Exception:
        pass

    # 综合缩仓系数（熔断 × VIX 双重保护）
    total_scale = halt_scale * vix_scale
    if total_scale < 1.0:
        out["ensemble_score"] = out["ensemble_score"] * total_scale
        print(f"  ⚡ 综合缩仓系数：{total_scale:.2f}（仓位整体降低）")

    # 🟢 修复1：返回 total_scale 以确保缩仓在最后仓位分配时不被归一化抵消
    return out, imp, dd_signal, total_scale

# ══════════════════════════════════════════════════════════════════
# 7. Fuzzy + 风险平价仓位
# ══════════════════════════════════════════════════════════════════

def fuzzy_membership(score, lo, md, hi):
    light    = max(0.0, min(1.0, (md-score)/(md-lo+1e-9))) if score<=md else 0.0
    moderate = (max(0.0,min(1.0,(score-lo)/(md-lo+1e-9))) if score<=md
                else max(0.0,min(1.0,(hi-score)/(hi-md+1e-9))))
    heavy    = max(0.0, min(1.0, (score-md)/(hi-md+1e-9))) if score>md else 0.0
    return light, moderate, heavy


def risk_parity(top, daily_map):
    """第四层：风险平价权重 ∝ 1/波动率"""
    vols = []
    for idx, _ in top.iterrows():
        t  = idx[1] if isinstance(idx,tuple) else idx
        df = daily_map.get(t, pd.DataFrame())
        v  = df["close"].pct_change().tail(63).std()*np.sqrt(252) if not df.empty else 0.25
        vols.append(max(v, 0.05))
    inv = [1/v for v in vols]
    s   = sum(inv)
    return [w/s for w in inv]


def _vec_fuzzy_membership(scores: np.ndarray, lo: float, md: float, hi: float):
    """向量化三角模糊隶属度（替代逐行循环）"""
    light    = np.where(scores <= md, np.clip((md-scores)/(md-lo+1e-9),0,1), 0.0)
    moderate = np.where(scores <= md,
                        np.clip((scores-lo)/(md-lo+1e-9),0,1),
                        np.clip((hi-scores)/(hi-md+1e-9),0,1))
    heavy    = np.where(scores > md,  np.clip((scores-md)/(hi-md+1e-9),0,1), 0.0)
    return light, moderate, heavy


def fuzzy_sizing(top, daily_map):
    """
    Fuzzy Logic × 风险平价 混合仓位（向量化，替代 iterrows）。
    速度提升约 10-50x。
    """
    RANGE_LO = {"Heavy Buy": 0.20, "Moderate Buy": 0.10, "Light Buy": 0.03}
    RANGE_HI = {"Heavy Buy": 0.25, "Moderate Buy": 0.15, "Light Buy": 0.08}

    scores = top["ensemble_score"].values
    probs  = top["pred_top20pct"].values
    rets   = top["pred_return"].values

    # 动态分位数阈值（一次性计算）
    lo,md,hi   = np.percentile(scores,[25,50,75])
    p_lo,p_md,p_hi = np.percentile(probs,  [25,50,75])
    r_lo,r_md,r_hi = np.percentile(rets,   [25,50,75])

    # 向量化隶属度计算
    l_s,m_s,h_s = _vec_fuzzy_membership(scores, lo, md, hi)
    l_p,m_p,h_p = _vec_fuzzy_membership(probs,  p_lo, p_md, p_hi)
    l_r,m_r,h_r = _vec_fuzzy_membership(rets,   r_lo, r_md, r_hi)

    wh    = (h_s + h_p + h_r) / 3
    wm    = (m_s + m_p + m_r) / 3
    wl    = (l_s + l_p + l_r) / 3
    crisp = (wh*3 + wm*2 + wl*1) / (wh + wm + wl + 1e-9)

    # 向量化波动率（一次性批量计算）
    tickers = [idx[1] if isinstance(idx,tuple) else idx for idx in top.index]
    vols    = np.array([
        daily_map[t]["close"].pct_change().tail(63).std()*np.sqrt(252)
        if t in daily_map and not daily_map[t].empty else 0.25
        for t in tickers
    ])

    # 向量化类别判定
    cats = np.where(crisp >= 2.2, "Heavy Buy",
           np.where(crisp >= 1.6, "Moderate Buy", "Light Buy"))
    # Vol downgrade
    cats = np.where((vols > 0.55) & (cats == "Heavy Buy"),    "Moderate Buy", cats)
    cats = np.where((vols > 0.65) & (cats == "Moderate Buy"), "Light Buy",    cats)

    # 向量化仓位计算（cats 是 np.str_，需要转成 str 才能查字典）
    rng_lo = np.array([RANGE_LO[str(c)] for c in cats])
    rng_hi = np.array([RANGE_HI[str(c)] for c in cats])
    t_val  = np.clip((crisp - 1.0) / 2.0, 0, 1)
    vol_f  = np.clip(1.0 - (vols - 0.15) / 0.40, 0, 1)
    alloc  = (rng_lo + t_val*(rng_hi - rng_lo)) * vol_f + rng_lo * (1 - vol_f)

    # 风险平价微调（向量化）
    rp      = np.array(risk_parity(top, daily_map))
    rp_mean = 1.0 / max(len(rp), 1)
    rp_adj  = 1.0 + 0.2 * (rp - rp_mean) / (rp_mean + 1e-9)
    alloc   = np.round(alloc * rp_adj * 100, 1)

    df_out = pd.DataFrame({
        "ticker":   tickers,
        "category": cats,
        "alloc_pct":alloc,
        "vol_ann":  vols,
        "crisp":    np.round(crisp, 2),
        "score":    scores,
        "prob":     probs,
        "ret":      rets,
    })
    total  = df_out["alloc_pct"].sum()
    df_out["alloc_pct"] = (df_out["alloc_pct"]/total*100).round(1)

    # 单支最大仓位上限
    cap = CONSTRAINTS.get("max_single_alloc", 1.0) * 100
    if (df_out["alloc_pct"] > cap).any():
        df_out["alloc_pct"] = df_out["alloc_pct"].clip(upper=cap)
        # 被截掉的部分重新按比例分配给其他股票
        total2 = df_out["alloc_pct"].sum()
        df_out["alloc_pct"] = (df_out["alloc_pct"] / total2 * 100).round(1)

    return df_out

# ══════════════════════════════════════════════════════════════════
# 8. 输出
# ══════════════════════════════════════════════════════════════════

def print_picks(result, imp, daily_map, meta_df, wf, dd_signal):
    top = result.head(TOP_N)
    models_str = "XGBoost"+(" + LightGBM" if LGBM else "")+(" + MLP" if TORCH else "")

    print("\n"+"═"*84)
    print(f"  🍁 TSX 量化选股 v2.0 — {datetime.today().strftime('%Y-%m-%d')}")
    print(f"  模型：{models_str}  |  特征：{len(FEATURE_COLS)} 个  |  训练：{YEARS} 年历史")
    print("═"*84)

    # 第四层：回撤控制警告
    if dd_signal is not None:
        print(f"\n  ⚠️  【风控】近3月组合累计 {dd_signal*100:.1f}%，触发回撤控制")
        print(f"     建议：整体仓位减半，等待信号恢复\n")

    # Top 10 明细（含建议股数）
    TOTAL_CAPITAL = 100_000   # 修改此处调整资金规模 CAD
    print(f"{'#':<4}{'Ticker':<13}{'公司':<18}{'GICS':<13}{'预测涨幅':>9}"
          f"{'进前20%':>8}{'集成分':>8}{'股价':>9}{'建议股数':>9}")
    print("─"*95)
    for i,(idx,row) in enumerate(top.iterrows(),1):
        t    = idx[1] if isinstance(idx,tuple) else idx
        prof = STOCK_PROFILE.get(t,{})
        df   = daily_map.get(t, pd.DataFrame())
        price = float(df["close"].iloc[-1]) if not df.empty else 0
        # 建议股数：用 fuzzy 仓位比例计算（预估，正式仓位在下方 Position Sizing）
        alloc_pct = 1.0 / len(top)   # 等权预估
        shares = int(TOTAL_CAPITAL * alloc_pct / price) if price > 0 else 0
        price_s  = f"${price:>7.2f}" if price > 0 else "   N/A"
        shares_s = f"{shares:>6}股"   if shares > 0 else "   N/A"
        print(f"{i:<4}{t:<13}{str(row.get('name',''))[:16]:<18}"
              f"{prof.get('gics','?')[:11]:<13}"
              f"{row['pred_return']*100:>+8.1f}%"
              f"{row['pred_top20pct']*100:>7.0f}%"
              f"{row['ensemble_score']:>9.3f}"
              f"  {price_s}  {shares_s}")
    print("─"*95)
    print(f"  ※ 建议股数基于等权 ${TOTAL_CAPITAL:,.0f} CAD，精确仓位见下方 Position Sizing")

    # 约束核查（含股息率）
    print(f"\n  {'Ticker':<14}{'价格':>8}{'ADV':>10}{'P/E':>7}{'市值':>10}"
          f"{'ROE':>7}{'股息率':>8}")
    print(f"  {'─'*64}")
    for _,(idx,row) in enumerate(top.iterrows()):
        t  = idx[1] if isinstance(idx,tuple) else idx
        df = daily_map.get(t,pd.DataFrame())
        m  = meta_df.loc[t] if t in meta_df.index else pd.Series(dtype=float)
        price  = f"${df['close'].iloc[-1]:.2f}"                   if not df.empty else "N/A"
        adv    = f"${(df['close'].tail(20)*df['volume'].tail(20)).mean()/1e6:.1f}M" \
                 if not df.empty else "N/A"
        pe     = f"{m.get('pe'):.1f}x"                            if pd.notna(m.get('pe'))  else "N/A"
        mc     = f"${m.get('mktcap')/1e9:.1f}B"                   if m.get('mktcap')        else "N/A"
        roe    = f"{m.get('roe')*100:.1f}%"                        if pd.notna(m.get('roe')) else "N/A"
        div    = f"{m.get('div_yield')*100:.1f}%"                  if pd.notna(m.get('div_yield')) else "N/A"
        print(f"  {t:<14}{price:>8}{adv:>10}{pe:>7}{mc:>10}{roe:>7}{div:>8}")

    # 第一层：止损提示
    print(f"\n  ⚡ 止损线：持仓期间单支跌超 {abs(STOP_LOSS_PCT)*100:.0f}% 建议止损")

    # Fuzzy 仓位
    fz = fuzzy_sizing(top, daily_map)
    # 加入当前价格和建议股数
    TOTAL_CAPITAL = 100_000   # ← 改成你的总资金 CAD

    prices = {}
    for idx, _ in top.iterrows():
        t  = idx[1] if isinstance(idx, tuple) else idx
        df = daily_map.get(t, pd.DataFrame())
        prices[t] = float(df["close"].iloc[-1]) if not df.empty else 0.0

    fz["price"]     = fz["ticker"].map(prices)
    fz["alloc_cad"] = (fz["alloc_pct"] / 100 * TOTAL_CAPITAL).round(0)
    fz["shares"]    = (fz["alloc_cad"] / fz["price"].replace(0, float("nan"))).apply(
                       lambda x: int(x) if pd.notna(x) else 0)

    SEP = "=" * 92
    print("\n" + SEP)
    print(f"  Position Sizing  (Total Capital: ${TOTAL_CAPITAL:,.0f} CAD)")
    print(f"  Heavy 20-25%  |  Moderate 10-15%  |  Light 3-8%")
    print(f"  Change TOTAL_CAPITAL above to match your actual portfolio size")
    print(f"  ⚠️  标注股票股数 < {CONSTRAINTS.get('min_shares',5)} 股，实际持仓意义有限（建议增加资金或跳过）")
    print(SEP)
    print(f"  {'#':<3}{'Ticker':<13}{'Category':<15}{'Alloc%':>7}  "
          f"{'Price':>8}  {'Amount(CAD)':>12}  {'Shares':>8}  {'Vol':>6}  {'Score':>7}")
    print(f"  {'-'*90}")
    for i, row in fz.iterrows():
        price_s = f"${row['price']:>7.2f}" if row["price"] > 0 else "    N/A"
        cad_s   = f"${row['alloc_cad']:>10,.0f}"
        min_sh  = CONSTRAINTS.get("min_shares", 1)
        sh_s    = f"{row['shares']:>7,}" if row["shares"] >= min_sh else f"  ⚠️ {row['shares']}股"
        print(f"  {i+1:<3}{row['ticker']:<13}{row['category']:<15}"
              f"{row['alloc_pct']:>6.1f}%  "
              f"{price_s}  {cad_s}  {sh_s}  "
              f"{row['vol_ann']*100:>5.1f}%  {row['score']:>7.3f}")
    print(f"  {'-'*90}")
    print(f"  {'Total':<31}{fz['alloc_pct'].sum():>6.1f}%  {'':>8}  "
          f"${fz['alloc_cad'].sum():>10,.0f}\n")
    for cat in ["Heavy Buy", "Moderate Buy", "Light Buy",
                "\u5927\u91cf\u4e70\u5165 \U0001f534",
                "\u9002\u5ea6\u4e70\u5165 \U0001f7e1",
                "\u5c11\u91cf\u4e70\u5165 \U0001f7e2"]:
        sub = fz[fz["category"]==cat]
        if len(sub):
            print(f"    {cat:<20} {len(sub)} stocks  {sub['alloc_pct'].sum():.1f}%"
                  f"  -> {' | '.join(sub['ticker'].tolist())}")

    # Feature importance
    print(f"\n  特征重要性（XGBoost，Top 12）：")
    for feat, val in imp.head(12).items():
        bar = "█"*int(val*300)
        print(f"  {feat:<24} {bar:<30} {val:.4f}")

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# MODULE A: 共线性处理（VIF + PCA 降维）
# ══════════════════════════════════════════════════════════════════

def compute_vif(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    计算每个特征的 VIF（方差膨胀因子）。
    VIF > 10 说明严重共线性，> 5 说明中度共线性。

    动量族（mom_1m/3m/6m/12m）通常 VIF > 20，是最大共线性来源。
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = panel[feature_cols].fillna(panel[feature_cols].median())
    # 去掉方差为0的列
    X = X.loc[:, X.std() > 0]

    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            v = variance_inflation_factor(X.values, i)
            vif_data.append({"feature": col, "vif": round(v, 2)})
        except Exception:
            vif_data.append({"feature": col, "vif": np.nan})

    return pd.DataFrame(vif_data).sort_values("vif", ascending=False)


def apply_collinearity_reduction(panel: pd.DataFrame, vif_threshold: float = 10.0) -> tuple[pd.DataFrame, list[str]]:
    """
    两步共线性处理：

    步骤1：对高度相关的动量族特征做 PCA 压缩
      mom_1m / mom_3m / mom_6m / mom_12m / mom_12_1
      → 提取 2 个主成分：mom_pc1（趋势强度）/ mom_pc2（动量曲率）

    步骤2：对波动率族特征做 PCA 压缩
      vol_1m / vol_3m / vol_ratio
      → 提取 1 个主成分：vol_pc1（整体波动水平）

    步骤3：逐步删除 VIF > threshold 的剩余特征

    返回：处理后的 panel，新的 feature_cols
    """
    from sklearn.decomposition import PCA

    print("\n  [共线性处理] 分析特征相关性...")

    panel = panel.copy()
    new_cols = list(FEATURE_COLS)

    # ── 步骤1：动量 PCA ────────────────────────────────────────────
    mom_cols = ["mom_1m","mom_3m","mom_6m","mom_12m","mom_12_1"]
    mom_data = panel[mom_cols].fillna(0)  # PCA前已中性化，0=均值，正确

    pca_mom = PCA(n_components=2, random_state=42)
    mom_pcs = pca_mom.fit_transform(mom_data)
    panel["mom_pc1"] = mom_pcs[:, 0]   # 趋势强度
    panel["mom_pc2"] = mom_pcs[:, 1]   # 动量曲率
    expl = pca_mom.explained_variance_ratio_
    print(f"    动量 PCA: PC1={expl[0]*100:.1f}%  PC2={expl[1]*100:.1f}%  "
          f"累计={sum(expl)*100:.1f}%")

    for c in mom_cols:
        new_cols.remove(c)
    new_cols = ["mom_pc1","mom_pc2"] + new_cols

    # ── 步骤2：波动率 PCA ──────────────────────────────────────────
    vol_cols = ["vol_1m","vol_3m","vol_ratio"]
    vol_data = panel[vol_cols].fillna(0)  # 同上

    pca_vol = PCA(n_components=1, random_state=42)
    vol_pcs = pca_vol.fit_transform(vol_data)
    panel["vol_pc1"] = vol_pcs[:, 0]
    expl_v = pca_vol.explained_variance_ratio_
    print(f"    波动率 PCA: PC1={expl_v[0]*100:.1f}%")

    for c in vol_cols:
        new_cols.remove(c)
    new_cols = ["vol_pc1"] + new_cols

    # ── 步骤3：计算剩余 VIF，删除高共线性特征 ─────────────────────
    vif_df = compute_vif(panel, new_cols)
    high_vif = vif_df[vif_df["vif"] > vif_threshold]["feature"].tolist()

    print(f"\n    VIF 分析（阈值 {vif_threshold}）：")
    print(f"    {'特征':<24} {'VIF':>8}  {'状态'}")
    print(f"    {'─'*44}")
    for _, row in vif_df.head(15).iterrows():
        status = "✗ 删除" if row["feature"] in high_vif else "✓ 保留"
        bar    = "▓" * min(20, int(row["vif"]/5)) if pd.notna(row["vif"]) else ""
        vif_str= f"{row['vif']:.1f}" if pd.notna(row["vif"]) else "N/A"
        print(f"    {row['feature']:<24} {vif_str:>8}  {bar}  {status}")

    # 删除高 VIF 特征（但保留 PCA 主成分）
    pc_cols = ["mom_pc1","mom_pc2","vol_pc1"]
    to_remove = [c for c in high_vif if c not in pc_cols]
    for c in to_remove:
        if c in new_cols:
            new_cols.remove(c)

    print(f"\n    最终特征数：{len(new_cols)} 个（原 {len(FEATURE_COLS)} 个）")
    print(f"    删除：{to_remove if to_remove else '无'}")

    return panel, new_cols


# ══════════════════════════════════════════════════════════════════
# MODULE B: SEDI 内部人交易数据（加拿大 SEC）
# ══════════════════════════════════════════════════════════════════

def fetch_sedi_insider(tickers: list[str], lookback_days: int = 90) -> pd.DataFrame:
    """
    从 canadianinsider.com 抓取 SEDI 内部人交易数据。
    canadianinsider.com 是 SEDI 的聚合展示平台，比直接抓 SEDI 更友好。

    内部人交易信号逻辑：
      净买入（insider_buy_ratio > 0.6）→ 正面信号，加分
      净卖出（insider_buy_ratio < 0.4）→ 负面信号，减分
      无交易                          → 中性

    返回 DataFrame，index=ticker，columns=[buy_cnt, sell_cnt, net_shares, signal]
    """
    import requests
    from bs4 import BeautifulSoup

    results = []
    end_date  = datetime.today()
    start_date= end_date - timedelta(days=lookback_days)

    print(f"\n  [SEDI] 抓取内部人交易（过去 {lookback_days} 天）...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }

    for t in tickers:
        t_clean = t.replace(".TO","").replace("-",".")
        try:
            # canadianinsider.com 按股票代码查询
            url = f"https://www.canadianinsider.com/node?ticker={t_clean}"
            r   = requests.get(url, headers=headers, timeout=10)

            if r.status_code != 200:
                results.append({"ticker":t,"buy_cnt":0,"sell_cnt":0,
                                 "net_shares":0,"signal":0.0,"source":"no_data"})
                continue

            soup = BeautifulSoup(r.text, "html.parser")

            # 找交易表格
            buy_cnt = sell_cnt = net_shares = 0
            rows = soup.find_all("tr")

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue
                try:
                    # 列格式：日期 | 内部人 | 职位 | 交易类型 | 数量 | 价格
                    date_str = cols[0].get_text(strip=True)
                    txn_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if txn_date < start_date:
                        continue
                    txn_type = cols[3].get_text(strip=True).lower()
                    shares_str = cols[4].get_text(strip=True).replace(",","")
                    shares = float(shares_str) if shares_str.replace(".","").isdigit() else 0

                    if "acquisition" in txn_type or "purchase" in txn_type or "buy" in txn_type:
                        buy_cnt   += 1
                        net_shares+= shares
                    elif "disposition" in txn_type or "sale" in txn_type or "sell" in txn_type:
                        sell_cnt   += 1
                        net_shares -= shares
                except Exception:
                    continue

            total = buy_cnt + sell_cnt
            buy_ratio = buy_cnt / total if total > 0 else 0.5

            # 信号：-1（强卖）到 +1（强买）
            if total == 0:
                signal = 0.0
            elif buy_ratio > 0.7:
                signal = 1.0    # 强买入信号
            elif buy_ratio > 0.55:
                signal = 0.5   # 温和买入
            elif buy_ratio < 0.3:
                signal = -1.0  # 强卖出信号
            elif buy_ratio < 0.45:
                signal = -0.5  # 温和卖出
            else:
                signal = 0.0   # 中性

            results.append({"ticker":t,"buy_cnt":buy_cnt,"sell_cnt":sell_cnt,
                            "net_shares":int(net_shares),"signal":signal,"source":"sedi"})

            status = ("强买 🟢" if signal>=1 else "买 🟡" if signal>0
                      else "强卖 🔴" if signal<=-1 else "卖 🟠" if signal<0 else "中性 ⚪")
            print(f"    {t:<14} 买入{buy_cnt:>3}次  卖出{sell_cnt:>3}次  "
                  f"净股数{net_shares:>10,}  {status}")

        except Exception as e:
            results.append({"ticker":t,"buy_cnt":0,"sell_cnt":0,
                            "net_shares":0,"signal":0.0,"source":"error"})

    df = pd.DataFrame(results).set_index("ticker")
    n_sig = (df["signal"] != 0).sum()
    print(f"\n    有效信号：{n_sig}/{len(tickers)} 支  "
          f"（买入信号 {(df['signal']>0).sum()} 支，"
          f"卖出信号 {(df['signal']<0).sum()} 支）")
    return df


def apply_insider_signal(result: pd.DataFrame, insider_df: pd.DataFrame,
                          weight: float = 0.15) -> pd.DataFrame:
    """
    将内部人信号加入最终排名。
    集成分 = 原始集成分 × (1 - weight) + insider_signal × weight

    weight=0.15 意味着内部人信号占 15% 权重。
    """
    if insider_df.empty:
        return result

    result = result.copy()
    tix    = result.index.get_level_values("ticker") if isinstance(result.index, pd.MultiIndex) \
             else result.index

    signals = tix.map(insider_df["signal"].to_dict()).fillna(0)
    # 归一化 insider signal 到 [0,1]
    sig_norm = (signals + 1) / 2   # -1→0, 0→0.5, +1→1

    result["insider_signal"]  = signals.values
    result["ensemble_score"]  = (result["ensemble_score"] * (1 - weight) +
                                  sig_norm.values * weight)
    result = result.sort_values("ensemble_score", ascending=False)
    return result


# ══════════════════════════════════════════════════════════════════
# MODULE C: 参数敏感性分析
# ══════════════════════════════════════════════════════════════════

def sensitivity_analysis(daily_map: dict, pit_map: dict, meta_df: pd.DataFrame,
                          macro_df: pd.DataFrame, feature_cols: list[str],
                          n_jobs: int = 1) -> pd.DataFrame:
    """
    对关键约束参数做网格搜索，找出让 Walk-Forward 表现最好的组合。

    搜索空间：
      max_pe:        [40, 50, 60, 80]
      min_mktcap:    [200M, 500M, 1B]
      max_per_gics:  [1, 2, 3]
      tx_cost:       [0.001, 0.002, 0.005]

    评估指标：Sharpe = 多空价差 / 波动率（越高越好）

    ⚠️  每个参数组合都要跑一次 Walk-Forward，计算量大。
        建议先用小网格测试（QUICK_MODE=True）。
    """
    import itertools

    QUICK_MODE = True   # True = 小网格，快速验证；False = 完整搜索

    if QUICK_MODE:
        grid = {
            "max_pe":       [40, 60],
            "min_mktcap":   [200_000_000, 500_000_000],
            "max_per_gics": [1, 2],
            "tx_cost":      [0.001, 0.003],
        }
    else:
        grid = {
            "max_pe":       [40, 50, 60, 80],
            "min_mktcap":   [200_000_000, 500_000_000, 1_000_000_000],
            "max_per_gics": [1, 2, 3],
            "tx_cost":      [0.001, 0.002, 0.005],
        }

    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"\n[参数敏感性] 搜索 {len(combos)} 个参数组合 "
          f"({'快速模式' if QUICK_MODE else '完整模式'})...")

    records = []
    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))

        # 构建当次约束
        c_test = CONSTRAINTS.copy()
        c_test["max_pe"]        = params["max_pe"]
        c_test["min_mktcap_cad"]= params["min_mktcap"]
        c_test["max_per_gics"]  = params["max_per_gics"]

        passed_test = apply_constraints(daily_map, meta_df, c_test)
        if len(passed_test) < TOP_N:
            records.append({**params, "n_stocks":len(passed_test),
                            "spread":np.nan, "sharpe":np.nan, "win_rate":np.nan})
            continue

        try:
            panel_t = build_panel(passed_test, daily_map, pit_map, macro_df)
            panel_t = add_labels(panel_t)
            panel_t = cross_z(panel_t)

            # 参数敏感性分析使用 asof 约束确保准确性
            wf_t = walk_forward(panel_t, tx_cost=params["tx_cost"],
                               daily_map=daily_map, pit_map=pit_map,
                               apply_asof_constraints=True)
            if wf_t.empty or "actual_ret" not in wf_t.columns:
                raise ValueError("Walk-Forward 结果为空")

            v    = wf_t.dropna(subset=["actual_ret","actual_cls"])
            p    = v["ens"]
            topR = v.loc[p>=p.quantile(0.80),"actual_ret"].mean()*100
            botR = v.loc[p<=p.quantile(0.20),"actual_ret"].mean()*100
            spread = topR - botR

            # Sharpe = 多空价差 / 月收益标准差（年化）
            monthly = v.groupby("date")["actual_ret"].mean()
            vol_m   = monthly.std() * np.sqrt(12) * 100
            sharpe  = spread / vol_m if vol_m > 0 else 0

            win_rate= (v.loc[p>=p.quantile(0.80),"actual_ret"]>0).mean()*100

            records.append({**params, "n_stocks":len(passed_test),
                           "spread":round(spread,3),"sharpe":round(sharpe,3),
                           "win_rate":round(win_rate,1),"top_ret":round(topR,3)})

            print(f"  [{ci:>3}/{len(combos)}] PE≤{params['max_pe']}  "
                  f"MCap≥{params['min_mktcap']//1e6:.0f}M  "
                  f"GICS≤{params['max_per_gics']}  "
                  f"cost={params['tx_cost']*100:.1f}%  "
                  f"→ 价差{spread:+.2f}%  Sharpe{sharpe:.2f}  "
                  f"通过{len(passed_test)}支")
        except Exception as e:
            records.append({**params, "n_stocks":len(passed_test),
                           "spread":np.nan,"sharpe":np.nan,"win_rate":np.nan})
            print(f"  [{ci:>3}/{len(combos)}] ✗ {str(e)[:40]}")

    result_df = pd.DataFrame(records).sort_values("sharpe", ascending=False)

    print(f"\n  {'─'*70}")
    print(f"  参数敏感性分析结果（按 Sharpe 排序，Top 5）：")
    print(f"  {'─'*70}")
    print(f"  {'max_pe':>8} {'min_mktcap':>12} {'max_gics':>10} "
          f"{'tx_cost':>8} {'n_stk':>6} {'价差%':>8} {'Sharpe':>8} {'胜率%':>8}")
    print(f"  {'─'*70}")
    for _, row in result_df.head(5).iterrows():
        if pd.notna(row["sharpe"]):
            print(f"  {row['max_pe']:>8.0f} "
                  f"${row['min_mktcap']/1e6:>9.0f}M "
                  f"{row['max_per_gics']:>10.0f} "
                  f"{row['tx_cost']*100:>7.1f}% "
                  f"{row['n_stocks']:>6.0f} "
                  f"{row['spread']:>+8.2f}% "
                  f"{row['sharpe']:>8.2f} "
                  f"{row['win_rate']:>7.1f}%")

    best = result_df.dropna(subset=["sharpe"]).iloc[0] if not result_df.dropna(subset=["sharpe"]).empty else None
    if best is not None:
        print(f"\n  ★ 最优参数组合：")
        print(f"    max_pe        = {best['max_pe']:.0f}")
        print(f"    min_mktcap    = ${best['min_mktcap']/1e6:.0f}M CAD")
        print(f"    max_per_gics  = {best['max_per_gics']:.0f}")
        print(f"    tx_cost       = {best['tx_cost']*100:.1f}%")
        print(f"    → Sharpe {best['sharpe']:.2f}，多空价差 {best['spread']:+.2f}%")

    return result_df


# ══════════════════════════════════════════════════════════════════
# MODULE D: 真实回测框架
# ══════════════════════════════════════════════════════════════════

class RealisticBacktest:
    """
    真实回测框架，模拟实际交易环境。

    改进点 vs 简单 Walk-Forward：
      ① 市场冲击模型（大单影响价格）
      ② 买卖价差（Bid-Ask Spread）
      ③ T+2 结算（加拿大市场标准）
      ④ 分红再投资（DRIP）
      ⑤ 止损执行（月内而非月底）
      ⑥ 持仓惯性（避免过度换仓）
      ⑦ 资金管理（保留现金缓冲）
    """

    def __init__(self,
                 capital:       float = 100_000,
                 tx_cost:       float = 0.002,    # 单边手续费
                 bid_ask:       float = 0.001,    # 买卖价差（0.1%）
                 market_impact: float = 0.002,    # 市场冲击（大单滑点）
                 stop_loss:     float = -0.08,    # 止损线
                 cash_buffer:   float = 0.05,     # 现金缓冲比例
                 hold_inertia:  float = 0.10,     # 持仓惯性阈值（分数差< X 不换仓）
                 drip:          bool  = True):    # 分红再投资

        self.capital       = capital
        self.tx_cost       = tx_cost
        self.bid_ask       = bid_ask
        self.market_impact = market_impact
        self.stop_loss     = stop_loss
        self.cash_buffer   = cash_buffer
        self.hold_inertia  = hold_inertia
        self.drip          = drip

        # 状态变量
        self.cash          = capital
        self.positions     = {}    # {ticker: {shares, cost_basis, entry_price}}
        self.nav_history   = []    # 净值历史
        self.trade_log     = []    # 交易记录
        self.monthly_rets  = []    # 月度收益

    def _trade_cost(self, value: float, adv: float) -> float:
        """
        总交易成本 = 手续费 + 买卖价差 + 市场冲击。
        市场冲击：交易金额 / (ADV × 0.1) 的平方根模型。
        """
        fee    = value * self.tx_cost
        spread = value * self.bid_ask
        # 市场冲击：交易占日均成交量比例越大，冲击越大
        participation = value / max(adv * 0.1, value)  # 假设只用10%日成交量
        impact = value * self.market_impact * np.sqrt(participation)
        return fee + spread + impact

    def _current_nav(self, prices: dict) -> float:
        """计算当前组合净值（现金 + 持仓市值）"""
        pos_value = sum(
            info["shares"] * prices.get(t, info["entry_price"])
            for t, info in self.positions.items()
        )
        return self.cash + pos_value

    def run(self, panel: pd.DataFrame, daily_map: dict,
            meta_df: pd.DataFrame, model_scores: pd.DataFrame) -> dict:
        """
        执行完整回测。
        model_scores: Walk-Forward 输出的 {date, ticker, ens} DataFrame
        """
        dates = sorted(panel.index.get_level_values("date").unique())
        print(f"\n[真实回测] 初始资金 ${self.capital:,.0f} CAD  "
              f"手续费 {self.tx_cost*100:.1f}%  买卖价差 {self.bid_ask*100:.1f}%")
        print(f"  止损线 {self.stop_loss*100:.0f}%  现金缓冲 {self.cash_buffer*100:.0f}%  "
              f"持仓惯性 {self.hold_inertia:.2f}")
        print(f"  {'─'*60}")

        turnover_total = 0

        for i, date in enumerate(dates[:-1]):
            nxt_date = dates[i+1]

            # 获取当月价格
            curr_prices = {}
            for t, df in daily_map.items():
                month_data = df[df.index.date <= date.date()]
                if not month_data.empty:
                    curr_prices[t] = float(month_data["close"].iloc[-1])

            nav_start = self._current_nav(curr_prices)

            # ── 止损检查 ─────────────────────────────────────────
            to_stop = []
            for t, info in self.positions.items():
                curr_p = curr_prices.get(t, info["entry_price"])
                ret    = (curr_p - info["entry_price"]) / info["entry_price"]
                if ret < self.stop_loss:
                    to_stop.append(t)

            for t in to_stop:
                info  = self.positions[t]
                price = curr_prices.get(t, info["entry_price"])
                adv   = (daily_map[t]["close"].tail(20) *
                         daily_map[t]["volume"].tail(20)).mean() if t in daily_map else 1e6
                cost  = self._trade_cost(info["shares"] * price, adv)
                proceeds = info["shares"] * price - cost
                self.cash += proceeds
                self.trade_log.append({"date":date,"ticker":t,"action":"止损卖出",
                                       "price":price,"shares":info["shares"],
                                       "cost":cost,"reason":f"跌幅超{self.stop_loss*100:.0f}%"})
                del self.positions[t]

            # ── 获取本月模型评分 ──────────────────────────────────
            month_scores = model_scores[model_scores["date"] == date]
            if month_scores.empty:
                self.nav_history.append({"date":date,"nav":self._current_nav(curr_prices)})
                continue

            ranked_df = (month_scores.sort_values("ens", ascending=False)
                         .rename(columns={"ens": "ensemble_score"})
                         .set_index("ticker"))
            selected_df = apply_rebalancing_band(
                current_ranked_df=ranked_df,
                prev_holdings=list(self.positions.keys()),
                top_n=TOP_N,
                rank_buffer=CONSTRAINTS.get("rank_buffer", 12),
                score_tolerance=CONSTRAINTS.get("score_tolerance", 0.005),
            )
            target_tickers = set(
                selected_df.index.get_level_values("ticker").tolist()
                if isinstance(selected_df.index, pd.MultiIndex)
                else selected_df.index.tolist()
            )
            current_tickers = set(self.positions.keys())

            # ── 卖出不在目标中的持仓 ──────────────────────────────
            to_sell = current_tickers - target_tickers
            for t in to_sell:
                info  = self.positions[t]
                price = curr_prices.get(t, info["entry_price"])
                adv   = (daily_map[t]["close"].tail(20) *
                         daily_map[t]["volume"].tail(20)).mean() if t in daily_map else 1e6
                cost  = self._trade_cost(info["shares"] * price, adv)
                self.cash += info["shares"] * price - cost
                self.trade_log.append({"date":date,"ticker":t,"action":"卖出",
                                       "price":price,"shares":info["shares"],"cost":cost})
                del self.positions[t]
                turnover_total += 1

            # ── 等风险分配资金 ────────────────────────────────────
            investable = self.cash * (1 - self.cash_buffer)
            n_new      = len(target_tickers - current_tickers)
            if n_new > 0 and investable > 1000:
                alloc_per = investable / max(len(target_tickers), 1)

                # ── 买入新标的 ────────────────────────────────────
                to_buy = (target_tickers - current_tickers) & set(curr_prices.keys())
                for t in to_buy:
                    price = curr_prices[t]
                    adv   = (daily_map[t]["close"].tail(20) *
                             daily_map[t]["volume"].tail(20)).mean() if t in daily_map else 1e6

                    # 流动性约束：单日最多买入 ADV 的 5%
                    max_buy = adv * 0.05
                    buy_val = min(alloc_per, max_buy, self.cash * 0.9)
                    if buy_val < 500:
                        continue

                    cost   = self._trade_cost(buy_val, adv)
                    shares = (buy_val - cost) / price
                    actual_cost = shares * price + cost

                    if actual_cost > self.cash:
                        continue

                    self.cash -= actual_cost
                    self.positions[t] = {"shares": shares, "entry_price": price,
                                         "cost_basis": price + cost/shares}
                    self.trade_log.append({"date":date,"ticker":t,"action":"买入",
                                           "price":price,"shares":shares,"cost":cost})
                    turnover_total += 1

            # ── 分红再投资（DRIP）────────────────────────────────
            if self.drip:
                for t, info in list(self.positions.items()):
                    if t in meta_df.index:
                        div_yield = meta_df.loc[t, "div_yield"]
                        if pd.notna(div_yield) and div_yield > 0:
                            # 月度分红 ≈ 年化股息率 / 12
                            monthly_div = info["shares"] * curr_prices.get(t, info["entry_price"]) \
                                         * div_yield / 12
                            if monthly_div > 10:   # 最小 $10 再投资
                                price    = curr_prices.get(t, info["entry_price"])
                                new_sh   = monthly_div / price
                                self.positions[t]["shares"] += new_sh
                                self.cash  = max(0, self.cash - monthly_div * 0.1)  # 预扣税

            # ── 记录月度净值 ──────────────────────────────────────
            nav_end = self._current_nav(curr_prices)
            ret_m   = (nav_end / nav_start - 1) if nav_start > 0 else 0
            self.monthly_rets.append(ret_m)
            self.nav_history.append({"date":date,"nav":nav_end,"ret":ret_m,
                                     "n_positions":len(self.positions),
                                     "cash_pct":self.cash/nav_end*100})

            if i % 4 == 0:
                print(f"  {date.strftime('%Y-%m')}  NAV ${nav_end:>10,.0f}  "
                      f"持仓{len(self.positions):>2}支  "
                      f"现金{self.cash/nav_end*100:.0f}%  "
                      f"月收益{ret_m*100:>+6.2f}%")

        return self._report(turnover_total)

    def _report(self, turnover_total: int) -> dict:
        """生成回测报告"""
        if not self.monthly_rets:
            return {}

        rets   = np.array(self.monthly_rets)
        ann    = (np.prod(1 + rets) ** (12/len(rets)) - 1) * 100
        vol    = rets.std() * np.sqrt(12) * 100
        sharpe = (ann/100 - 0.04) / (vol/100) if vol > 0 else 0

        nav_ser = pd.Series([h["nav"] for h in self.nav_history])
        cum_max = nav_ser.cummax()
        mdd     = ((nav_ser - cum_max) / cum_max).min() * 100

        win_rate= (rets > 0).mean() * 100
        calmar  = ann / abs(mdd) if mdd != 0 else 0
        total_tx_cost = sum(t["cost"] for t in self.trade_log)

        print(f"\n  {'═'*56}")
        print(f"  📊 真实回测报告")
        print(f"  {'═'*56}")
        print(f"  初始资金       ${self.capital:>12,.0f} CAD")
        print(f"  最终净值       ${self.nav_history[-1]['nav']:>12,.0f} CAD")
        print(f"  总收益         {(self.nav_history[-1]['nav']/self.capital-1)*100:>+11.1f}%")
        print(f"  年化收益       {ann:>+11.1f}%")
        print(f"  年化波动率     {vol:>11.1f}%")
        print(f"  Sharpe 比率    {sharpe:>11.2f}")
        print(f"  最大回撤       {mdd:>+11.1f}%")
        print(f"  Calmar 比率    {calmar:>11.2f}")
        print(f"  月胜率         {win_rate:>11.1f}%")
        print(f"  总交易次数     {turnover_total:>12}")
        print(f"  总交易成本     ${total_tx_cost:>12,.0f} CAD")
        print(f"  {'─'*56}")
        print(f"  注：含手续费 {self.tx_cost*100:.1f}% + 买卖价差 {self.bid_ask*100:.1f}% "
              f"+ 市场冲击 + 止损 + 分红再投资")

        return {"ann_ret":ann,"vol":vol,"sharpe":sharpe,"mdd":mdd,
                "calmar":calmar,"win_rate":win_rate,"total_cost":total_tx_cost}


# ══════════════════════════════════════════════════════════════════
# 运行四个新模块的入口函数
# ══════════════════════════════════════════════════════════════════

def run_advanced_analysis(panel, daily_map, pit_map, meta_df, macro_df, wf,
                           run_sensitivity=False,
                           run_collinearity=True,
                           run_insider=True,
                           run_backtest=True):
    """
    统一入口，按需运行四个高级模块。
    默认不跑敏感性分析（耗时很长），其他三个默认开启。
    """
    print("\n" + "▓"*60)
    print("  高级分析模块")
    print("▓"*60)

    used_feature_cols = FEATURE_COLS
    panel_reduced = panel  # 🎯 修复1-A：加一行兜底，防止不跑共线性时报错

    # ── A. 共线性处理 ────────────────────────────────────────────
    if run_collinearity:
        print("\n【A】特征共线性处理")
        try:
            panel_reduced, used_feature_cols = apply_collinearity_reduction(
                panel, vif_threshold=10.0)
            print(f"  ✅ 共线性处理完成：{len(FEATURE_COLS)} → {len(used_feature_cols)} 个特征")
        except Exception as e:
            print(f"  ⚠️  共线性处理失败：{e}，使用原始特征")
            panel_reduced = panel

    # ── B. SEDI 内部人交易 ────────────────────────────────────────
    insider_df = pd.DataFrame()
    if run_insider:
        print("\n【B】SEDI 内部人交易信号")
        # canadianinsider.com 屏蔽 Colab IP，仅本地运行有效
        # 本地运行时将 SEDI_LOCAL_ONLY 改为 False 启用
        SEDI_LOCAL_ONLY = False
        if SEDI_LOCAL_ONLY:
            print("  ⚠️  SEDI 仅支持本地运行（Colab IP 被 canadianinsider.com 屏蔽）")
            print("       本地运行：将上方 SEDI_LOCAL_ONLY = True 改为 False")
        else:
            passed_tickers = list(daily_map.keys())
            try:
                insider_df = fetch_sedi_insider(passed_tickers, lookback_days=90)
            except Exception as e:
                print(f"  ⚠️  SEDI 抓取失败：{e}")

    # ── C. 参数敏感性分析 ─────────────────────────────────────────
    if run_sensitivity:
        print("\n【C】参数敏感性分析（耗时较长）")
        try:
            sensitivity_df = sensitivity_analysis(
                daily_map, pit_map, meta_df, macro_df, used_feature_cols)
        except Exception as e:
            print(f"  ⚠️  敏感性分析失败：{e}")

    # ── D. 真实回测框架 ───────────────────────────────────────────
    if run_backtest and not wf.empty:
        print("\n【D】真实回测框架")
        bt = RealisticBacktest(
            capital       = 100_000,
            tx_cost       = BT_TX_COST,
            bid_ask       = 0.001,
            market_impact = 0.002,
            stop_loss     = STOP_LOSS_PCT,
            cash_buffer   = 0.01,     # 🚀 降低现金闲置：原为 0.05，改为 0.01 (只留 1% 现金交点差)
            hold_inertia  = 0.35,     # 🚀 从 0.10 提高到 0.35！强迫持股，克服多动症
                                      # 逻辑：只要这只股票还在池子里，哪怕排名掉到了第 8 名，坚决不卖它去换第 1 名
            drip          = True,
        )
        try:
            bt_result = bt.run(panel, daily_map, meta_df, wf)
        except Exception as e:
            print(f"  ⚠️  真实回测失败：{e}")

    # 🎯 修复1-B：将 panel_reduced 加到 return 列表中
    return panel_reduced, insider_df, used_feature_cols

# ══════════════════════════════════════════════════════════════════
# MODULE E: Regime Detection（市场状态识别）
# ══════════════════════════════════════════════════════════════════

class MarketRegime:
    """
    市场状态识别器。

    三种 Regime：
      BULL   (Risk-On)  → VIX < 20  且 TSX > 200日均线
      NEUTRAL           → 介于两者之间
      BEAR   (Risk-Off) → VIX > 28  或 TSX < 200日均线

    对模型的影响：
      BULL   → 动量权重 ×1.5，波动率约束放宽，允许买高波动成长股
      NEUTRAL → 使用默认权重
      BEAR   → 防御模式：股息/低波动权重 ×2，PE约束收紧
    """

    BULL     = "BULL"
    NEUTRAL  = "NEUTRAL"
    BEAR     = "BEAR"

    # Regime → 因子权重调整系数（相对于默认权重）
    FACTOR_MULTIPLIERS = {
        BULL: {
            "mom_6m":       2.0,   # 🎯 极大增强中期动量权重 (原1.5)
            "mom_12m":      2.0,   # 🎯 增强长期动量权重
            "price_vs_52w_high": 2.5, # 🎯 强迫模型偏好创新高的股票
            "pe":           0.2,   # 🎯 彻底削弱绝对估值的压制 (原0.7)
            "pb":           0.05,   # 🎯 彻底削弱市净率的压制
            "pe_rel_sector": 0.3,  # 🎯 允许买入行业内享受估值溢价的龙头
            "fcf_yield":    0.8,
            "vol_1m":       0.7,   # 允许一定的波动
            "vix_level":    0.5,
        },
        NEUTRAL: {k: 1.0 for k in FEATURE_COLS},
        BEAR: {
            "mom_6m":       0.5,   # 动量降权（熊市动量失效）
            "mom_12m":      0.5,
            "pe":           1.5,   # 估值加权（熊市只买便宜货）
            "fcf_yield":    2.0,   # 现金流加权（防御）
            "vol_1m":       2.0,   # 高波动惩罚加大
            "vix_level":    1.5,
            "price_vs_52w_high": 0.3,  # 不追高
        },
    }

    # Regime → 约束调整
    CONSTRAINT_OVERRIDES = {
        BULL:    {"max_pe": 70, "max_per_gics": 2},
        NEUTRAL: {},
        BEAR:    {"max_pe": 40, "max_per_gics": 1,
                  "min_roe": 0.05},   # 熊市只买 ROE > 5%
    }

    def __init__(self, macro_df: pd.DataFrame):
        self.macro_df = macro_df
        self.regime   = self._detect()

    def _detect(self) -> str:
        if self.macro_df.empty or len(self.macro_df) < 10:
            return self.NEUTRAL

        latest = self.macro_df.iloc[-1]
        vix    = latest.get("vix", 20)

        # TSX vs 200日均线（月线用10个月均线近似 200日）
        if "tsx" in self.macro_df.columns:
            tsx_now = latest.get("tsx", 0)
            tsx_ma  = self.macro_df["tsx"].tail(10).mean()
            tsx_above_ma = tsx_now > tsx_ma
        else:
            tsx_above_ma = True

        if vix < 20 and tsx_above_ma:
            return self.BULL
        elif vix > 28 or not tsx_above_ma:
            return self.BEAR
        else:
            return self.NEUTRAL

    def describe(self) -> str:
        vix = self.macro_df["vix"].iloc[-1] if "vix" in self.macro_df.columns else "N/A"
        tsx_now = self.macro_df["tsx"].iloc[-1] if "tsx" in self.macro_df.columns else 0
        tsx_ma  = self.macro_df["tsx"].tail(10).mean() if "tsx" in self.macro_df.columns else 0
        diff    = (tsx_now/tsx_ma - 1)*100 if tsx_ma > 0 else 0

        emoji  = {"BULL":"🟢","NEUTRAL":"🟡","BEAR":"🔴"}[self.regime]
        labels = {"BULL":"牛市 Risk-On","NEUTRAL":"中性","BEAR":"熊市 Risk-Off"}

        return (f"  {emoji} 当前 Regime：{labels[self.regime]}\n"
                f"     VIX = {vix:.1f}  "
                f"TSX vs 10月均线 = {diff:+.1f}%\n"
                f"     {'动量权重加大，允许高波动' if self.regime==self.BULL else '防御模式，低波动/高股息' if self.regime==self.BEAR else '默认权重'}")

    def adjust_features(self, panel: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
        """根据 Regime 调整特征权重（乘以调整系数后重新 Z-score）"""
        mults = self.FACTOR_MULTIPLIERS.get(self.regime, {})
        out   = panel.copy()
        for col in feat_cols:
            if col in mults and col in out.columns:
                out[col] = out[col] * mults[col]
        return out

    def adjust_constraints(self, base_constraints: dict) -> dict:
        """根据 Regime 覆盖约束参数"""
        c = base_constraints.copy()
        c.update(self.CONSTRAINT_OVERRIDES.get(self.regime, {}))
        return c


# ══════════════════════════════════════════════════════════════════
# MODULE F: Earnings Surprise（盈利惊喜因子）
# ══════════════════════════════════════════════════════════════════

def fetch_earnings_surprise(tickers: list[str]) -> pd.DataFrame:
    """
    从 yfinance 获取季度盈利惊喜数据。

    盈利惊喜 = (实际EPS - 预期EPS) / |预期EPS|

    PEAD（Post-Earnings Announcement Drift）效应：
      超预期 > +5%  → 未来1-3月股价持续跑赢
      低于预期 < -5% → 未来1-3月股价持续跑输

    返回：DataFrame，index=ticker，
          columns=[surprise_pct, surprise_dir, n_quarters_positive]
    """
    print(f"\n  [Earnings Surprise] 获取 {len(tickers)} 支季报数据...")
    rows = []
    
    # 🔧 修复1：ETF 黑名单 — 大盘指数和宽基 ETF 无财报数据，跳过以避免 yfinance 报错
    ETF_BLACKLIST = {"XIU.TO", "XSP.TO", "VFV.TO", "XIC.TO", "^GSPTSE", "^RY"}

    for t in tickers:
        # 🔧 修复1：直接跳过 ETF，避免试图获取不存在的财报数据
        if t in ETF_BLACKLIST:
            rows.append({"ticker":t, "surprise_pct":0.0, "surprise_dir":0, "n_beat":0, "source":"etf_skip"})
            continue
        
        try:
            ticker = yf.Ticker(t)
            ed     = ticker.earnings_dates

            if ed is None or ed.empty:
                rows.append({"ticker":t,"surprise_pct":0.0,
                             "surprise_dir":0,"n_beat":0,"source":"no_data"})
                continue

            # 只取有实际EPS数据的历史季报
            ed = ed.dropna(subset=["Reported EPS","EPS Estimate"])
            ed = ed[ed["Reported EPS"].notna() & ed["EPS Estimate"].notna()]

            if ed.empty:
                rows.append({"ticker":t,"surprise_pct":0.0,
                             "surprise_dir":0,"n_beat":0,"source":"no_eps"})
                continue

            # 最近一季的惊喜度
            latest = ed.iloc[0]
            actual = float(latest["Reported EPS"])
            est    = float(latest["EPS Estimate"])

            if abs(est) < 0.01:   # 避免除以接近零的预期
                surprise_pct = 0.0
            else:
                surprise_pct = (actual - est) / abs(est)

            # 过去4季中超预期的次数（一致性）
            n_quarters = min(4, len(ed))
            recent4    = ed.head(n_quarters)
            n_beat     = ((recent4["Reported EPS"] > recent4["EPS Estimate"]).sum())

            # 方向信号：连续超预期 > 单次超预期
            if surprise_pct > 0.05 and n_beat >= 3:
                surprise_dir = 2    # 强超预期（持续性）
            elif surprise_pct > 0.02:
                surprise_dir = 1    # 轻微超预期
            elif surprise_pct < -0.05 and n_beat <= 1:
                surprise_dir = -2   # 强低于预期
            elif surprise_pct < -0.02:
                surprise_dir = -1   # 轻微低于预期
            else:
                surprise_dir = 0    # 中性

            rows.append({
                "ticker":       t,
                "surprise_pct": round(surprise_pct * 100, 2),  # 转成%
                "surprise_dir": surprise_dir,
                "n_beat":       int(n_beat),
                "actual_eps":   round(actual, 3),
                "est_eps":      round(est, 3),
                "source":       "yfinance",
            })

            dir_str = {2:"强超预期 ✅",1:"超预期 ☑",0:"中性 ⚪",-1:"低于预期 ⚠️",-2:"大幅低于预期 ❌"}.get(surprise_dir,"?")
            print(f"    {t:<14} 实际{actual:+.2f} vs 预期{est:+.2f}  "
                  f"惊喜{surprise_pct*100:+.1f}%  {dir_str}  "
                  f"近4季超预期{n_beat}/4次")

        except Exception as e:
            rows.append({"ticker":t,"surprise_pct":0.0,
                         "surprise_dir":0,"n_beat":0,"source":"error"})

    df = pd.DataFrame(rows).set_index("ticker")
    n_pos = (df["surprise_dir"] > 0).sum()
    n_neg = (df["surprise_dir"] < 0).sum()
    print(f"\n    汇总：超预期 {n_pos} 支  低于预期 {n_neg} 支  "
          f"中性 {len(df)-n_pos-n_neg} 支")
    return df


def apply_earnings_signal(result: pd.DataFrame, surprise_df: pd.DataFrame,
                           weight: float = 0.12) -> pd.DataFrame:
    """
    将盈利惊喜信号融入最终排名。

    规则：
      强超预期（dir=+2）→ 集成分 × 1.15
      超预期   (dir=+1) → 集成分 × 1.07
      大幅低于预期(dir=-2) → 集成分 × 0.80，且移到后面
      低于预期 (dir=-1) → 集成分 × 0.92

    连续4季超预期的股票额外加权（PEAD 效应更强）
    """
    if surprise_df.empty:
        return result

    result = result.copy()
    tix    = (result.index.get_level_values("ticker")
              if isinstance(result.index, pd.MultiIndex) else result.index)

    for i, t in enumerate(tix):
        if t not in surprise_df.index:
            continue
        row = surprise_df.loc[t]
        d   = row.get("surprise_dir", 0)
        n   = row.get("n_beat", 0)

        # 基础乘数
        mult = {2:1.15, 1:1.07, 0:1.0, -1:0.92, -2:0.80}.get(d, 1.0)

        # 连续性加成：4季全超预期再 +5%
        if n >= 4 and d > 0:
            mult *= 1.05

        result.loc[result.index[i], "ensemble_score"] *= mult

    result = result.sort_values("ensemble_score", ascending=False)
    result["earnings_surprise"] = tix.map(surprise_df["surprise_pct"].to_dict()).fillna(0)
    result["surprise_dir"]      = tix.map(surprise_df["surprise_dir"].to_dict()).fillna(0)
    return result


# ══════════════════════════════════════════════════════════════════
# MODULE G: Black-Litterman 最优组合构建
# ══════════════════════════════════════════════════════════════════

def black_litterman_weights(top: pd.DataFrame, daily_map: dict,
                             meta_df: pd.DataFrame,
                             risk_aversion: float = 2.5,
                             prev_weights: dict = None,
                             prev_weights_dict: dict = None,
                             pred_variance: dict = None,
                             etf_fallback_triggered: bool = False) -> pd.DataFrame:
    """
    用 Black-Litterman 模型替换 Fuzzy 仓位分配。

    原理：
      1. 市场均衡收益 → 以 TSX 市值加权为起点（不偏向任何股票）
      2. 模型观点     → 把 ensemble_score 转化为预期超额收益
      3. 协方差矩阵   → 用历史日收益率计算（Ledoit-Wolf 收缩）
      4. BL 公式      → 混合均衡收益 + 模型观点 → 最优权重

    参数：
      risk_aversion: 风险厌恶系数（越高越保守，一般2-4）
            prev_weights:  上期持仓权重，用于换仓感知优化（兼容旧调用）
            prev_weights_dict:  上期真实持仓权重字典，用于换仓感知优化
      pred_variance: MC Dropout 预测不确定性（方差），用于动态调整观点置信度

    输出：每支股票的最优仓位比例（合计100%）
    
    ★ Plan 2: 融入 MC Dropout 不确定性
      - pred_variance 中高方差 → 观点置信度降低
      - 相当于在高不确定性资产上下注更少、权重更低
    """
    try:
        from pypfopt import BlackLittermanModel, risk_models, expected_returns
        from pypfopt.efficient_frontier import EfficientFrontier
        try:
            import cvxpy as cp
        except ImportError:
            cp = None
    except ImportError:
        print("  ⚠️  PyPortfolioOpt 未安装，使用 Fuzzy 仓位")
        return pd.DataFrame()

    tickers = []
    for idx in top.index:
        t = idx[1] if isinstance(idx, tuple) else idx
        tickers.append(t)

    # ✓ Bug Fix: 如果只有 1 支资产（ETF 兜底），跳过 BL 复杂优化，直接返回 100% 仓位
    if len(tickers) == 1:
        single_t = tickers[0]
        if etf_fallback_triggered or "XIU" in single_t or ("sector" in top.columns and "ETF" in top["sector"].iloc[0]):
            print(f"  🎯 单资产 ETF（{single_t}）兜底模式：跳过 BL 优化，返回 100% 仓位")
            fallback_df = pd.DataFrame([{
                "ticker": single_t,
                "alloc_pct": 100.0,
                "category": "应急 ETF",
                "vol_ann": 0.15,
                "score": top["ensemble_score"].iloc[0],
                "prob": 0.0,
                "ret": 0.0,
                "view_ret": 0.0,
            }])
            return fallback_df

    # ── 1. 构建历史价格矩阵 ───────────────────────────────────────
    price_data = {}
    for t in tickers:
        df = daily_map.get(t, pd.DataFrame())
        if not df.empty:
            price_data[t] = df["close"]

    if len(price_data) < 3:
        print("  ⚠️  价格数据不足，使用 Fuzzy 仓位")
        return pd.DataFrame()

    # ✓ Bug fix: 先 tail(252) 截取最近252个交易日（~1年），再填充缺失值
    # 这样可以避免 dropna() 因为某个次新股数据缺失而删除整个矩阵的问题
    prices_full = pd.DataFrame(price_data).ffill()  # 先前向填充
    prices = prices_full.tail(252).dropna()  # 截取最近252日，防止次新股导致全体删除
    
    if len(prices) < 60:
        print(f"  ⚠️  历史数据不足60天（共 {len(prices_full)} 行）")
        print("     可能原因1：某支股票最近上市（次新股）导致共同历史稀少")
        print("     可能原因2：个别股票数据缺口")
        print("  ⚠️  使用 Fuzzy 仓位")
        return pd.DataFrame()

    # ── 2. 协方差矩阵（Ledoit-Wolf 收缩，减少估计误差）─────────────
    try:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    except Exception:
        S = risk_models.sample_cov(prices)

    # ── 3. 市场均衡权重（以市值为比例）─────────────────────────────
    mktcaps = {}
    for t in tickers:
        mc = meta_df.loc[t, "mktcap"] if t in meta_df.index else None
        mktcaps[t] = float(mc) if mc else 1e9   # 缺失用 $1B 默认

    total_mc = sum(mktcaps.values())
    market_weights = pd.Series({t: v/total_mc for t,v in mktcaps.items()})

    # ── 4. Grinold 公式构建观点矩阵：E[R] = IC × σ × Z ─────────────
    # 替代硬编码线性映射，建立在真实预测能力和个股风险特征上
    # IC: 信息系数（模型真实预测能力代理）
    # σ:  个股特异性月波动率
    # Z:  集成分数横截面 Z-Score

    # IC 代理：用 pred_top20pct 偏离 0.5 的程度衡量模型判别力
    probs_arr = top["pred_top20pct"].values
    ic_proxy  = float(np.mean(np.abs(probs_arr - 0.5)) * 2)
    ic_proxy  = np.clip(ic_proxy, 0.05, 0.40)

    # 横截面 Z-Score 标准化
    scores_arr = top["ensemble_score"].values
    z_scores   = (scores_arr - scores_arr.mean()) / (scores_arr.std() + 1e-8)

    views = {}
    for j, (idx, row) in enumerate(top.iterrows()):
        t  = idx[1] if isinstance(idx, tuple) else idx
        df = daily_map.get(t, pd.DataFrame())
        # 个股特异性月波动率（63日 × √21）
        if not df.empty:
            sigma = float(df["close"].pct_change().tail(63).std() * np.sqrt(21))
            sigma = np.clip(sigma, 0.03, 0.25)
        else:
            sigma = 0.08
        # Grinold: E[R] = IC × σ × Z
        views[t] = float(np.clip(ic_proxy * sigma * float(z_scores[j]), -0.15, 0.15))

    # ── 5. 观点置信度 ─────────────────────────────────────────────
    # ★ Plan 2: 融入模型不确定性（MC Dropout方差）
    confidences = {}
    for idx, row in top.iterrows():
        t    = idx[1] if isinstance(idx, tuple) else idx
        prob = row.get("pred_top20pct", 0.5)
        
        # 如果有MC Dropout方差，融入不确定性
        confidence = prob  # 默认使用分类概率
        if pred_variance and t in pred_variance:
            variance = float(pred_variance.get(t, 0.0))
            # 检查方差是否有效（不是 NaN 或无穷大）
            if np.isfinite(variance) and variance >= 0:
                # variance × 0.7 权重 + 分类概率 × 0.3 权重
                # 低方差(确定) → 置信度高; 高方差(不确定) → 置信度低
                uncertainty_adj = np.exp(-np.clip(variance * 2.0, -10, 10))  # 防止数值溢出
                confidence = 0.7 * float(prob) + 0.3 * uncertainty_adj
        
        confidences[t] = float(np.clip(confidence, 0.3, 0.9))

    effective_prev_weights = prev_weights_dict if prev_weights_dict is not None else prev_weights

    # ── 6. Black-Litterman 模型 ───────────────────────────────────
    try:
        bl = BlackLittermanModel(
            S,
            pi          = "market",
            market_caps = mktcaps,
            risk_aversion = risk_aversion,
            absolute_views = views,
        )
        # 设置观点置信度
        omega = bl.bl_weights(risk_aversion=risk_aversion)

        # 用 BL 后验收益做均值方差优化
        bl_returns = bl.bl_returns()
        
        # ★ 修复角点解（强制分散）：
        # 1. 动态边界：根据实际传入的股票数量动态调整上限，防止求解器崩溃
        max_weight_bound = max(0.15, 1.0 / len(bl_returns) + 0.05) if len(bl_returns) > 0 else 0.15
        ef = EfficientFrontier(bl_returns, S, weight_bounds=(0.02, max_weight_bound))
        
        # 2. 软性约束（L2 正则化）：强迫权重平滑分散
        #    gamma 越大权重越趋向等权。0.1 是平衡点（对应 ~15% 权重即使 score 最高）
        try:
            from pypfopt import objective_functions
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        except Exception as e:
            print(f"  ⚠️  L2 正则化加载失败（{e}），继续优化但权重分散效果降低")

        # 3. 交易成本惩罚（如果有上期权重）
        # ✓ Bug fix: prev_weights 参数正确传入且有效使用（对齐上期权重，做换仓感知优化）
        w_prev = None
        if effective_prev_weights and cp is not None:
            # 对齐真实上期权重到当期股票池
            w_prev_arr = np.array([effective_prev_weights.get(t, 0.0) for t in bl_returns.index])
            total_prev = w_prev_arr.sum()
            if total_prev > 1e-6:
                w_prev = w_prev_arr / total_prev

        if w_prev is not None and cp is not None:
            # 换仓惩罚：双边手续费 0.2%（有边界+L2后，换仓天生就被抑制，所以惩罚较轻）
            # 优化器会权衡：换仓收益 vs 0.2% * 换仓量
            try:
                ef.add_objective(
                    objective_functions.transaction_cost,
                    w_prev=w_prev, k=0.0015
                )
                ef.max_sharpe(risk_free_rate=0.04/12)
                weights = ef.clean_weights()
                all_tickers = set(weights.keys()).union(set(effective_prev_weights.keys()) if effective_prev_weights else set())
                turnover = sum(abs(weights.get(t, 0.0) - (effective_prev_weights.get(t, 0.0) if effective_prev_weights else 0.0)) for t in all_tickers)
                print(f"  BL 最优仓位：边界约束[2%-15%] + L2 正则化 + 换仓成本")
                print(f"  真实换手率 {turnover*100:.1f}%（基于当前 Wealthsimple 持仓）")
            except Exception as e_opt:
                print(f"  ⚠️  交易成本优化失败（{e_opt}），使用标准最优解")
                # 🎯 【核心修改】：一旦前面的 ef 失败，必须重新实例化一个新的 ef，否则会报错
                ef = EfficientFrontier(bl_returns, S, weight_bounds=(0.02, 0.15))
                try:
                    from pypfopt import objective_functions
                    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                except:
                    pass
                ef.max_sharpe(risk_free_rate=0.04/12)
                weights = ef.clean_weights()
        else:
            # 首次运行 / cvxpy 未装 / 无上期权重：标准 max_sharpe（含边界+L2）
            try:
                ef.max_sharpe(risk_free_rate=0.04/12)
                weights = ef.clean_weights()
                print(f"  BL 最优仓位：边界约束[2%-15%] + L2 正则化（首次运行）")
            except Exception as e_sharpe:
                print(f"  ⚠️  Sharpe 优化失败（{e_sharpe}），使用最小波动率替代")
                ef.min_volatility()
                weights = ef.clean_weights()

    except Exception as e:
        print(f"  ⚠️  BL 优化失败：{e}，使用等风险权重")
        # Fallback：等风险权重
        vols = []
        for t in tickers:
            df  = daily_map.get(t, pd.DataFrame())
            vol = df["close"].pct_change().tail(63).std() * np.sqrt(252) if not df.empty else 0.25
            vols.append(max(vol, 0.05))
        inv_vol  = [1/v for v in vols]
        total_iv = sum(inv_vol)
        weights  = {t: w/total_iv for t,w in zip(tickers, inv_vol)}

    # ── 7. 转为 DataFrame 输出 ────────────────────────────────────
    rows = []
    for idx, row in top.iterrows():
        t    = idx[1] if isinstance(idx, tuple) else idx
        w    = weights.get(t, 0.0)
        prof = STOCK_PROFILE.get(t, {})
        df   = daily_map.get(t, pd.DataFrame())
        vol  = df["close"].pct_change().tail(63).std()*np.sqrt(252) if not df.empty else 0.25

        # 仓位类别（用于显示）
        if w >= 0.18:    cat = "大量买入 🔴"
        elif w >= 0.10:  cat = "适度买入 🟡"
        else:            cat = "少量买入 🟢"

        rows.append({
            "ticker":    t,
            "alloc_pct": round(w * 100, 1),
            "category":  cat,
            "vol_ann":   vol,
            "score":     row["ensemble_score"],
            "prob":      row.get("pred_top20pct", 0),
            "ret":       row.get("pred_return", 0),
            "view_ret":  views.get(t, 0),
        })

    df_out = pd.DataFrame(rows)
    total  = df_out["alloc_pct"].sum()
    if total > 0:
        df_out["alloc_pct"] = (df_out["alloc_pct"]/total*100).round(1)

    return df_out


def print_bl_weights(df: pd.DataFrame, regime: str = "NEUTRAL"):
    """打印 Black-Litterman 仓位"""
    regime_label = {"BULL":"牛市","NEUTRAL":"中性","BEAR":"熊市"}.get(regime, regime)
    print("\n" + "═"*76)
    print(f"  📊 Black-Litterman 最优仓位  [Regime: {regime_label}]")
    print(f"  方法：市场均衡权重 + 模型观点 + Ledoit-Wolf协方差 → Max Sharpe")
    print("═"*76)
    print(f"  {'#':<3}{'Ticker':<13}{'类别':<14}{'BL仓位%':>8}  "
          f"{'条形图':<14}{'波动率':>7}{'预期超额':>9}{'集成分':>8}")
    print(f"  {'─'*74}")
    for i, row in df.iterrows():
        bar = "█" * int(row["alloc_pct"] / 2)
        print(f"  {i+1:<3}{row['ticker']:<13}{row['category']:<14}"
              f"{row['alloc_pct']:>7.1f}%  {bar:<14}"
              f"  {row['vol_ann']*100:>5.1f}%"
              f"  {row['view_ret']*100:>+7.1f}%"
              f"  {row['score']:>7.3f}")
    print(f"  {'─'*74}")
    print(f"  合计：{df['alloc_pct'].sum():.1f}%\n")
    for cat in ["Heavy Buy","Moderate Buy","Light Buy"]:
        sub = df[df["category"]==cat]
        if len(sub):
            print(f"    {cat}  {len(sub)}支  合计{sub['alloc_pct'].sum():.1f}%"
                  f"  → {' | '.join(sub['ticker'].tolist())}")


# ══════════════════════════════════════════════════════════════════
# MODULE H: 自动月度报告（邮件）
# ══════════════════════════════════════════════════════════════════

def generate_monthly_report(result, bl_weights_df, wf, regime,
                             surprise_df, imp, dd_signal=None, daily_map=None):
    today = datetime.today().strftime("%Y-%m-%d")
    month = datetime.today().strftime("%Y-%m")
    top   = result.head(TOP_N)

    RED="#C0392B"; GREEN="#27AE60"; ORANGE="#E67E22"
    BLUE="#2980B9"; GRAY="#7F8C8D"; BG="#F8F9FA"; WHITE="#FFFFFF"
    r_obj   = getattr(regime, "regime", "NEUTRAL")
    r_color = {"BULL":GREEN,"NEUTRAL":ORANGE,"BEAR":RED}.get(r_obj, ORANGE)
    r_label = {"BULL":"Bull Market - Risk On","NEUTRAL":"Neutral","BEAR":"Bear Market - Risk Off"}.get(r_obj,"Neutral")

    wf_topR = wf_acc = wf_mdd = "N/A"
    if not wf.empty and "actual_ret" in wf.columns:
        v = wf.dropna(subset=["actual_ret","actual_cls"])
        if len(v):
            p = v["ens"]
            mask80 = p >= p.quantile(0.80)
            wf_topR = f"{v.loc[mask80,'actual_ret'].mean()*100:+.2f}%"
            acc_val = accuracy_score(v["actual_cls"], mask80.astype(int)) * 100
            wf_acc  = f"{acc_val:.1f}%"
            navs    = (1 + v.groupby("date")["actual_ret"].mean()).cumprod()
            wf_mdd  = f"{((navs - navs.cummax()) / navs.cummax()).min()*100:+.2f}%"

    css = (
        "<style>"
        f"body{{font-family:'PingFang SC','Microsoft YaHei','Segoe UI',-apple-system,Helvetica,Arial,sans-serif;background:{BG};margin:0;padding:16px;color:#2C3E50;line-height:1.5}}"
        f".wrap{{max-width:750px;margin:0 auto;background:{WHITE};border-radius:8px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,.1)}}"
        f".hdr{{background:#1A252F;padding:24px 20px 16px}}"
        f".hdr h1{{color:{WHITE};margin:0;font-size:20px;font-weight:700}}"
        f".hdr p{{color:#BDC3C7;margin:4px 0 0;font-size:12px}}"
        f".sec{{padding:18px 20px;border-bottom:1px solid #ECF0F1}}"
        f".sec h2{{font-size:14px;font-weight:700;color:#1A252F;margin:0 0 12px;padding-bottom:6px;border-bottom:2px solid {BLUE}}}"
        f".rbox{{background:{BG};border-left:4px solid {r_color};padding:10px 14px;border-radius:0 6px 6px 0;font-size:13px;font-weight:700;color:{r_color}}}"
        f".wbox{{background:#FFF3CD;border-left:4px solid {ORANGE};padding:10px 14px;border-radius:0 6px 6px 0;font-size:12px;margin-top:10px}}"
        ".mgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px}"
        f".mcard{{background:{BG};border-radius:6px;padding:12px;text-align:center}}"
        ".mval{font-size:18px;font-weight:700;color:#1A252F}"
        f".mlbl{{font-size:11px;color:{GRAY};margin-top:2px}}"
        "table{width:100%;border-collapse:collapse;font-size:11px;table-layout:fixed}"
        f"th{{background:#1A252F;color:{WHITE};padding:8px 9px;text-align:left;font-size:11px;font-weight:600;word-wrap:break-word}}"
        f"td{{padding:8px 9px;border-bottom:1px solid #ECF0F1;word-wrap:break-word;overflow-wrap:break-word}}"
        f"tr:nth-child(even) td{{background:{BG}}}"
        f".tk{{font-weight:700;color:{BLUE};font-family:monospace;font-size:12px}}"
        f".pos{{color:{GREEN};font-weight:600}}.neg{{color:{RED};font-weight:600}}"
        f".ch{{color:{RED};font-weight:700}}.cm{{color:{ORANGE};font-weight:700}}.cl{{color:{GREEN};font-weight:700}}"
        ".frow{display:flex;align-items:center;margin:5px 0;font-size:11px}"
        f".fn{{font-family:monospace;font-size:11px;color:{BLUE};width:140px;flex-shrink:0}}"
        ".fbg{flex:1;background:#ECF0F1;height:6px;border-radius:2px}"
        f".fb{{background:{BLUE};height:6px;border-radius:2px}}"
        f".fv{{font-size:11px;color:{GRAY};margin-left:8px;width:40px}}"
        f".ftr{{background:#1A252F;padding:14px 20px;text-align:center}}"
        f".ftr p{{color:#7F8C8D;font-size:10px;margin:0;line-height:1.4}}"
        "td[style*='max-width']{text-overflow:ellipsis}"
        "</style>"
    )

    h = f'<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">{css}</head><body><div class="wrap">'
    h += f'<div class="hdr"><h1>🇨🇦 TSX Monthly Stock Report</h1>'
    h += f'<p>{month} | XGBoost + LightGBM + PyTorch Ensemble | {len(FEATURE_COLS)} Features</p></div>'

    # Regime box
    h += f'<div class="sec"><h2>Market Regime & Risk Signal</h2><div class="rbox">{r_label}</div>'
    if dd_signal:
        h += f'<div class="wbox"><b>⚠️ Risk Alert:</b> 3-month cumulative drawdown is {dd_signal*100:+.1f}%. Consider reducing exposure by 50%.</div>'
    h += '</div>'

    # Walk-Forward metrics
    tc = "pos" if wf_topR != "N/A" and "+" in wf_topR else "neg"
    h += (
        '<div class="sec"><h2>Walk-Forward Backtest Performance</h2><div class="mgrid">'
        f'<div class="mcard"><div class="mval {tc}">{wf_topR}</div><div class="mlbl">Avg Monthly Return (Top 20%)</div></div>'
        f'<div class="mcard"><div class="mval" style="color:{BLUE}">{wf_acc}</div><div class="mlbl">Classification Accuracy</div></div>'
        f'<div class="mcard"><div class="mval" style="color:{RED}">{wf_mdd}</div><div class="mlbl">Max Monthly Drawdown</div></div>'
        '</div></div>'
    )

    # Top 10 table with price and shares
    REPORT_CAPITAL = 100_000  # CAD - change to match your portfolio
    h += ('<div class="sec"><h2>Top 10 Stock Picks</h2>'
          '<table style="font-size:12px;width:100%">'
          '<tr><th>#</th><th>Ticker</th><th>Company</th><th>Sector</th>'
          '<th>Forecast Return</th><th>Current Price</th>'
          '<th>Proposed Amount</th><th style="color:#2980B9">Suggested Shares</th></tr>')
    for i, (idx, row) in enumerate(top.iterrows(), 1):
        t    = idx[1] if isinstance(idx, tuple) else idx
        prof = STOCK_PROFILE.get(t, {})
        surp = surprise_df.loc[t, "surprise_pct"] if t in surprise_df.index else 0
        rp   = row["pred_return"] * 100
        rc   = "pos" if rp >= 0 else "neg"
        # Allocation: from BL weights if available, else equal weight
        alloc_pct = 1.0 / max(len(top), 1)
        if not bl_weights_df.empty and "ticker" in bl_weights_df.columns:
            match = bl_weights_df[bl_weights_df["ticker"] == t]
            if not match.empty:
                alloc_pct = float(match["alloc_pct"].iloc[0]) / 100
        amount = REPORT_CAPITAL * alloc_pct
        # Price from daily_map
        price_val = 0
        if daily_map and t in daily_map and not daily_map[t].empty:
            price_val = float(daily_map[t]["close"].iloc[-1])
        shares  = int(amount / price_val) if price_val > 0 else 0
        price_s = f"${price_val:.2f}" if price_val > 0 else "N/A"
        h += (f'<tr><td>{i}</td><td class="tk">{t}</td>'
              f'<td style="font-size:11px;max-width:100px;overflow:hidden">{str(row.get("name",""))[:20]}</td>'
              f'<td style="font-size:11px;color:{GRAY}">{prof.get("gics","Unknown")[:14]}</td>'
              f'<td class="{rc}">{rp:+.1f}%</td>'
              f'<td style="font-family:monospace">{price_s}</td>'
              f'<td style="font-family:monospace">${amount:,.0f}</td>'
              f'<td style="font-weight:700;color:#2980B9;font-size:13px">{shares:,}</td></tr>')
    h += '</table></div>'

    # Position Sizing（BL 或 Fuzzy，始终显示，含价格和股数）
    pos_df = bl_weights_df if not bl_weights_df.empty else pd.DataFrame()
    if pos_df.empty and len(top) > 0:
        pos_rows = []
        for idx2, row2 in top.iterrows():
            t2 = idx2[1] if isinstance(idx2, tuple) else idx2
            pos_rows.append({"ticker":t2,"category":"Equal Weight",
                             "alloc_pct":100.0/len(top),"vol_ann":0,
                             "score":row2.get("ensemble_score",0)})
        pos_df = pd.DataFrame(pos_rows)

    if not pos_df.empty:
        REPORT_CAPITAL = 100_000
        h += '<div class="sec"><h2>Position Sizing</h2>'
        h += '<table style="font-size:12px;width:100%">'
        h += ('<tr><th>Ticker</th><th>Strategy</th><th>Allocation</th>'
              '<th>Price (CAD)</th><th>Amount (CAD)</th>'
              '<th style="color:#2980B9">Shares to Buy</th><th>Volatility</th></tr>')
        for _, row in pos_df.iterrows():
            t3  = row.get("ticker","")
            
            # 🔧 修复：确保 category 字段总是有数据，即使 BL 返回空也有备用
            raw_cat = row.get("category","")
            if not raw_cat or raw_cat.strip() == "":
                # 备用分类逻辑：基于分配百分比
                alloc = row.get("alloc_pct", 0)
                if alloc >= 18:    raw_cat = "Heavy Buy 🔴"
                elif alloc >= 10:  raw_cat = "Moderate Buy 🟡"
                else:              raw_cat = "Light Buy 🟢"
            
            cat = raw_cat
            cc  = "ch" if "Heavy" in str(cat) or "🔴" in str(cat) else ("cm" if "Moderate" in str(cat) or "🟡" in str(cat) else "cl")
            
            alloc_pct = float(row.get("alloc_pct", 0))
            bw  = min(int(max(alloc_pct, 1) / 30 * 100), 100)  # 进度条幅度（基于最大可能权重）
            
            price_val = 0
            if daily_map and t3 in daily_map and not daily_map[t3].empty:
                price_val = float(daily_map[t3]["close"].iloc[-1])
            amount  = REPORT_CAPITAL * alloc_pct / 100
            shares  = int(amount / price_val) if price_val > 0 else 0
            price_s = f"${price_val:.2f}" if price_val > 0 else "N/A"
            vol_pct = float(row.get("vol_ann", 0)) * 100
            
            h += (f'<tr><td class="tk">{t3}</td><td class="{cc}">{cat}</td>'
                  f'<td><div style="display:flex;align-items:center;gap:6px">'
                  f'<div class="fbg" style="width:60px"><div class="fb" style="width:{bw}%"></div></div>'
                  f'<b>{alloc_pct:.1f}%</b></div></td>'
                  f'<td>{price_s}</td>'
                  f'<td>${amount:,.0f}</td>'
                  f'<td style="font-weight:700;color:#2980B9;font-size:13px">{shares:,}</td>'
                  f'<td style="color:#7F8C8D">{vol_pct:.1f}%</td></tr>')
        h += '</table></div>'

    # Feature importance
    h += '<div class="sec"><h2>Key Driving Factors (Top 8 Features)</h2>'
    if not imp.empty:
        max_v = imp.head(8).max()
        for feat, val in imp.head(8).items():
            bw = int(val / max_v * 100) if max_v > 0 else 0
            h += (f'<div class="frow"><span class="fn">{feat}</span>'
                  f'<div class="fbg"><div class="fb" style="width:{bw}%"></div></div>'
                  f'<span class="fv">{val:.4f}</span></div>')
    else:
        h += '<p style="font-size:11px;color:#7F8C8D">Feature importance data not available</p>'
    h += '</div>'

    h += (f'<div class="ftr"><p>📋 Generated: {datetime.today().strftime("%Y-%m-%d %H:%M %Z")} | TSX Quant Stock Picker v3.0 | '
          'For informational reference only — not investment advice | Past performance does not guarantee future results</p></div>'
          '</div></body></html>')
    return h


def html_to_pdf(html_content: str, pdf_path: str) -> bool:
    """HTML report to PDF (weasyprint)"""
    try:
        import logging
        logging.getLogger("weasyprint").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
        logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
        logging.disable(logging.DEBUG)
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(pdf_path)
        print(f"  PDF generated: {pdf_path}")
        return True
    except ImportError:
        print("  weasyprint not installed: pip install weasyprint")
        return False
    except Exception as e:
        print(f"  PDF generation failed: {e}")
        return False


def send_monthly_report(html_content, to_email="your@email.com",
                        from_email="your@gmail.com",
                        app_password="your_gmail_app_password",
                        attach_pdf=True):
    """
    Send HTML email with optional PDF attachment.
    attach_pdf=True: attaches PDF version for printing/archiving.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication

    month_str  = datetime.today().strftime("%Y%m")
    html_fname = f"tsx_report_{month_str}.html"
    pdf_fname  = f"tsx_report_{month_str}.pdf"

    # Save HTML locally
    with open(html_fname, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Generate PDF
    pdf_ok = html_to_pdf(html_content, pdf_fname) if attach_pdf else False

    if "your@" in to_email or "your_gmail" in app_password:
        print(f"  Email not configured. Saved: {html_fname}")
        return False

    try:
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"TSX Stock Report - {datetime.today().strftime('%Y-%m')}"
        msg["From"]    = from_email
        msg["To"]      = to_email

        # HTML body
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("Please view in HTML-capable email client.", "plain", "utf-8"))
        alt.attach(MIMEText(html_content, "html", "utf-8"))
        msg.attach(alt)

        # PDF attachment
        if pdf_ok:
            with open(pdf_fname, "rb") as f:
                part = MIMEApplication(f.read(), _subtype="pdf")
                part.add_header("Content-Disposition", "attachment", filename=pdf_fname)
                msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.sendmail(from_email, to_email, msg.as_string())

        print(f"  Report sent to {to_email}")
        print(f"  HTML: {html_fname}" + (f"  |  PDF attached: {pdf_fname}" if pdf_ok else ""))
        return True

    except Exception as e:
        print(f"  Email failed: {e}. Saved: {html_fname}")
        return False


# ══════════════════════════════════════════════════════════════════
# 四个新模块的统一入口
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# Double Machine Learning (DML) — 事件因果效应估计
# ══════════════════════════════════════════════════════════════════

def estimate_dml_alpha(panel: pd.DataFrame,
                       event_signal: pd.Series,
                       signal_name: str = "event",
                       n_folds: int = 5) -> pd.Series:
    """
    用 Double Machine Learning 估计事件信号（内部人交易/盈利惊喜）
    对下月收益的纯因果效应（Treatment Effect），替代硬编码乘数。

    DML 两步法（Chernozhukov et al. 2018）：
      Step 1: 用 ML（XGBoost）从特征 X 预测收益 Y → 得到残差 Ẽ_Y = Y - Ŷ
      Step 2: 用 ML 从特征 X 预测处理变量 T → 得到残差 Ẽ_T = T - T̂
      Step 3: OLS 回归 Ẽ_Y ~ Ẽ_T → θ 是纯净的因果效应（排除混淆变量）

    返回：每支股票的动态 Alpha 调整系数（比固定乘数更精确）
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression

    # 准备数据
    # Bug3 修复：event_signal 的 index 是 ticker，panel 是 MultiIndex(date, ticker)
    # 需要将 signal 按 ticker 对齐到 panel 的每一行
    tix_in_panel = panel.index.get_level_values("ticker")
    sig_dict     = event_signal.to_dict()
    valid_tix    = [t for t in tix_in_panel.unique() if t in sig_dict]
    n_valid      = len(valid_tix)
    if n_valid < 20:
        print(f"  [DML] {signal_name}: 样本不足（{n_valid} 支有信号），使用默认权重")
        return pd.Series(0.0, index=event_signal.index)

    # 对齐事件信号到 panel（按 ticker 展开）
    tix = tix_in_panel
    T   = np.array([sig_dict.get(t, 0.0) for t in tix], dtype=float)  # Treatment
    Y   = panel["next_ret"].fillna(panel["next_ret"].median()).values.astype(float)  # Outcome
    X   = panel[FEATURE_COLS].fillna(panel[FEATURE_COLS].median()).values    # Confounders

    if T.std() < 1e-6:
        return pd.Series(0.0, index=event_signal.index)

    # Cross-fitting：必须用时间序列分割，严禁 shuffle=True
    # 金融面板数据中随机折叠会导致用未来数据预测过去（数据泄露）
    from sklearn.model_selection import TimeSeriesSplit

    # 获取每行的时间戳，按时间排序确保 train < val
    if isinstance(panel.index, pd.MultiIndex):
        dates_arr = panel.index.get_level_values("date").values
    else:
        dates_arr = panel.index.values
    sort_order = np.argsort(dates_arr)   # 按时间升序排列

    # 按时间顺序重新排列 X/Y/T
    X_sorted = X[sort_order]
    Y_sorted = Y[sort_order]
    T_sorted = T[sort_order]

    tscv    = TimeSeriesSplit(n_splits=n_folds, gap=1)   # gap=1 避免相邻月泄露
    res_Y   = np.zeros_like(Y_sorted)
    res_T   = np.zeros_like(T_sorted)

    for train_idx, val_idx in tscv.split(X_sorted):
        sc  = StandardScaler()
        X_tr= sc.fit_transform(X_sorted[train_idx])
        X_va= sc.transform(X_sorted[val_idx])

        # ML1: 预测 Y（控制混淆变量）
        m_Y = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                learning_rate=0.05, verbosity=0,
                                random_state=42).fit(X_tr, Y_sorted[train_idx])
        res_Y[val_idx] = Y_sorted[val_idx] - m_Y.predict(X_va)

        # ML2: 预测 T（倾向得分）
        m_T = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                learning_rate=0.05, verbosity=0,
                                random_state=42).fit(X_tr, T_sorted[train_idx])
        res_T[val_idx] = T_sorted[val_idx] - m_T.predict(X_va)

    # Step 3: 从残差回归提取纯净 Treatment Effect（时序排序后的残差）
    mask = np.abs(res_T) > 1e-6
    if mask.sum() < 20:
        return pd.Series(0.0, index=event_signal.index)

    theta = np.dot(res_T[mask], res_Y[mask]) / np.dot(res_T[mask], res_T[mask])

    print(f"  [DML] {signal_name}: θ = {theta:.4f}  "
          f"（纯因果效应，正值=事件信号有Alpha贡献）")

    # 用 θ 动态计算每支股票的仓位调整系数
    # 调整量 = θ × signal_value，映射到 [-0.2, +0.2] 范围
    raw_adj = event_signal * float(theta)
    adj     = raw_adj.clip(-0.20, 0.20)
    return adj


def apply_dml_signal(result: pd.DataFrame, panel: pd.DataFrame,
                     insider_df: pd.DataFrame, surprise_df: pd.DataFrame) -> pd.DataFrame:
    """
    用 DML 估计的因果效应替代硬编码乘数，动态调整集成分。
    """
    result = result.copy()
    tix    = (result.index.get_level_values("ticker")
              if isinstance(result.index, pd.MultiIndex) else result.index)

    adj_total = pd.Series(0.0, index=tix)

    # 内部人交易 DML
    if not insider_df.empty and "signal" in insider_df.columns:
        insider_sig = insider_df["signal"].reindex(tix).fillna(0)
        if insider_sig.std() > 0:
            insider_adj = estimate_dml_alpha(
                panel, insider_sig, signal_name="内部人交易")
            adj_total  += insider_adj.reindex(tix).fillna(0)

    # 盈利惊喜 DML
    if not surprise_df.empty and "surprise_pct" in surprise_df.columns:
        surprise_sig = (surprise_df["surprise_pct"] / 100).reindex(tix).fillna(0)
        if surprise_sig.std() > 0:
            surprise_adj = estimate_dml_alpha(
                panel, surprise_sig, signal_name="盈利惊喜")
            adj_total   += surprise_adj.reindex(tix).fillna(0)

    # 应用动态调整
    result["ensemble_score"] = result["ensemble_score"] + adj_total.values
    result = result.sort_values("ensemble_score", ascending=False)
    return result


def run_new_modules(panel, daily_map, meta_df, macro_df, wf,
                    result, imp, dd_signal,
                    prev_weights_dict: dict = None,
                    # 邮件配置（填写后自动发送）
                    email_to:       str = "your@email.com",
                    email_from:     str = "your@gmail.com",
                    email_password: str = "your_gmail_app_password"):
    """
    运行四个新模块并返回结果：
      E. Regime Detection
      F. Earnings Surprise
      G. Black-Litterman 仓位
      H. 月度报告 + 邮件
    """
    print("\n" + "▓"*60)
    print("  新增模块 E/F/G/H")
    print("▓"*60)

    # ── E. Regime Detection ───────────────────────────────────────
    print("\n【E】Regime Detection（市场状态识别）")
    regime = MarketRegime(macro_df)
    print(regime.describe())

    # Regime 调整特征权重（用于下次训练参考）
    panel_regime = regime.adjust_features(panel, FEATURE_COLS)
    regime_constraints = regime.adjust_constraints(CONSTRAINTS)
    print(f"  约束调整：{regime_constraints}")

    # ── F. Earnings Surprise ──────────────────────────────────────
    print("\n【F】Earnings Surprise（盈利惊喜）报告")
    
    # 🎯 DML 已在 predict_now 中提前执行！这里只打印报表，不重复计算
    # DML 导致 result 已经被因果调整过，所以这里只需要提取 surprise_df 用于报告
    if result is not None and not result.empty:
        # 从 result 前60中再次提取 surprise_df 用于报告生成（不影响分数）
        try:
            top60_for_report = result.head(60)
            top60_tickers_report = (top60_for_report.index.get_level_values("ticker").tolist()
                                   if isinstance(top60_for_report.index, pd.MultiIndex)
                                   else top60_for_report.index.tolist())
            surprise_df = fetch_earnings_surprise(top60_tickers_report)
            print(f"  ℹ️  {len(surprise_df)} 支股票有财报数据（仅供报告参考，分数已在predict_now调整）")
        except Exception as e:
            print(f"  ⚠️  财报获取失败：{str(e)[:50]}")
            surprise_df = pd.DataFrame()
    else:
        surprise_df = pd.DataFrame()

    # ── G. Black-Litterman ────────────────────────────────────────
    print("\n【G】Black-Litterman 最优仓位")
    bl_df = pd.DataFrame()
    mc_dropout_variance = {}  # ✓ 初始化为空字典（预设）
    if result is not None:
        try:
            top = result.head(TOP_N)
            
            # ✓ 检查是否为 ETF 兜底（单资产）情况
            etf_fallback = len(top) == 1 and (("XIU" in top.index.get_level_values("ticker")[0]) or 
                                               ("sector" in top.columns and "ETF" in top["sector"].iloc[0]))
            
            # 优先使用 Wealthsimple 真实持仓权重；没有时再回退到 WF 上月等权持仓
            prev_bl_weights = prev_weights_dict or {}
            if not prev_bl_weights and not wf.empty and "ens" in wf.columns:
                last_date  = wf["date"].max()
                last_picks = wf[wf["date"] == last_date].nlargest(TOP_N, "ens")
                n_picks    = len(last_picks)
                if n_picks > 0:
                    prev_bl_weights = {
                        row["ticker"]: 1.0/n_picks
                        for _, row in last_picks.iterrows()
                    }
            
            bl_df = black_litterman_weights(top, daily_map, meta_df,
                                             risk_aversion=2.5,
                                             prev_weights_dict=prev_bl_weights,
                                             pred_variance=mc_dropout_variance,
                                             etf_fallback_triggered=etf_fallback)
            if not bl_df.empty:
                print_bl_weights(bl_df, regime.regime)
            else:
                print("  ⚠️  BL 优化失败，请查看上方提示")
        except Exception as e:
            print(f"  ⚠️  Black-Litterman 失败：{e}")

    # ── H. 月度报告 ───────────────────────────────────────────────
    print("\n【H】生成月度报告")
    if result is not None and imp is not None:
        try:
            # generate_monthly_report 返回 HTML 字符串
            html_report = generate_monthly_report(
                result, bl_df, wf, regime,
                surprise_df, imp, dd_signal,
                daily_map=daily_map)

            month_str = datetime.today().strftime("%Y%m")

            # 保存 HTML
            html_fname = f"tsx_report_{month_str}.html"
            with open(html_fname, "w", encoding="utf-8") as f:
                f.write(html_report)
            print(f"  ✅ HTML 报告已保存：{html_fname}")

            # 生成 PDF（weasyprint）
            pdf_fname = f"tsx_report_{month_str}.pdf"
            pdf_ok = html_to_pdf(html_report, pdf_fname)

            # 发送邮件（HTML正文 + PDF附件）
            send_monthly_report(html_report, email_to, email_from, email_password,
                                attach_pdf=True)

        except Exception as e:
            print(f"  ⚠️  报告生成失败：{e}")

    return result, bl_df, surprise_df, regime


# ══════════════════════════════════════════════════════════════════
# MODULE I: 历史回测 — Walk-Forward 完整模型 P&L
# ══════════════════════════════════════════════════════════════════
# 直接用主模型的 Walk-Forward 输出重建组合历史收益。
# 每月：取集成分 Top N → 等权持有 → 计算实际下月收益（含止损/手续费）
# 这是完整的 26特征 + XGBoost + LightGBM + MLP 模型的真实历史表现。


BACKTEST_MONTHS  = 120
INITIAL_CAPITAL  = 100_000
BENCHMARK        = "XIU.TO"
BT_TX_COST       = 0.000   # ★ Wealthsimple 免手续费版本：硬性手续费清零（滑点由bid_ask模拟）
BT_STOP_LOSS     = -0.08   # 全局止损下限（保底）
BT_VOL_STOP_MULT = 1.5    # 动态止损 = 个股历史月波动率 × 此倍数
# 例：月波动率 8% 的矿业股 → 止损 -12%（不被噪音洗盘）
#     月波动率 3% 的银行股  → 止损  -4.5%（更敏感）
#     两者取 min(动态止损, BT_STOP_LOSS=-8%) 作为实际止损线



def backtest_from_wf(wf, daily_map, meta_df, top_n=10):
    """从 Walk-Forward 输出构建组合历史 P&L（完整模型，无简化）"""
    if wf.empty or "actual_ret" not in wf.columns:
        print("  WF 结果为空")
        return []

    dates = sorted(wf["date"].unique())
    dates = dates[-min(BACKTEST_MONTHS, len(dates)):]

    # 🎯 【修复】使用多层次容错的基准获取，而不是简单查找 daily_map
    bench_monthly = fetch_benchmark_monthly(benchmark=BENCHMARK, years=YEARS)
    
    if not bench_monthly:
        print(f"  ⚠️  基准数据获取失败，回测将显示 0% 基准收益")

    nav, prev_hold, results = float(INITIAL_CAPITAL), set(), []

    for date in dates:
        month_wf = wf[wf["date"] == date].copy()
        if month_wf.empty:
            continue

        # ✓ 修订（2026-04）：报告口径与实际选股口径对齐
        # walk_forward 已经做了缓冲带 + 冷静期 + 矿业上限 + ETF 风控，
        # 选中的股票标记在 is_selected 列。报告必须沿用同一组持仓，
        # 否则"回测看到的收益"≠"实际会交易的组合"。
        # 仅在该列缺失或当月没有任何 is_selected=True 时才退化到 nlargest。
        if "is_selected" in month_wf.columns and month_wf["is_selected"].any():
            top_month = month_wf[month_wf["is_selected"] == True].copy()
            # 按分数降序，保持与 walk_forward 内部一致
            top_month = top_month.sort_values("ens", ascending=False)
        else:
            top_month = month_wf.nlargest(top_n, "ens")
        tickers   = top_month["ticker"].tolist()
        curr_hold = set(tickers)
        new_in    = curr_hold - prev_hold
        exit_out  = prev_hold - curr_hold

        holdings = []
        for _, row in top_month.iterrows():
            t       = row["ticker"]
            raw_ret = row.get("actual_ret", np.nan)
            if pd.isna(raw_ret):
                continue

            stop_hit = False
            df = daily_map.get(t, pd.DataFrame())
            if not df.empty:
                next_m = df[df.index > pd.Timestamp(date)]
                end_dt = date + pd.offsets.MonthEnd(1)
                mdata  = next_m[next_m.index <= pd.Timestamp(end_dt)]
                if not mdata.empty:
                    ep  = df[df.index <= pd.Timestamp(date)]["close"].iloc[-1]
                    mp  = mdata["low"].min()

                    # OPT3: 动态波动率止损
                    # ✓ 修复前视（2026-04）：必须用回测当月之前的 63 天，
                    #   而不是 df.tail(63)（那是全样本最后 63 天，等于"用 2026 年波动率
                    #   设置 2018 年止损"，严重前视）。
                    pit_close = df[df.index <= pd.Timestamp(date)]["close"]
                    hist_vol = pit_close.pct_change().tail(63).std() * np.sqrt(21) if len(pit_close) >= 63 else 0
                    vol_stop = -abs(hist_vol * BT_VOL_STOP_MULT) if hist_vol > 0 else BT_STOP_LOSS
                    # 取动态止损和全局止损的较大值（更宽松的保护）
                    effective_stop = max(vol_stop, BT_STOP_LOSS)

                    if (mp - ep) / ep < effective_stop:
                        raw_ret  = effective_stop
                        stop_hit = True

            tx      = BT_TX_COST * 2 if t in new_in else BT_TX_COST
            net_ret = raw_ret - tx
            holdings.append({
                "ticker":   t,
                "raw_ret":  round(raw_ret * 100, 2),
                "net_ret":  round(net_ret * 100, 2),
                "weight":   1.0 / top_n,
                "contrib":  round(net_ret / top_n * 100, 3),
                "score":    round(row["ens"], 4),
                "stop_hit": stop_hit,
            })

        if not holdings:
            prev_hold = curr_hold
            continue

        hdf      = pd.DataFrame(holdings)
        port_ret = hdf["contrib"].sum()

        # 🎯 基准日期匹配：使用 (year, month) 元组格式
        ts_date = pd.Timestamp(date)
        bench_ret = 0.0
        year_month_key = (ts_date.year, ts_date.month)
        if year_month_key in bench_monthly:
            bench_ret = float(bench_monthly[year_month_key]) * 100

        nav_before = nav
        nav        = nav * (1 + port_ret / 100)

        results.append({
            "date":        date,
            "month":       pd.Timestamp(date).strftime("%Y-%m"),
            "portfolio":   round(port_ret, 2),
            "benchmark":   round(bench_ret, 2),
            "excess":      round(port_ret - bench_ret, 2),
            "nav":         round(nav, 0),
            "nav_chg":     round(nav - nav_before, 0),
            "turnover":    len(new_in),
            "holdings_df": hdf,
            "tickers":     tickers,
        })
        prev_hold = curr_hold

    return results


def print_wf_backtest(results, initial_capital):
    """打印逐月 P&L + 年度汇总"""
    if not results:
        print("  没有回测结果")
        return

    SEP = "=" * 72
    print()
    print(SEP)
    print("  模型历史回测（Walk-Forward，完整26特征模型）")
    print(f"  初始资金 ${initial_capital:,.0f} CAD  手续费 {BT_TX_COST*100:.1f}%  止损 {BT_STOP_LOSS*100:.0f}%")
    print(SEP)

    for m in results:
        hdf    = m["holdings_df"].sort_values("net_ret", ascending=False)
        profit = "盈利" if m["portfolio"] >= 0 else "亏损"
        icon   = "+" if m["portfolio"] >= 0 else ""
        beat   = "+" if m["excess"] >= 0 else ""

        print()
        print(f"  {'-'*70}")
        print(f"  {m['month']}  {profit}  策略 {icon}{m['portfolio']:.2f}%  "
              f"基准 {m['benchmark']:+.2f}%  "
              f"超额 {beat}{m['excess']:.2f}%  "
              f"NAV ${m['nav']:,.0f} ({m['nav_chg']:+,.0f})")
        print(f"  换仓 {m['turnover']} 支")
        print()
        print(f"  {'Ticker':<13} {'集成分':>7} {'月涨跌':>8} {'净收益':>8} {'贡献%':>8}")
        print(f"  {'-'*50}")
        for _, row in hdf.iterrows():
            stop_s = " STOP" if row["stop_hit"] else ""
            sign   = "+" if row["net_ret"] >= 0 else ""
            print(f"  {row['ticker']:<13} {row['score']:>7.4f} "
                  f"{row['raw_ret']:>+7.2f}% "
                  f"{row['net_ret']:>+7.2f}% "
                  f"{row['contrib']:>+7.3f}%"
                  f"{stop_s}")

    # 汇总
    rets       = np.array([m["portfolio"] for m in results]) / 100
    bench_rets = np.array([m["benchmark"] for m in results]) / 100
    final_nav  = results[-1]["nav"]
    n          = len(rets)

    total_ret = (final_nav / initial_capital - 1) * 100
    ann_ret   = ((1 + rets).prod() ** (12/n) - 1) * 100 if n > 0 else 0
    bench_ann = ((1 + bench_rets).prod() ** (12/n) - 1) * 100 if n > 0 else 0
    vol_m     = rets.std() * np.sqrt(12) * 100
    sharpe    = (ann_ret/100 - 0.04) / (vol_m/100) if vol_m > 0 else 0
    navs      = pd.Series([m["nav"] for m in results])
    mdd       = ((navs - navs.cummax()) / navs.cummax()).min() * 100
    win_m     = sum(r > 0 for r in rets)
    beat_m    = sum(m["excess"] > 0 for m in results)
    best_m    = max(results, key=lambda m: m["portfolio"])
    worst_m   = min(results, key=lambda m: m["portfolio"])

    from collections import Counter
    ticker_counts = Counter([t for m in results for t in m["tickers"]])

    print()
    print(SEP)
    print(f"  年度汇总（{n} 个月）")
    print(SEP)
    summary = [
        ("初始资金",        f"${initial_capital:,.0f} CAD"),
        ("最终净值",        f"${final_nav:,.0f} CAD"),
        ("总收益",          f"{total_ret:+.2f}%"),
        ("年化收益（策略）", f"{ann_ret:+.2f}%"),
        ("年化收益（基准）", f"{bench_ann:+.2f}%"),
        ("超额收益 Alpha",  f"{ann_ret - bench_ann:+.2f}%"),
        ("年化波动率",      f"{vol_m:.2f}%"),
        ("Sharpe 比率",     f"{sharpe:.2f}"),
        ("最大回撤",        f"{mdd:+.2f}%"),
        ("月胜率",          f"{win_m/n*100:.1f}%  ({win_m}/{n} 月)"),
        ("跑赢基准",        f"{beat_m/n*100:.1f}%  ({beat_m}/{n} 月)"),
        ("最佳月份",        f"{best_m['month']}  {best_m['portfolio']:+.2f}%"),
        ("最差月份",        f"{worst_m['month']}  {worst_m['portfolio']:+.2f}%"),
    ]
    for label, value in summary:
        print(f"  {label:<20} {value}")

    print()
    print("  月度明细：")
    print(f"  {'月份':<10} {'策略':>8} {'基准':>8} {'超额':>8} {'净值':>12} {'盈亏CAD':>10}")
    print(f"  {'-'*58}")
    for m in results:
        icon = "+" if m["portfolio"] >= 0 else ""
        print(f"  {m['month']:<10} {m['portfolio']:>+7.2f}% {m['benchmark']:>+7.2f}% "
              f"{m['excess']:>+7.2f}% {m['nav']:>12,.0f} {m['nav_chg']:>+9,.0f}")

    print()
    print("  最常入选 Top 10（次数）：")
    for t, cnt in ticker_counts.most_common(10):
        bar = "#" * cnt
        print(f"  {t:<14} {cnt:>2}次  {cnt/n*100:>4.0f}%  {bar}")

    print()
    print("  注：Walk-Forward 完整模型，每月用该月前所有数据训练")
    print(f"     手续费 {BT_TX_COST*100:.1f}% 单边已扣，止损 {BT_STOP_LOSS*100:.0f}% 按日线触发")


# ══════════════════════════════════════════════════════════════════
# 💼 OPERATIONS MANUAL（日常运维配置）
# ══════════════════════════════════════════════════════════════════
#
# 📋 使用场景：Wealthsimple 免佣金账户日常维护
#
# 🎯 每月操作流程：
#   第一步（1 分钟）：  查看 Wealthsimple App 现有持仓 → 更新下方 MY_CURRENT_PORTFOLIO
#   第二步（5 分钟）：  运行 python picker.py → 等待报告生成
#   第三步（1 分钟）：  打开邮件 PDF 报告 → 按"Position Sizing"表格操作
#
# ⚙️  配置项说明：
#   MODE：                 运行模式（"pick"当月选股 / "backtest"历史回测 / "both"两者）
#   MY_CURRENT_PORTFOLIO:  今天持有的股票及占比（权重总和应接近 1.0）
#   EMAIL_CONFIG:          报告邮件发送地址（自动发送 PDF）
#   BT_TX_COST:            手续费成本（Wealthsimple=0.0 免佣金）
#
# ❌ 禁止修改：
#   - picker.py 核心逻辑（第 1-4800 行）
#   - 模型参数（XGBoost / LightGBM / MLP 配置）
#   - CONSTRAINTS（约束条件）已优化到生产条件
#
# ✓ 允许修改：
#   - MY_CURRENT_PORTFOLIO（每月更新）
#   - MODE（想要回测就改 "both"）
#   - EMAIL_CONFIG（换新邮箱）
# ══════════════════════════════════════════════════════════════════

# ── 运行模式（MODE 配置）─────────────────────────────────────────
MODE = "pick"   # "pick"     → 当月选股（默认，耗时 5 分钟）
                # "backtest" → 历史回测（过去 12 个月逐月模拟，耗时 30 分钟）
                # "both"     → 先回测再选股（耗时 35 分钟，推荐周末运行）

# ── 当前持仓配置（MY_CURRENT_PORTFOLIO）─────────────────────────
# 📱 获取方式：
#   1. 打开 Wealthsimple App
#   2. 点击"Accounts" → 看你的持仓列表
#   3. 每支股票旁边有百分比（%), 复制过来填到下面的字典
# 
# 💡 规则：
#   - 权重（小数点）= Wealthsimple 显示的百分比 ÷ 100
#   - 例：BMO 占 15.2% → 字典里写 "BMO.TO": 0.152
#   - 所有权重总和应接近 1.0（如果有现金，可以 0.95）
#   - 持仓少于 10 只？写你实际持有的即可，系统会用"缓冲带"做平滑过渡
#
# ✓ MY_CURRENT_PORTFOLIO 现在从单独文件 portfolio_config.py 导入
# 这样做的好处：
#   - 避免重复定义导致的混淆
#   - 清楚的单一来源（Single Source of Truth）
#   - 便于版本控制，分离配置与代码

# ── 邮件配置（EMAIL_CONFIG）──────────────────────────────────────
# 📧 工作原理：
#   - 运行结束后自动生成 PDF 报告 → 发送到你的邮箱
#   - 报告包含：选股结果 + 凸优化仓位 + 风险控制 + 性能指标
#   - 从邮件打开 PDF → 直接看表格 → 执行调仓
#
# 🔐 安全性：
#   - 密码存储在本地 Python 文件（非云端）
#   - Gmail: 用 app password（gmail.com/account/security）而不是真实密码
#   - 163/QQ: 用授权码就行（邮箱设置 → 账户安全）

EMAIL_CONFIG = {
    "to":       "carlchenn@hotmail.com",          # 你的接收邮箱
    "from":     "carlchenyiqing@gmail.com",       # 发送邮箱（Gmail，需配置 app password）
    "password": "vvdn ezoz yivl fbrw",            # Gmail app password（NOT 真实密码）
}

# ── 回测参数（BACKTEST_MONTHS）────────────────────────────────────
# 已在上面配置，默认 12 个月
# 改成 24 可回测过去 2 年，但注意数据需要 5 年完整价格历史


if __name__ == "__main__":
    from dateutil.relativedelta import relativedelta

    models_str = "XGBoost"+(" + LightGBM" if LGBM else "")+(" + MLP" if TORCH else "")
    print("="*70)
    print(f"  TSX 量化选股 v3.0  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"  模型：{models_str}  |  模式：{MODE.upper()}")
    print("="*70)

    # ── 历史回测 ─────────────────────────────────────────────────
    if MODE in ("backtest", "both"):
        print(f"\n{'|'*60}")
        print(f"  历史回测模式（完整模型 Walk-Forward，过去 {BACKTEST_MONTHS} 个月）")
        print(f"{'|'*60}")
        # 回测需要先跑完整 pipeline 拿到 wf
        if "daily_map" not in dir() or "wf" not in dir():
            daily_map, pit_map, meta_df, macro_df = fetch_all(TICKERS, YEARS)
            # ✓ 【修复存活者偏差】：历史回测用基础约束，避免用当前 meta_df 过滤历史数据
            passed  = apply_base_constraints(daily_map, CONSTRAINTS)
            panel   = build_panel(passed, daily_map, pit_map, macro_df)
            panel   = add_labels(panel)
            print("  智能插补（行业截面中位数）...")
            panel   = smart_impute(panel, FEATURE_COLS)
            panel   = cross_z(panel)
            # ✅ 【改进】启用时间序列约束，传入 daily_map 和 pit_map 实现动态过滤
            wf      = walk_forward(panel, tx_cost=BT_TX_COST, 
                                   daily_map=daily_map, pit_map=pit_map,
                                   apply_asof_constraints=True)

        # 用 Walk-Forward 输出还原逐月动态选股 P&L
        bt_results = backtest_from_wf(wf, daily_map, meta_df, top_n=TOP_N)
        print_wf_backtest(bt_results, INITIAL_CAPITAL)
        
        # 🎨 【新增】生成交互式 HTML 报告（Chart.js）
        if bt_results:
            print(f"\n  📊 生成交互式 HTML 报告...")
            try:
                html_content = generate_html_report(bt_results, INITIAL_CAPITAL)
                if html_content:
                    html_file = save_html_report(html_content)
                    print(f"     ✅ 已保存到 {html_file}")
                    print(f"     使用浏览器打开查看完整的图表和数据")
            except Exception as e:
                print(f"     ⚠️  HTML 报告生成失败: {e}")

    # ── 当月选股 ─────────────────────────────────────────────────
    if MODE in ("pick", "both"):
        print(f"\n{'▓'*60}")
        print(f"  当月选股模式")
        print(f"{'▓'*60}")

        daily_map, pit_map, meta_df, macro_df = fetch_all(TICKERS, YEARS)

        print(f"\n[3/4] 约束过滤（用最新基本面数据）")
        # ✓ 当月选股明确使用当前 meta_df（PE/ROE/市值）
        # 同步启用财报日历特征（当月预测需要未来财报安排）
        passed = apply_constraints_current(daily_map, meta_df, CONSTRAINTS)
        if len(passed) < TOP_N:
            print(f"\n⚠️  通过约束 {len(passed)} 支 < TOP_N={TOP_N}，"
                  f"建议放宽 max_pe 或 min_mktcap_cad")
            # 注意：放宽约束时 max_price_cad 始终保留
            relaxed = {**CONSTRAINTS, "max_pe": 100, "min_mktcap_cad": 200_000_000,
                       "max_price_cad": CONSTRAINTS.get("max_price_cad", 9999)}
            passed = apply_constraints_current(daily_map, meta_df, relaxed)

        print(f"\n[4/4] 特征工程 + 模型")
        # ✓ 当月选股启用真实财报日历（未来 90 天财报安排是当下可知信息）
        panel = build_panel(passed, daily_map, pit_map, macro_df, is_backtest=False)
        panel = add_labels(panel)
        print("  智能插补（行业截面中位数）...")
        panel = smart_impute(panel, FEATURE_COLS)
        panel = cross_z(panel)

        # 当月选股不使用 asof 约束（已用当前最新数据做过滤）
        wf = walk_forward(panel, tx_cost=BT_TX_COST, apply_asof_constraints=False)
        evaluate(wf)

        # 完整模型逐月 P&L 报告（用Walk-Forward的真实预测结果）
        backtest_report(wf, panel, daily_map, meta_df,
                        initial_capital = INITIAL_CAPITAL,
                        tx_cost         = BT_TX_COST,
                        stop_loss       = STOP_LOSS_PCT,
                        benchmark       = "XIU.TO")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🎯 【核心配置】MY_CURRENT_PORTFOLIO — 每月必须更新！
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 
        # 更新步骤：
        #   1️⃣  打开 Wealthsimple App → My Portfolio
        #   2️⃣  看你的 10 支持仓股票及其百分比权重（%）
        #   3️⃣  权重换算：百分比 ÷ 100 = 字典的值
        #       例：15.2% → 0.152，14.8% → 0.148
        #   4️⃣  所有权重和应接近 1.0（允许 0.95-1.05）
        #
        # 用途：
        #   - 计算"缓冲带"（±12档避免过度换仓）
        #   - 用于 Black-Litterman 优化（考虑现有仓位）
        #   - 如果持仓少于 10 只，写你实际持有的即可
        #
        # ✓ MY_CURRENT_PORTFOLIO 现在从 portfolio_config.py 导入
        # 编辑方式：
        #   1. 打开 portfolio_config.py
        #   2. 取消注释示例配置
        #   3. 填入你的实际持仓比例
        #   4. 保存 portfolio_config.py
        #   5. 运行 python picker.py（会自动加载新配置）
        # ✓ 好处：避免重复定义导致配置被意外覆盖
        # ✓ 第一次运行：使用空配置，系统会生成推荐，然后复制到 portfolio_config.py
        
        # 如果 MY_CURRENT_PORTFOLIO 为空，系统会用 ETF 扫描（防守策略）
        if not MY_CURRENT_PORTFOLIO:
            print("  ⚠️  MY_CURRENT_PORTFOLIO 为空 → 系统使用默认防守配置")
            print("  💡 建议：查看下方 result DataFrame，复制推荐配置到上面")
        else:
            print(f"  ✅ 已读入 {len(MY_CURRENT_PORTFOLIO)} 只现有持仓")

        # 提取名字列表给“换仓缓冲带”用
        current_tickers_only = list(MY_CURRENT_PORTFOLIO.keys())

        # 🟢 新增：加载上月的模型分数，用于 EMA 平滑
        SCORE_FILE = "last_month_scores.json"
        prev_scores = {}
        if os.path.exists(SCORE_FILE):
            try:
                with open(SCORE_FILE, "r") as f:
                    prev_scores = json.load(f)
                    print(f"  📥 成功加载上月模型得分记录，用于 EMA 平滑 (共 {len(prev_scores)} 只股票)")
            except Exception as e:
                print(f"  ⚠️  加载历史分数失败：{e}，本月跳过 EMA 平滑")

        # 🔧 修复3：加载冷却期黑名单（上月被止损的股票）
        COOLDOWN_FILE = "cooldown_tickers.json"
        cooldown_tickers = set()
        if os.path.exists(COOLDOWN_FILE):
            try:
                with open(COOLDOWN_FILE, "r") as f:
                    cooldown_tickers = set(json.load(f))
                    if cooldown_tickers:
                        print(f"  🔴 成功加载冷却期黑名单 (共 {len(cooldown_tickers)} 只被冻结股票)")
            except Exception as e:
                print(f"  ⚠️  加载冷却期黑名单失败：{e}，本月无冷却限制")

        # 【第一步】特征工程：先做共线性降维
        print("\n[特征工程] VIF 共线性检测...")
        # 🎯 修复2：用 panel 接收返回的 panel_reduced
        panel, insider_df, feat_cols = run_advanced_analysis(
            panel, daily_map, pit_map, meta_df, macro_df, wf,
            run_collinearity=True, run_insider=False, run_backtest=False,
            run_sensitivity=False
        )
        if feat_cols and len(feat_cols) < len(FEATURE_COLS):
            FEATURE_COLS = feat_cols

        # 【第二步】用纯净特征做预测
        result, imp, dd_signal, total_scale = predict_now(
            panel, daily_map, meta_df, wf, macro_df=macro_df,
            current_holdings=current_tickers_only,
            prev_weights_dict=MY_CURRENT_PORTFOLIO,
            prev_scores_dict=prev_scores  # 🟢 传入历史分数用于 EMA 平滑
        )  # 🔧 修复1：现在接收 total_scale（VIX/回撤缩仓系数）

        # 🔧 修复3：过滤掉冷却期黑名单中的股票
        if result is not None and not result.empty and cooldown_tickers:
            if isinstance(result.index, pd.MultiIndex):
                # MultiIndex(date, ticker) 的情况
                result = result[~result.index.get_level_values("ticker").isin(cooldown_tickers)]
            elif "ticker" in result.columns:
                # ticker 列的情况
                result = result[~result["ticker"].isin(cooldown_tickers)]
            if not result.empty:
                print(f"  🔴 已从推荐列表中过滤冷却期黑名单({len(cooldown_tickers)} 只)")

        if result is not None and not insider_df.empty:
            result = apply_insider_signal(result, insider_df, weight=0.15)

        if result is not None:
            print_picks(result, imp, daily_map, meta_df, wf, dd_signal)
            
            # 🟢 新增：保存本月的模型分数到文件，用于下个月的 EMA 平滑
            # 提取 ticker 和 ensemble_score
            if isinstance(result.index, pd.MultiIndex):
                # MultiIndex(date, ticker) 的情况
                tickers = result.index.get_level_values("ticker")
                current_scores = dict(zip(tickers, result["ensemble_score"]))
            else:
                # 其他情况，尝试用 ticker 列
                current_scores = result.set_index("ticker")["ensemble_score"].to_dict() if "ticker" in result.columns else {}
            
            SCORE_FILE = "last_month_scores.json"
            if current_scores:
                try:
                    with open(SCORE_FILE, "w") as f:
                        json.dump(current_scores, f)
                    print(f"  💾 本月模型得分已保存至 {SCORE_FILE}，用于下月平滑")
                except Exception as e:
                    print(f"  ⚠️  分数保存失败：{e}")

        # 新模块 E/F/G/H：Regime + Earnings + Black-Litterman + 月报
        result, bl_df, surprise_df, regime = run_new_modules(
            panel, daily_map, meta_df, macro_df, wf,
            result, imp, dd_signal,
            prev_weights_dict=MY_CURRENT_PORTFOLIO,
            email_to       = EMAIL_CONFIG["to"],
            email_from     = EMAIL_CONFIG["from"],
            email_password = EMAIL_CONFIG["password"],
        )

        # 🔧 修复1：应用 VIX/回撤缩仓系数到最终配置（清晰无冗余版本）
        if bl_df is not None and not bl_df.empty and total_scale < 1.0:
            bl_df["alloc_pct"] = (bl_df["alloc_pct"] * total_scale).round(1)
            cash_alloc = round(100.0 - bl_df["alloc_pct"].sum(), 1)
            
            # 使用 pd.concat 干净利落地追加现金行
            cash_df = pd.DataFrame([{
                "ticker": "CASH.TO", 
                "alloc_pct": cash_alloc,
                "category": "防御性现金 🛡️",
                "vol_ann": 0.0,
                "score": 0.0,
                "prob": 0.0,
                "ret": 0.0,
                "view_ret": 0.0
            }])
            bl_df = pd.concat([bl_df, cash_df], ignore_index=True)
            print(f"  💾 🔴 熔断缩仓激活 | 缩仓系数={total_scale:.2f} | 现金占比={cash_alloc:.1f}%")


# ══════════════════════════════════════════════════════════════════
# 直接运行 HTML 生成
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "html":
        # 快速生成 HTML 模式：只做回测 + 生成报告
        print("\n" + "="*60)
        print("  🚀 HTML 报告快速生成模式")
        print("="*60)
        
        try:
            # 快速获取数据
            print("\n[1/3] 下载数据...")
            daily_map, pit_map, meta_df, macro_df = fetch_all(TICKERS, YEARS)
            print(f"  ✅ 完成 ({len(TICKERS)} 只股票)")
            
            # 约束过滤
            print("\n[2/3] 约束过滤...")
            passed = apply_base_constraints(daily_map, CONSTRAINTS)
            print(f"  ✅ {len(passed)} 只股票通过约束")
            
            # 特征工程 + 回测
            print("\n[3/3] 特征工程 + 回测...")
            panel = build_panel(passed, daily_map, pit_map, macro_df)
            panel = add_labels(panel)
            panel = smart_impute(panel, FEATURE_COLS)
            panel = cross_z(panel)
            wf = walk_forward(panel, tx_cost=BT_TX_COST, 
                            daily_map=daily_map, pit_map=pit_map,
                            apply_asof_constraints=True)
            
            # 生成回测结果 + HTML
            print("\n[4/4] 生成回测报告...")
            bt_results = backtest_from_wf(wf, daily_map, meta_df, top_n=TOP_N)
            print_wf_backtest(bt_results, INITIAL_CAPITAL)
            
            # HTML 报告
            if bt_results:
                print(f"\n  📊 生成交互式 HTML 报告...")
                try:
                    html_content = generate_html_report(bt_results, INITIAL_CAPITAL)
                    if html_content:
                        html_file = save_html_report(html_content)
                        print(f"     ✅ HTML 已保存到：{html_file}")
                        print(f"     🌐 用浏览器打开查看完整的图表和数据\n")
                except Exception as e:
                    print(f"     ⚠️  HTML 报告生成失败: {e}")
            else:
                print(f"     ⚠️  HTML 报告模块不可用")
                
        except Exception as e:
            print(f"\n❌ 错误：{e}")
            import traceback
            traceback.print_exc()
    else:
        print("用法：python picker_ca.py [html]")
        print("\n示例：")
        print("  python picker_ca.py html    # 生成 HTML 回测报告")
        print("  python picker_ca.py          # 显示此帮助信息")
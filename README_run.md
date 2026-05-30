# 运行手册 (README_run)

一个入口：`./run.sh <command>`（`./run.sh help` 看全部）。
报告邮件发到 **carlchenn@hotmail.com**，dashboard 自动更新。

---

## 📅 每月必跑（核心选股流程）

```bash
./run.sh monthly        # 加拿大 + 美股 一起跑，发邮件 + 更新 dashboard
# 或分开：
./run.sh ca             # 仅加拿大 (picker.py pick)
./run.sh us             # 仅美股科技 (picker_us.py pick)
```

> **执行原则：出了名单当天市价买。** 择时已实测无效（见下「不用再跑」），
> 别等企稳、别挂限价抄底（美股尤其，平均反而贵 ~3%）。

自动化：`run_monthly.sh` 已含 CA+US，可交给 launchd 定时（见 SCHEDULING.md）。

---

## 📊 每季跑（银行财报季后，约 2 / 5 / 8 / 12 月底）

```bash
./run.sh banks          # 7 家银行基本面评分 & 排名
```
⚠️ **跑前手动更新**：CET1 / PCL / 分部收入 yfinance 抓不到，需从各行季报
填进 `bank_score_ca.py` 顶部的 `DATA` 字典（数据源见 `bank_deep_ca.py` 注释里的
SEC EDGAR / IR / SEDAR+ 链接）。当前填的是 FY2026 Q2。

```bash
./run.sh bankdeep       # 银行 yfinance 趋势（ROE/EPS/BVPS/经营杠杆）— 全自动
./run.sh fund           # 全 30 只基本面粗筛（Value/Quality/Growth）— 全自动
```

银行评分权重（`bank_score_ca.py`，可改）：
ROE 25 · PCL 20 · CET1 15 · 经营杠杆 10 · 分部质量 10 · 收入增长 10 · BVPS 10。
> 银行最重要的三项：**ROE（赚钱）+ PCL 趋势（信用风险）+ CET1（资本安全）**。

---

## 🔍 按需跑（验证）

```bash
./run.sh backtest       # 加拿大 walk-forward (Sharpe/IR)
./run.sh backtest-us    # 美股 walk-forward
./run.sh rigor          # 过拟合审计 (DSR / CPCV / PBO)
./run.sh monitor        # 每日持仓监控
```

当前基准：CA Sharpe ~1.75 / IR ~0.96；US Sharpe ~0.94 / IR ~1.02（DSR：CA MODERATE，US WEAK）。

---

## ✅ 不用再跑（结论已定，留作框架）

| 脚本 | 结论 |
|------|------|
| `entry_timing.py` / `limit_buy_test.py` | 择时是负期望 → 当天市价买，不抄底、不等确认 |

模型本身也已定稿：特征 / 调参 / 扩池三条提收益的路都试过且失败，
**唯一提升靠每月实盘积累 OOS 数据**，别再调模型。

---

## 文件速查

| 文件 | 用途 |
|------|------|
| `picker.py` | 加拿大模型（30 只 / 4 行业 / 基准 XIU.TO）|
| `picker_us.py` | 美股科技模型（22 只 / 3 行业 / 基准 QQQ）|
| `bank_score_ca.py` | 7 家银行基本面评分（需季度更新 DATA）|
| `bank_deep_ca.py` | 银行 yfinance 深度趋势 + 数据源 TODO |
| `fundamentals_ca.py` | 全 30 只通用基本面筛 |
| `entry_timing.py` / `limit_buy_test.py` | 择时研究（已完成）|
| `run.sh` | 总入口 | `run_monthly.sh` | 定时自动化 |
| 邮箱 | `email_config.py`（gitignored，EMAIL_TO = hotmail）|

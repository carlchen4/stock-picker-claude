# RISK_AI_REPORT — 生成 Prompt / Runbook

复用这份说明即可重跑 `RISK_AI_REPORT.md`（每季或每次想刷新时）。
数据是定性 web 研究 + 已有基本面打分，**没有 API、不可自动化**，靠按下面流程做。

---

## 目标
对 US(21) + CA(30) universe 每只股票产出两栏：
1. **最大风险/弱点** = (a) 已有基本面里最弱的维度 + (b) 联网近期风险/分析师担忧
2. **AI 投入/趋势** = 联网查 capex / AI 项目 / 管理层表态，给方向箭头

方向：`↑↑` 强受益 · `↑` 受益 · `↗` 轻度/间接 · `→` 中性 · `↘` 落后/受冲击

---

## 第 1 步：拉已有基本面"弱点"（本地，先跑）
```bash
./run.sh fund-us        # 美股 21 只 Value/Quality/Growth → 每只最弱维度
./run.sh fund           # 加拿大 30 只
./run.sh banks          # 7 银行深度(CET1/PCL/ROE) — 先更新 bank_data_ca.csv
python3 financials_score_ca.py   # 加拿大全 12 金融
```
→ 每只"基本面最弱项"从这些输出读（如 Tesla 毛利最低、INTC 亏损、EQB 恶化、BNS PCL 最高）。

## 第 2 步：联网补"当前风险 + AI"（用下面这些 query，按主题批量，别一只一搜）
实际用过的 WebSearch 查询（替换年份后照搬）：
1. `US tech AI capex 2026 Microsoft Amazon Google Meta Nvidia hyperscaler data center spending trend biggest risk`
2. `semiconductor stocks 2026 risks NVIDIA AMD TSMC Micron Broadcom Qualcomm AMAT Lam China export controls memory cycle`
3. `Tesla Apple Intel Palantir Snowflake Cloudflare Arista 2026 biggest risk valuation AI exposure`
4. `Canadian energy utilities AI data center electricity demand 2026 Enbridge TC Energy Fortis power; Canadian banks AI investment`
5. `Canadian energy stocks 2026 risk oil price Canadian Natural Suncor Cenovus pipeline regulatory debt`
6. `Celestica AI server demand 2026 Thomson Reuters AI legal Canadian National Railway CP rail freight risk WSP`

→ 6 次批量搜索 ≈ 覆盖全部 51 只。缺口股(如 ANET/AVGO/具体银行)用知识补或单独补一搜。

## 第 3 步：合成 `RISK_AI_REPORT.md`
结构：跨市场大背景 → 🇺🇸按子行业 → 🇨🇦按板块 → 一句话总结 → Sources。
每只 2 行：风险一行、AI 一行 + 箭头。**必带 Sources 链接。**

---

## 关键判断原则（写报告时的"prompt 风格"）
- **诚实**：基本面差≠会跌、贵≠会崩，只陈述事实+暴露，不给买卖信号。
- **AI 角度分层**：美股科技=直接(capex/产品)；加拿大=间接"卖铲子"(管道供气、公用供电、Celestica 造服务器、TRI 法律AI、银行提效)，**别硬凑**。
- **区分"AI 受益"与"AI 估值雷"**：基本面好+便宜=真受益(TSM)；基本面好+极贵=雷(PLTR/ARM)。
- **标出软肋**：现金流转负、周期顶、信用恶化、债务高、最弱 ROE 这些要点名。

---

## 复用方式（给未来的自己 / AI 的一句话 prompt）
> "按 RISK_AI_REPORT_PROMPT.md：先跑 fund-us/fund/banks 拿基本面最弱项，
>  再用里面的 6 条 WebSearch query 批量联网拿当前风险+AI capex，
>  合成 RISK_AI_REPORT.md，每只两行(风险/AI+箭头)，带 Sources，不给买卖建议。"

⚠️ 局限：定性、时效性强(季度刷新)、~50 只研究量大、不进 picker 模型(纯参考)。

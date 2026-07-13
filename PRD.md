# PRD: AI 产品增长与运营分析平台

## 1. 项目背景

模拟公司是一家 AI SaaS 产品公司，产品形态类似 ChatGPT、Notion AI、Kimi、Coze 和企业 AI Agent 平台。核心功能包括 AI 对话、文件分析、代码助手、数据分析、Agent 工作流、团队协作、付费订阅和用户反馈。

随着用户规模扩大，产品和运营团队需要一套统一的数据分析平台，回答增长、活跃、留存、转化、模型成本和用户满意度问题，并通过机器学习模型提前识别高流失风险用户和高付费潜力用户。

## 2. 项目目标

1. 建立从用户、事件、对话、订阅、反馈到实验的完整数据模型。
2. 生成接近真实业务规律的 synthetic data，用于作品集展示和分析练习。
3. 使用 Pandas、DuckDB、SQL 完成指标计算和分析复现。
4. 构建交互式 Streamlit Dashboard，服务产品、运营和增长团队。
5. 训练两个可解释的基线模型：
   - 用户未来 14 天流失预测。
   - 免费用户未来 30 天付费转化预测。
6. 输出可复现代码、SQL、notebook、README、PRD 和分析报告。

## 3. 目标用户

| 用户角色 | 关注问题 | 使用场景 |
| --- | --- | --- |
| 产品经理 | 激活、留存、功能采用、反馈痛点 | 周会复盘、版本评估、功能优先级 |
| 增长运营 | 渠道质量、转化漏斗、付费潜力用户 | 拉新投放、用户分层、召回策略 |
| 数据分析师 | 指标口径、SQL 复现、Cohort 和漏斗 | 分析报告、专题诊断 |
| 数据科学家 | 流失预测、转化预测、特征解释 | 模型实验、用户打分、策略评估 |
| 管理层 | 增长、收入、成本、满意度 | 月度经营复盘 |

## 4. 业务问题

### 增长与渠道

- 每日/每周新增用户趋势如何？
- 哪些渠道带来更多高质量用户？
- 不同渠道的激活率、留存率和付费率有什么差异？

### 行为与活跃

- 用户最常使用哪些 AI 功能？
- Agent、文件分析、数据分析等高级功能是否提高留存和转化？
- 不同 persona 的使用深度是否不同？

### 留存与流失

- 不同注册 cohort 的 D1/D7/D14/D30 留存如何？
- 响应延迟、失败率、低评分是否会提高流失概率？
- 哪些用户应被优先召回？

### 付费转化

- 从注册、激活、高级功能使用、访问付费页到付费的漏斗如何？
- 哪些渠道、persona 和功能行为更容易转化？
- 免费用户未来 30 天付费概率如何预测？

### 模型成本

- 不同模型、功能和用户层级贡献了多少 token 和成本？
- 高成本模型调用是否集中在高价值用户或付费用户中？
- 延迟和失败率是否影响评分、留存和付费？

### 用户反馈

- NPS、评分和负面反馈主题如何变化？
- 主要负面反馈来自价格、性能、准确性、稳定性还是易用性？
- 哪些反馈主题和用户行为关联最强？

## 5. 功能模块

| 模块 | 功能 | 关键输出 |
| --- | --- | --- |
| 数据生成 | 生成用户、事件、对话、订阅、反馈、实验数据 | CSV、DuckDB |
| 指标计算 | 增长、活跃、留存、漏斗、转化、成本、反馈 | SQL 和 Pandas 指标表 |
| Dashboard | 7 个交互页面 | Streamlit + Plotly |
| 机器学习 | 流失预测、付费转化预测 | AUC、PR-AUC、特征重要性 |
| 分析报告 | 业务洞察、建议、风险说明 | `reports/final_report.md` |

## 6. 指标体系

### 北极星指标

**Weekly Successful AI Workflows per Active User**

定义：每周每个活跃用户成功完成的 AI 工作流数量，包括 AI 对话、文件分析、代码助手、数据分析和 Agent 工作流。该指标同时反映使用频率、任务完成和产品价值。

### 一级指标

| 指标族 | 指标 | 定义 |
| --- | --- | --- |
| 增长 | 注册用户数 | 当日新注册用户数 |
| 增长 | 激活率 | 注册后 24 小时内完成首次 AI 对话或高级功能使用的用户占比 |
| 活跃 | DAU/WAU/MAU | 在对应窗口内有有效事件或对话的用户数 |
| 活跃 | 人均对话数 | 对话数 / 活跃用户数 |
| 留存 | D1/D7/D14/D30 留存 | 注册后第 N 天仍活跃的用户占比 |
| 转化 | 付费转化率 | 付费用户数 / 注册用户数 |
| 转化 | 免费到付费 30 天转化率 | 注册后 30 天内从免费转为付费的用户占比 |
| 收入 | MRR | 当前有效订阅月经常性收入 |
| 成本 | 模型调用成本 | 输入/输出 token 按模型单价计算的成本 |
| 质量 | 失败率 | 失败对话数 / 总对话数 |
| 质量 | P95 延迟 | 对话响应延迟第 95 百分位 |
| 满意度 | 平均评分 | 用户对对话或功能反馈的平均评分 |
| 满意度 | NPS | 推荐者占比 - 贬损者占比 |

### 关键分群维度

- 注册渠道：organic、paid_search、social、referral、content、partner、app_store。
- 用户 persona：casual、creator、developer、analyst、operator、team_admin。
- 公司规模：individual、small_team、mid_market、enterprise。
- 订阅计划：Free、Pro、Team、Enterprise、Canceled。
- 功能：chat、file_analysis、code_assistant、data_analysis、agent_workflow。
- 模型：fast-chat、balanced、reasoning-pro、vision-pro、agentic-pro。
- 地区、设备、实验分组。

## 7. 页面结构

| 页面 | 面向问题 | 主要组件 |
| --- | --- | --- |
| 首页总览 | 当前业务健康度如何？ | KPI 卡片、DAU 趋势、收入成本趋势、反馈概览 |
| 增长分析 | 增长来自哪里，质量如何？ | 注册趋势、渠道注册/激活/付费率、persona 构成 |
| 留存分析 | 哪些 cohort 留得住？ | Cohort heatmap、D1/D7/D14/D30 留存曲线 |
| 付费转化 | 用户如何从注册走向付费？ | 漏斗图、分渠道转化、分 persona 转化、订阅概览 |
| 模型成本 | 成本由哪些模型/功能驱动？ | 成本趋势、token 趋势、模型成本矩阵、延迟和失败率 |
| 用户反馈 | 用户满意度和痛点是什么？ | 评分分布、NPS 趋势、反馈主题、负面反馈表 |
| 预测模型 | 哪些用户可能流失或付费？ | 模型指标、特征重要性、高风险用户样例 |

## 8. 模拟数据生成逻辑

### 8.1 用户注册

- 注册日期包含增长趋势、工作日/周末差异和营销活动峰值。
- 渠道存在质量差异：
  - referral、content 用户质量更高，激活和留存更好。
  - paid_search 规模较大但质量分布更混杂。
  - social 注册波动更大，付费转化相对较低。
- persona 影响使用深度：
  - developer 更常用 code_assistant。
  - analyst 更常用 data_analysis 和 file_analysis。
  - team_admin 更可能转化为 Team/Enterprise。
  - casual 活跃和转化较低。

### 8.2 行为事件

- 高活跃用户拥有更多 active days 和 conversations。
- 使用高级功能的用户更可能访问付费页和转化。
- 已付费用户的人均对话数、Agent 使用率和高级模型占比更高。
- 失败事件、低评分和高延迟会提高后续流失风险。

### 8.3 对话与模型调用

- 不同功能对应不同 token 分布：
  - chat token 较低。
  - file_analysis、data_analysis 输入 token 较高。
  - code_assistant 和 agent_workflow 输出 token 更高。
- 不同模型成本不同：
  - fast-chat 成本最低、延迟低。
  - reasoning-pro 和 agentic-pro 成本高、延迟高。
  - 高级功能更容易调用高级模型。
- 对话成功率受模型、功能复杂度、用户质量风险和延迟影响。

### 8.4 付费转化

付费概率受以下因素提升：

- 注册渠道质量高。
- 注册后快速激活。
- 高频使用。
- 使用 Agent、文件分析、数据分析等高级功能。
- 团队或企业用户。

付费概率受以下因素降低：

- 失败率高。
- 响应延迟高。
- 平均评分低。
- 早期使用深度低。

### 8.5 流失

流失风险受以下因素提升：

- 未激活或低频使用。
- 对话失败率高。
- P95 延迟高。
- 用户评分低或负面反馈多。
- 仅使用基础 chat 且未形成高级工作流。

流失风险受以下因素降低：

- 高频使用。
- 付费订阅。
- 使用高级功能。
- 团队/企业场景。

## 9. 数据表设计与数据字典

### 9.1 `users`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| user_id | string | 用户唯一 ID |
| signup_date | date | 注册日期 |
| signup_ts | timestamp | 注册时间 |
| acquisition_channel | string | 注册渠道 |
| country | string | 国家/地区 |
| device_type | string | 首次注册设备 |
| persona | string | 用户 persona |
| company_size | string | 公司规模 |
| email_domain_type | string | personal、business、education |
| marketing_campaign | string | 首次归因活动 |
| activated_24h | boolean | 注册 24 小时内是否激活 |
| advanced_propensity | float | 高级功能使用倾向，0-1 |
| latent_engagement_score | float | 合成用户活跃潜力分 |
| quality_risk_score | float | 合成体验风险分 |
| paid_converted | boolean | 是否曾付费转化 |
| conversion_date | date | 首次付费日期 |
| current_plan | string | 当前计划：Free、Pro、Team、Enterprise、Canceled |
| last_active_date | date | 最近活跃日期 |
| is_churned_14d | boolean | 以数据截止日判断，是否 14 天未活跃 |

### 9.2 `events`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| event_id | string | 事件唯一 ID |
| user_id | string | 用户 ID |
| event_time | timestamp | 事件时间 |
| event_date | date | 事件日期 |
| event_name | string | 事件名称 |
| feature | string | 相关功能 |
| model_name | string | 相关模型 |
| session_id | string | 会话 ID |
| conversation_id | string | 对话 ID |
| device_type | string | 设备 |
| country | string | 国家/地区 |
| acquisition_channel | string | 注册渠道 |
| plan_at_time | string | 事件发生时订阅计划 |
| latency_ms | float | 相关延迟 |
| success_flag | boolean | 是否成功 |
| tokens | int | 相关 token 数 |
| cost_usd | float | 相关模型成本 |

### 9.3 `conversations`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| conversation_id | string | 对话唯一 ID |
| user_id | string | 用户 ID |
| conversation_started_at | timestamp | 对话开始时间 |
| conversation_date | date | 对话日期 |
| feature | string | 功能类型 |
| model_name | string | 使用模型 |
| messages_count | int | 消息轮次 |
| input_tokens | int | 输入 token |
| output_tokens | int | 输出 token |
| total_tokens | int | 总 token |
| response_latency_ms | float | 响应延迟 |
| is_success | boolean | 是否成功 |
| error_type | string | 失败类型 |
| cost_usd | float | 成本 |
| user_rating | float | 用户评分，1-5，可为空 |
| sentiment_score | float | 情感分，-1 到 1 |
| is_advanced_feature | boolean | 是否高级功能 |
| plan_at_time | string | 对话发生时计划 |

### 9.4 `subscriptions`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| subscription_id | string | 订阅唯一 ID |
| user_id | string | 用户 ID |
| plan_name | string | Pro、Team、Enterprise |
| billing_cycle | string | monthly、annual |
| started_at | date | 订阅开始日期 |
| ended_at | date | 订阅结束日期，可为空 |
| status | string | active、canceled、past_due |
| mrr_usd | float | 月经常性收入 |
| arr_usd | float | 年化收入 |
| seats | int | 席位数 |
| payment_method | string | 支付方式 |
| currency | string | 币种 |
| created_at | timestamp | 创建时间 |
| updated_at | timestamp | 更新时间 |

### 9.5 `feedback`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| feedback_id | string | 反馈唯一 ID |
| user_id | string | 用户 ID |
| conversation_id | string | 相关对话，可为空 |
| submitted_at | timestamp | 提交时间 |
| feedback_type | string | conversation_rating、nps、support_ticket |
| rating | float | 评分，1-5 |
| nps_score | int | NPS 分数，0-10，可为空 |
| sentiment | string | positive、neutral、negative |
| sentiment_score | float | 情感分 |
| category | string | 反馈主题 |
| feedback_text | string | 合成反馈文本 |
| response_latency_ms | float | 相关延迟 |
| model_name | string | 相关模型 |
| feature | string | 相关功能 |
| source | string | in_app、email、support |

### 9.6 `experiments`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| experiment_id | string | 实验 ID |
| experiment_name | string | 实验名称 |
| hypothesis | string | 实验假设 |
| primary_metric | string | 主指标 |
| start_date | date | 开始日期 |
| end_date | date | 结束日期 |
| status | string | running、completed、paused |
| owner | string | 负责人角色 |

### 9.7 `experiment_assignments`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| assignment_id | string | 分流记录 ID |
| experiment_id | string | 实验 ID |
| user_id | string | 用户 ID |
| variant | string | control、variant_a、variant_b |
| assigned_at | timestamp | 分流时间 |
| exposed | boolean | 是否实际曝光 |

## 10. 第一版开发计划

### Milestone 1: 数据与指标基础

- 完成 PRD、数据字典和指标口径。
- 实现 `generate_data.py`，生成 CSV 和 DuckDB。
- 实现 `create_tables.sql`、`user_metrics.sql`、`retention.sql`、`funnel.sql`。
- 验证数据规模、字段类型和关键业务规律。

### Milestone 2: 核心分析

- 完成增长、活跃、漏斗、留存、转化、成本、反馈分析。
- 输出 EDA notebook 和分析报告初稿。
- 校验主要指标是否互相一致。

### Milestone 3: 机器学习

- 构建用户级特征表。
- 训练流失预测模型和付费转化预测模型。
- 输出 AUC、PR-AUC、召回率、特征重要性。
- 可选加入 XGBoost/LightGBM 和 SHAP。

### Milestone 4: Dashboard

- Streamlit 实现 7 个页面。
- 增加全局筛选器：日期、渠道、persona、订阅计划。
- 将模型结果接入预测模型页面。
- 进行本地冒烟测试和截图。

### Milestone 5: 作品集包装

- 完善 README、PRD、final_report。
- 加入截图、项目亮点、业务价值说明。
- 标注 synthetic data 限制和真实数据接入方案。

## 11. 成功标准

- 一条命令可生成完整数据集。
- 一条命令可启动 Dashboard。
- SQL、Pandas 和 Dashboard 指标口径一致。
- 至少包含 7 个 Dashboard 页面。
- 两个模型可训练、可评估、可解释。
- 报告包含业务洞察和行动建议，而不是只放图表。

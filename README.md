# AI 产品增长与运营分析平台

这是一个作品集级别的数据分析 / 数据科学项目，模拟一家 AI SaaS 产品公司的增长、运营、成本、反馈和预测建模场景。项目覆盖从数据建模、模拟数据生成、SQL 指标计算、业务分析、机器学习建模到 Streamlit Dashboard 的完整链路。

## 这个项目为什么有价值

模拟数据本身不能证明某家真实公司的业务结论，但它可以证明三件真实工作中非常重要的能力：

1. **业务建模能力**：你能不能把一个 AI 产品拆成用户、事件、对话、订阅、反馈、实验、模型成本等可分析对象。
2. **分析工程能力**：你能不能从零搭建可复现的数据管道、指标口径、SQL 查询、特征工程和 Dashboard。
3. **数据科学产品能力**：你能不能把增长、留存、转化、成本、满意度和预测模型组织成一个产品团队能使用的分析平台。

它的正确定位不是“伪造业务成绩”，而是“可复现地展示你的方法论和工程能力”。在作品集中，建议清楚标注 synthetic data，并重点展示数据生成假设、指标体系、分析方法、模型评估和可扩展到真实数据的接口设计。

## 项目结构

```text
ai-product-growth-analytics/
├── README.md
├── PRD.md
├── requirements.txt
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── database/
├── notebooks/
├── src/
│   ├── generate_data.py
│   ├── data_cleaning.py
│   ├── metrics.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── visualization.py
├── sql/
├── reports/
└── assets/
```

## 快速开始

```bash
cd ai-product-growth-analytics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py --users 10000 --seed 42
streamlit run app.py
```

默认数据规模约为：

- 10,000 名用户
- 200,000+ 条事件
- 80,000+ 条 AI 对话记录
- 订阅、反馈、实验分流和模型调用成本数据

## Dashboard 页面

- 首页总览：核心 KPI、DAU、收入、成本和满意度概览
- 增长分析：注册趋势、渠道质量、活跃趋势
- 留存分析：Cohort 留存矩阵、留存曲线
- 付费转化：激活到付费漏斗、渠道/人群转化率
- 模型成本：token、成本、延迟、失败率、模型/功能拆解
- 用户反馈：评分、NPS、反馈主题、负面反馈驱动因素
- 预测模型：流失预测和付费转化预测的基线模型效果

## 核心业务问题

- 哪些渠道带来的用户增长质量最高？
- 哪些功能行为最能预测留存和付费？
- 高级模型和 Agent 工作流是否带来足够的商业价值来覆盖成本？
- 响应延迟、失败率和低评分是否会显著推高流失风险？
- 免费用户应该如何分层运营，以提升未来 30 天付费转化率？

## 数据说明

本项目数据为合成数据。生成逻辑包含业务规律，例如渠道质量差异、用户 persona 差异、高频使用与付费倾向、延迟/失败/低评分与流失风险、高级功能与付费转化、模型 token 成本差异等。

合成数据不应被解释为真实市场结论，但可以用于展示完整的数据产品建设能力。

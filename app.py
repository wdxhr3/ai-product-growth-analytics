from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_cleaning import load_all_tables, validate_core_tables
from src.feature_engineering import build_user_features, get_model_columns
from src.metrics import (
    activation_funnel,
    cohort_retention,
    compute_kpis,
    daily_growth,
    feedback_summary,
    model_cost_summary,
    paid_conversion_by_segment,
    retention_summary,
)
from src.modeling import train_logistic_baseline


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "database" / "ai_product_analytics.duckdb"

st.set_page_config(
    page_title="AI 产品增长与运营分析平台",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(db_path: str) -> dict[str, pd.DataFrame]:
    return load_all_tables(db_path)


def pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1%}"


def money(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"${value:,.0f}"


def filter_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    users = tables["users"].copy()
    users["signup_date"] = pd.to_datetime(users["signup_date"])

    st.sidebar.title("筛选器")
    min_date = users["signup_date"].min().date()
    max_date = users["signup_date"].max().date()
    date_range = st.sidebar.date_input("注册日期", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = [pd.Timestamp(x) for x in date_range]
    else:
        start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)

    channels = st.sidebar.multiselect("注册渠道", sorted(users["acquisition_channel"].dropna().unique()), default=sorted(users["acquisition_channel"].dropna().unique()))
    personas = st.sidebar.multiselect("Persona", sorted(users["persona"].dropna().unique()), default=sorted(users["persona"].dropna().unique()))
    plans = st.sidebar.multiselect("当前计划", sorted(users["current_plan"].dropna().unique()), default=sorted(users["current_plan"].dropna().unique()))

    users_f = users[
        (users["signup_date"].between(start_date, end_date))
        & (users["acquisition_channel"].isin(channels))
        & (users["persona"].isin(personas))
        & (users["current_plan"].isin(plans))
    ].copy()
    selected_users = set(users_f["user_id"])

    filtered = {"users": users_f}
    for name in ["events", "conversations", "subscriptions", "feedback", "experiment_assignments"]:
        df = tables[name]
        filtered[name] = df[df["user_id"].isin(selected_users)].copy() if "user_id" in df.columns else df.copy()
    filtered["experiments"] = tables["experiments"].copy()
    return filtered


def metric_row(kpis: dict[str, float]) -> None:
    cols = st.columns(5)
    cols[0].metric("用户数", f"{kpis['users']:,.0f}")
    cols[1].metric("30日活跃用户", f"{kpis['active_30d_users']:,.0f}")
    cols[2].metric("激活率", pct(kpis["activation_rate"]))
    cols[3].metric("付费转化率", pct(kpis["paid_conversion_rate"]))
    cols[4].metric("MRR", money(kpis["mrr_usd"]))

    cols = st.columns(5)
    cols[0].metric("付费用户", f"{kpis['paid_users']:,.0f}")
    cols[1].metric("模型成本", money(kpis["model_cost_usd"]))
    cols[2].metric("成功工作流/活跃用户", f"{kpis['successful_workflows_per_active_user']:.1f}")
    cols[3].metric("平均评分", "-" if pd.isna(kpis["avg_rating"]) else f"{kpis['avg_rating']:.2f}")
    cols[4].metric("NPS", "-" if pd.isna(kpis["nps"]) else f"{kpis['nps']:.0f}")


def page_overview(tables: dict[str, pd.DataFrame]) -> None:
    st.title("AI 产品增长与运营分析平台")
    kpis = compute_kpis(tables["users"], tables["events"], tables["conversations"], tables["subscriptions"], tables["feedback"])
    metric_row(kpis)

    growth = daily_growth(tables["users"], tables["events"], tables["subscriptions"])
    cost_daily, _ = model_cost_summary(tables["conversations"])

    left, right = st.columns(2)
    left.plotly_chart(px.line(growth, x="date", y=["signups", "dau"], title="注册与 DAU 趋势"), use_container_width=True)
    right.plotly_chart(px.line(cost_daily, x="date", y="cost_usd", title="模型调用成本趋势"), use_container_width=True)

    funnel = activation_funnel(tables["users"], tables["events"], tables["conversations"])
    st.plotly_chart(px.funnel(funnel, x="users", y="step", title="注册到付费漏斗"), use_container_width=True)


def page_growth(tables: dict[str, pd.DataFrame]) -> None:
    st.title("增长分析")
    growth = daily_growth(tables["users"], tables["events"], tables["subscriptions"])
    st.plotly_chart(px.line(growth, x="date", y=["signups", "signup_7d_avg"], title="新增注册与 7 日均线"), use_container_width=True)

    by_channel = paid_conversion_by_segment(tables["users"], "acquisition_channel")
    left, right = st.columns(2)
    left.plotly_chart(px.bar(by_channel, x="acquisition_channel", y="users", title="各渠道注册用户数"), use_container_width=True)
    right.plotly_chart(px.bar(by_channel, x="acquisition_channel", y=["activated_rate", "paid_conversion_rate"], barmode="group", title="各渠道激活率与付费率"), use_container_width=True)

    by_persona = paid_conversion_by_segment(tables["users"], "persona")
    st.plotly_chart(px.scatter(by_persona, x="activated_rate", y="paid_conversion_rate", size="users", color="persona", title="Persona 质量矩阵"), use_container_width=True)
    st.dataframe(by_channel, use_container_width=True)


def page_retention(tables: dict[str, pd.DataFrame]) -> None:
    st.title("留存分析")
    retention = cohort_retention(tables["users"], tables["events"], max_days=30)
    matrix = retention.pivot_table(index="cohort_month", columns="days_since_signup", values="retention_rate")
    st.plotly_chart(
        px.imshow(matrix, aspect="auto", color_continuous_scale="Teal", title="Cohort 留存热力图"),
        use_container_width=True,
    )
    summary = retention_summary(tables["users"], tables["events"])
    st.subheader("关键留存节点")
    st.dataframe(summary.style.format({1: "{:.1%}", 7: "{:.1%}", 14: "{:.1%}", 30: "{:.1%}"}), use_container_width=True)


def page_conversion(tables: dict[str, pd.DataFrame]) -> None:
    st.title("付费转化")
    funnel = activation_funnel(tables["users"], tables["events"], tables["conversations"])
    st.plotly_chart(px.funnel(funnel, x="users", y="step", title="核心转化漏斗"), use_container_width=True)

    left, right = st.columns(2)
    by_channel = paid_conversion_by_segment(tables["users"], "acquisition_channel")
    by_persona = paid_conversion_by_segment(tables["users"], "persona")
    left.plotly_chart(px.bar(by_channel, x="acquisition_channel", y="paid_conversion_rate", title="渠道付费转化率"), use_container_width=True)
    right.plotly_chart(px.bar(by_persona, x="persona", y="paid_conversion_rate", title="Persona 付费转化率"), use_container_width=True)

    st.subheader("订阅明细")
    st.dataframe(tables["subscriptions"].sort_values("mrr_usd", ascending=False).head(200), use_container_width=True)


def page_cost(tables: dict[str, pd.DataFrame]) -> None:
    st.title("模型成本")
    daily, by_model = model_cost_summary(tables["conversations"])
    cols = st.columns(3)
    cols[0].metric("总成本", money(by_model["cost_usd"].sum()))
    cols[1].metric("总 token", f"{by_model['total_tokens'].sum():,.0f}")
    cols[2].metric("平均失败率", pct(by_model["failure_rate"].mean()))

    left, right = st.columns(2)
    left.plotly_chart(px.area(daily, x="date", y="cost_usd", title="每日模型成本"), use_container_width=True)
    right.plotly_chart(px.line(daily, x="date", y="avg_latency_ms", title="平均响应延迟"), use_container_width=True)

    st.plotly_chart(px.bar(by_model.head(20), x="cost_usd", y="model_name", color="feature", orientation="h", title="模型 x 功能成本 Top 20"), use_container_width=True)
    st.dataframe(by_model, use_container_width=True)


def page_feedback(tables: dict[str, pd.DataFrame]) -> None:
    st.title("用户反馈")
    by_category, daily = feedback_summary(tables["feedback"])
    left, right = st.columns(2)
    left.plotly_chart(px.bar(by_category, x="category", y="feedback_count", color="sentiment", title="反馈主题与情感"), use_container_width=True)
    right.plotly_chart(px.line(daily, x="date", y=["avg_rating", "avg_sentiment"], title="评分与情感趋势"), use_container_width=True)

    negative = tables["feedback"][tables["feedback"]["sentiment"].eq("negative")].copy()
    st.subheader("负面反馈样例")
    st.dataframe(negative[["submitted_at", "user_id", "category", "rating", "nps_score", "feedback_text"]].head(100), use_container_width=True)


@st.cache_data(show_spinner=True)
def train_models_cached(users: pd.DataFrame, events: pd.DataFrame, conversations: pd.DataFrame, feedback: pd.DataFrame) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    frame = build_user_features(users, events, conversations, feedback, observation_days=14)
    categorical, numeric = get_model_columns(frame)
    churn_model = train_logistic_baseline(frame, "churn_14d", categorical, numeric)

    conversion_frame = frame[frame["current_plan"].isin(["Free", "Canceled", "Pro", "Team", "Enterprise"])].copy()
    paid_model = train_logistic_baseline(conversion_frame, "paid_conversion_30d", categorical, numeric)
    return churn_model, paid_model, frame


def page_models(tables: dict[str, pd.DataFrame]) -> None:
    st.title("预测模型")
    st.caption("当前为基线模型：使用注册后 14 天行为特征预测 14 天流失与 30 天付费转化。后续可替换为 XGBoost/LightGBM + SHAP。")

    try:
        churn_model, paid_model, frame = train_models_cached(tables["users"], tables["events"], tables["conversations"], tables["feedback"])
    except ValueError as exc:
        st.warning(str(exc))
        return

    left, right = st.columns(2)
    left.subheader("流失预测模型")
    left.json(churn_model["metrics"])
    left.plotly_chart(px.bar(churn_model["importance"], x="abs_coefficient", y="feature", orientation="h", title="流失模型 Top 特征"), use_container_width=True)

    right.subheader("付费转化预测模型")
    right.json(paid_model["metrics"])
    right.plotly_chart(px.bar(paid_model["importance"], x="abs_coefficient", y="feature", orientation="h", title="付费转化模型 Top 特征"), use_container_width=True)

    st.subheader("用户级特征表样例")
    st.dataframe(frame.head(200), use_container_width=True)


def main() -> None:
    if not DB_PATH.exists():
        st.error("未找到 DuckDB 数据库。请先运行：`python src/generate_data.py --users 10000 --seed 42`")
        st.stop()

    tables = load_data(str(DB_PATH))
    validation = validate_core_tables(tables)
    if not validation["ok"]:
        st.error("核心表校验失败")
        st.write(validation["issues"])
        st.stop()

    filtered = filter_tables(tables)
    page = st.sidebar.radio(
        "页面",
        ["首页总览", "增长分析", "留存分析", "付费转化", "模型成本", "用户反馈", "预测模型"],
    )

    if page == "首页总览":
        page_overview(filtered)
    elif page == "增长分析":
        page_growth(filtered)
    elif page == "留存分析":
        page_retention(filtered)
    elif page == "付费转化":
        page_conversion(filtered)
    elif page == "模型成本":
        page_cost(filtered)
    elif page == "用户反馈":
        page_feedback(filtered)
    elif page == "预测模型":
        page_models(filtered)


if __name__ == "__main__":
    main()

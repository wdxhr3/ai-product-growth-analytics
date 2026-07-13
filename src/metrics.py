from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "database" / "ai_product_analytics.duckdb"


def _as_datetime(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def compute_kpis(
    users: pd.DataFrame,
    events: pd.DataFrame,
    conversations: pd.DataFrame,
    subscriptions: pd.DataFrame,
    feedback: pd.DataFrame,
) -> dict[str, float]:
    users = _as_datetime(users, ["signup_date", "last_active_date"])
    events = _as_datetime(events, ["event_date"])
    conversations = _as_datetime(conversations, ["conversation_date"])

    anchor = max(
        [
            events["event_date"].max() if not events.empty else pd.NaT,
            conversations["conversation_date"].max() if not conversations.empty else pd.NaT,
        ]
    )
    if pd.isna(anchor):
        anchor = users["signup_date"].max()

    last_30 = anchor - pd.Timedelta(days=29)
    active_30_users = pd.concat(
        [
            events.loc[events["event_date"] >= last_30, "user_id"],
            conversations.loc[conversations["conversation_date"] >= last_30, "user_id"],
        ]
    ).nunique()
    successful_workflows = int(conversations["is_success"].sum()) if "is_success" in conversations else 0
    active_users = max(1, pd.concat([events["user_id"], conversations["user_id"]]).nunique())

    active_subs = subscriptions[subscriptions["status"].eq("active")] if not subscriptions.empty else subscriptions
    mrr = float(active_subs["mrr_usd"].sum()) if "mrr_usd" in active_subs else 0.0
    total_cost = float(conversations["cost_usd"].sum()) if "cost_usd" in conversations else 0.0
    avg_rating = float(feedback["rating"].dropna().mean()) if "rating" in feedback and feedback["rating"].notna().any() else np.nan

    nps_rows = feedback[feedback["nps_score"].notna()] if "nps_score" in feedback else pd.DataFrame()
    if nps_rows.empty:
        nps = np.nan
    else:
        promoters = (nps_rows["nps_score"] >= 9).mean()
        detractors = (nps_rows["nps_score"] <= 6).mean()
        nps = float((promoters - detractors) * 100)

    return {
        "users": float(len(users)),
        "paid_users": float(users["paid_converted"].sum()) if "paid_converted" in users else 0.0,
        "active_30d_users": float(active_30_users),
        "activation_rate": float(users["activated_24h"].mean()) if "activated_24h" in users else np.nan,
        "paid_conversion_rate": float(users["paid_converted"].mean()) if "paid_converted" in users else np.nan,
        "mrr_usd": mrr,
        "model_cost_usd": total_cost,
        "successful_workflows_per_active_user": successful_workflows / active_users,
        "avg_rating": avg_rating,
        "nps": nps,
    }


def daily_growth(users: pd.DataFrame, events: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
    users = _as_datetime(users, ["signup_date", "conversion_date"])
    events = _as_datetime(events, ["event_date"])

    signups = users.groupby(users["signup_date"].dt.date).agg(signups=("user_id", "nunique")).reset_index()
    signups = signups.rename(columns={"signup_date": "date"})

    active = events.groupby(events["event_date"].dt.date).agg(dau=("user_id", "nunique")).reset_index()
    active = active.rename(columns={"event_date": "date"})

    paid = users[users["conversion_date"].notna()].groupby(users["conversion_date"].dt.date).agg(new_paid_users=("user_id", "nunique")).reset_index()
    paid = paid.rename(columns={"conversion_date": "date"})

    out = signups.merge(active, on="date", how="outer").merge(paid, on="date", how="outer").sort_values("date")
    out[["signups", "dau", "new_paid_users"]] = out[["signups", "dau", "new_paid_users"]].fillna(0)
    out["date"] = pd.to_datetime(out["date"])
    out["wau"] = out["dau"].rolling(7, min_periods=1).sum()
    out["signup_7d_avg"] = out["signups"].rolling(7, min_periods=1).mean()
    out["new_paid_7d_avg"] = out["new_paid_users"].rolling(7, min_periods=1).mean()
    return out


def activation_funnel(users: pd.DataFrame, events: pd.DataFrame, conversations: pd.DataFrame) -> pd.DataFrame:
    registered = set(users["user_id"])
    activated = set(users.loc[users["activated_24h"], "user_id"]) if "activated_24h" in users else set()
    core = set(conversations["user_id"]) if not conversations.empty else set()
    advanced = set(conversations.loc[conversations["is_advanced_feature"], "user_id"]) if "is_advanced_feature" in conversations else set()
    billing = set(events.loc[events["event_name"].eq("billing_viewed"), "user_id"]) if not events.empty else set()
    clicked = set(events.loc[events["event_name"].eq("subscribe_clicked"), "user_id"]) if not events.empty else set()
    paid = set(users.loc[users["paid_converted"], "user_id"]) if "paid_converted" in users else set()

    raw_steps = [
        ("Registered", registered),
        ("Activated 24h", activated),
        ("Used AI workflow", core),
        ("Used advanced feature", advanced),
        ("Viewed billing", billing),
        ("Clicked subscribe", clicked),
        ("Paid", paid),
    ]
    base = max(1, len(registered))
    rows = []
    previous = None
    previous_set = None
    for step, raw_user_set in raw_steps:
        user_set = raw_user_set if previous_set is None else previous_set & raw_user_set
        count = len(user_set)
        rows.append(
            {
                "step": step,
                "users": count,
                "conversion_from_registered": count / base,
                "conversion_from_previous": np.nan if previous is None or previous == 0 else count / previous,
            }
        )
        previous = count
        previous_set = user_set
    return pd.DataFrame(rows)


def cohort_retention(users: pd.DataFrame, events: pd.DataFrame, max_days: int = 30) -> pd.DataFrame:
    users = _as_datetime(users, ["signup_date"])
    events = _as_datetime(events, ["event_date"])
    activity = events[["user_id", "event_date"]].drop_duplicates()
    joined = activity.merge(users[["user_id", "signup_date"]], on="user_id", how="inner")
    joined["days_since_signup"] = (joined["event_date"] - joined["signup_date"]).dt.days
    joined = joined[(joined["days_since_signup"] >= 0) & (joined["days_since_signup"] <= max_days)]
    joined["cohort_month"] = joined["signup_date"].dt.to_period("M").astype(str)

    cohort_size = users.assign(cohort_month=users["signup_date"].dt.to_period("M").astype(str)).groupby("cohort_month")["user_id"].nunique()
    retained = joined.groupby(["cohort_month", "days_since_signup"])["user_id"].nunique().reset_index(name="retained_users")
    retained["cohort_users"] = retained["cohort_month"].map(cohort_size)
    retained["retention_rate"] = retained["retained_users"] / retained["cohort_users"]
    return retained


def retention_summary(users: pd.DataFrame, events: pd.DataFrame, checkpoints: tuple[int, ...] = (1, 7, 14, 30)) -> pd.DataFrame:
    retention = cohort_retention(users, events, max(checkpoints))
    out = retention[retention["days_since_signup"].isin(checkpoints)].copy()
    return out.pivot_table(index="cohort_month", columns="days_since_signup", values="retention_rate").reset_index()


def paid_conversion_by_segment(users: pd.DataFrame, segment: str) -> pd.DataFrame:
    users = users.copy()
    return (
        users.groupby(segment)
        .agg(
            users=("user_id", "nunique"),
            activated_rate=("activated_24h", "mean"),
            paid_conversion_rate=("paid_converted", "mean"),
            avg_advanced_propensity=("advanced_propensity", "mean"),
        )
        .reset_index()
        .sort_values("paid_conversion_rate", ascending=False)
    )


def model_cost_summary(conversations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conversations = _as_datetime(conversations, ["conversation_date"])
    daily = (
        conversations.groupby(conversations["conversation_date"].dt.date)
        .agg(
            conversations=("conversation_id", "nunique"),
            total_tokens=("total_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
            avg_latency_ms=("response_latency_ms", "mean"),
            failure_rate=("is_success", lambda s: 1 - s.mean()),
        )
        .reset_index()
        .rename(columns={"conversation_date": "date"})
    )
    by_model = (
        conversations.groupby(["model_name", "feature"])
        .agg(
            conversations=("conversation_id", "nunique"),
            total_tokens=("total_tokens", "sum"),
            cost_usd=("cost_usd", "sum"),
            avg_latency_ms=("response_latency_ms", "mean"),
            failure_rate=("is_success", lambda s: 1 - s.mean()),
        )
        .reset_index()
        .sort_values("cost_usd", ascending=False)
    )
    return daily, by_model


def feedback_summary(feedback: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feedback = _as_datetime(feedback, ["submitted_at"])
    by_category = (
        feedback.groupby(["category", "sentiment"])
        .agg(feedback_count=("feedback_id", "nunique"), avg_rating=("rating", "mean"), avg_nps=("nps_score", "mean"))
        .reset_index()
        .sort_values("feedback_count", ascending=False)
    )
    daily = (
        feedback.groupby(feedback["submitted_at"].dt.date)
        .agg(
            feedback_count=("feedback_id", "nunique"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
        .rename(columns={"submitted_at": "date"})
    )
    return by_category, daily

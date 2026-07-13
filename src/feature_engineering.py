from __future__ import annotations

import numpy as np
import pandas as pd


def _to_datetime(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def build_user_features(
    users: pd.DataFrame,
    events: pd.DataFrame,
    conversations: pd.DataFrame,
    feedback: pd.DataFrame,
    observation_days: int = 14,
) -> pd.DataFrame:
    users = _to_datetime(users, ["signup_date", "conversion_date", "last_active_date"])
    events = _to_datetime(events, ["event_date"])
    conversations = _to_datetime(conversations, ["conversation_date"])
    feedback = _to_datetime(feedback, ["submitted_at"])

    base = users[
        [
            "user_id",
            "signup_date",
            "acquisition_channel",
            "country",
            "device_type",
            "persona",
            "company_size",
            "email_domain_type",
            "activated_24h",
            "advanced_propensity",
            "latent_engagement_score",
            "quality_risk_score",
            "paid_converted",
            "conversion_date",
            "current_plan",
            "last_active_date",
            "is_churned_14d",
        ]
    ].copy()

    event_window = events.merge(base[["user_id", "signup_date"]], on="user_id", how="inner")
    event_window["days_since_signup"] = (event_window["event_date"] - event_window["signup_date"]).dt.days
    event_window = event_window[(event_window["days_since_signup"] >= 0) & (event_window["days_since_signup"] <= observation_days)]
    event_agg = (
        event_window.groupby("user_id")
        .agg(
            events_14d=("event_id", "nunique"),
            active_days_14d=("event_date", "nunique"),
            billing_views_14d=("event_name", lambda s: (s == "billing_viewed").sum()),
            subscribe_clicks_14d=("event_name", lambda s: (s == "subscribe_clicked").sum()),
            errors_14d=("event_name", lambda s: (s == "error_occurred").sum()),
        )
        .reset_index()
    )

    conv_window = conversations.merge(base[["user_id", "signup_date"]], on="user_id", how="inner")
    conv_window["days_since_signup"] = (conv_window["conversation_date"] - conv_window["signup_date"]).dt.days
    conv_window = conv_window[(conv_window["days_since_signup"] >= 0) & (conv_window["days_since_signup"] <= observation_days)]
    conv_agg = (
        conv_window.groupby("user_id")
        .agg(
            conversations_14d=("conversation_id", "nunique"),
            successful_conversations_14d=("is_success", "sum"),
            avg_latency_14d=("response_latency_ms", "mean"),
            p95_latency_14d=("response_latency_ms", lambda s: s.quantile(0.95)),
            failure_rate_14d=("is_success", lambda s: 1 - s.mean()),
            total_tokens_14d=("total_tokens", "sum"),
            total_cost_14d=("cost_usd", "sum"),
            advanced_conversations_14d=("is_advanced_feature", "sum"),
            avg_rating_14d=("user_rating", "mean"),
            avg_sentiment_14d=("sentiment_score", "mean"),
        )
        .reset_index()
    )

    for feature in ["file_analysis", "code_assistant", "data_analysis", "agent_workflow"]:
        used = conv_window.groupby("user_id")["feature"].apply(lambda s, f=feature: int((s == f).any())).reset_index(name=f"used_{feature}_14d")
        conv_agg = conv_agg.merge(used, on="user_id", how="left")

    feedback_window = feedback.merge(base[["user_id", "signup_date"]], on="user_id", how="inner")
    feedback_window["days_since_signup"] = (feedback_window["submitted_at"] - feedback_window["signup_date"]).dt.days
    feedback_window = feedback_window[(feedback_window["days_since_signup"] >= 0) & (feedback_window["days_since_signup"] <= observation_days)]
    feedback_agg = (
        feedback_window.groupby("user_id")
        .agg(
            feedback_count_14d=("feedback_id", "nunique"),
            negative_feedback_14d=("sentiment", lambda s: (s == "negative").sum()),
            avg_nps_14d=("nps_score", "mean"),
        )
        .reset_index()
    )

    frame = base.merge(event_agg, on="user_id", how="left").merge(conv_agg, on="user_id", how="left").merge(feedback_agg, on="user_id", how="left")
    numeric_fill_zero = [
        "events_14d",
        "active_days_14d",
        "billing_views_14d",
        "subscribe_clicks_14d",
        "errors_14d",
        "conversations_14d",
        "successful_conversations_14d",
        "total_tokens_14d",
        "total_cost_14d",
        "advanced_conversations_14d",
        "feedback_count_14d",
        "negative_feedback_14d",
        "used_file_analysis_14d",
        "used_code_assistant_14d",
        "used_data_analysis_14d",
        "used_agent_workflow_14d",
    ]
    for col in numeric_fill_zero:
        if col in frame.columns:
            frame[col] = frame[col].fillna(0)

    frame["advanced_feature_share_14d"] = np.where(
        frame["conversations_14d"] > 0,
        frame["advanced_conversations_14d"] / frame["conversations_14d"],
        0,
    )
    frame["days_to_conversion"] = (frame["conversion_date"] - frame["signup_date"]).dt.days
    frame["paid_conversion_30d"] = (frame["paid_converted"]) & (frame["days_to_conversion"].between(0, 30))
    frame["days_since_last_active"] = (frame["last_active_date"].max() - frame["last_active_date"]).dt.days
    frame["churn_14d"] = frame["is_churned_14d"].astype(bool)

    return frame


def get_model_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["acquisition_channel", "country", "device_type", "persona", "company_size", "email_domain_type"]
    numeric = [
        "activated_24h",
        "advanced_propensity",
        "events_14d",
        "active_days_14d",
        "billing_views_14d",
        "subscribe_clicks_14d",
        "errors_14d",
        "conversations_14d",
        "successful_conversations_14d",
        "avg_latency_14d",
        "p95_latency_14d",
        "failure_rate_14d",
        "total_tokens_14d",
        "total_cost_14d",
        "advanced_conversations_14d",
        "advanced_feature_share_14d",
        "avg_rating_14d",
        "avg_sentiment_14d",
        "feedback_count_14d",
        "negative_feedback_14d",
        "avg_nps_14d",
        "used_file_analysis_14d",
        "used_code_assistant_14d",
        "used_data_analysis_14d",
        "used_agent_workflow_14d",
    ]
    categorical = [col for col in categorical if col in frame.columns]
    numeric = [col for col in numeric if col in frame.columns]
    return categorical, numeric

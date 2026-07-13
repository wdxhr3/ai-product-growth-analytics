from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_DIR = PROJECT_ROOT / "data" / "database"
DEFAULT_DB_PATH = DB_DIR / "ai_product_analytics.duckdb"

START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2026-06-30")

CHANNELS = {
    "organic": {"p": 0.24, "quality": 0.12, "campaign": "brand_search"},
    "paid_search": {"p": 0.22, "quality": -0.05, "campaign": "search_ai_tools"},
    "social": {"p": 0.16, "quality": -0.18, "campaign": "creator_social"},
    "referral": {"p": 0.12, "quality": 0.42, "campaign": "invite_rewards"},
    "content": {"p": 0.13, "quality": 0.25, "campaign": "seo_templates"},
    "partner": {"p": 0.07, "quality": 0.35, "campaign": "partner_bundle"},
    "app_store": {"p": 0.06, "quality": -0.08, "campaign": "mobile_launch"},
}

PERSONAS = {
    "casual": {"p": 0.27, "engagement": -0.35, "advanced": -0.40},
    "creator": {"p": 0.17, "engagement": 0.08, "advanced": 0.08},
    "developer": {"p": 0.18, "engagement": 0.28, "advanced": 0.34},
    "analyst": {"p": 0.16, "engagement": 0.24, "advanced": 0.42},
    "operator": {"p": 0.13, "engagement": 0.14, "advanced": 0.18},
    "team_admin": {"p": 0.09, "engagement": 0.34, "advanced": 0.50},
}

COMPANY_SIZES = {
    "individual": {"p": 0.48, "paid": -0.22, "seats": (1, 1)},
    "small_team": {"p": 0.26, "paid": 0.16, "seats": (2, 8)},
    "mid_market": {"p": 0.17, "paid": 0.38, "seats": (8, 45)},
    "enterprise": {"p": 0.09, "paid": 0.68, "seats": (40, 220)},
}

COUNTRIES = {
    "US": 0.32,
    "CN": 0.18,
    "IN": 0.10,
    "JP": 0.07,
    "GB": 0.07,
    "DE": 0.06,
    "BR": 0.05,
    "CA": 0.05,
    "SG": 0.04,
    "AU": 0.03,
    "FR": 0.03,
}

DEVICE_TYPES = {"desktop": 0.58, "mobile": 0.33, "tablet": 0.09}
EMAIL_DOMAIN_TYPES = {"personal": 0.58, "business": 0.36, "education": 0.06}

FEATURES = ["chat", "file_analysis", "code_assistant", "data_analysis", "agent_workflow"]
ADVANCED_FEATURES = {"file_analysis", "code_assistant", "data_analysis", "agent_workflow"}

MODEL_PRICING = {
    "fast-chat": {"input_per_m": 0.18, "output_per_m": 0.60, "latency": 850, "fail": 0.012},
    "balanced": {"input_per_m": 0.65, "output_per_m": 2.20, "latency": 1350, "fail": 0.018},
    "reasoning-pro": {"input_per_m": 3.00, "output_per_m": 12.00, "latency": 3100, "fail": 0.030},
    "vision-pro": {"input_per_m": 2.20, "output_per_m": 8.00, "latency": 2600, "fail": 0.026},
    "agentic-pro": {"input_per_m": 4.00, "output_per_m": 16.00, "latency": 4200, "fail": 0.040},
}

FEATURE_TOKEN_PROFILE = {
    "chat": {"input": 900, "output": 700, "latency": 1.0, "complexity": 0.00},
    "file_analysis": {"input": 5600, "output": 1300, "latency": 1.45, "complexity": 0.04},
    "code_assistant": {"input": 2400, "output": 2200, "latency": 1.35, "complexity": 0.03},
    "data_analysis": {"input": 4800, "output": 1800, "latency": 1.55, "complexity": 0.05},
    "agent_workflow": {"input": 6200, "output": 3100, "latency": 1.90, "complexity": 0.07},
}


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


def normalize(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr / arr.sum()


def choice_from_dict(rng: np.random.Generator, weights: dict[str, float], size: int) -> np.ndarray:
    keys = list(weights.keys())
    probs = normalize(list(weights.values()))
    return rng.choice(keys, size=size, p=probs)


def build_signup_dates(rng: np.random.Generator, n_users: int) -> pd.DatetimeIndex:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    day_index = np.arange(len(dates))
    trend = np.linspace(0.62, 1.55, len(dates))
    weekday = np.where(dates.dayofweek < 5, 1.12, 0.78)

    campaign = np.ones(len(dates))
    for center, lift, width in [
        ("2025-03-18", 0.45, 12),
        ("2025-08-05", 0.70, 18),
        ("2026-01-20", 0.55, 14),
        ("2026-05-14", 0.62, 16),
    ]:
        center_idx = (pd.Timestamp(center) - START_DATE).days
        campaign += lift * np.exp(-0.5 * ((day_index - center_idx) / width) ** 2)

    weights = trend * weekday * campaign
    weights = weights / weights.sum()
    return pd.DatetimeIndex(rng.choice(dates, size=n_users, p=weights))


def generate_users(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    signup_dates = build_signup_dates(rng, n_users)
    signup_ts = signup_dates + pd.to_timedelta(rng.integers(0, 24 * 60 * 60, n_users), unit="s")

    channel_probs = {k: v["p"] for k, v in CHANNELS.items()}
    persona_probs = {k: v["p"] for k, v in PERSONAS.items()}
    company_probs = {k: v["p"] for k, v in COMPANY_SIZES.items()}

    channels = choice_from_dict(rng, channel_probs, n_users)
    personas = choice_from_dict(rng, persona_probs, n_users)
    company_sizes = choice_from_dict(rng, company_probs, n_users)
    countries = choice_from_dict(rng, COUNTRIES, n_users)
    devices = choice_from_dict(rng, DEVICE_TYPES, n_users)
    email_domains = choice_from_dict(rng, EMAIL_DOMAIN_TYPES, n_users)

    channel_quality = np.array([CHANNELS[c]["quality"] for c in channels])
    persona_engagement = np.array([PERSONAS[p]["engagement"] for p in personas])
    persona_advanced = np.array([PERSONAS[p]["advanced"] for p in personas])
    company_paid = np.array([COMPANY_SIZES[c]["paid"] for c in company_sizes])

    latent_engagement = (
        rng.normal(0, 0.75, n_users)
        + channel_quality
        + persona_engagement
        + 0.25 * company_paid
        + np.where(email_domains == "business", 0.10, 0.0)
    )
    advanced_propensity = sigmoid(-0.22 + 0.75 * latent_engagement + persona_advanced + 0.35 * company_paid)
    quality_risk = np.clip(rng.beta(2.2, 8.0, n_users) + np.maximum(-latent_engagement, 0) * 0.08, 0, 1)

    activated_prob = sigmoid(-0.15 + 0.95 * latent_engagement + 0.55 * channel_quality)
    activated_24h = rng.random(n_users) < activated_prob

    paid_prob = sigmoid(
        -2.65
        + 0.95 * latent_engagement
        + 1.05 * advanced_propensity
        + 0.55 * activated_24h
        + 0.95 * company_paid
        - 0.65 * quality_risk
    )
    paid_candidate = rng.random(n_users) < paid_prob
    conv_offsets = np.maximum(1, rng.gamma(shape=2.0, scale=9.5, size=n_users).astype(int))
    days_to_end = (END_DATE - signup_dates).days
    conversion_dates = signup_dates + pd.to_timedelta(conv_offsets, unit="D")
    paid_candidate = paid_candidate & (conv_offsets < np.maximum(days_to_end - 7, 1))

    churn_prob = sigmoid(
        -1.15
        - 0.92 * latent_engagement
        + 1.75 * quality_risk
        + 0.85 * (~activated_24h)
        - 0.55 * paid_candidate
        - 0.35 * advanced_propensity
    )
    is_churned = rng.random(n_users) < churn_prob
    lifetime_if_churned = np.maximum(
        1,
        rng.gamma(shape=2.0, scale=24.0, size=n_users).astype(int)
        + (activated_24h * rng.integers(8, 45, n_users)),
    )
    active_end = signup_dates + pd.to_timedelta(lifetime_if_churned, unit="D")
    recent_gap = pd.to_timedelta(rng.integers(0, 8, n_users), unit="D")
    active_end = pd.DatetimeIndex(np.where(is_churned, active_end, END_DATE - recent_gap))
    active_end = pd.DatetimeIndex(np.minimum(active_end.values.astype("datetime64[ns]"), np.array(END_DATE, dtype="datetime64[ns]")))
    active_end = pd.DatetimeIndex(np.maximum(active_end.values.astype("datetime64[ns]"), signup_dates.values.astype("datetime64[ns]")))

    paid_converted = paid_candidate & (conversion_dates <= active_end)
    conversion_dates = pd.Series(conversion_dates).where(paid_converted, pd.NaT)

    plan_names = []
    for paid, company, persona in zip(paid_converted, company_sizes, personas):
        if not paid:
            plan_names.append("Free")
        elif company == "enterprise" or persona == "team_admin":
            plan_names.append(rng.choice(["Team", "Enterprise"], p=[0.55, 0.45]))
        elif company == "mid_market":
            plan_names.append(rng.choice(["Pro", "Team"], p=[0.45, 0.55]))
        else:
            plan_names.append(rng.choice(["Pro", "Team"], p=[0.82, 0.18]))

    current_plan = []
    for paid, plan, churned, last_active in zip(paid_converted, plan_names, is_churned, active_end):
        if not paid:
            current_plan.append("Free")
        elif churned and pd.Timestamp(last_active) <= END_DATE - pd.Timedelta(days=14):
            current_plan.append("Canceled")
        else:
            current_plan.append(plan)

    users = pd.DataFrame(
        {
            "user_id": [f"u_{i:05d}" for i in range(1, n_users + 1)],
            "signup_date": signup_dates.date,
            "signup_ts": signup_ts,
            "acquisition_channel": channels,
            "country": countries,
            "device_type": devices,
            "persona": personas,
            "company_size": company_sizes,
            "email_domain_type": email_domains,
            "marketing_campaign": [CHANNELS[c]["campaign"] for c in channels],
            "activated_24h": activated_24h,
            "advanced_propensity": np.round(advanced_propensity, 4),
            "latent_engagement_score": np.round(latent_engagement, 4),
            "quality_risk_score": np.round(quality_risk, 4),
            "paid_converted": paid_converted,
            "conversion_date": pd.to_datetime(conversion_dates).dt.date,
            "current_plan": current_plan,
            "last_active_date": active_end.date,
            "is_churned_14d": active_end <= END_DATE - pd.Timedelta(days=14),
        }
    )
    return users


def feature_probabilities(persona: str, advanced_propensity: float, paid: bool) -> np.ndarray:
    probs = np.array([0.66, 0.10, 0.10, 0.08, 0.06], dtype=float)
    if persona == "developer":
        probs += np.array([-0.11, -0.02, 0.16, -0.01, -0.02])
    elif persona == "analyst":
        probs += np.array([-0.14, 0.06, -0.03, 0.13, -0.02])
    elif persona == "creator":
        probs += np.array([-0.04, 0.08, -0.03, -0.02, 0.01])
    elif persona == "operator":
        probs += np.array([-0.07, 0.02, -0.02, 0.04, 0.03])
    elif persona == "team_admin":
        probs += np.array([-0.17, 0.03, -0.01, 0.04, 0.11])

    probs += advanced_propensity * np.array([-0.20, 0.05, 0.05, 0.05, 0.05])
    if paid:
        probs += np.array([-0.12, 0.03, 0.03, 0.02, 0.04])
    probs = np.clip(probs, 0.01, None)
    return probs / probs.sum()


def model_probabilities(feature: str, paid: bool, advanced_propensity: float) -> np.ndarray:
    if feature == "chat":
        probs = np.array([0.54, 0.33, 0.08, 0.03, 0.02])
    elif feature == "file_analysis":
        probs = np.array([0.16, 0.39, 0.12, 0.28, 0.05])
    elif feature == "code_assistant":
        probs = np.array([0.17, 0.35, 0.34, 0.03, 0.11])
    elif feature == "data_analysis":
        probs = np.array([0.12, 0.35, 0.32, 0.06, 0.15])
    else:
        probs = np.array([0.07, 0.26, 0.28, 0.04, 0.35])

    if paid:
        probs += np.array([-0.08, -0.03, 0.05, 0.01, 0.05])
    probs += advanced_propensity * np.array([-0.09, -0.02, 0.04, 0.01, 0.06])
    probs = np.clip(probs, 0.01, None)
    return probs / probs.sum()


def plan_at_time(row: pd.Series, event_time: pd.Timestamp) -> str:
    if not row["paid_converted"] or pd.isna(row["conversion_date"]):
        return "Free"
    if event_time.date() < row["conversion_date"]:
        return "Free"
    if row["current_plan"] == "Canceled":
        return "Paid"
    return row["current_plan"]


def generate_conversations_and_events(users: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    conversation_records: list[dict] = []
    event_records: list[dict] = []
    model_names = list(MODEL_PRICING.keys())

    conv_idx = 1
    session_idx = 1

    for row in users.itertuples(index=False):
        user = pd.Series(row._asdict())
        signup = pd.Timestamp(user.signup_date)
        last_active = pd.Timestamp(user.last_active_date)
        active_span_days = max(1, (last_active - signup).days + 1)
        paid = bool(user.paid_converted)

        mean_conversations = (
            2.4
            + 5.4 * bool(user.activated_24h)
            + 10.5 * float(user.advanced_propensity)
            + 11.0 * paid
            + 4.2 * max(float(user.latent_engagement_score), 0)
        )
        mean_conversations *= np.clip(active_span_days / 120, 0.25, 2.4)
        n_conversations = int(rng.negative_binomial(2, 2 / (2 + mean_conversations)))
        if bool(user.activated_24h):
            n_conversations = max(n_conversations, 1)

        registered_time = pd.Timestamp(user.signup_ts)
        event_records.append(
            {
                "user_id": user.user_id,
                "event_time": registered_time,
                "event_date": registered_time.date(),
                "event_name": "user_registered",
                "feature": "account",
                "model_name": None,
                "session_id": None,
                "conversation_id": None,
                "device_type": user.device_type,
                "country": user.country,
                "acquisition_channel": user.acquisition_channel,
                "plan_at_time": "Free",
                "latency_ms": np.nan,
                "success_flag": True,
                "tokens": 0,
                "cost_usd": 0.0,
            }
        )

        if bool(user.activated_24h):
            event_records.append(
                {
                    "user_id": user.user_id,
                    "event_time": registered_time + pd.to_timedelta(rng.integers(3, 22 * 60), unit="m"),
                    "event_date": (registered_time + pd.to_timedelta(1, unit="h")).date(),
                    "event_name": "first_login",
                    "feature": "account",
                    "model_name": None,
                    "session_id": None,
                    "conversation_id": None,
                    "device_type": user.device_type,
                    "country": user.country,
                    "acquisition_channel": user.acquisition_channel,
                    "plan_at_time": "Free",
                    "latency_ms": np.nan,
                    "success_flag": True,
                    "tokens": 0,
                    "cost_usd": 0.0,
                }
            )

        if n_conversations == 0:
            continue

        raw_offsets = rng.beta(1.1, 1.8, n_conversations)
        offsets = np.floor(raw_offsets * active_span_days).astype(int)
        if not bool(user.is_churned_14d):
            recency_boost = rng.random(n_conversations) < 0.22
            offsets[recency_boost] = rng.integers(max(0, active_span_days - 30), active_span_days, recency_boost.sum())

        for offset in offsets:
            conv_time = signup + pd.Timedelta(days=int(offset)) + pd.to_timedelta(rng.integers(8 * 3600, 23 * 3600), unit="s")
            if conv_time > last_active + pd.Timedelta(hours=23):
                conv_time = last_active + pd.to_timedelta(rng.integers(8 * 3600, 21 * 3600), unit="s")

            current_plan = plan_at_time(user, conv_time)
            paid_at_time = current_plan != "Free"
            feature = rng.choice(FEATURES, p=feature_probabilities(user.persona, float(user.advanced_propensity), paid_at_time))
            model_name = rng.choice(model_names, p=model_probabilities(feature, paid_at_time, float(user.advanced_propensity)))

            token_profile = FEATURE_TOKEN_PROFILE[feature]
            model_profile = MODEL_PRICING[model_name]
            input_tokens = int(max(80, rng.gamma(2.2, token_profile["input"] / 2.2)))
            output_tokens = int(max(50, rng.gamma(2.0, token_profile["output"] / 2.0)))
            messages_count = int(np.clip(rng.poisson(4 + token_profile["complexity"] * 35) + 1, 1, 24))

            load_factor = 1.0 + (0.12 if conv_time.hour in [10, 11, 14, 15, 20, 21] else 0.0)
            latency = float(
                rng.lognormal(
                    mean=np.log(model_profile["latency"] * token_profile["latency"] * load_factor),
                    sigma=0.32,
                )
            )
            fail_prob = (
                model_profile["fail"]
                + token_profile["complexity"]
                + float(user.quality_risk_score) * 0.055
                + (0.020 if latency > 6000 else 0.0)
            )
            success = bool(rng.random() > min(fail_prob, 0.28))
            error_type = None
            if not success:
                error_type = rng.choice(["timeout", "model_error", "tool_error", "rate_limit"], p=[0.42, 0.25, 0.22, 0.11])

            cost_usd = (
                input_tokens * model_profile["input_per_m"] / 1_000_000
                + output_tokens * model_profile["output_per_m"] / 1_000_000
            )

            rating_probability = 0.20 + 0.12 * (not success) + 0.06 * (latency > 5000)
            if rng.random() < rating_probability:
                rating_base = 4.65 - 0.55 * (latency > 3500) - 1.8 * (not success) - 0.65 * float(user.quality_risk_score)
                user_rating = float(np.clip(np.round(rng.normal(rating_base, 0.75)), 1, 5))
            else:
                user_rating = np.nan

            sentiment_score = float(
                np.clip(
                    0.55
                    + 0.10 * success
                    - 0.00010 * max(latency - 2000, 0)
                    - 0.90 * (not success)
                    + rng.normal(0, 0.18),
                    -1,
                    1,
                )
            )

            conversation_id = f"c_{conv_idx:07d}"
            session_id = f"s_{session_idx:07d}"
            conv_idx += 1
            session_idx += 1

            conversation_records.append(
                {
                    "conversation_id": conversation_id,
                    "user_id": user.user_id,
                    "conversation_started_at": conv_time,
                    "conversation_date": conv_time.date(),
                    "feature": feature,
                    "model_name": model_name,
                    "messages_count": messages_count,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "response_latency_ms": round(latency, 2),
                    "is_success": success,
                    "error_type": error_type,
                    "cost_usd": round(cost_usd, 6),
                    "user_rating": user_rating,
                    "sentiment_score": round(sentiment_score, 4),
                    "is_advanced_feature": feature in ADVANCED_FEATURES,
                    "plan_at_time": current_plan,
                }
            )

            base_event = {
                "user_id": user.user_id,
                "event_date": conv_time.date(),
                "feature": feature,
                "model_name": model_name,
                "session_id": session_id,
                "conversation_id": conversation_id,
                "device_type": user.device_type,
                "country": user.country,
                "acquisition_channel": user.acquisition_channel,
                "plan_at_time": current_plan,
                "latency_ms": round(latency, 2),
                "success_flag": success,
                "tokens": input_tokens + output_tokens,
                "cost_usd": round(cost_usd, 6),
            }
            event_records.append({**base_event, "event_time": conv_time, "event_name": "conversation_started"})
            event_records.append(
                {
                    **base_event,
                    "event_time": conv_time + pd.to_timedelta(rng.integers(5, 90), unit="s"),
                    "event_name": "ai_message_sent",
                }
            )
            if feature in ADVANCED_FEATURES:
                event_records.append(
                    {
                        **base_event,
                        "event_time": conv_time + pd.to_timedelta(rng.integers(20, 180), unit="s"),
                        "event_name": f"{feature}_run",
                    }
                )
            if not success:
                event_records.append(
                    {
                        **base_event,
                        "event_time": conv_time + pd.to_timedelta(rng.integers(10, 120), unit="s"),
                        "event_name": "error_occurred",
                    }
                )
            if not np.isnan(user_rating):
                event_records.append(
                    {
                        **base_event,
                        "event_time": conv_time + pd.to_timedelta(rng.integers(60, 600), unit="s"),
                        "event_name": "feedback_submitted",
                    }
                )
            if (not paid_at_time) and (feature in ADVANCED_FEATURES) and rng.random() < 0.06 + 0.07 * float(user.advanced_propensity):
                event_records.append(
                    {
                        **base_event,
                        "event_time": conv_time + pd.to_timedelta(rng.integers(60, 900), unit="s"),
                        "event_name": "billing_viewed",
                        "latency_ms": np.nan,
                        "tokens": 0,
                        "cost_usd": 0.0,
                    }
                )
                near_conversion = (
                    bool(user.paid_converted)
                    and pd.notna(user.conversion_date)
                    and abs((pd.Timestamp(user.conversion_date) - conv_time).days) <= 14
                )
                click_probability = 0.035 + 0.055 * float(user.advanced_propensity) + (0.34 if near_conversion else 0.0)
                if rng.random() < click_probability:
                    event_records.append(
                        {
                            **base_event,
                            "event_time": conv_time + pd.to_timedelta(rng.integers(120, 1200), unit="s"),
                            "event_name": "subscribe_clicked",
                            "latency_ms": np.nan,
                            "tokens": 0,
                            "cost_usd": 0.0,
                        }
                    )

    conversations = pd.DataFrame(conversation_records)
    events = pd.DataFrame(event_records)
    events = events.sort_values("event_time").reset_index(drop=True)
    events.insert(0, "event_id", [f"e_{i:08d}" for i in range(1, len(events) + 1)])
    return conversations, events


def generate_subscriptions(users: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    records = []
    paid_users = users[users["paid_converted"]].copy()
    for idx, row in enumerate(paid_users.itertuples(index=False), start=1):
        plan_name = row.current_plan if row.current_plan != "Canceled" else rng.choice(["Pro", "Team", "Enterprise"], p=[0.58, 0.30, 0.12])
        if plan_name == "Pro":
            seats = 1
            mrr = 20.0
        elif plan_name == "Team":
            low, high = COMPANY_SIZES[row.company_size]["seats"]
            seats = int(np.clip(rng.integers(max(2, low), max(3, high + 1)), 2, 80))
            mrr = round(seats * 16.0, 2)
        else:
            low, high = COMPANY_SIZES[row.company_size]["seats"]
            seats = int(np.clip(rng.integers(max(20, low), max(21, high + 1)), 20, 260))
            mrr = round(499 + seats * 9.5, 2)

        billing_cycle = rng.choice(["monthly", "annual"], p=[0.74, 0.26])
        if billing_cycle == "annual":
            mrr = round(mrr * 0.85, 2)

        started_at = pd.Timestamp(row.conversion_date)
        churned_paid = row.current_plan == "Canceled"
        if churned_paid:
            status = "canceled"
            ended_at = pd.Timestamp(row.last_active_date).date()
        else:
            status = rng.choice(["active", "past_due"], p=[0.96, 0.04])
            ended_at = pd.NaT

        records.append(
            {
                "subscription_id": f"sub_{idx:06d}",
                "user_id": row.user_id,
                "plan_name": plan_name,
                "billing_cycle": billing_cycle,
                "started_at": started_at.date(),
                "ended_at": ended_at,
                "status": status,
                "mrr_usd": mrr if status != "canceled" else 0.0,
                "arr_usd": round(mrr * 12, 2),
                "seats": seats,
                "payment_method": rng.choice(["card", "invoice", "paypal"], p=[0.78, 0.17, 0.05]),
                "currency": "USD",
                "created_at": started_at + pd.to_timedelta(rng.integers(0, 3600), unit="s"),
                "updated_at": END_DATE - pd.to_timedelta(rng.integers(0, 20), unit="D"),
            }
        )
    return pd.DataFrame(records)


def generate_feedback(conversations: pd.DataFrame, users: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rated = conversations[conversations["user_rating"].notna()].copy()
    if len(rated) > 0:
        rated = rated.sample(frac=min(1.0, 18000 / len(rated)), random_state=7)

    category_by_condition = {
        "latency": "performance",
        "failure": "reliability",
        "low_rating": "quality",
        "positive": "product_value",
    }
    text_templates = {
        "performance": "The answer quality is useful, but the response time feels too slow for daily work.",
        "reliability": "The workflow failed before finishing, so I could not trust it for an important task.",
        "quality": "The response missed important context and needed too much correction.",
        "pricing": "The advanced features are valuable, but the paid plan feels expensive for my usage.",
        "usability": "The product is powerful, but the workflow setup could be easier.",
        "product_value": "This saved time and helped me finish a real work task.",
    }

    records = []
    for idx, row in enumerate(rated.itertuples(index=False), start=1):
        if not row.is_success:
            category = category_by_condition["failure"]
        elif row.response_latency_ms > 5200:
            category = category_by_condition["latency"]
        elif row.user_rating <= 3:
            category = category_by_condition["low_rating"]
        elif rng.random() < 0.08:
            category = rng.choice(["pricing", "usability"], p=[0.55, 0.45])
        else:
            category = category_by_condition["positive"]

        rating = float(row.user_rating)
        sentiment = "positive" if rating >= 4 else "neutral" if rating == 3 else "negative"
        records.append(
            {
                "feedback_id": f"fb_{idx:07d}",
                "user_id": row.user_id,
                "conversation_id": row.conversation_id,
                "submitted_at": pd.Timestamp(row.conversation_started_at) + pd.to_timedelta(rng.integers(1, 900), unit="s"),
                "feedback_type": "conversation_rating",
                "rating": rating,
                "nps_score": np.nan,
                "sentiment": sentiment,
                "sentiment_score": row.sentiment_score,
                "category": category,
                "feedback_text": text_templates[category],
                "response_latency_ms": row.response_latency_ms,
                "model_name": row.model_name,
                "feature": row.feature,
                "source": "in_app",
            }
        )

    sample_users = users.sample(n=min(1800, len(users)), random_state=19)
    start_idx = len(records) + 1
    for offset, row in enumerate(sample_users.itertuples(index=False), start=0):
        base_nps = 7.2 + 1.0 * row.paid_converted + 0.65 * row.activated_24h - 2.2 * row.quality_risk_score
        nps = int(np.clip(np.round(rng.normal(base_nps, 1.7)), 0, 10))
        sentiment = "positive" if nps >= 9 else "neutral" if nps >= 7 else "negative"
        category = "product_value" if nps >= 9 else rng.choice(["pricing", "performance", "quality", "usability"], p=[0.28, 0.26, 0.26, 0.20])
        records.append(
            {
                "feedback_id": f"fb_{start_idx + offset:07d}",
                "user_id": row.user_id,
                "conversation_id": None,
                "submitted_at": pd.Timestamp(row.signup_date) + pd.to_timedelta(rng.integers(5, 120), unit="D"),
                "feedback_type": "nps",
                "rating": np.nan,
                "nps_score": nps,
                "sentiment": sentiment,
                "sentiment_score": round((nps - 5) / 5, 3),
                "category": category,
                "feedback_text": text_templates.get(category, text_templates["product_value"]),
                "response_latency_ms": np.nan,
                "model_name": None,
                "feature": None,
                "source": rng.choice(["in_app", "email", "support"], p=[0.76, 0.15, 0.09]),
            }
        )

    return pd.DataFrame(records)


def generate_experiments(users: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiments = pd.DataFrame(
        [
            {
                "experiment_id": "exp_001",
                "experiment_name": "onboarding_v2",
                "hypothesis": "A guided first prompt improves activation and D7 retention.",
                "primary_metric": "activated_24h_rate",
                "start_date": pd.Timestamp("2025-03-01").date(),
                "end_date": pd.Timestamp("2025-05-15").date(),
                "status": "completed",
                "owner": "product_growth",
            },
            {
                "experiment_id": "exp_002",
                "experiment_name": "paywall_copy",
                "hypothesis": "Outcome-oriented upgrade copy improves free-to-paid conversion.",
                "primary_metric": "paid_conversion_30d",
                "start_date": pd.Timestamp("2025-08-01").date(),
                "end_date": pd.Timestamp("2025-10-01").date(),
                "status": "completed",
                "owner": "growth",
            },
            {
                "experiment_id": "exp_003",
                "experiment_name": "model_router_latency",
                "hypothesis": "Routing simple tasks to faster models reduces latency without hurting satisfaction.",
                "primary_metric": "successful_workflows_per_active_user",
                "start_date": pd.Timestamp("2026-01-10").date(),
                "end_date": pd.Timestamp("2026-03-15").date(),
                "status": "completed",
                "owner": "ml_platform",
            },
            {
                "experiment_id": "exp_004",
                "experiment_name": "agent_template_gallery",
                "hypothesis": "Template discovery increases agent workflow adoption.",
                "primary_metric": "agent_workflow_adoption_rate",
                "start_date": pd.Timestamp("2026-05-01").date(),
                "end_date": pd.Timestamp("2026-07-15").date(),
                "status": "running",
                "owner": "product_agent",
            },
        ]
    )

    assignments = []
    assignment_idx = 1
    for exp in experiments.itertuples(index=False):
        eligible = users[pd.to_datetime(users["signup_date"]) <= pd.Timestamp(exp.end_date)].copy()
        eligible = eligible[pd.to_datetime(eligible["signup_date"]) >= pd.Timestamp(exp.start_date) - pd.Timedelta(days=14)]
        if eligible.empty:
            continue
        sample = eligible.sample(frac=0.58, random_state=int(exp.experiment_id[-3:]))
        for row in sample.itertuples(index=False):
            assigned_at = max(pd.Timestamp(row.signup_ts), pd.Timestamp(exp.start_date) + pd.to_timedelta(rng.integers(0, 10), unit="D"))
            assignments.append(
                {
                    "assignment_id": f"asgn_{assignment_idx:07d}",
                    "experiment_id": exp.experiment_id,
                    "user_id": row.user_id,
                    "variant": rng.choice(["control", "variant_a", "variant_b"], p=[0.40, 0.40, 0.20]),
                    "assigned_at": assigned_at,
                    "exposed": bool(rng.random() < (0.74 if row.activated_24h else 0.38)),
                }
            )
            assignment_idx += 1

    return experiments, pd.DataFrame(assignments)


def write_outputs(tables: dict[str, pd.DataFrame], db_path: Path) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)

    con = duckdb.connect(str(db_path))
    try:
        for name, df in tables.items():
            con.register(f"{name}_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {name}_df")
            con.unregister(f"{name}_df")
    finally:
        con.close()


def generate_dataset(n_users: int, seed: int, db_path: Path = DEFAULT_DB_PATH) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    users = generate_users(n_users, rng)
    conversations, events = generate_conversations_and_events(users, rng)
    subscriptions = generate_subscriptions(users, rng)
    feedback = generate_feedback(conversations, users, rng)
    experiments, experiment_assignments = generate_experiments(users, rng)

    tables = {
        "users": users,
        "events": events,
        "conversations": conversations,
        "subscriptions": subscriptions,
        "feedback": feedback,
        "experiments": experiments,
        "experiment_assignments": experiment_assignments,
    }
    write_outputs(tables, db_path)
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic AI SaaS product analytics data.")
    parser.add_argument("--users", type=int, default=10_000, help="Number of synthetic users to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="DuckDB output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = generate_dataset(args.users, args.seed, args.db_path)
    print("Synthetic dataset generated successfully.")
    for name, df in tables.items():
        print(f"- {name}: {len(df):,} rows")
    print(f"DuckDB: {args.db_path}")


if __name__ == "__main__":
    main()

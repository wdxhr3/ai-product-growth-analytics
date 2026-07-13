from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


DATE_COLUMNS = {
    "users": ["signup_date", "signup_ts", "conversion_date", "last_active_date"],
    "events": ["event_time", "event_date"],
    "conversations": ["conversation_started_at", "conversation_date"],
    "subscriptions": ["started_at", "ended_at", "created_at", "updated_at"],
    "feedback": ["submitted_at"],
    "experiments": ["start_date", "end_date"],
    "experiment_assignments": ["assigned_at"],
}


def load_table(db_path: str | Path, table_name: str) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(f"SELECT * FROM {table_name}").df()
    for col in DATE_COLUMNS.get(table_name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_all_tables(db_path: str | Path) -> dict[str, pd.DataFrame]:
    table_names = [
        "users",
        "events",
        "conversations",
        "subscriptions",
        "feedback",
        "experiments",
        "experiment_assignments",
    ]
    return {name: load_table(db_path, name) for name in table_names}


def validate_core_tables(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    required = {
        "users": ["user_id", "signup_date", "acquisition_channel", "persona"],
        "events": ["event_id", "user_id", "event_time", "event_name"],
        "conversations": ["conversation_id", "user_id", "feature", "model_name", "cost_usd"],
        "subscriptions": ["subscription_id", "user_id", "plan_name", "mrr_usd"],
        "feedback": ["feedback_id", "user_id", "feedback_type", "category"],
    }
    issues: list[str] = []
    for table_name, columns in required.items():
        if table_name not in tables:
            issues.append(f"Missing table: {table_name}")
            continue
        missing = sorted(set(columns) - set(tables[table_name].columns))
        if missing:
            issues.append(f"{table_name} missing columns: {', '.join(missing)}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "row_counts": {name: len(df) for name, df in tables.items()},
    }

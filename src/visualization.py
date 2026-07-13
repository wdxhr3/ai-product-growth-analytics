from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def line_chart(df: pd.DataFrame, x: str, y: str | list[str], title: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
    return fig


def retention_heatmap(retention: pd.DataFrame, title: str = "Cohort retention") -> go.Figure:
    matrix = retention.pivot_table(index="cohort_month", columns="days_since_signup", values="retention_rate")
    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale="Teal",
        labels=dict(x="Days since signup", y="Signup cohort", color="Retention"),
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig

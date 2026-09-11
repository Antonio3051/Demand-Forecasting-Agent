"""
src/visualizer.py
=================

All **Plotly** charts live here.

Why a separate module?
----------------------
Separation of concerns:  ``data_handler`` knows about files, ``agent`` knows
about decisions, ``visualizer`` knows about pixels.  ``app.py`` only glues
them together.  If tomorrow we replace Streamlit with a Flask API or a
Jupyter notebook, these functions still work — they return
``plotly.graph_objects.Figure`` objects, not Streamlit widgets.

Every function follows the same shape:

    def plot_xxx(<pandas objects>, <cosmetic options>) -> go.Figure

so they are trivial to test and to reuse.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# A small, colour-blind-friendly palette reused across charts.
COLORS = {
    "history": "#1f77b4",  # blue
    "test": "#ff7f0e",  # orange
    "forecast": "#2ca02c",  # green
    "band": "rgba(44, 160, 44, 0.18)",  # translucent green
}


def _base_layout(fig: go.Figure, title: str, y_title: str = "Demand") -> go.Figure:
    """Apply consistent styling so all charts look like one product."""
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",  # one tooltip for all traces at a given date
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    return fig


# ----------------------------------------------------------------------
# 1. Raw history
# ----------------------------------------------------------------------
def plot_history(series: pd.Series, title: str = "Historical demand") -> go.Figure:
    """Simple line chart of the cleaned series — the first thing users see."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="History",
            line=dict(color=COLORS["history"], width=1.5),
        )
    )
    return _base_layout(fig, title)


# ----------------------------------------------------------------------
# 2. Forecast with uncertainty band
# ----------------------------------------------------------------------
def plot_forecast(
    history: pd.Series,
    forecast: pd.DataFrame,
    model_name: str,
    show_last: int | None = 180,
) -> go.Figure:
    """History + future forecast + shaded confidence interval.

    Parameters
    ----------
    show_last:
        Only draw the last *n* historical points so the forecast is not a
        tiny sliver at the far right of a long chart. ``None`` shows all.
    """
    hist = history if show_last is None else history.iloc[-show_last:]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist.values,
            mode="lines",
            name="History",
            line=dict(color=COLORS["history"], width=1.5),
        )
    )

    # Shaded band: draw the upper bound, then the lower bound with
    # ``fill="tonexty"`` so Plotly fills the area between them.
    if {"yhat_lower", "yhat_upper"} <= set(forecast.columns):
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=COLORS["band"],
                name="Uncertainty",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["yhat"],
            mode="lines+markers",
            name=f"Forecast ({model_name})",
            line=dict(color=COLORS["forecast"], width=2.5),
            marker=dict(size=4),
        )
    )

    # Vertical dashed line marking "today" (end of known data).
    fig.add_vline(x=hist.index[-1], line_dash="dash", line_color="grey")
    return _base_layout(fig, f"Demand forecast — {model_name}")


# ----------------------------------------------------------------------
# 3. Model comparison on the hold-out window
# ----------------------------------------------------------------------
def plot_backtest(
    train: pd.Series,
    test: pd.Series,
    predictions: dict[str, pd.DataFrame],
    show_last: int | None = 90,
) -> go.Figure:
    """Overlay each candidate's test-window prediction against the truth.

    This is the chart that makes the agent's *decision* visible: students can
    see with their own eyes which line hugs the orange "actual" line best.

    Parameters
    ----------
    predictions:
        ``{model_name: DataFrame with a 'yhat' column}`` — typically built
        from ``AgentReport.evaluations``.
    """
    hist = train if show_last is None else train.iloc[-show_last:]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist.values,
            mode="lines",
            name="Train",
            line=dict(color=COLORS["history"], width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test.values,
            mode="lines+markers",
            name="Actual (test)",
            line=dict(color=COLORS["test"], width=2.5),
            marker=dict(size=5),
        )
    )
    for name, pred in predictions.items():
        if pred.empty or "yhat" not in pred:
            continue  # model failed during evaluation — nothing to draw
        fig.add_trace(
            go.Scatter(
                x=pred.index,
                y=pred["yhat"],
                mode="lines",
                name=name,
                line=dict(width=1.8, dash="dot"),
            )
        )
    fig.add_vline(x=train.index[-1], line_dash="dash", line_color="grey")
    return _base_layout(fig, "Backtest: how each model did on unseen data")


# ----------------------------------------------------------------------
# 4. Leaderboard bar chart
# ----------------------------------------------------------------------
def plot_leaderboard(leaderboard: pd.DataFrame, metric: str = "MAE") -> go.Figure:
    """Horizontal bars of one error metric per model (lower is better)."""
    ok = leaderboard[leaderboard["Status"] == "ok"].sort_values(metric, ascending=False)
    fig = go.Figure(
        go.Bar(
            x=ok[metric],
            y=ok["Model"],
            orientation="h",
            marker_color=COLORS["forecast"],
            text=ok[metric].round(2),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Model comparison by {metric} (lower is better)",
        xaxis_title=metric,
        yaxis_title="",
        template="plotly_white",
        margin=dict(l=40, r=60, t=60, b=40),
    )
    return fig

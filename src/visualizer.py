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

Every public function follows the same shape:

    def plot_xxx(<pandas objects>, <cosmetic options>) -> go.Figure

so they are trivial to test and to reuse.

The forecast chart is built from small, composable helpers (``_add_history``,
``_add_confidence_band``, ``_add_forecast``).  Each adds one visual layer to
a figure — mix and match them to build your own charts.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# A small palette reused across charts.  History is blue, anything the
# models produce (forecast, band) is orange — a convention users pick up in
# one glance.
COLORS = {
    "history": "#1f77b4",  # blue
    "forecast": "#ff7f0e",  # orange
    "band": "rgba(255, 127, 14, 0.20)",  # translucent orange
    "actual": "#2b2b2b",  # near-black, used for the test truth in backtests
}

#: Columns a forecast DataFrame must have for us to draw an interval.
INTERVAL_COLUMNS = ("yhat_lower", "yhat_upper")


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
# Building blocks for the forecast chart
# ----------------------------------------------------------------------
def _add_history(fig: go.Figure, history: pd.Series, name: str = "History") -> None:
    """Blue solid line with the observed values."""
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.to_numpy(),
            mode="lines",
            name=name,
            line=dict(color=COLORS["history"], width=1.8),
            hovertemplate="%{y:,.1f}<extra>" + name + "</extra>",
        )
    )


def has_confidence_interval(forecast: pd.DataFrame) -> bool:
    """``True`` when the strategy that produced ``forecast`` supplied bounds.

    Every strategy returns the same three columns, but a model without a
    notion of uncertainty leaves the bounds as ``NaN`` — in that case we
    simply do not draw a band.
    """
    return all(col in forecast.columns for col in INTERVAL_COLUMNS) and bool(
        forecast[list(INTERVAL_COLUMNS)].notna().all(axis=None)
    )


def _add_confidence_band(fig: go.Figure, forecast: pd.DataFrame, name: str) -> None:
    """Translucent orange area between ``yhat_lower`` and ``yhat_upper``.

    Plotly trick: draw the upper edge first (invisible line), then the lower
    edge with ``fill="tonexty"`` so the area *between* the two is shaded.
    """
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hovertemplate="upper %{y:,.1f}<extra></extra>",
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
            name=name,
            hovertemplate="lower %{y:,.1f}<extra></extra>",
        )
    )


def _add_forecast(
    fig: go.Figure,
    forecast: pd.DataFrame,
    name: str,
    anchor: pd.Series | None = None,
) -> None:
    """Orange line with the predicted values.

    ``anchor`` is the last observed point; prepending it makes the orange
    line start exactly where the blue one ends instead of leaving a gap.
    """
    yhat = forecast["yhat"]
    if anchor is not None and len(anchor):
        yhat = pd.concat([anchor.iloc[[-1]].rename("yhat"), yhat])
    fig.add_trace(
        go.Scatter(
            x=yhat.index,
            y=yhat.to_numpy(),
            mode="lines+markers",
            name=name,
            line=dict(color=COLORS["forecast"], width=2.5),
            marker=dict(size=5),
            hovertemplate="%{y:,.1f}<extra>" + name + "</extra>",
        )
    )


# ----------------------------------------------------------------------
# 1. Raw history
# ----------------------------------------------------------------------
def plot_history(series: pd.Series, title: str = "Historical demand") -> go.Figure:
    """Simple line chart of the cleaned series — the first thing users see."""
    fig = go.Figure()
    _add_history(fig, series)
    return _base_layout(fig, title)


# ----------------------------------------------------------------------
# 2. History + forecast (+ confidence interval when available)
# ----------------------------------------------------------------------
def plot_forecast(
    history: pd.Series,
    forecast: pd.DataFrame,
    model_name: str | None = None,
    show_last: int | None = None,
    show_interval: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Historical data in **blue**, forecast horizon in **orange**.

    Parameters
    ----------
    history:
        The observed series (DatetimeIndex).
    forecast:
        Standard model output: DatetimeIndex + ``yhat`` (+ optional
        ``yhat_lower`` / ``yhat_upper``).
    model_name:
        Used in the legend and the default title.
    show_last:
        Only draw the last *n* historical points so the forecast is not a
        tiny sliver at the far right of a long chart. ``None`` shows all.
    show_interval:
        Draw the shaded confidence band — only happens if the model actually
        produced bounds (see ``has_confidence_interval``).
    """
    hist = history if show_last is None else history.iloc[-show_last:]
    label = f"Forecast ({model_name})" if model_name else "Forecast"
    fig = go.Figure()

    _add_history(fig, hist)
    if show_interval and has_confidence_interval(forecast):
        _add_confidence_band(fig, forecast, name="Confidence interval")
    _add_forecast(fig, forecast, name=label, anchor=hist)

    # Vertical dashed line marking "today" (end of known data).
    fig.add_vline(x=hist.index[-1], line_dash="dash", line_color="grey")
    return _base_layout(fig, title or f"Demand forecast — {model_name or 'agent'}")


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
    see with their own eyes which dotted line hugs the black "actual" line.

    Parameters
    ----------
    predictions:
        ``{model_name: DataFrame with a 'yhat' column}`` — typically built
        from ``AgentReport.evaluations``.
    """
    hist = train if show_last is None else train.iloc[-show_last:]
    fig = go.Figure()
    _add_history(fig, hist, name="Train")
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test.to_numpy(),
            mode="lines+markers",
            name="Actual (test)",
            line=dict(color=COLORS["actual"], width=2.5),
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
def plot_leaderboard(leaderboard: pd.DataFrame, metric: str = "MAPE (%)") -> go.Figure:
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

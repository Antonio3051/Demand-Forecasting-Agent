"""
app.py — Streamlit entry point of the AI Demand Forecasting Agent
=================================================================

Run it with:

    streamlit run app.py

What this file does (and does NOT do)
-------------------------------------
``app.py`` is deliberately **thin**.  It only:

1. draws widgets (sidebar controls, buttons, tabs),
2. calls the three building blocks — ``DataHandler`` → ``ForecastingAgent``
   → ``visualizer`` — and
3. shows their results.

There is no forecasting logic here.  If you find yourself writing
``pandas`` transformations or model code in this file, it probably belongs
in ``src/``.  Keeping the UI "dumb" is what lets the same core be reused
from a notebook, a CLI or an API later on.

How Streamlit works in 30 seconds
---------------------------------
Streamlit re-runs this whole script **top to bottom** every time the user
touches a widget.  Anything expensive (loading a file, training models) must
therefore be cached (``@st.cache_data``) or stored in ``st.session_state`` so
it survives re-runs.  You will see both techniques below.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from src import visualizer as viz
from src.agent import DEFAULT_CANDIDATES, METRICS, TEST_FRACTION, ForecastingAgent
from src.data_handler import SUPPORTED_FREQUENCIES, DataHandler, DatasetSummary
from src.models import MODEL_REGISTRY, UNAVAILABLE_MODELS, available_models

#: Sentinel shown in the column pickers meaning "let DataHandler guess".
AUTO = "(auto-detect)"

# ----------------------------------------------------------------------
# Page configuration — must be the first Streamlit call in the script.
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Demand Forecasting Agent",
    page_icon="📈",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached helpers.  ``st.cache_data`` memoises the return value keyed on the
# arguments, so re-running the script does not reload/recompute.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_uploaded(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read the uploaded file once; bytes are hashable so caching works."""
    return DataHandler.load_file(BytesIO(file_bytes), filename=filename)


@st.cache_data(show_spinner=False)
def _sample_data(periods: int, freq: str) -> pd.DataFrame:
    return DataHandler.generate_sample_data(periods=periods, freq=freq)


# ----------------------------------------------------------------------
# Sidebar: every user-controlled parameter lives here.  It is drawn in two
# halves because the forecast defaults (season length) depend on the
# frequency we only know *after* the data has been loaded.
# ----------------------------------------------------------------------
def render_data_controls() -> dict:
    """Sidebar section 1 — where the data comes from."""
    st.sidebar.title("⚙️ Settings")
    st.sidebar.header("1. Data")
    source = st.sidebar.radio(
        "Source",
        ["Sample dataset", "Upload CSV / Excel"],
        help="Start with the synthetic sample to explore, then bring your own data.",
    )
    uploaded = None
    if source == "Upload CSV / Excel":
        uploaded = st.sidebar.file_uploader(
            "File with a date column and a demand column",
            type=["csv", "xlsx", "xls"],
        )

    freq_label = st.sidebar.selectbox(
        "Frequency",
        list(SUPPORTED_FREQUENCIES),
        index=0,
        help="Auto-detect looks at the spacing between your dates. Override if needed.",
    )
    return dict(source=source, uploaded=uploaded, freq=SUPPORTED_FREQUENCIES[freq_label])


def render_forecast_controls(summary: DatasetSummary) -> dict:
    """Sidebar sections 2 + 3 — how the agent should work."""
    st.sidebar.header("2. Forecast")
    horizon = st.sidebar.slider("Horizon (periods ahead)", 1, 120, 30)
    season_length = st.sidebar.number_input(
        "Season length",
        min_value=1,
        max_value=365,
        value=summary.season_length,
        # Keying on the frequency re-applies the default when it changes.
        key=f"season_{summary.frequency}",
        help="7 = weekly pattern in daily data, 12 = yearly pattern in monthly data.",
    )
    test_share = st.sidebar.slider(
        "Hold-out share for evaluation (%)",
        5,
        40,
        int(TEST_FRACTION * 100),
        help="The most recent slice of history the models never see during evaluation.",
    )
    metric = st.sidebar.selectbox(
        "Decision metric",
        list(METRICS),
        format_func=str.upper,
        help="The agent picks the model with the lowest value of this metric.",
    )

    # --- 3. Strategy selection ------------------------------------------
    st.sidebar.header("3. Models (Strategies)")
    candidates = st.sidebar.multiselect(
        "Candidates the agent may choose from",
        options=available_models(),
        default=[m for m in DEFAULT_CANDIDATES if m in available_models()],
    )
    mode = st.sidebar.radio(
        "Selection mode",
        ["Let the agent decide", "Force a specific model"],
    )
    forced = None
    if mode == "Force a specific model" and candidates:
        forced = st.sidebar.selectbox("Model to use", candidates)

    if UNAVAILABLE_MODELS:
        with st.sidebar.expander("⚠️ Models not available", expanded=False):
            for cls, reason in UNAVAILABLE_MODELS.items():
                st.caption(f"**{cls}** — {reason}")

    return dict(
        horizon=int(horizon),
        season_length=int(season_length),
        test_size=test_share / 100,
        metric=metric,
        candidates=candidates,
        forced=forced,
    )


# ----------------------------------------------------------------------
# Step 1 — PERCEIVE: obtain a raw DataFrame and turn it into a Series.
# ----------------------------------------------------------------------
def load_data(cfg: dict) -> tuple[pd.Series, DatasetSummary] | None:
    """Return the prepared series + summary, or ``None`` if we are still waiting."""
    date_col: str | None = None
    value_col: str | None = None
    if cfg["source"] == "Sample dataset":
        raw = _sample_data(periods=730, freq="D")
    else:
        if cfg["uploaded"] is None:
            st.info("👈 Upload a file in the sidebar to get started.")
            return None
        raw = _load_uploaded(cfg["uploaded"].getvalue(), cfg["uploaded"].name)
        # DataHandler guesses the columns; the pickers let the user override.
        options = [AUTO, *map(str, raw.columns)]
        c1, c2 = st.columns(2)
        picked_date = c1.selectbox("Date column", options, key="date_col")
        picked_value = c2.selectbox("Demand column", options, key="value_col")
        date_col = None if picked_date == AUTO else picked_date
        value_col = None if picked_value == AUTO else picked_value

    with st.expander("🔍 Raw data preview", expanded=False):
        st.dataframe(raw.head(20), width="stretch")

    try:
        series, summary = DataHandler.prepare_time_series(raw, date_col, value_col, cfg["freq"])
    except ValueError as exc:
        st.error(f"Could not prepare the data: {exc}")
        return None

    # Key facts about the series in a row of metric cards.
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Periods", summary.n_periods)
    m2.metric("Frequency", summary.frequency_label.capitalize())
    m3.metric("From", summary.start.date().isoformat())
    m4.metric("To", summary.end.date().isoformat())
    m5.metric("Mean demand", f"{summary.mean:,.1f}")

    notes = [
        f"date column **{summary.date_col}**, demand column **{summary.value_col}**",
        f"frequency {'inferred as' if summary.frequency_inferred else 'set to'} "
        f"**{summary.frequency_label}**",
    ]
    if summary.duplicates_merged:
        notes.append(f"{summary.duplicates_merged} rows on repeated dates were summed")
    if summary.missing_filled:
        notes.append(f"{summary.missing_filled} missing periods were interpolated")
    st.caption("ℹ️ " + "; ".join(notes) + ".")

    st.plotly_chart(viz.plot_history(series), width="stretch")
    return series, summary


# ----------------------------------------------------------------------
# Fingerprints: detect when a stored report no longer matches the inputs.
# ----------------------------------------------------------------------
def _data_key(series: pd.Series) -> int:
    """Cheap, deterministic fingerprint of the prepared series."""
    return int(pd.util.hash_pandas_object(series, index=True).sum())


def _settings_key(cfg: dict) -> tuple:
    """The sidebar values that influence the agent's result."""
    return (
        cfg["horizon"],
        cfg["metric"],
        cfg["season_length"],
        cfg["test_size"],
        tuple(cfg["candidates"]),
        cfg["forced"],
    )


# ----------------------------------------------------------------------
# Step 2 — REASON / DECIDE / ACT: run the agent.
# ----------------------------------------------------------------------
def run_agent(series: pd.Series, cfg: dict) -> None:
    """Button handler: execute the agent and stash the report in session_state."""
    # A report belongs to the dataset it was computed on. If the user loaded
    # or re-mapped data, the old report is meaningless: forget it.
    if st.session_state.get("report_data_key") != _data_key(series):
        st.session_state.pop("report", None)

    if not cfg["candidates"]:
        st.warning("Select at least one candidate model in the sidebar.")
        return

    if st.button("🚀 Run forecasting agent", type="primary"):
        agent = ForecastingAgent(
            candidates=cfg["candidates"],
            metric=cfg["metric"],
            season_length=cfg["season_length"],
            test_size=cfg["test_size"],
        )
        with st.spinner("The agent is evaluating strategies…"):
            try:
                report = agent.run(
                    series,
                    horizon=cfg["horizon"],
                    forced_model=cfg["forced"],
                )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return
        # Persist so the report survives the next widget interaction, along
        # with fingerprints of the inputs it was computed from.
        st.session_state["report"] = report
        st.session_state["report_data_key"] = _data_key(series)
        st.session_state["report_settings_key"] = _settings_key(cfg)


# ----------------------------------------------------------------------
# Step 3 — EXPLAIN: show what the agent did.
# ----------------------------------------------------------------------
def render_report(series: pd.Series, cfg: dict) -> None:
    report = st.session_state.get("report")
    if report is None:
        return

    # Same data but different sidebar settings: the report is still valid
    # for what it was computed with, so keep it but flag it as outdated.
    if st.session_state.get("report_settings_key") != _settings_key(cfg):
        st.info(
            "Settings changed since this report was generated — "
            "press **Run forecasting agent** to refresh it."
        )

    st.success(f"🏆 Selected strategy: **{report.chosen_model}**")
    st.markdown(report.rationale)

    tab_fc, tab_bt, tab_lb, tab_learn = st.tabs(
        ["📈 Forecast", "🧪 Backtest", "🏁 Leaderboard", "🎓 Learn"]
    )

    with tab_fc:
        st.plotly_chart(
            viz.plot_forecast(series, report.forecast, report.chosen_model),
            width="stretch",
        )
        st.dataframe(report.forecast.round(2), width="stretch")
        st.download_button(
            "⬇️ Download forecast as CSV",
            data=report.forecast.round(2).to_csv().encode(),
            file_name="forecast.csv",
            mime="text/csv",
        )

    with tab_bt:
        preds = {e.model_name: e.test_prediction for e in report.evaluations}
        st.plotly_chart(
            viz.plot_backtest(report.train, report.test, preds),
            width="stretch",
        )

    with tab_lb:
        lb = report.leaderboard
        st.dataframe(lb, width="stretch", hide_index=True)
        st.plotly_chart(
            viz.plot_leaderboard(lb, metric=cfg["metric"].upper().replace("MAPE", "MAPE (%)")),
            width="stretch",
        )
        failed = [e for e in report.evaluations if e.error]
        for e in failed:
            st.warning(f"**{e.model_name}** failed: {e.error}")

    with tab_learn:
        st.markdown(
            """
            ### How this app is built — the Strategy pattern

            * `src/models/base.py` defines **`BaseForecaster`**, an abstract class
              with two methods: `fit(series)` and `predict(horizon)`.
            * Every algorithm (Naive, ARIMA, Prophet, LightGBM) is a **Strategy**:
              a subclass that implements those two methods its own way.
            * `src/agent.py` is the **Context**: it holds out the most recent
              15 % of history, fits every strategy on the rest, scores each one
              with MAPE on the unseen slice, picks the best and explains why —
              without knowing anything about how the models work internally.
              `model, why = ForecastingAgent().select_model(series)` is the
              whole API.
            * Adding a new algorithm = one new file in `src/models/` + one line in
              the registry. The UI and the agent need **zero** changes.

            ### The models available right now
            """
        )
        for name, cls in MODEL_REGISTRY.items():
            st.markdown(f"* **{name}** — {cls.description}")


# ----------------------------------------------------------------------
# Main — the script's "story line".
# ----------------------------------------------------------------------
def main() -> None:
    st.title("📈 AI Demand Forecasting Agent")
    st.caption(
        "An educational agent that perceives your sales history, compares several "
        "forecasting strategies, decides which one to trust, and explains its choice."
    )

    cfg = render_data_controls()
    loaded = load_data(cfg)  # PERCEIVE
    if loaded is None:
        return
    series, summary = loaded
    cfg.update(render_forecast_controls(summary))
    st.divider()
    run_agent(series, cfg)  # REASON → DECIDE → ACT
    render_report(series, cfg)  # EXPLAIN


if __name__ == "__main__":
    main()

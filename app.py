"""
app.py — Streamlit entry point of the AI Demand Forecasting Agent
=================================================================

Run it with:

    streamlit run app.py

What this file does (and does NOT do)
-------------------------------------
``app.py`` is deliberately **thin**.  It only:

1. draws widgets (upload box, dropdowns, the *Run Forecast* button),
2. calls the three building blocks — ``DataHandler`` → ``ForecastingAgent``
   → ``visualizer`` — and
3. shows their results (explanation, chart, download).

There is no forecasting logic here.  If you find yourself writing
``pandas`` transformations or model code in this file, it probably belongs
in ``src/``.  Keeping the UI "dumb" is what lets the same core be reused
from a notebook, a CLI or an API later on.

How Streamlit works in 30 seconds
---------------------------------
Streamlit re-runs this whole script **top to bottom** every time the user
touches a widget.  Anything expensive (reading a file, training models) must
therefore be cached (``@st.cache_data``) or stored in ``st.session_state`` so
it survives re-runs.  You will see both techniques below.

Error handling philosophy
-------------------------
A demo that crashes with a red traceback is a bad demo.  Every step that can
fail on user input (reading the file, preparing the series, fitting models,
drawing charts) is wrapped in ``try/except`` and turns the problem into a
friendly ``st.error`` — with the technical details tucked away in an
expander for the curious.  ``main()`` has a final safety net so that even an
unexpected bug never takes the whole page down.
"""

from __future__ import annotations

import traceback
from io import BytesIO

import pandas as pd
import streamlit as st

from src import visualizer as viz
from src.agent import DEFAULT_CANDIDATES, METRICS, TEST_FRACTION, AgentReport, ForecastingAgent
from src.data_handler import SUPPORTED_FREQUENCIES, DataHandler, DatasetSummary
from src.models import MODEL_REGISTRY, UNAVAILABLE_MODELS, available_models

#: Sentinel shown in the date-column picker meaning "let DataHandler guess".
AUTO = "(auto-detect)"

# ----------------------------------------------------------------------
# Page configuration — must be the first Streamlit call in the script.
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Demand Forecasting Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------
def show_error(message: str, exc: Exception | None = None) -> None:
    """Friendly error banner; the traceback is one click away, never in your face."""
    st.error(message)
    if exc is not None:
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(exc)), language="text")


def numeric_columns(df: pd.DataFrame, exclude: str | None = None) -> list[str]:
    """Columns that can serve as the forecast target (numeric, not the date)."""
    cols = [c for c in df.select_dtypes("number").columns if c != exclude]
    if not cols:
        # Numbers stored as text ("1,234") still count if they mostly convert.
        for c in df.columns:
            if c == exclude:
                continue
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().mean() >= 0.9:
                cols.append(c)
    return [str(c) for c in cols]


# ----------------------------------------------------------------------
# Cached helpers.  ``st.cache_data`` memoises the return value keyed on the
# arguments, so re-running the script does not reload/recompute.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read the uploaded file once; bytes are hashable so caching works."""
    return DataHandler.load_file(BytesIO(file_bytes), filename=filename)


@st.cache_data(show_spinner=False)
def _sample_data(periods: int, freq: str) -> pd.DataFrame:
    return DataHandler.generate_sample_data(periods=periods, freq=freq)


# ----------------------------------------------------------------------
# Fingerprints: detect when a stored report no longer matches the inputs.
# ----------------------------------------------------------------------
def _data_key(series: pd.Series) -> int:
    """Cheap, deterministic fingerprint of the prepared series."""
    return int(pd.util.hash_pandas_object(series, index=True).sum())


def _settings_key(cfg: dict) -> tuple:
    """The settings that influence the agent's result."""
    return (
        cfg["horizon"],
        cfg["metric"],
        cfg["season_length"],
        cfg["test_size"],
        tuple(cfg["candidates"]),
        cfg["forced"],
    )


# ----------------------------------------------------------------------
# Step 1 — PERCEIVE: get a raw DataFrame from the user.
# ----------------------------------------------------------------------
def render_upload() -> pd.DataFrame | None:
    """Sidebar upload box (or the built-in sample). ``None`` while waiting."""
    st.sidebar.title("📈 Forecasting Agent")
    st.sidebar.header("1. Data")
    uploaded = st.sidebar.file_uploader(
        "Upload a CSV with a date column and at least one numeric column",
        type=["csv", "xlsx", "xls"],
        help="One row per period. Column names are free — you pick the target below.",
    )
    use_sample = st.sidebar.toggle(
        "Use the sample dataset instead",
        value=uploaded is None,
        help="Two years of synthetic daily sales with trend, weekly pattern and promotions.",
    )

    if uploaded is not None and not use_sample:
        try:
            return _read_upload(uploaded.getvalue(), uploaded.name)
        except Exception as exc:  # unreadable / corrupt / unsupported file
            show_error(f"Could not read **{uploaded.name}**. Is it a valid CSV or Excel file?", exc)
            return None
    if use_sample:
        return _sample_data(periods=730, freq="D")

    st.info("👈 Upload a CSV in the sidebar — or switch on the sample dataset to explore.")
    return None


def render_column_controls(raw: pd.DataFrame) -> dict | None:
    """Target-variable dropdown + horizon input, populated from the uploaded file."""
    st.sidebar.header("2. What to forecast")

    # Date column: DataHandler guesses; the dropdown lets the user override.
    try:
        guessed_date = DataHandler.detect_datetime_column(raw)
    except ValueError:
        guessed_date = None
    date_options = [AUTO, *map(str, raw.columns)]
    date_pick = st.sidebar.selectbox(
        "Date column",
        date_options,
        index=0,
        help=f"Auto-detected: **{guessed_date}**" if guessed_date else "No date column found.",
    )
    date_col = None if date_pick == AUTO else date_pick

    # Target column: numeric columns only, pre-selected with the handler's guess.
    targets = numeric_columns(raw, exclude=date_col or guessed_date)
    if not targets:
        show_error("No numeric column found to forecast. Add a column with the demand values.")
        return None
    try:
        guessed_target = DataHandler.detect_value_column(
            raw, exclude=date_col or guessed_date or ""
        )
    except ValueError:
        guessed_target = targets[0]
    value_col = st.sidebar.selectbox(
        "Target variable (numeric)",
        targets,
        index=targets.index(guessed_target) if guessed_target in targets else 0,
        help="The quantity you want to predict — sales, units, orders…",
    )

    horizon = st.sidebar.number_input(
        "Forecast horizon (periods ahead)",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        help="Periods are in the data's own frequency: days, weeks or months.",
    )
    return dict(date_col=date_col, value_col=value_col, horizon=int(horizon))


def render_advanced_controls(summary: DatasetSummary) -> dict:
    """Optional knobs, collapsed by default so the main flow stays simple."""
    with st.sidebar.expander("⚙️ Advanced settings", expanded=False):
        freq_label = st.selectbox(
            "Frequency",
            list(SUPPORTED_FREQUENCIES),
            index=0,
            help="Auto-detect looks at the spacing between your dates. Override if needed.",
        )
        season_length = st.number_input(
            "Season length",
            min_value=1,
            max_value=365,
            value=summary.season_length,
            # Keying on the frequency re-applies the default when it changes.
            key=f"season_{summary.frequency}",
            help="7 = weekly pattern in daily data, 12 = yearly pattern in monthly data.",
        )
        test_share = st.slider(
            "Hold-out share for evaluation (%)",
            5,
            40,
            int(TEST_FRACTION * 100),
            help="The most recent slice of history the models never see during evaluation.",
        )
        metric = st.selectbox(
            "Decision metric",
            list(METRICS),
            format_func=str.upper,
            help="The agent picks the model with the lowest value of this metric.",
        )
        candidates = st.multiselect(
            "Candidate models",
            options=available_models(),
            default=[m for m in DEFAULT_CANDIDATES if m in available_models()],
        )
        forced = st.selectbox(
            "Force a specific model",
            ["Let the agent decide", *candidates],
        )
        if UNAVAILABLE_MODELS:
            st.caption("Not available in this environment:")
            for cls, reason in UNAVAILABLE_MODELS.items():
                st.caption(f"• **{cls}** — {reason}")

    return dict(
        freq=SUPPORTED_FREQUENCIES[freq_label],
        season_length=int(season_length),
        test_size=test_share / 100,
        metric=metric,
        candidates=candidates,
        forced=None if forced == "Let the agent decide" else forced,
    )


def prepare_series(raw: pd.DataFrame, cfg: dict) -> tuple[pd.Series, DatasetSummary] | None:
    """Raw DataFrame → clean regular series, or ``None`` with an error shown."""
    try:
        return DataHandler.prepare_time_series(raw, cfg["date_col"], cfg["value_col"], cfg["freq"])
    except ValueError as exc:  # expected: bad columns, too few rows, no dates ...
        show_error(f"Could not prepare the data: {exc}")
    except Exception as exc:  # unexpected: still no traceback on screen
        show_error("Something went wrong while preparing the data.", exc)
    return None


def render_data_overview(raw: pd.DataFrame, series: pd.Series, summary: DatasetSummary) -> None:
    """Metric cards + history chart so the user can sanity-check the input."""
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Periods", summary.n_periods)
    m2.metric("Frequency", summary.frequency_label.capitalize())
    m3.metric("From", summary.start.date().isoformat())
    m4.metric("To", summary.end.date().isoformat())
    m5.metric("Mean demand", f"{summary.mean:,.1f}")

    notes = [
        f"date column **{summary.date_col}**, target **{summary.value_col}**",
        f"frequency {'inferred as' if summary.frequency_inferred else 'set to'} "
        f"**{summary.frequency_label}**",
    ]
    if summary.duplicates_merged:
        notes.append(f"{summary.duplicates_merged} rows on repeated dates were summed")
    if summary.missing_filled:
        notes.append(f"{summary.missing_filled} missing periods were interpolated")
    st.caption("ℹ️ " + "; ".join(notes) + ".")

    with st.expander("🔍 Raw data preview", expanded=False):
        st.dataframe(raw.head(50), width="stretch")
    st.plotly_chart(viz.plot_history(series), width="stretch")


# ----------------------------------------------------------------------
# Step 2 — REASON / DECIDE / ACT: run the agent.
# ----------------------------------------------------------------------
def run_forecast(series: pd.Series, cfg: dict) -> None:
    """Button handler: execute the agent pipeline and stash the report."""
    # A report belongs to the dataset it was computed on. If the user loaded
    # or re-mapped data, the old report is meaningless: forget it.
    if st.session_state.get("report_data_key") != _data_key(series):
        st.session_state.pop("report", None)

    disabled = not cfg["candidates"]
    if disabled:
        st.warning("Select at least one candidate model in **Advanced settings**.")

    if not st.button("🚀 Run Forecast", type="primary", disabled=disabled):
        return

    try:
        agent = ForecastingAgent(
            candidates=cfg["candidates"],
            metric=cfg["metric"],
            season_length=cfg["season_length"],
            test_size=cfg["test_size"],
        )
        with st.spinner("The agent is evaluating strategies on a hold-out window…"):
            report = agent.run(series, horizon=cfg["horizon"], forced_model=cfg["forced"])
    except (ValueError, RuntimeError) as exc:  # expected: too little data, all models failed
        show_error(f"The agent could not produce a forecast: {exc}")
        return
    except Exception as exc:  # unexpected library failure — keep the UI alive
        show_error("An unexpected error occurred while running the agent.", exc)
        return

    # Persist so the report survives the next widget interaction, along
    # with fingerprints of the inputs it was computed from.
    st.session_state["report"] = report
    st.session_state["report_data_key"] = _data_key(series)
    st.session_state["report_settings_key"] = _settings_key(cfg)


# ----------------------------------------------------------------------
# Step 3 — EXPLAIN: show what the agent did.
# ----------------------------------------------------------------------
def render_explanation(report: AgentReport) -> None:
    """The agent's reasoning, front and centre."""
    st.subheader(f"🏆 Selected strategy: {report.chosen_model}")
    with st.container(border=True):
        st.markdown("#### 🤖 Why the agent chose this model")
        st.markdown(report.rationale)


def render_forecast_chart(series: pd.Series, report: AgentReport) -> None:
    show_last = len(series)
    if len(series) > 30:
        show_last = st.slider(
            "Historical periods to display",
            min_value=30,
            max_value=len(series),
            value=min(len(series), max(90, report.horizon * 4)),
            help="Zoom the history so the forecast stays readable.",
        )
    try:
        fig = viz.plot_forecast(series, report.forecast, report.chosen_model, show_last=show_last)
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        show_error("The forecast chart could not be drawn.", exc)
    if not viz.has_confidence_interval(report.forecast):
        st.caption("This model does not provide confidence intervals, so no band is shown.")


def render_download(report: AgentReport, summary: DatasetSummary) -> None:
    """Forecast table + CSV download."""
    table = report.forecast.round(2).rename(
        columns={"yhat": "forecast", "yhat_lower": "lower_95", "yhat_upper": "upper_95"}
    )
    st.dataframe(table, width="stretch")
    st.download_button(
        "⬇️ Download forecast CSV",
        data=table.to_csv().encode("utf-8"),
        file_name=f"forecast_{summary.value_col}_{report.chosen_model.replace(' ', '_')}.csv",
        mime="text/csv",
        type="secondary",
    )


def render_details(report: AgentReport, cfg: dict) -> None:
    """Backtest chart + leaderboard: the evidence behind the decision."""
    with st.expander("🧪 How the candidates compared on unseen data", expanded=False):
        try:
            preds = {e.model_name: e.test_prediction for e in report.evaluations}
            st.plotly_chart(viz.plot_backtest(report.train, report.test, preds), width="stretch")
            lb = report.leaderboard
            st.dataframe(lb, width="stretch", hide_index=True)
            metric_col = cfg["metric"].upper().replace("MAPE", "MAPE (%)")
            st.plotly_chart(viz.plot_leaderboard(lb, metric=metric_col), width="stretch")
        except Exception as exc:
            show_error("The comparison details could not be rendered.", exc)
        for e in report.evaluations:
            if e.error:
                st.warning(f"**{e.model_name}** failed during evaluation: {e.error}")


def render_learn() -> None:
    with st.expander("🎓 How this app is built — the Strategy pattern", expanded=False):
        st.markdown(
            """
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
            * `src/visualizer.py` turns pandas objects into Plotly figures;
              `app.py` (this file) only wires the pieces together.
            * Adding a new algorithm = one new file in `src/models/` + one line in
              the registry. The UI and the agent need **zero** changes.

            **Models available right now**
            """
        )
        for name, cls in MODEL_REGISTRY.items():
            st.markdown(f"* **{name}** — {cls.description}")


def render_report(series: pd.Series, summary: DatasetSummary, cfg: dict) -> None:
    report: AgentReport | None = st.session_state.get("report")
    if report is None:
        return

    # Same data but different settings: the report is still valid for what
    # it was computed with, so keep it but flag it as outdated.
    if st.session_state.get("report_settings_key") != _settings_key(cfg):
        st.info(
            "Settings changed since this forecast was generated — press **Run Forecast** again."
        )

    render_explanation(report)
    st.subheader("📈 Forecast")
    render_forecast_chart(series, report)
    render_download(report, summary)
    render_details(report, cfg)


# ----------------------------------------------------------------------
# Main — the script's "story line".
# ----------------------------------------------------------------------
def main() -> None:
    st.title("📈 AI Demand Forecasting Agent")
    st.caption(
        "Upload your sales history, pick what to forecast, and let the agent compare "
        "several forecasting strategies, choose the most accurate one and explain why."
    )

    raw = render_upload()  # PERCEIVE (1/2): raw table
    if raw is None:
        render_learn()
        return

    cfg = render_column_controls(raw)
    if cfg is None:
        return

    # Frequency has to be known before the season-length default can be set,
    # so prepare once with auto-frequency, then honour any override.
    prepared = prepare_series(raw, {**cfg, "freq": None})
    if prepared is None:
        return
    _, summary = prepared
    cfg.update(render_advanced_controls(summary))
    if cfg["freq"] is not None:
        prepared = prepare_series(raw, cfg)
        if prepared is None:
            return
    series, summary = prepared  # PERCEIVE (2/2): clean series

    render_data_overview(raw, series, summary)
    st.divider()
    run_forecast(series, cfg)  # REASON → DECIDE → ACT
    render_report(series, summary, cfg)  # EXPLAIN
    render_learn()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # last line of defence: never show a raw traceback
        show_error("The app hit an unexpected error. Please adjust the inputs and try again.", exc)

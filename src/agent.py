"""
src/agent.py
============

The **ForecastingAgent** — the brain of the project and the *Context* of the
Strategy pattern.

What makes this an "agent" and not just a script?
-------------------------------------------------
A (simple) AI agent follows a loop that students will meet again and again:

    PERCEIVE  ->  REASON  ->  DECIDE  ->  ACT  ->  EXPLAIN

* **Perceive** – receive a clean time series (from ``DataHandler``) and
                 measure its properties: length, frequency, trend,
                 seasonality, volatility, intermittency.
* **Reason**   – run an *empirical evaluation*: hold out the last 15 % of the
                 history, fit every candidate strategy on the first 85 %,
                 and score each one with MAPE on the part it never saw.
* **Decide**   – pick the strategy with the lowest error (or honour the
                 user's explicit choice).
* **Act**      – refit the winner on *all* the data so it is ready to
                 forecast.
* **Explain**  – produce a plain-English report connecting the data
                 properties and the error metrics to the decision
                 (transparency is what separates an agent from a black box).

The agent has **no idea** how any model works internally.  It only relies on
the ``BaseForecaster`` contract (``fit`` / ``predict``).  That is the Strategy
pattern in action: you can plug in a new algorithm and the agent will
evaluate it alongside the others without any code change here.

Public API in two lines
-----------------------
>>> model, explanation = ForecastingAgent().select_model(series)
>>> forecast = model.predict(horizon=30)

Evaluation metrics (all "lower is better")
-----------------------------------------
* **MAPE** – Mean Absolute Percentage Error: the decision metric.  It is
             scale-free (a % number), so "5 %" means the same for a product
             selling 10 units/day and one selling 10 000.
* **MAE**  – Mean Absolute Error, in units — easier to relate to stock.
* **RMSE** – like MAE but punishes big misses more.  Reported for context.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_handler import DataHandler, DataProperties, season_length_for
from src.models import available_models, create_model
from src.models.base import BaseForecaster

#: Strategies the agent compares by default (the three "serious" models).
#: ``Seasonal Naive`` is available too — add it to see the baseline.
DEFAULT_CANDIDATES: tuple[str, ...] = ("Auto-ARIMA", "Prophet", "LightGBM")

#: Share of the history held out for the empirical evaluation.
TEST_FRACTION: float = 0.15

#: Supported decision metrics.
METRICS: tuple[str, ...] = ("mape", "mae", "rmse")


# ----------------------------------------------------------------------
# Small data classes: typed containers are friendlier than loose dicts.
# ----------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """Scores of one strategy on the hold-out (test) period."""

    model_name: str
    mae: float
    rmse: float
    mape: float
    fit_seconds: float
    test_prediction: pd.DataFrame  # kept so the UI can plot it
    error: str | None = None  # set when the model crashed

    def score(self, metric: str) -> float:
        """Value of the requested metric; NaN is treated as worst possible."""
        value = {"mae": self.mae, "rmse": self.rmse, "mape": self.mape}[metric]
        return float("inf") if np.isnan(value) else value


@dataclass
class ModelSelection:
    """Outcome of ``ForecastingAgent.select_model``.

    ``model`` is already **fitted on the full history**, so the caller can
    immediately ask ``model.predict(horizon)``.  ``explanation`` is the
    agent's reasoning in Markdown.

    Supports tuple unpacking for a compact API::

        model, explanation = agent.select_model(series)
    """

    model: BaseForecaster
    model_name: str
    explanation: str
    evaluations: list[EvaluationResult]  # best first
    properties: DataProperties
    train: pd.Series
    test: pd.Series
    metric: str

    def __iter__(self) -> Iterator:
        yield self.model
        yield self.explanation

    @property
    def leaderboard(self) -> pd.DataFrame:
        """Evaluation table (nice for ``st.dataframe``)."""
        rows = [
            {
                "Model": e.model_name,
                "MAPE (%)": round(e.mape, 2),
                "MAE": round(e.mae, 2),
                "RMSE": round(e.rmse, 2),
                "Fit time (s)": round(e.fit_seconds, 2),
                "Status": "error" if e.error else "ok",
            }
            for e in self.evaluations
        ]
        return pd.DataFrame(rows)


@dataclass
class AgentReport:
    """``ModelSelection`` + the actual forecast, ready for the UI."""

    selection: ModelSelection
    forecast: pd.DataFrame  # future predictions (yhat, yhat_lower, yhat_upper)
    horizon: int
    extra: dict = field(default_factory=dict)

    # Convenience pass-throughs so the UI reads naturally.
    @property
    def chosen_model(self) -> str:
        return self.selection.model_name

    @property
    def rationale(self) -> str:
        return self.selection.explanation

    @property
    def evaluations(self) -> list[EvaluationResult]:
        return self.selection.evaluations

    @property
    def leaderboard(self) -> pd.DataFrame:
        return self.selection.leaderboard

    @property
    def train(self) -> pd.Series:
        return self.selection.train

    @property
    def test(self) -> pd.Series:
        return self.selection.test


# ----------------------------------------------------------------------
# Metric helper
# ----------------------------------------------------------------------
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE in %, ignoring periods with zero demand (division by zero)."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ----------------------------------------------------------------------
# The agent itself
# ----------------------------------------------------------------------
class ForecastingAgent:
    """Context of the Strategy pattern + an empirical decision loop.

    Parameters
    ----------
    candidates:
        Names of the strategies to compare (see ``src.models.available_models``).
        Defaults to Auto-ARIMA, Prophet and LightGBM.
    metric:
        Decision metric: ``"mape"`` (default), ``"mae"`` or ``"rmse"``.
    season_length:
        Seasonal period passed to the strategies.  ``None`` = derive it from
        the series frequency (7 for daily, 52 for weekly, 12 for monthly).
    test_size:
        Hold-out used for the evaluation: a fraction (``0.15`` = last 15 %)
        or an absolute number of periods.
    """

    def __init__(
        self,
        candidates: list[str] | tuple[str, ...] | None = None,
        metric: str = "mape",
        season_length: int | None = None,
        test_size: int | float = TEST_FRACTION,
    ) -> None:
        installed = available_models()
        wanted = list(candidates) if candidates else list(DEFAULT_CANDIDATES)
        self.candidates = [name for name in wanted if name in installed]
        if not self.candidates:
            raise ValueError(f"None of {wanted} is installed. Available: {installed}")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}.")
        self.metric = metric
        self.season_length = season_length
        self.test_size = test_size

    # ------------------------------------------------------------------
    # Strategy construction (the only place that knows about kwargs)
    # ------------------------------------------------------------------
    def _build(self, name: str, season_length: int, n_obs: int) -> BaseForecaster:
        """Instantiate a strategy with sensible, data-aware defaults.

        Every model accepts different constructor arguments, so this helper
        translates two facts about the data (season length, number of
        observations) into what each strategy understands.  A model that
        needs no arguments simply falls through to ``create_model(name)``.
        """
        m = season_length
        if name in {"Seasonal Naive", "Auto-ARIMA"}:
            return create_model(name, season_length=m)
        if name == "LightGBM":
            # Seasonal lags only when there is enough history to learn them.
            lags = {1, 2, 3}
            if n_obs > m + 10:
                lags.add(m)
            if n_obs > 2 * m + 10:
                lags.add(2 * m)
            return create_model(name, lags=tuple(sorted(lags)))
        return create_model(name)

    # ------------------------------------------------------------------
    # REASON: evaluate every candidate on the hold-out window
    # ------------------------------------------------------------------
    def evaluate(
        self, train: pd.Series, test: pd.Series, season_length: int | None = None
    ) -> list[EvaluationResult]:
        """Fit each candidate on ``train`` and score it on ``test``.

        A model that raises an exception is *not* allowed to crash the agent:
        it gets an infinite error so it ranks last, and the message is stored
        for the UI.  Robustness is a core property of agents.
        """
        m = season_length or self.season_length or season_length_for(train.index.freqstr or "")
        results: list[EvaluationResult] = []
        horizon = len(test)
        y_true = test.to_numpy(dtype=float)

        for name in self.candidates:
            start = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # statsmodels is noisy
                    model = self._build(name, m, len(train)).fit(train)
                    pred = model.predict(horizon)
                y_pred = pred["yhat"].to_numpy(dtype=float)
                results.append(
                    EvaluationResult(
                        model_name=name,
                        mae=float(mean_absolute_error(y_true, y_pred)),
                        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                        mape=_mape(y_true, y_pred),
                        fit_seconds=time.perf_counter() - start,
                        test_prediction=pred,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - we *want* to catch everything
                results.append(
                    EvaluationResult(
                        model_name=name,
                        mae=float("inf"),
                        rmse=float("inf"),
                        mape=float("inf"),
                        fit_seconds=time.perf_counter() - start,
                        test_prediction=pd.DataFrame(),
                        error=f"{exc.__class__.__name__}: {exc}",
                    )
                )

        # Rank: best (lowest metric) first.
        return sorted(results, key=lambda r: r.score(self.metric))

    # ------------------------------------------------------------------
    # PERCEIVE + REASON + DECIDE + ACT + EXPLAIN  ->  ModelSelection
    # ------------------------------------------------------------------
    def select_model(
        self,
        series: pd.Series,
        forced_model: str | None = None,
    ) -> ModelSelection:
        """Empirically choose the best strategy and return it **fitted**.

        Parameters
        ----------
        series:
            Clean, regular time series (output of ``DataHandler.prepare_time_series``).
        forced_model:
            Skip the decision and use this strategy.  Candidates are still
            evaluated so the explanation can compare them.

        Returns
        -------
        ModelSelection
            ``.model`` (fitted on the full history) and ``.explanation``.
        """
        # --- PERCEIVE ---------------------------------------------------
        props = DataHandler.describe_series(series, self.season_length)
        m = props.season_length
        train, test = DataHandler.train_test_split(series, self.test_size)

        # --- REASON -----------------------------------------------------
        evaluations = self.evaluate(train, test, season_length=m)
        healthy = [e for e in evaluations if e.error is None]
        if not healthy:
            details = "; ".join(f"{e.model_name}: {e.error}" for e in evaluations)
            raise RuntimeError(f"Every candidate model failed. Details: {details}")

        # --- DECIDE -----------------------------------------------------
        if forced_model is not None:
            if forced_model not in self.candidates:
                raise ValueError(f"{forced_model!r} is not among the candidates.")
            preference = [forced_model]
        else:
            # Ordered wish-list: best first, runner-ups as fallbacks.
            preference = [e.model_name for e in healthy]

        # --- ACT --------------------------------------------------------
        # Refit on the *entire* history: more data -> better final forecast.
        # A model that behaved on the training window can still fail on the
        # full series (e.g. seasonal ARIMA on very short data), so we walk
        # down the preference list instead of giving up.
        chosen, model, failures = self._fit_first_working(preference, series, m)

        # --- EXPLAIN ----------------------------------------------------
        explanation = self._compose_explanation(
            props=props,
            ranked=evaluations,
            chosen=chosen,
            forced=forced_model,
            fallback_failures=failures,
            test=test,
        )

        return ModelSelection(
            model=model,
            model_name=chosen,
            explanation=explanation,
            evaluations=evaluations,
            properties=props,
            train=train,
            test=test,
            metric=self.metric,
        )

    def run(
        self,
        series: pd.Series,
        horizon: int,
        forced_model: str | None = None,
    ) -> AgentReport:
        """Convenience wrapper: ``select_model`` + ``predict``."""
        selection = self.select_model(series, forced_model=forced_model)
        forecast = selection.model.predict(horizon)
        return AgentReport(selection=selection, forecast=forecast, horizon=horizon)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fit_first_working(
        self, preference: list[str], series: pd.Series, season_length: int
    ) -> tuple[str, BaseForecaster, list[tuple[str, str]]]:
        """Fit models in order of preference; return the first that succeeds.

        Returns ``(model_name, fitted_model, failures)`` where ``failures``
        lists ``(model_name, error_message)`` for every model skipped.
        """
        failures: list[tuple[str, str]] = []
        for name in preference:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = self._build(name, season_length, len(series)).fit(series)
                return name, model, failures
            except Exception as exc:  # noqa: BLE001
                failures.append((name, f"{exc.__class__.__name__}: {exc}"))
        details = "; ".join(f"{n}: {e}" for n, e in failures)
        raise RuntimeError(f"No model could be fitted on the full history. Details: {details}")

    # ------------------------------------------------------------------
    # EXPLAIN: turn numbers into an argument a human can follow
    # ------------------------------------------------------------------
    def _compose_explanation(
        self,
        props: DataProperties,
        ranked: list[EvaluationResult],
        chosen: str,
        forced: str | None,
        fallback_failures: list[tuple[str, str]],
        test: pd.Series,
    ) -> str:
        """Build the Markdown rationale: data -> experiment -> decision -> why."""
        parts = [
            self._describe_data(props),
            self._describe_experiment(ranked, test),
            self._describe_decision(props, ranked, chosen, forced, fallback_failures),
        ]
        return "\n\n".join(parts)

    # -- 1. data properties -------------------------------------------------
    @staticmethod
    def _describe_data(p: DataProperties) -> str:
        seasons = p.n_obs / p.season_length if p.season_length > 1 else 0
        if p.seasonal_strength >= 0.5:
            level = "**strong**"
        elif p.seasonal_strength >= 0.2:
            level = "**moderate**"
        else:
            level = "**weak or no**"
        season_txt = (
            f"{level} seasonality (autocorrelation {p.seasonal_strength:.2f} "
            f"at lag {p.season_length})"
        )
        if not p.enough_for_seasonality:
            season_txt += (
                f" — only {seasons:.1f} full seasons available, so seasonal patterns "
                "are hard to learn"
            )

        if abs(p.trend_pct) < 5:
            trend_txt = "essentially **flat** (no meaningful trend)"
        else:
            direction = "upward" if p.trend_pct > 0 else "downward"
            trend_txt = f"a clear **{direction} trend** ({p.trend_pct:+.0f} % across the sample)"

        if p.cv < 0.15:
            vol_txt = "**low** volatility"
        elif p.cv < 0.4:
            vol_txt = "**moderate** volatility"
        else:
            vol_txt = "**high** volatility"
        vol_txt += f" (coefficient of variation {p.cv:.2f})"

        text = (
            f"**1. What the data looks like.** {p.n_obs} {p.frequency_label} observations "
            f"(mean demand {p.mean:,.1f}), showing {trend_txt}, {season_txt}, and {vol_txt}."
        )
        if p.zero_share >= 0.2:
            text += (
                f" {p.zero_share:.0%} of the periods have zero demand: this is *intermittent* "
                "demand, which every candidate here struggles with (a Croston-type model "
                "would be a good addition)."
            )
        return text

    # -- 2. the experiment -----------------------------------------------
    def _describe_experiment(self, ranked: list[EvaluationResult], test: pd.Series) -> str:
        scores = []
        for e in ranked:
            if e.error:
                scores.append(f"{e.model_name}: *failed* ({e.error.split(':')[0]})")
            else:
                scores.append(f"{e.model_name}: MAPE {e.mape:.2f} %, MAE {e.mae:,.1f}")
        share = (
            f", ≈{self.test_size:.0%} of the history"
            if isinstance(self.test_size, float) and self.test_size < 1
            else ""
        )
        return (
            f"**2. Empirical evaluation.** The last {len(test)} periods "
            f"({test.index[0].date()} → {test.index[-1].date()}{share}) were held out. "
            f"Each candidate was fitted on the earlier data only and scored on that unseen "
            f"window using **{self.metric.upper()}** as the decision metric:\n\n"
            + "\n".join(f"- {s}" for s in scores)
        )

    # -- 3. the decision and the *why* ------------------------------------
    def _describe_decision(
        self,
        p: DataProperties,
        ranked: list[EvaluationResult],
        chosen: str,
        forced: str | None,
        fallback_failures: list[tuple[str, str]],
    ) -> str:
        healthy = [e for e in ranked if e.error is None]
        by_name = {e.model_name: e for e in ranked}
        winner_eval = by_name.get(chosen)

        lines: list[str] = []
        if forced is not None:
            lines.append(
                f"**3. Decision.** You asked explicitly for **{chosen}**, so the agent used it "
                f"regardless of the ranking."
            )
        else:
            best = healthy[0]
            lines.append(
                f"**3. Decision.** The agent selected **{chosen}** because it achieved the lowest "
                f"{self.metric.upper()} on unseen data ({best.score(self.metric):,.2f})."
            )
            if len(healthy) > 1:
                runner = healthy[1]
                rv = runner.score(self.metric)
                if rv > 0 and np.isfinite(rv):
                    gain = (rv - best.score(self.metric)) / rv * 100
                    lines.append(
                        f"It beat the runner-up, {runner.model_name} ({rv:,.2f}), by "
                        f"**{gain:.1f} %**"
                        + (
                            " — a narrow margin, so both are reasonable choices."
                            if gain < 5
                            else "."
                        )
                    )

        if fallback_failures:
            skipped = "; ".join(f"{n} ({err})" for n, err in fallback_failures)
            lines.append(
                f"⚠️ Fallback applied: {skipped} could not be refitted on the full history, "
                f"so **{chosen}** was used instead."
            )

        # Why the winner fits these data properties.
        lines.append(f"**Why {chosen} suits this data:** {self._strength_of(chosen, p)}")

        # Why each loser fell short — data-aware, not just "higher error".
        losers = [e for e in ranked if e.model_name != chosen]
        if losers:
            reasons = []
            for e in losers:
                if e.error:
                    reasons.append(f"- **{e.model_name}** could not be fitted: `{e.error}`.")
                else:
                    diff = (
                        f" ({self.metric.upper()} {e.score(self.metric):,.2f} vs "
                        f"{winner_eval.score(self.metric):,.2f})"
                        if winner_eval and winner_eval.error is None
                        else ""
                    )
                    reasons.append(
                        f"- **{e.model_name}**{diff}: {self._weakness_of(e.model_name, p)}"
                    )
            lines.append("**Why not the others:**\n" + "\n".join(reasons))

        if p.cv >= 0.4:
            lines.append(
                "Because volatility is high, expect wide uncertainty bands — plan safety stock "
                "on `yhat_upper`, not only on the point forecast."
            )
        return "\n\n".join(lines)

    # -- model knowledge base ------------------------------------------------
    # These sentences encode *domain knowledge* about each algorithm. They are
    # combined with the measured data properties to produce an argument that
    # is specific to the dataset at hand.
    @staticmethod
    def _strength_of(name: str, p: DataProperties) -> str:
        if name == "Auto-ARIMA":
            txt = (
                "ARIMA models the value as a function of its own recent past and past errors, "
                "which captures the short-term autocorrelation typical of demand data"
            )
            if p.enough_for_seasonality and p.seasonal_strength >= 0.2:
                txt += (
                    f"; with seasonality enabled (m={p.season_length}) it also repeats "
                    "the seasonal cycle"
                )
            if abs(p.trend_pct) >= 5:
                txt += ", and differencing removes the trend before modelling"
            return txt + "."
        if name == "Prophet":
            txt = "Prophet decomposes the series into trend + seasonality"
            if abs(p.trend_pct) >= 5:
                txt += f"; its piecewise trend fits the {p.trend_pct:+.0f} % drift well"
            if p.seasonal_strength >= 0.2:
                txt += " and its Fourier seasonality exploits the repeating pattern"
            return txt + ". It is also robust to outliers such as promotions."
        if name == "LightGBM":
            txt = (
                "Gradient-boosted trees learn *non-linear* interactions between lagged demand "
                "and calendar features (weekday, month)"
            )
            if p.n_obs >= 300:
                txt += f", and {p.n_obs} observations give the trees enough rows to generalise"
            return txt + "."
        if name == "Seasonal Naive":
            return (
                "Repeating the last season is hard to beat when the pattern is stable and the "
                "history is short — complexity did not pay off here."
            )
        return "It produced the lowest error on the hold-out window."

    @staticmethod
    def _weakness_of(name: str, p: DataProperties) -> str:
        if name == "Auto-ARIMA":
            if not p.enough_for_seasonality and p.season_length > 1:
                return (
                    "with fewer than three full seasons the seasonal ARIMA terms cannot be "
                    "estimated reliably, so it falls back to a non-seasonal model."
                )
            if p.cv >= 0.4:
                return "ARIMA assumes roughly constant variance; the high volatility violates that."
            return "its linear structure misses part of the dynamics the winner captured."
        if name == "Prophet":
            if p.seasonal_strength < 0.2:
                return (
                    "Prophet's strength is seasonality, and this series shows little of it, so its "
                    "extra flexibility mostly fits noise."
                )
            if p.n_obs < 2 * p.season_length:
                return "Prophet's changepoint detection needs a longer history to be stable."
            return "its smooth trend + seasonality decomposition under-reacts to recent shifts."
        if name == "LightGBM":
            if p.n_obs < 200:
                return (
                    f"tree models are data-hungry; {p.n_obs} rows are too few to learn lag "
                    "patterns without overfitting."
                )
            return (
                "it forecasts recursively (each prediction feeds the next), so errors compound "
                "over the hold-out window."
            )
        if name == "Seasonal Naive":
            return "it ignores trend and noise, serving only as the baseline to beat."
        return "it scored a higher error on the hold-out window."

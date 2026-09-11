"""
src/agent.py
============

The **ForecastingAgent** — the brain of the project and the *Context* of the
Strategy pattern.

What makes this an "agent" and not just a script?
-------------------------------------------------
A (simple) AI agent follows a loop that students will meet again and again:

    PERCEIVE  ->  REASON / DECIDE  ->  ACT  ->  EXPLAIN

* **Perceive** – receive a clean time series (from ``DataHandler``).
* **Reason**   – try several candidate strategies on a hold-out period,
                 measure how wrong each one is, and rank them.
* **Decide**   – pick the strategy with the lowest error (or honour the
                 user's explicit choice).
* **Act**      – refit the winner on *all* the data and produce the forecast.
* **Explain**  – return a human-readable rationale for the decision so the
                 UI can show *why* a model was chosen (transparency!).

The agent has **no idea** how any model works internally.  It only relies on
the ``BaseForecaster`` contract (``fit`` / ``predict``).  That is the Strategy
pattern in action: you can plug in a new algorithm and the agent will
evaluate it alongside the others without any code change here.

Evaluation metrics (all "lower is better")
-----------------------------------------
* **MAE**  – Mean Absolute Error: average size of the mistakes, in units.
* **RMSE** – Root Mean Squared Error: like MAE but punishes big misses more.
* **MAPE** – Mean Absolute Percentage Error: MAE expressed in %, handy to
             compare products with very different volumes.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_handler import DataHandler
from src.models import available_models, create_model
from src.models.base import BaseForecaster


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
class AgentReport:
    """Everything the agent produced, ready for the UI or a notebook."""

    chosen_model: str
    forecast: pd.DataFrame  # future predictions (yhat, bands)
    evaluations: list[EvaluationResult]  # one per candidate, best first
    rationale: str  # plain-English explanation
    train: pd.Series
    test: pd.Series
    horizon: int
    extra: dict = field(default_factory=dict)

    @property
    def leaderboard(self) -> pd.DataFrame:
        """Evaluation table sorted by the primary metric (nice for st.dataframe)."""
        rows = [
            {
                "Model": e.model_name,
                "MAE": round(e.mae, 2),
                "RMSE": round(e.rmse, 2),
                "MAPE (%)": round(e.mape, 2),
                "Fit time (s)": round(e.fit_seconds, 2),
                "Status": "error" if e.error else "ok",
            }
            for e in self.evaluations
        ]
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Metric helper
# ----------------------------------------------------------------------
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE that ignores periods with zero demand (division by zero)."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ----------------------------------------------------------------------
# The agent itself
# ----------------------------------------------------------------------
class ForecastingAgent:
    """Context of the Strategy pattern + a tiny decision loop.

    Parameters
    ----------
    candidates:
        Names of the strategies to consider (see ``src.models.available_models``).
        ``None`` means "every model that is installed".
    metric:
        Which column of the leaderboard decides the winner: ``"mae"``,
        ``"rmse"`` or ``"mape"``.
    season_length:
        Passed to the strategies that accept it (Naive, ARIMA, LightGBM lags).
    """

    def __init__(
        self,
        candidates: list[str] | None = None,
        metric: str = "mae",
        season_length: int = 7,
    ) -> None:
        self.candidates = candidates or available_models()
        if metric not in {"mae", "rmse", "mape"}:
            raise ValueError("metric must be one of 'mae', 'rmse', 'mape'.")
        self.metric = metric
        self.season_length = season_length

    # ------------------------------------------------------------------
    # Strategy construction (the only place that knows about kwargs)
    # ------------------------------------------------------------------
    def _build(self, name: str) -> BaseForecaster:
        """Instantiate a strategy with sensible, frequency-aware defaults.

        Every model accepts different constructor arguments, so this helper
        translates *one* agent-level setting (``season_length``) into what
        each strategy understands.  If you add a model that needs no
        arguments, it simply falls through to ``create_model(name)``.
        """
        m = self.season_length
        if name in {"Seasonal Naive", "Auto-ARIMA"}:
            return create_model(name, season_length=m)
        if name == "LightGBM":
            lags = tuple(sorted({1, 2, 3, m, 2 * m}))
            return create_model(name, lags=lags)
        return create_model(name)

    # ------------------------------------------------------------------
    # REASON: evaluate every candidate on the hold-out window
    # ------------------------------------------------------------------
    def evaluate(self, train: pd.Series, test: pd.Series) -> list[EvaluationResult]:
        """Fit each candidate on ``train`` and score it on ``test``.

        A model that raises an exception is *not* allowed to crash the agent:
        it gets an infinite error so it ranks last, and the message is stored
        for the UI.  Robustness is a core property of agents.
        """
        results: list[EvaluationResult] = []
        horizon = len(test)
        y_true = test.to_numpy(dtype=float)

        for name in self.candidates:
            start = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # statsmodels is noisy
                    model = self._build(name).fit(train)
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
    # DECIDE + ACT + EXPLAIN: the full loop
    # ------------------------------------------------------------------
    def run(
        self,
        series: pd.Series,
        horizon: int,
        test_size: int | None = None,
        forced_model: str | None = None,
    ) -> AgentReport:
        """Execute the whole perceive → reason → decide → act → explain loop.

        Parameters
        ----------
        series:
            Clean, regular time series (output of ``DataHandler.prepare_series``).
        horizon:
            How many future periods to forecast.
        test_size:
            Length of the hold-out window used for model selection.
            Defaults to ``horizon`` (evaluate on a window as long as the one
            we are about to predict — a common rule of thumb), capped so at
            least 70 % of the data remains for training.
        forced_model:
            If given, skip the decision step and use this strategy.  We still
            evaluate all candidates so the user can compare.
        """
        # --- PERCEIVE ---------------------------------------------------
        if test_size is None:
            test_size = max(1, min(horizon, int(len(series) * 0.3)))
        train, test = DataHandler.train_test_split(series, test_size)

        # --- REASON -----------------------------------------------------
        evaluations = self.evaluate(train, test)
        healthy = [e for e in evaluations if e.error is None]
        if not healthy:
            details = "; ".join(f"{e.model_name}: {e.error}" for e in evaluations)
            raise RuntimeError(f"Every candidate model failed. Details: {details}")

        # --- DECIDE -----------------------------------------------------
        if forced_model is not None:
            if forced_model not in self.candidates:
                raise ValueError(f"{forced_model!r} is not among the candidates.")
            preference = [forced_model]
            reason = f"You asked explicitly for **{forced_model}**."
        else:
            # Ordered wish-list: best first, runner-ups as fallbacks.
            preference = [e.model_name for e in healthy]
            reason = self._explain_choice(healthy)

        # --- ACT --------------------------------------------------------
        # Refit on the *entire* history: more data -> better final forecast.
        # A model that behaved on the training window can still fail on the
        # full series (e.g. seasonal ARIMA on very short data), so we walk
        # down the preference list instead of giving up.
        chosen, forecast, failures = self._refit_first_working(preference, series, horizon)
        if failures:
            skipped = "; ".join(f"{name} ({err})" for name, err in failures)
            reason += (
                f"\n\n⚠️ Fallback applied: {skipped} could not be refitted on the full "
                f"history, so **{chosen}** was used instead."
            )

        # --- EXPLAIN ----------------------------------------------------
        rationale = (
            f"{reason}\n\n"
            f"Selection was based on a hold-out window of **{test_size}** periods "
            f"({test.index[0].date()} → {test.index[-1].date()}) using **{self.metric.upper()}** "
            f"as the decision metric. The winner was then refitted on all "
            f"**{len(series)}** observations to forecast the next **{horizon}** periods."
        )

        return AgentReport(
            chosen_model=chosen,
            forecast=forecast,
            evaluations=evaluations,
            rationale=rationale,
            train=train,
            test=test,
            horizon=horizon,
        )

    # ------------------------------------------------------------------
    def _refit_first_working(
        self, preference: list[str], series: pd.Series, horizon: int
    ) -> tuple[str, pd.DataFrame, list[tuple[str, str]]]:
        """Fit models in order of preference; return the first that succeeds.

        Returns ``(model_name, forecast, failures)`` where ``failures`` lists
        ``(model_name, error_message)`` for every model that was skipped.
        """
        failures: list[tuple[str, str]] = []
        for name in preference:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = self._build(name).fit(series).predict(horizon)
                return name, forecast, failures
            except Exception as exc:  # noqa: BLE001
                failures.append((name, f"{exc.__class__.__name__}: {exc}"))
        details = "; ".join(f"{n}: {e}" for n, e in failures)
        raise RuntimeError(f"No model could be fitted on the full history. Details: {details}")

    def _explain_choice(self, ranked: list[EvaluationResult]) -> str:
        """Turn the ranking into a sentence a business user can read."""
        best = ranked[0]
        metric_value = best.score(self.metric)
        text = (
            f"The agent chose **{best.model_name}** because it had the lowest "
            f"{self.metric.upper()} on unseen data ({metric_value:,.2f})."
        )
        if len(ranked) > 1:
            runner = ranked[1]
            runner_value = runner.score(self.metric)
            if runner_value > 0:
                gain = (runner_value - metric_value) / runner_value * 100
                text += (
                    f" It beat the runner-up, {runner.model_name} "
                    f"({runner_value:,.2f}), by {gain:.1f} %."
                )
        baseline = next((r for r in ranked if r.model_name == "Seasonal Naive"), None)
        if baseline is not None and baseline is not best:
            text += (
                " Good news: the winner outperforms the naive baseline, "
                "so the extra complexity pays off."
            )
        elif baseline is best:
            text += (
                " Note: the naive baseline won. The series may be too short or too "
                "noisy for the advanced models to add value — try more history."
            )
        return text

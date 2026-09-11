"""
src/models/base.py
==================

The *abstract* base class that every forecasting model must inherit from.

Why does this file exist?  (The Strategy design pattern)
---------------------------------------------------------
The **Strategy pattern** lets us define a *family of algorithms* (ARIMA,
Prophet, LightGBM, ...) that all solve the same problem (forecasting demand)
and make them **interchangeable** at run-time.

    +------------------+          uses          +--------------------+
    | ForecastingAgent |----------------------->|   BaseForecaster   |  <- abstract "Strategy"
    |   (Context)      |                        |  + fit(series)     |
    +------------------+                        |  + predict(h)      |
                                                +--------------------+
                                                          ^
                                   inherits               |  inherits
                        +-----------------+---------------+------------------+
                        |                 |               |                  |
                +---------------+ +---------------+ +---------------+ +---------------+
                | NaiveForecaster| |ArimaForecaster| |ProphetForecaster| |LightGBMForecaster|
                +---------------+ +---------------+ +---------------+ +---------------+
                             (concrete Strategies live in src/models/*.py)

The *Context* (our ``ForecastingAgent`` in ``src/agent.py``) never needs to
know **which** algorithm it is talking to; it only relies on the contract
defined here: "you can be *fitted* on a series and you can *predict* the
next *h* steps".  Because of that, adding a brand new algorithm never
requires touching the agent or the UI — you only:

    1. create ``src/models/my_model.py``,
    2. subclass ``BaseForecaster`` and implement ``fit`` and ``predict``,
    3. register it in ``src/models/__init__.py``.

That is the whole point of the pattern: **open for extension, closed for
modification** (the "O" in the SOLID principles).

Contract summary
----------------
* ``fit(series)``      -> learns the patterns of a ``pandas.Series`` whose index
                          is a ``DatetimeIndex`` with a regular frequency.
* ``predict(horizon)`` -> returns a ``pandas.DataFrame`` with the columns
                          ``yhat`` (point forecast) and optionally
                          ``yhat_lower`` / ``yhat_upper`` (uncertainty band),
                          indexed by the *future* timestamps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Abstract Strategy: the interface shared by every forecasting model.

    ``ABC`` (Abstract Base Class) together with the ``@abstractmethod``
    decorator makes Python *refuse* to instantiate a subclass that forgets to
    implement ``fit`` or ``predict``.  This turns a runtime surprise into an
    immediate, easy-to-read error for students extending the project.
    """

    #: Human-friendly name shown in the UI and in evaluation tables.
    #: Concrete subclasses override this class attribute.
    name: str = "BaseForecaster"

    #: One-sentence description used for tooltips / educational text.
    description: str = "Abstract forecaster. Do not use directly."

    def __init__(self) -> None:
        # Every strategy keeps a reference to the training data so that
        # ``predict`` can build the future timestamps (index) without the
        # caller having to pass them explicitly.
        self._history: pd.Series | None = None
        # Flag toggled by ``fit``; ``predict`` checks it to fail loudly
        # instead of returning garbage when the model was never trained.
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract contract — MUST be implemented by every concrete strategy.
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, series: pd.Series) -> BaseForecaster:
        """Learn from historical demand.

        Parameters
        ----------
        series:
            Historical demand.  Must have a ``pandas.DatetimeIndex`` with a
            regular frequency (daily, weekly, monthly ...).  Missing values
            should already be handled by ``src/data_handler.py``.

        Returns
        -------
        BaseForecaster
            ``self``, so calls can be chained: ``model.fit(s).predict(30)``.

        Implementation tip
        ------------------
        Concrete classes should call ``self._remember(series)`` first; it
        stores the history and validates the input for them.
        """

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Forecast the next ``horizon`` periods after the training data.

        Parameters
        ----------
        horizon:
            Number of future time steps to predict (e.g. 30 for the next
            thirty days when the series is daily).

        Returns
        -------
        pandas.DataFrame
            Indexed by the future timestamps, with at least a ``yhat`` column
            (point forecast).  Models that estimate uncertainty also return
            ``yhat_lower`` and ``yhat_upper``.

        Implementation tip
        ------------------
        Call ``self._check_is_fitted()`` first and use
        ``self._future_index(horizon)`` to build the output index so every
        strategy returns exactly the same shape.
        """

    # ------------------------------------------------------------------
    # Shared helpers — concrete strategies inherit these for free.
    # (Template-method flavour: common plumbing lives here, the algorithm-
    # specific part lives in the subclass.)
    # ------------------------------------------------------------------
    def _remember(self, series: pd.Series) -> None:
        """Validate the training series and store it for later use."""
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"{self.name}.fit expects a pandas.Series, got {type(series).__name__}."
            )
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(f"{self.name}.fit expects a Series indexed by dates (DatetimeIndex).")
        if series.index.freq is None:
            # Try to infer the frequency ("D", "W", "MS", ...). Models such as
            # Prophet or ARIMA need it to know how far apart future steps are.
            inferred = pd.infer_freq(series.index)
            if inferred is None:
                raise ValueError(
                    "Could not infer a regular frequency from the index. "
                    "Resample the data first (see DataHandler.prepare_series)."
                )
            series = series.asfreq(inferred)
        if series.isna().any():
            raise ValueError(f"{self.name}.fit received missing values; fill them before fitting.")
        self._history = series.astype(float)
        self._is_fitted = True

    def _check_is_fitted(self) -> None:
        """Raise a clear error if ``predict`` is called before ``fit``."""
        if not self._is_fitted or self._history is None:
            raise RuntimeError(
                f"{self.name} is not fitted yet. Call .fit(series) before .predict()."
            )

    def _future_index(self, horizon: int) -> pd.DatetimeIndex:
        """Build the timestamps that come right *after* the training data.

        Example: history ends on 2024-01-31 with daily frequency and
        ``horizon=3``  ->  [2024-02-01, 2024-02-02, 2024-02-03].
        """
        self._check_is_fitted()
        if horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        assert self._history is not None  # for type checkers
        freq = self._history.index.freq
        # ``periods=horizon + 1`` includes the last known date, which we drop.
        return pd.date_range(start=self._history.index[-1], periods=horizon + 1, freq=freq)[1:]

    def _build_output(
        self,
        horizon: int,
        yhat,
        yhat_lower=None,
        yhat_upper=None,
    ) -> pd.DataFrame:
        """Package raw predictions into the standard output DataFrame.

        Having one helper guarantees every strategy returns identical columns,
        which makes the agent and visualizer code model-agnostic.
        """
        index = self._future_index(horizon)
        frame = pd.DataFrame({"yhat": list(yhat)}, index=index)
        if yhat_lower is not None and yhat_upper is not None:
            frame["yhat_lower"] = list(yhat_lower)
            frame["yhat_upper"] = list(yhat_upper)
        # Demand cannot be negative: clip is a tiny but important business rule.
        return frame.clip(lower=0)

    # ------------------------------------------------------------------
    # Dunder niceties
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self._is_fitted else "not fitted"
        return f"<{self.__class__.__name__} name={self.name!r} ({state})>"

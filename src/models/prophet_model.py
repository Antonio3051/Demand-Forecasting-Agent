"""
src/models/prophet_model.py
===========================

Concrete Strategy #3: **Prophet** (open-sourced by Meta).

What is Prophet?
----------------
Prophet decomposes a series into interpretable pieces:

    y(t) = trend(t) + seasonality(t) + holidays(t) + noise

* **trend**       – piecewise linear (or logistic) growth with automatic
                    change-point detection.
* **seasonality** – Fourier series for yearly / weekly / daily cycles.
* **holidays**    – optional list of special dates (not used here, but easy
                    to add — a nice student exercise!).

It is robust to missing data and outliers and needs very little tuning,
which made it popular for business forecasting.

Prophet-specific quirk
----------------------
Prophet insists on a DataFrame with two columns named exactly ``ds`` (dates)
and ``y`` (values).  Our Strategy hides that detail: the agent keeps passing
a plain ``pandas.Series`` and this class does the translation.  That is one
of the benefits of the pattern — each strategy *adapts* its library to the
common interface.
"""

from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet

from src.models.base import BaseForecaster

# Prophet (via cmdstanpy) is very chatty; keep the console readable.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetForecaster(BaseForecaster):
    """Meta's Prophet wrapped as a Strategy.

    Parameters
    ----------
    yearly_seasonality, weekly_seasonality, daily_seasonality:
        Passed straight to Prophet. ``"auto"`` lets Prophet decide based on
        the amount of history available.
    interval_width:
        Width of the uncertainty band (0.95 = 95 % interval).
    """

    name = "Prophet"
    description = (
        "Additive model from Meta: trend + seasonalities + holidays. Robust and easy to interpret."
    )

    def __init__(
        self,
        yearly_seasonality: str | bool = "auto",
        weekly_seasonality: str | bool = "auto",
        daily_seasonality: str | bool = False,
        interval_width: float = 0.95,
    ) -> None:
        super().__init__()
        self._kwargs = dict(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=interval_width,
        )
        self._model: Prophet | None = None

    # -- Strategy contract ------------------------------------------------
    def fit(self, series: pd.Series) -> ProphetForecaster:
        self._remember(series)
        # Adapter step: Series -> DataFrame(ds, y) that Prophet understands.
        train_df = pd.DataFrame({"ds": self._history.index, "y": self._history.to_numpy()})
        self._model = Prophet(**self._kwargs)
        self._model.fit(train_df)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        self._check_is_fitted()
        assert self._model is not None
        # Prophet wants the future dates as a DataFrame too.
        future = pd.DataFrame({"ds": self._future_index(horizon)})
        forecast = self._model.predict(future)
        return self._build_output(
            horizon,
            forecast["yhat"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
        )

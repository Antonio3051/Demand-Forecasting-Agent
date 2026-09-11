"""
src/models/prophet.py
=====================

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

Prophet-specific quirks this Strategy hides
-------------------------------------------
1. **Input format.**  Prophet insists on a DataFrame with two columns named
   exactly ``ds`` (dates) and ``y`` (values).  The agent keeps passing a
   plain ``pandas.Series``; this class does the translation.  Each strategy
   *adapts* its library to the common interface — that is the pattern.
2. **Noise.**  Prophet's Stan backend (``cmdstanpy``) logs every sampling
   step and re-creates its logger the first time it runs, so a one-off
   ``setLevel`` is not enough.  ``_quiet()`` silences loggers *and* the
   process-level stdout/stderr for the duration of ``fit`` only.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
from prophet import Prophet

from src.models.base import BaseForecaster

#: Loggers that become chatty while Prophet fits.
_NOISY_LOGGERS = ("prophet", "cmdstanpy", "stan", "pystan")


@contextmanager
def _quiet() -> Iterator[None]:
    """Temporarily silence Prophet / Stan output.

    Two layers are needed:
    * Python ``logging`` — cmdstanpy attaches its own ``StreamHandler`` and
      resets its level to DEBUG on first use, so we raise the level *inside*
      the fit and restore it afterwards.
    * File descriptors 1 and 2 — the compiled Stan binary writes straight to
      the OS-level stdout/stderr, bypassing Python entirely.  ``os.dup2`` to
      ``/dev/null`` is the only way to catch that.
    """
    previous_levels = {}
    for name in _NOISY_LOGGERS:
        logger = logging.getLogger(name)
        previous_levels[name] = logger.level
        logger.setLevel(logging.CRITICAL)

    sys.stdout.flush()
    sys.stderr.flush()
    saved_fds = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        for fd in (*saved_fds, devnull):
            os.close(fd)
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


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
        with _quiet():
            self._model.fit(train_df)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        self._check_is_fitted()
        assert self._model is not None
        # Prophet wants the future dates as a DataFrame too.
        future = pd.DataFrame({"ds": self._future_index(horizon)})
        with _quiet():
            forecast = self._model.predict(future)
        # Back to the shared contract: DataFrame indexed by future dates with
        # yhat / yhat_lower / yhat_upper columns.
        return self._build_output(
            horizon,
            forecast["yhat"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
        )

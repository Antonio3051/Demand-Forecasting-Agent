"""
src/models/naive.py
===================

The simplest possible Strategy: **Seasonal Naive**.

    "Tomorrow's demand will be the same as demand one season ago."

Why start with something so simple?
-----------------------------------
1. It is the **baseline** every serious model must beat.  If ARIMA or
   LightGBM cannot outperform "copy last week", they are not worth their
   complexity.
2. It has **zero external dependencies**, so it also proves the Strategy
   pattern works even when heavy libraries (Prophet, pmdarima) fail to
   install.
3. It is short enough to read in one minute — perfect as the first concrete
   example of how to implement ``fit`` / ``predict``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Repeat the last observed season (or the last value if no season).

    Parameters
    ----------
    season_length:
        Number of periods in one season (7 for weekly patterns in daily data,
        12 for yearly patterns in monthly data).  ``1`` turns the model into
        the classic "last value" naive forecast.
    """

    name = "Seasonal Naive"
    description = (
        "Baseline that repeats the last observed season. Every other model should beat this one."
    )

    def __init__(self, season_length: int = 7) -> None:
        super().__init__()  # never forget: initialises _history / _is_fitted
        self.season_length = max(1, int(season_length))
        self._last_season: np.ndarray | None = None

    # -- Strategy contract ------------------------------------------------
    def fit(self, series: pd.Series) -> NaiveForecaster:
        self._remember(series)  # validation + storage (from BaseForecaster)
        # If the series is shorter than one season, fall back to what we have.
        window = min(self.season_length, len(series))
        self._last_season = self._history.to_numpy()[-window:]
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        self._check_is_fitted()
        assert self._last_season is not None
        # np.resize repeats the pattern cyclically until it reaches `horizon`.
        yhat = np.resize(self._last_season, horizon)
        # Crude uncertainty band: +/- one historical standard deviation.
        std = float(self._history.std()) if len(self._history) > 1 else 0.0
        return self._build_output(horizon, yhat, yhat - std, yhat + std)

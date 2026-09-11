"""
src/models/lightgbm_model.py
============================

Concrete Strategy #4: **LightGBM** — a machine-learning approach.

How can a tree model forecast a time series?
--------------------------------------------
Trees do not understand "time" natively, so we *reframe* forecasting as a
regular supervised-learning problem (this is called **feature engineering**):

    +------------+-------+-------+-------+---------+-------+
    | date       | lag_1 | lag_2 | lag_7 | weekday | y     |   <- one row per day
    +------------+-------+-------+-------+---------+-------+
    | 2024-01-08 | 120   | 118   | 101   | 0 (Mon) | 125   |
    | 2024-01-09 | 125   | 120   | 105   | 1 (Tue) | 130   |
    +------------+-------+-------+-------+---------+-------+

* **lag_k**   – demand *k* periods ago (the model's "memory").
* **calendar**– weekday / month capture seasonality.

Prediction is done **recursively**: we predict one step, append the result
to the history, rebuild the features, predict the next step, and so on.
This is the main difference from statistical models, and also a known
weakness: errors accumulate the further we go.

Why include it?
---------------
Gradient-boosted trees win most tabular forecasting competitions (M5 on
Kaggle, for instance) and they can ingest *external* features (price,
promotions, weather) with almost no extra work.  A great next step for
students is adding such exogenous variables to ``_make_features``.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BaseForecaster


class LightGBMForecaster(BaseForecaster):
    """Gradient boosting with lag + calendar features, recursive forecasting.

    Parameters
    ----------
    lags:
        Which past periods to use as features.  Include the season length
        (7 for daily data with weekly pattern) so the model can "see" it.
    n_estimators, learning_rate:
        Standard boosting hyper-parameters; the defaults train in a second.
    """

    name = "LightGBM"
    description = (
        "Machine-learning model: gradient-boosted trees trained on lag and "
        "calendar features, forecasting step by step."
    )

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 2, 3, 7, 14),
        n_estimators: int = 300,
        learning_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.lags = tuple(sorted({int(lag) for lag in lags}))
        self._model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=15,  # small trees generalise better on short series
            min_child_samples=5,  # allow learning from limited history
            verbose=-1,  # silence LightGBM's log spam
        )
        self._residual_std: float = 0.0

    # -- Feature engineering ---------------------------------------------
    def _make_features(self, series: pd.Series) -> pd.DataFrame:
        """Turn a series into a supervised-learning table (X only)."""
        feats = pd.DataFrame(index=series.index)
        for lag in self.lags:
            feats[f"lag_{lag}"] = series.shift(lag)
        # Calendar features. They are cheap and surprisingly powerful.
        feats["dayofweek"] = series.index.dayofweek
        feats["month"] = series.index.month
        feats["dayofyear"] = series.index.dayofyear
        return feats

    # -- Strategy contract ------------------------------------------------
    def fit(self, series: pd.Series) -> LightGBMForecaster:
        self._remember(series)
        max_lag = max(self.lags)
        if len(self._history) <= max_lag + 5:
            raise ValueError(
                f"{self.name} needs more than {max_lag + 5} observations for "
                f"lags={self.lags}. Use fewer/shorter lags or more data."
            )
        X = self._make_features(self._history)
        y = self._history
        # The first `max_lag` rows contain NaNs (no past to look at): drop them.
        mask = X.notna().all(axis=1)
        self._model.fit(X[mask], y[mask])
        # In-sample residual spread -> quick-and-dirty uncertainty band.
        residuals = y[mask] - self._model.predict(X[mask])
        self._residual_std = float(np.std(residuals))
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        self._check_is_fitted()
        # Recursive strategy: extend the series one step at a time.
        extended = self._history.copy()
        future_index = self._future_index(horizon)
        preds: list[float] = []
        for ts in future_index:
            # Append a placeholder for `ts` so shift() produces its lag row.
            extended.loc[ts] = np.nan
            X_next = self._make_features(extended).iloc[[-1]]
            y_next = float(self._model.predict(X_next)[0])
            extended.loc[ts] = y_next  # feed the prediction back in
            preds.append(y_next)
        yhat = np.asarray(preds)
        # Widen the band with sqrt(step) — uncertainty grows with the horizon.
        steps = np.arange(1, horizon + 1)
        band = 1.96 * self._residual_std * np.sqrt(steps)
        return self._build_output(horizon, yhat, yhat - band, yhat + band)

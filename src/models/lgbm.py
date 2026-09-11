"""
src/models/lgbm.py
==================

Concrete Strategy #4: **LightGBM** — a machine-learning approach.

How can a tree model forecast a time series?
--------------------------------------------
Trees do not understand "time" natively, so we *reframe* forecasting as a
regular supervised-learning problem.  ``_make_features`` turns the single
column of demand into a table (this is called **feature engineering**):

    +------------+-------+-------+-------+------------+-----------+---------+-----+
    | date       | lag_1 | lag_2 | lag_7 | roll_mean_7| roll_std_7| weekday | y   |
    +------------+-------+-------+-------+------------+-----------+---------+-----+
    | 2024-01-08 | 120   | 118   | 101   | 111.4      | 8.2       | 0 (Mon) | 125 |
    | 2024-01-09 | 125   | 120   | 105   | 114.9      | 7.9       | 1 (Tue) | 130 |
    +------------+-------+-------+-------+------------+-----------+---------+-----+

* **lag_k**        – demand *k* periods ago (the model's short-term memory).
* **roll_<stat>_w**– mean / std / min / max of the previous *w* periods
                     (the model's sense of *level* and *volatility*).
* **calendar**     – weekday / month / day-of-year capture seasonality.

Every feature is built **only from the past** (note the ``shift(1)`` before
each rolling window): using today's value to predict today would be a
*leak*, and the model would look perfect in training and useless in
production.

Prediction is done **recursively**: predict one step, append the result to
the history, rebuild the features, predict the next step, and so on.  This
is the main difference from statistical models, and also a known weakness:
errors accumulate the further we go.

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
    """Gradient boosting on lag + rolling + calendar features, forecasting recursively.

    Parameters
    ----------
    lags:
        Which past periods to use as features.  Include the season length
        (7 for daily data with weekly pattern) so the model can "see" it.
    rolling_windows:
        Window sizes for the rolling statistics (mean, std, min, max).
    n_estimators, learning_rate:
        Standard boosting hyper-parameters; the defaults train in a second.
    """

    name = "LightGBM"
    description = (
        "Machine-learning model: gradient-boosted trees trained on lag, rolling-window "
        "and calendar features, forecasting step by step."
    )

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 2, 3, 7, 14),
        rolling_windows: tuple[int, ...] = (3, 7),
        n_estimators: int = 300,
        learning_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.lags = tuple(sorted({int(lag) for lag in lags}))
        self.rolling_windows = tuple(sorted({int(w) for w in rolling_windows}))
        self._model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=15,  # small trees generalise better on short series
            min_child_samples=5,  # allow learning from limited history
            verbose=-1,  # silence LightGBM's log spam
        )
        self._residual_std: float = 0.0

    # -- Feature engineering ---------------------------------------------
    @property
    def _warmup(self) -> int:
        """Rows at the start of the table that will contain NaNs."""
        return max((*self.lags, *self.rolling_windows), default=0)

    def _make_features(self, series: pd.Series) -> pd.DataFrame:
        """Build the supervised-learning table (X only) from a univariate series.

        Three families of columns are created, all strictly from past data:

        1. **Lags** — ``lag_k = y[t-k]``.
        2. **Rolling statistics** — computed over the ``w`` values *before*
           ``t`` (``shift(1)`` first, then ``rolling(w)``): mean, std, min, max.
        3. **Calendar** — day-of-week, month, day-of-year.
        """
        feats = pd.DataFrame(index=series.index)

        for lag in self.lags:
            feats[f"lag_{lag}"] = series.shift(lag)

        past = series.shift(1)  # exclude the current period -> no leakage
        for w in self.rolling_windows:
            window = past.rolling(window=w, min_periods=w)
            feats[f"roll_mean_{w}"] = window.mean()
            feats[f"roll_std_{w}"] = window.std()
            feats[f"roll_min_{w}"] = window.min()
            feats[f"roll_max_{w}"] = window.max()

        # Calendar features. They are cheap and surprisingly powerful.
        feats["dayofweek"] = series.index.dayofweek
        feats["month"] = series.index.month
        feats["dayofyear"] = series.index.dayofyear
        return feats

    # -- Strategy contract ------------------------------------------------
    def fit(self, series: pd.Series) -> LightGBMForecaster:
        self._remember(series)
        if len(self._history) <= self._warmup + 5:
            raise ValueError(
                f"{self.name} needs more than {self._warmup + 5} observations for "
                f"lags={self.lags} and rolling_windows={self.rolling_windows}. "
                "Use shorter lags/windows or more data."
            )
        X = self._make_features(self._history)
        y = self._history
        # The first `_warmup` rows contain NaNs (no past to look at): drop them.
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
            # Append a placeholder for `ts` so shift() produces its feature row.
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

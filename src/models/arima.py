"""
src/models/arima.py
===================

Concrete Strategy #2: **ARIMA** via ``pmdarima.auto_arima``.

What is ARIMA?
--------------
ARIMA = **A**uto**R**egressive **I**ntegrated **M**oving **A**verage.
It is the classic statistical approach to time series:

* **AR(p)**  – today's value depends on the previous *p* values.
* **I(d)**   – the series is differenced *d* times to remove trend.
* **MA(q)**  – today's value depends on the previous *q* forecast errors.

Choosing p, d, q by hand is tedious, so ``auto_arima`` does it for us:

* **d** is picked with a unit-root test (KPSS): difference until stationary.
* **p, q** are searched *stepwise* — start from a few sensible orders, move
  to a neighbour whenever it lowers the **AIC**, stop when nothing improves.

With ``seasonal=True`` it becomes **SARIMA** and additionally tunes the
seasonal orders (P, D, Q) for a cycle of length ``m``.

Where does it fit in the Strategy pattern?
------------------------------------------
Exactly like ``NaiveForecaster``: it subclasses ``BaseForecaster`` and
implements ``fit`` and ``predict``.  Everything else (validation, output
format, future index) is inherited.
"""

from __future__ import annotations

import pandas as pd
import pmdarima as pm

from src.models.base import BaseForecaster


class ArimaForecaster(BaseForecaster):
    """Auto-tuned (S)ARIMA model.

    Parameters
    ----------
    season_length:
        Seasonal period ``m`` (7 = weekly seasonality on daily data).
        Use ``1`` to disable seasonality and fit a plain ARIMA.
    max_order:
        Upper bound on p + q.  Smaller values train faster; the default is
        conservative so the Streamlit app stays responsive.
    """

    name = "Auto-ARIMA"
    description = (
        "Classical statistical model that explains a value from its own past "
        "values and past errors. Hyper-parameters are searched automatically."
    )

    def __init__(self, season_length: int = 7, max_order: int = 5) -> None:
        super().__init__()
        self.season_length = max(1, int(season_length))
        self.max_order = int(max_order)
        self._model: pm.arima.ARIMA | None = None

    # -- Strategy contract ------------------------------------------------
    def fit(self, series: pd.Series) -> ArimaForecaster:
        self._remember(series)
        # Seasonal differencing eats one full season of data, so require a
        # comfortable margin (3 seasons) before turning seasonality on.
        seasonal = self.season_length > 1 and len(series) >= 3 * self.season_length
        self._model = pm.auto_arima(
            self._history.to_numpy(),
            # --- what to optimise -------------------------------------
            start_p=1,
            start_q=1,
            max_p=self.max_order,
            max_q=self.max_order,
            max_order=self.max_order,  # cap on p + q keeps the search fast
            d=None,  # None = choose d with the KPSS stationarity test
            information_criterion="aic",
            # --- seasonality ------------------------------------------
            seasonal=seasonal,
            m=self.season_length if seasonal else 1,
            D=None,  # None = choose D with the OCSB seasonal test
            # --- search behaviour ---------------------------------------
            stepwise=True,  # greedy neighbour search: much faster than a full grid
            suppress_warnings=True,
            error_action="ignore",  # skip parameter combos that fail to converge
            trace=False,  # keep the console quiet
        )
        return self

    @property
    def order(self) -> tuple[int, int, int] | None:
        """The ``(p, d, q)`` that ``auto_arima`` settled on (after ``fit``)."""
        return None if self._model is None else self._model.order

    def predict(self, horizon: int) -> pd.DataFrame:
        self._check_is_fitted()
        assert self._model is not None
        # ``return_conf_int`` gives us a 95 % interval alongside the mean.
        yhat, conf = self._model.predict(n_periods=horizon, return_conf_int=True)
        return self._build_output(horizon, yhat, conf[:, 0], conf[:, 1])

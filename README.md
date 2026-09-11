# AI Demand Forecasting Agent

An educational, production-shaped Python project that shows how an **AI agent**
can perceive sales history, compare several **forecasting strategies**, decide
which one to trust and explain its choice — all behind a Streamlit UI.

The code base is built around the **Strategy design pattern** so that adding a
new algorithm never requires touching the agent or the UI.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Pick *Sample dataset* in the sidebar and press **Run forecasting agent**.

## Project structure

```
.
├── app.py                    # Streamlit entry point (thin UI layer)
├── requirements.txt
└── src/
    ├── data_handler.py       # Load / validate / clean / split data   (perception)
    ├── agent.py              # ForecastingAgent: evaluate → decide → forecast → explain
    ├── visualizer.py         # Plotly charts
    └── models/
        ├── base.py           # BaseForecaster — abstract Strategy (fit / predict)
        ├── naive.py          # Seasonal-naive baseline
        ├── arima.py          # Auto-ARIMA (pmdarima)
        ├── prophet.py        # Prophet (Stan output silenced during fit)
        ├── lgbm.py           # LightGBM with lag + rolling-window features
        └── __init__.py       # Model registry (discovery + factory)
```

## The Strategy pattern in this project

```
ForecastingAgent (Context) ──uses──> BaseForecaster (Strategy interface)
                                        ├── NaiveForecaster
                                        ├── ArimaForecaster
                                        ├── ProphetForecaster
                                        └── LightGBMForecaster
```

* `BaseForecaster` (`src/models/base.py`) declares two abstract methods,
  `fit(series)` and `predict(horizon)`, plus shared helpers for validation
  and output formatting.
* Each concrete model adapts its library to that contract.
* `ForecastingAgent` (`src/agent.py`) only speaks the contract: it holds out
  the last 15 % of history, fits Auto-ARIMA, Prophet and LightGBM on the rest,
  scores each with MAPE on the unseen slice, refits the winner on all data and
  explains the decision from the data properties and the error metrics.

### Using the core without Streamlit

```python
import pandas as pd
from src.data_handler import DataHandler
from src.agent import ForecastingAgent

raw = pd.read_csv("my_sales.csv")  # any date column + any numeric column
series, summary = DataHandler.prepare_time_series(raw)  # auto-detects columns & frequency
print(summary.frequency_label, summary.missing_filled)  # e.g. "daily", 3 gaps interpolated

model, explanation = ForecastingAgent().select_model(series)
print(explanation)  # Markdown: data -> experiment -> decision -> why
forecast = model.predict(horizon=30)  # the returned model is already fitted
```

### Adding a new algorithm

1. Create `src/models/my_model.py`:

   ```python
   from src.models.base import BaseForecaster


   class MyForecaster(BaseForecaster):
       name = "My Model"
       description = "One sentence for the UI."

       def fit(self, series):
           self._remember(series)  # validation + storage
           ...  # learn something
           return self

       def predict(self, horizon):
           self._check_is_fitted()
           yhat = ...  # array of length `horizon`
           return self._build_output(horizon, yhat)
   ```

2. Register it in `src/models/__init__.py` by adding
   `("my_model", "MyForecaster")` to `_CANDIDATES`.

That's it — it now shows up in the sidebar and in the agent's comparison.

## Learning path

Read the files in this order; each one has an extensive module docstring:

1. `src/models/base.py` — the contract.
2. `src/models/naive.py` — the simplest implementation.
3. `src/models/__init__.py` — how strategies are discovered.
4. `src/data_handler.py` — time-series preprocessing.
5. `src/agent.py` — the perceive → reason → decide → act → explain loop.
6. `src/visualizer.py` and `app.py` — presentation.

## Data format

Any CSV / Excel file with a **date** column and a numeric **demand** column.
Duplicate dates are summed, gaps are filled with 0, and the series is
resampled to the frequency chosen in the sidebar (daily, weekly or monthly).

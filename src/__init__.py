"""
src/
====

Top-level package of the AI Demand Forecasting Agent.

Module map (read them in this order if you are new to the project):

    src/models/base.py       -> the BaseForecaster contract (Strategy interface)
    src/models/naive.py      -> simplest concrete Strategy, great first read
    src/models/__init__.py   -> registry: how strategies are discovered
    src/data_handler.py      -> loading & cleaning data (perception)
    src/agent.py             -> ForecastingAgent: evaluates, decides, forecasts
    src/visualizer.py        -> Plotly charts
    app.py (project root)    -> Streamlit UI that wires everything together
"""

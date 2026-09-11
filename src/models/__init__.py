"""
src/models/__init__.py
======================

The **model registry**: a single place that knows which forecasting
Strategies exist and how to build them.

Why a registry?
---------------
The Strategy pattern says the *Context* (``ForecastingAgent``) should not
depend on concrete classes.  But *somebody* has to know that "Prophet" maps
to ``ProphetForecaster``.  Putting that knowledge here means:

* ``src/agent.py`` and ``app.py`` only deal with **names** (strings), and
* adding a new algorithm is a **one-line change** in ``MODEL_REGISTRY``.

Graceful degradation
--------------------
Heavy libraries (Prophet, pmdarima, LightGBM) sometimes fail to install on a
student's laptop.  Instead of crashing the whole app, each import is
attempted individually; models whose library is missing are simply skipped
and listed in ``UNAVAILABLE_MODELS`` with the reason, so the UI can show a
helpful message.

How to add a new algorithm
--------------------------
1. Create ``src/models/my_model.py`` with ``class MyForecaster(BaseForecaster)``.
2. Add one entry to ``_CANDIDATES`` below:  ``("my_model", "MyForecaster")``.
3. Done.  It now appears in the Streamlit sidebar and in the agent's
   automatic model comparison.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from src.models.base import BaseForecaster

# Type alias: something you can call with keyword args to obtain a strategy.
ForecasterFactory = Callable[..., BaseForecaster]

#: (module name inside src.models, class name) for every concrete strategy.
#: Order matters: it is the order shown in the UI.
_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("naive", "NaiveForecaster"),
    ("arima", "ArimaForecaster"),
    ("prophet", "ProphetForecaster"),
    ("lgbm", "LightGBMForecaster"),
)

#: display name  ->  class.  Filled at import time by ``_discover``.
MODEL_REGISTRY: dict[str, type[BaseForecaster]] = {}

#: display name  ->  reason it could not be loaded (missing library, ...).
UNAVAILABLE_MODELS: dict[str, str] = {}


def _discover() -> None:
    """Import every candidate module and register the classes that load."""
    for module_name, class_name in _CANDIDATES:
        try:
            module = importlib.import_module(f"src.models.{module_name}")
            cls: type[BaseForecaster] = getattr(module, class_name)
        except ImportError as exc:  # library not installed / broken build
            UNAVAILABLE_MODELS[class_name] = f"{exc.__class__.__name__}: {exc}"
            continue
        MODEL_REGISTRY[cls.name] = cls


_discover()


def available_models() -> list[str]:
    """Names of the strategies that can be instantiated right now."""
    return list(MODEL_REGISTRY)


def create_model(name: str, **kwargs) -> BaseForecaster:
    """Factory: build a strategy from its display name.

    Example
    -------
    >>> model = create_model("Seasonal Naive", season_length=7)
    >>> model.fit(series).predict(14)
    """
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}") from exc
    return cls(**kwargs)


__all__ = [
    "BaseForecaster",
    "MODEL_REGISTRY",
    "UNAVAILABLE_MODELS",
    "available_models",
    "create_model",
]

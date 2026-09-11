"""
src/data_handler.py
===================

Everything related to **getting data in and making it model-ready**.

In an AI-agent architecture this module is the agent's *perception* layer:
it turns raw files into a clean, regular time series the reasoning layer
(``src/agent.py``) can work with.  Keeping I/O and cleaning here means the
agent and the models never have to worry about CSV quirks.

Responsibilities
----------------
1. **Load** a CSV / Excel file (``load_file``).
2. **Sanity-check** the columns the user picked (``validate``).
3. **Prepare** a regular ``pandas.Series`` indexed by date
   (``prepare_series``): parse dates, aggregate duplicates, resample to a
   fixed frequency and fill gaps.
4. **Split** history into train / test for honest evaluation
   (``train_test_split``).
5. **Generate** a realistic synthetic dataset so the app works out of the
   box without any file (``generate_sample_data``).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

#: Frequencies we support in the UI: label -> pandas offset alias.
SUPPORTED_FREQUENCIES: dict[str, str] = {
    "Daily": "D",
    "Weekly": "W-MON",
    "Monthly": "MS",  # "Month Start"
}

#: Reasonable default season length for each frequency (used by the models).
DEFAULT_SEASON_LENGTH: dict[str, int] = {
    "D": 7,  # weekly pattern in daily data
    "W-MON": 52,  # yearly pattern in weekly data
    "MS": 12,  # yearly pattern in monthly data
}


@dataclass
class DatasetSummary:
    """Small, UI-friendly description of a prepared series."""

    start: pd.Timestamp
    end: pd.Timestamp
    n_periods: int
    frequency: str
    mean: float
    std: float
    missing_filled: int  # how many gaps we had to fill during resampling


class DataHandler:
    """Load, clean and split demand data.

    The class is intentionally *stateless* (every method is a
    ``@staticmethod``): it is a namespace of pure functions.  Pure functions
    are easier to unit-test and to reason about — a habit worth learning.
    """

    # ------------------------------------------------------------------
    # 1. Loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_file(file: str | Path | BytesIO, filename: str | None = None) -> pd.DataFrame:
        """Read a CSV or Excel file into a DataFrame.

        ``file`` can be a path **or** the in-memory object Streamlit gives us
        from ``st.file_uploader``.  ``filename`` is only needed in the latter
        case to detect the extension.
        """
        name = filename or str(file)
        suffix = Path(name).suffix.lower()
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(file)
        # Default to CSV; ``sep=None`` lets pandas sniff "," vs ";".
        return pd.read_csv(file, sep=None, engine="python")

    # ------------------------------------------------------------------
    # 2. Validation
    # ------------------------------------------------------------------
    @staticmethod
    def validate(df: pd.DataFrame, date_col: str, value_col: str) -> None:
        """Fail fast with a human-readable message if the columns are unusable."""
        if date_col not in df.columns:
            raise ValueError(f"Date column {date_col!r} not found in the file.")
        if value_col not in df.columns:
            raise ValueError(f"Value column {value_col!r} not found in the file.")
        if date_col == value_col:
            raise ValueError("Date and value columns must be different.")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            # Try a soft conversion: "1,234" or "12.5" strings are common.
            converted = pd.to_numeric(df[value_col], errors="coerce")
            if converted.isna().all():
                raise ValueError(f"Column {value_col!r} contains no numeric values.")
        if df.empty:
            raise ValueError("The file is empty.")

    # ------------------------------------------------------------------
    # 3. Preparation
    # ------------------------------------------------------------------
    @staticmethod
    def prepare_series(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        freq: str = "D",
    ) -> tuple[pd.Series, DatasetSummary]:
        """Convert two columns into a clean, regular time series.

        Steps (each one is a classic time-series preprocessing move):
        1. Parse dates and drop rows where parsing failed.
        2. Coerce values to numbers.
        3. **Aggregate** duplicates: several sales on the same day -> one total.
        4. **Resample** to the requested frequency so every period exists.
        5. **Fill gaps** with 0 — a missing day in sales data usually means
           "nothing sold", which is different from "unknown".  (Change to
           interpolation if your data means the latter.)
        """
        DataHandler.validate(df, date_col, value_col)

        work = df[[date_col, value_col]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[date_col, value_col])
        if work.empty:
            raise ValueError("No valid (date, value) pairs left after cleaning.")

        # Steps 3 + 4: group by date, then resample. ``sum`` is the right
        # aggregation for demand (units sold); use ``mean`` for prices.
        series = work.groupby(date_col)[value_col].sum().sort_index()
        resampled = series.resample(freq).sum(min_count=1)
        missing = int(resampled.isna().sum())
        resampled = resampled.fillna(0.0).astype(float)
        resampled.name = "demand"
        resampled.index.name = "date"

        summary = DatasetSummary(
            start=resampled.index[0],
            end=resampled.index[-1],
            n_periods=len(resampled),
            frequency=freq,
            mean=float(resampled.mean()),
            std=float(resampled.std()) if len(resampled) > 1 else 0.0,
            missing_filled=missing,
        )
        return resampled, summary

    # ------------------------------------------------------------------
    # 4. Splitting
    # ------------------------------------------------------------------
    @staticmethod
    def train_test_split(series: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series]:
        """Chronological split — **never** shuffle time series!

        The last ``test_size`` observations become the hold-out set that the
        agent uses to score each model before trusting its forecast.
        """
        if test_size < 1:
            raise ValueError("test_size must be at least 1.")
        if test_size >= len(series):
            raise ValueError(
                f"test_size={test_size} leaves no training data (series has {len(series)} points)."
            )
        return series.iloc[:-test_size], series.iloc[-test_size:]

    # ------------------------------------------------------------------
    # 5. Synthetic data
    # ------------------------------------------------------------------
    @staticmethod
    def generate_sample_data(
        periods: int = 365 * 2,
        freq: str = "D",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Create a realistic-looking demand dataset for demos and tests.

        The signal is built the same way Prophet *thinks* about series —
        a good way to make the model's assumptions tangible for students:

            demand = base + trend + weekly season + yearly season + noise
        """
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2022-01-01", periods=periods, freq=freq)
        t = np.arange(periods)

        base = 200.0
        trend = 0.15 * t  # slow growth
        weekly = 25 * np.sin(2 * np.pi * t / 7)  # weekend peaks
        yearly = 40 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)  # summer high
        noise = rng.normal(0, 15, periods)
        # Occasional promotions: random spikes to make life interesting.
        promos = rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 80], size=periods)

        demand = np.clip(base + trend + weekly + yearly + noise + promos, 0, None)
        return pd.DataFrame({"date": dates, "demand": demand.round(0)})

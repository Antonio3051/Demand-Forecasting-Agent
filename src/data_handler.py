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
2. **Detect** which column holds the dates and which holds the demand
   (``detect_datetime_column``, ``detect_value_column``).
3. **Infer** the temporal frequency — daily, weekly or monthly — even when
   the data has gaps (``infer_frequency``).
4. **Prepare** a regular ``pandas.Series`` indexed by date
   (``prepare_time_series``): parse, aggregate duplicates, resample and
   interpolate missing values.
5. **Describe** the statistical properties of the series so the agent can
   reason about them (``describe_series``).
6. **Split** history into train / test for honest evaluation
   (``train_test_split``).
7. **Generate** a realistic synthetic dataset so the app works out of the
   box without any file (``generate_sample_data``).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Frequency vocabulary
# ----------------------------------------------------------------------
#: Canonical pandas offset aliases we normalise everything to.
DAILY, WEEKLY, MONTHLY = "D", "W-MON", "MS"  # MS = "Month Start"

#: Human label -> pandas alias (``None`` = let the code infer it).
SUPPORTED_FREQUENCIES: dict[str, str | None] = {
    "Auto-detect": None,
    "Daily": DAILY,
    "Weekly": WEEKLY,
    "Monthly": MONTHLY,
}

#: pandas alias -> human label (reverse map, handy for messages).
FREQUENCY_LABELS: dict[str, str] = {DAILY: "daily", WEEKLY: "weekly", MONTHLY: "monthly"}

#: Reasonable default season length for each frequency (used by the models).
DEFAULT_SEASON_LENGTH: dict[str, int] = {
    DAILY: 7,  # weekly pattern in daily data
    WEEKLY: 52,  # yearly pattern in weekly data
    MONTHLY: 12,  # yearly pattern in monthly data
}


def season_length_for(freq: str) -> int:
    """Default seasonal period for any supported alias (``"W-SUN"`` -> 52)."""
    if freq.startswith("W"):
        return DEFAULT_SEASON_LENGTH[WEEKLY]
    return DEFAULT_SEASON_LENGTH.get(freq, 1)


def frequency_label_for(freq: str) -> str:
    """Human label for any supported alias (``"W-SUN"`` -> ``"weekly"``)."""
    if freq.startswith("W"):
        return FREQUENCY_LABELS[WEEKLY]
    return FREQUENCY_LABELS.get(freq, freq)


#: Column-name hints used by the auto-detection heuristics (lower-case).
_DATE_HINTS = ("date", "time", "day", "week", "month", "period", "fecha", "dia", "mes", "ds")
_VALUE_HINTS = (
    "demand",
    "sales",
    "quantity",
    "qty",
    "units",
    "volume",
    "orders",
    "ventas",
    "demanda",
    "cantidad",
    "y",
    "value",
    "target",
)


# ----------------------------------------------------------------------
# Small typed containers (friendlier than loose dicts)
# ----------------------------------------------------------------------
@dataclass
class DatasetSummary:
    """UI-friendly description of a prepared series."""

    date_col: str
    value_col: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_periods: int
    frequency: str  # pandas alias: "D", "W-MON" or "MS"
    frequency_inferred: bool  # True if we guessed it, False if the user chose it
    mean: float
    std: float
    missing_filled: int  # periods that did not exist in the raw data
    duplicates_merged: int  # raw rows collapsed into an existing period

    @property
    def frequency_label(self) -> str:
        return frequency_label_for(self.frequency)

    @property
    def season_length(self) -> int:
        return season_length_for(self.frequency)


@dataclass
class DataProperties:
    """Statistical fingerprint of a series — the agent's *perception*.

    Every number here is later turned into a sentence of the agent's
    explanation, so keep the fields interpretable.
    """

    n_obs: int
    frequency: str
    season_length: int
    mean: float
    cv: float  # coefficient of variation = std / mean  (volatility)
    trend_pct: float  # total linear trend over the sample, as % of the mean
    seasonal_strength: float  # autocorrelation at lag = season_length  (0..1)
    zero_share: float  # share of periods with zero demand  (intermittency)
    enough_for_seasonality: bool  # do we have >= 3 full seasons?

    @property
    def frequency_label(self) -> str:
        return frequency_label_for(self.frequency)


# ----------------------------------------------------------------------
# The handler
# ----------------------------------------------------------------------
class DataHandler:
    """Load, detect, clean, describe and split demand data.

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
    # 2. Column detection
    # ------------------------------------------------------------------
    @staticmethod
    def parse_dates(column: pd.Series) -> pd.Series:
        """Parse a column to datetimes, resolving the day/month ambiguity.

        "03/04/2024" is 4 March in Europe and 3 April in the US.  We parse
        both ways and keep the interpretation that (1) produces more valid
        dates and, on a tie, (2) is more chronologically ordered — real
        exports are almost always written in time order.
        """
        if pd.api.types.is_datetime64_any_dtype(column):
            return pd.to_datetime(column)
        text = column.astype(str).str.strip()
        month_first = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=False)
        day_first = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=True)

        def quality(parsed: pd.Series) -> tuple[int, float]:
            valid = parsed.dropna()
            ordered = (
                float((valid.diff().dropna() >= pd.Timedelta(0)).mean()) if len(valid) > 1 else 0.0
            )
            return len(valid), ordered

        return day_first if quality(day_first) > quality(month_first) else month_first

    @staticmethod
    def _parse_ratio(column: pd.Series) -> float:
        """Share of non-null cells that parse as a date (0..1)."""
        sample = column.dropna()
        if sample.empty:
            return 0.0
        # Pure numbers ("2021", "150") technically parse as dates; skip them.
        if pd.api.types.is_numeric_dtype(sample):
            return 0.0
        parsed = DataHandler.parse_dates(sample.head(500))
        return float(parsed.notna().mean())

    @staticmethod
    def detect_datetime_column(df: pd.DataFrame) -> str:
        """Find the column most likely to contain the timestamps.

        Heuristic, in order of trust:
        1. A column that already has a datetime dtype.
        2. A column whose *name* hints at time ("date", "fecha", "month", ...)
           **and** whose values mostly parse as dates.
        3. Any column whose values mostly (>= 90 %) parse as dates.
        """
        if df.empty:
            raise ValueError("The DataFrame is empty.")

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return str(col)

        by_name = [c for c in df.columns if any(h in str(c).lower() for h in _DATE_HINTS)]
        for col in by_name:
            if DataHandler._parse_ratio(df[col]) >= 0.8:
                return str(col)

        scores = {col: DataHandler._parse_ratio(df[col]) for col in df.columns}
        best_col, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score >= 0.9:
            return str(best_col)

        raise ValueError(
            "Could not find a datetime column. Make sure one column contains "
            f"dates (checked: {list(df.columns)})."
        )

    @staticmethod
    def detect_value_column(df: pd.DataFrame, exclude: str) -> str:
        """Find the numeric column that most likely holds the demand.

        Prefers names such as "demand", "sales", "ventas"; otherwise takes the
        first numeric (or numeric-coercible) column that is not the date.
        """
        candidates = [c for c in df.columns if str(c) != exclude]
        if not candidates:
            raise ValueError("The DataFrame needs at least one column besides the dates.")

        def is_numeric(col: str) -> bool:
            if pd.api.types.is_numeric_dtype(df[col]):
                return True
            coerced = pd.to_numeric(df[col], errors="coerce")
            return bool(coerced.notna().mean() >= 0.8)

        by_name = [c for c in candidates if any(h == str(c).lower() for h in _VALUE_HINTS)]
        by_name += [c for c in candidates if any(h in str(c).lower() for h in _VALUE_HINTS)]
        for col in by_name:
            if is_numeric(col):
                return str(col)
        for col in candidates:
            if is_numeric(col):
                return str(col)
        raise ValueError(f"No numeric column found to use as demand (checked: {candidates}).")

    # ------------------------------------------------------------------
    # 3. Frequency inference
    # ------------------------------------------------------------------
    @staticmethod
    def infer_frequency(index: pd.DatetimeIndex) -> str:
        """*Forcefully* classify the spacing of timestamps as D, W-<day> or MS.

        ``pd.infer_freq`` only works on perfectly regular indexes, which real
        data rarely is (gaps, duplicates).  So we look at the **median gap**
        between consecutive distinct timestamps — a statistic that is robust
        to a few holes — and snap it to the nearest supported frequency:

            gap <= 1.5 days   -> daily   (sub-daily data gets aggregated to days)
            gap <= 10  days   -> weekly  (anchored on the most common weekday)
            otherwise         -> monthly
        """
        idx = pd.DatetimeIndex(index).dropna().unique().sort_values()
        if len(idx) < 2:
            raise ValueError("Need at least two distinct dates to infer a frequency.")

        # Fast path: pandas recognises a perfectly regular index.
        regular = pd.infer_freq(idx) if len(idx) >= 3 else None
        if regular is not None:
            base = regular.split("-")[0].rstrip("0123456789")
            if base in {"D", "B", "C"}:
                return DAILY
            if base == "W":
                return regular if "-" in regular else WEEKLY
            if base in {"M", "MS", "ME", "BM", "BMS", "SM", "SMS"}:
                return MONTHLY

        # Robust path: median spacing.
        median_gap = pd.Series(idx[1:] - idx[:-1]).median()
        days = median_gap / pd.Timedelta(days=1)
        if days <= 1.5:
            return DAILY
        if days <= 10:
            weekday = int(pd.Series(idx.dayofweek).mode().iloc[0])
            return f"W-{['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][weekday]}"
        return MONTHLY

    # ------------------------------------------------------------------
    # 4. Preparation
    # ------------------------------------------------------------------
    @staticmethod
    def prepare_time_series(
        df: pd.DataFrame,
        date_col: str | None = None,
        value_col: str | None = None,
        freq: str | None = None,
    ) -> tuple[pd.Series, DatasetSummary]:
        """Turn a raw DataFrame into a clean, regular demand series.

        All three keyword arguments are optional: leave them as ``None`` and
        the handler will detect the datetime column, the demand column and
        the frequency on its own.  Pass them explicitly to override.

        Steps (each one is a classic time-series preprocessing move):
        1. Detect / validate columns; parse dates; coerce values to numbers.
        2. Drop rows where either failed.
        3. Set the date as the index and **aggregate duplicates**
           (several sales lines on the same day -> one daily total).
        4. **Infer the frequency** (or use the one provided).
        5. **Resample** so every period exists, even those with no rows.
        6. **Interpolate** the gaps created in step 5 (time-weighted linear),
           padding the edges with the nearest known value.
        7. Clip negatives — demand cannot be below zero.
        """
        if df.empty:
            raise ValueError("The DataFrame is empty.")

        # -- 1. columns ----------------------------------------------------
        date_col = date_col or DataHandler.detect_datetime_column(df)
        value_col = value_col or DataHandler.detect_value_column(df, exclude=date_col)
        if date_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"Columns {date_col!r} / {value_col!r} not found in the data.")
        if date_col == value_col:
            raise ValueError("Date and value columns must be different.")

        work = pd.DataFrame(
            {
                "date": DataHandler.parse_dates(df[date_col]),
                "value": pd.to_numeric(df[value_col], errors="coerce"),
            }
        )
        # -- 2. drop unusable rows ----------------------------------------
        work = work.dropna()
        if work.empty:
            raise ValueError(
                f"No valid (date, value) pairs: check that {date_col!r} holds dates "
                f"and {value_col!r} holds numbers."
            )

        # -- 3. index + duplicates ----------------------------------------
        # ``sum`` is right for demand (units sold); use ``mean`` for prices.
        by_date = work.groupby("date")["value"].sum().sort_index()
        duplicates_merged = len(work) - len(by_date)

        # -- 4. frequency --------------------------------------------------
        inferred = freq is None
        freq = freq or DataHandler.infer_frequency(by_date.index)

        # -- 5. resample ---------------------------------------------------
        resampled = by_date.resample(freq).sum(min_count=1)
        missing = int(resampled.isna().sum())

        # -- 6. interpolate ------------------------------------------------
        filled = (
            resampled.interpolate(method="time")  # inside gaps
            .bfill()  # leading gap (rare, but possible)
            .ffill()  # trailing gap
        )
        # -- 7. business rule ---------------------------------------------
        series = filled.clip(lower=0).astype(float)
        series.name = "demand"
        series.index.name = "date"
        series = series.asfreq(freq)  # attach freq metadata for the models

        summary = DatasetSummary(
            date_col=str(date_col),
            value_col=str(value_col),
            start=series.index[0],
            end=series.index[-1],
            n_periods=len(series),
            frequency=freq,
            frequency_inferred=inferred,
            mean=float(series.mean()),
            std=float(series.std()) if len(series) > 1 else 0.0,
            missing_filled=missing,
            duplicates_merged=int(duplicates_merged),
        )
        return series, summary

    # ------------------------------------------------------------------
    # 5. Description (what the agent "sees")
    # ------------------------------------------------------------------
    @staticmethod
    def describe_series(series: pd.Series, season_length: int | None = None) -> DataProperties:
        """Compute interpretable statistics the agent can reason about."""
        freq = series.index.freqstr or DataHandler.infer_frequency(series.index)
        m = season_length or season_length_for(freq)
        values = series.to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean()) if n else 0.0

        cv = float(values.std() / mean) if mean > 0 and n > 1 else 0.0

        # Linear trend: fit y = a + b*t, express total change b*n as % of mean.
        trend_pct = 0.0
        if n >= 3 and mean > 0:
            slope = np.polyfit(np.arange(n), values, 1)[0]
            trend_pct = float(slope * n / mean * 100)

        # Seasonal strength: autocorrelation at the seasonal lag.
        seasonal_strength = 0.0
        if m > 1 and n > 2 * m:
            centered = values - mean
            denom = float((centered**2).sum())
            if denom > 0:
                seasonal_strength = float((centered[m:] * centered[:-m]).sum() / denom)
                seasonal_strength = max(0.0, seasonal_strength)

        return DataProperties(
            n_obs=n,
            frequency=freq,
            season_length=m,
            mean=mean,
            cv=cv,
            trend_pct=trend_pct,
            seasonal_strength=seasonal_strength,
            zero_share=float((values == 0).mean()) if n else 0.0,
            enough_for_seasonality=bool(m > 1 and n >= 3 * m),
        )

    # ------------------------------------------------------------------
    # 6. Splitting
    # ------------------------------------------------------------------
    @staticmethod
    def train_test_split(
        series: pd.Series, test_size: int | float = 0.15
    ) -> tuple[pd.Series, pd.Series]:
        """Chronological split — **never** shuffle time series!

        ``test_size`` can be an absolute number of periods (``30``) or a
        fraction of the data (``0.15`` = last 15 %).  The last chunk becomes
        the hold-out set the agent uses to score models honestly.
        """
        n = len(series)
        if isinstance(test_size, float) and 0 < test_size < 1:
            test_size = max(1, int(round(n * test_size)))
        test_size = int(test_size)
        if test_size < 1:
            raise ValueError("test_size must be at least 1 period.")
        if test_size >= n:
            raise ValueError(
                f"test_size={test_size} leaves no training data (series has {n} points)."
            )
        return series.iloc[:-test_size], series.iloc[-test_size:]

    # ------------------------------------------------------------------
    # 7. Synthetic data
    # ------------------------------------------------------------------
    @staticmethod
    def generate_sample_data(
        periods: int = 365 * 2,
        freq: str = DAILY,
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

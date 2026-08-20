"""Deterministic simulated real-time release calendar for nowcast testing.

This module builds a small but structurally realistic real-time dataset:

* **4 quarterly series**, each published at a *different* point relative to the
  quarter end, and each subsequently revised.  ``q_preflash`` is deliberately
  published **before** the quarter ends (a flash/survey style release), which is
  the case that exercises the negative ``k`` branch.
* **4 monthly series**, each with a *different* publication lag, including lags
  longer than one month.
* A **nowcast issued on every release date** in the calendar, for every target
  series, at horizons -1 / 0 / +1.

Values are white noise around a flat level: the point of the fixture is the
*timing* structure (release calendar, revision ordering, mixed frequency), not
the numbers.  Every generator is seeded so the data is reproducible.
"""

import numpy as np
import pandas as pd

# Days relative to the END of the target period at which each release happens.
# Negative => published before the period has finished (flash / survey release).
# The first entry is the first release; later entries are revisions.
QUARTERLY_RELEASE_OFFSETS: dict[str, list[int]] = {
    "q_preflash": [-10, 20, 50],  # first estimate lands BEFORE quarter end
    "q_flash": [15, 45, 75],
    "q_standard": [30, 60, 90],
    "q_late": [55, 85],
}

MONTHLY_RELEASE_OFFSETS: dict[str, list[int]] = {
    "m_fast": [5, 35],
    "m_mid": [20, 50],
    "m_slow": [40, 70],  # lag exceeds one month
    "m_verylate": [75],  # lag exceeds two months
}

SOURCES = ("nowcast_a", "nowcast_b")

_START = "2021-01-01"
_END = "2022-12-31"

# Base level per series, so different variables are distinguishable.
_BASE = {
    "q_preflash": 100.0,
    "q_flash": 200.0,
    "q_standard": 300.0,
    "q_late": 400.0,
    "m_fast": 10.0,
    "m_mid": 20.0,
    "m_slow": 30.0,
    "m_verylate": 40.0,
}


def _period_ends(freq: str) -> pd.DatetimeIndex:
    """End-of-period dates for the fixture window at the given frequency."""
    return pd.date_range(_START, _END, freq="QE" if freq == "Q" else "ME")


def create_realtime_outturns() -> pd.DataFrame:
    """Build the outturn release calendar for all 8 series.

    Each row is one *publication event*: the value of ``variable`` for target
    period ``date`` as it appeared in the vintage published on ``vintage_date``.
    """
    rng = np.random.default_rng(0)
    rows = []

    for freq, offsets_by_var in (("Q", QUARTERLY_RELEASE_OFFSETS), ("M", MONTHLY_RELEASE_OFFSETS)):
        for variable, offsets in offsets_by_var.items():
            for period_end in _period_ends(freq):
                # Successive releases revise the same underlying truth.
                truth = _BASE[variable] + rng.normal(0, 1)
                for offset in offsets:
                    rows.append(
                        {
                            "date": period_end,
                            "variable": variable,
                            "vintage_date": period_end + pd.Timedelta(days=offset),
                            "frequency": freq,
                            "value": round(truth + rng.normal(0, 0.1), 4),
                        }
                    )

    return pd.DataFrame(rows)


def release_calendar(outturns: pd.DataFrame) -> pd.DatetimeIndex:
    """Every distinct date on which *some* series was published."""
    return pd.DatetimeIndex(sorted(outturns["vintage_date"].unique()))


def create_realtime_forecasts(outturns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Issue a nowcast for every target series on every release date.

    For each release date in the calendar, each source produces a forecast for
    the previous, current and next period. The previous-period path has
    information horizon 0 and is identified as a vintage-relative backcast by
    ``target_minus_vintage`` after ingestion.
    """
    if outturns is None:
        outturns = create_realtime_outturns()

    rng = np.random.default_rng(1)
    calendar = release_calendar(outturns)
    rows = []

    for freq, offsets_by_var in (("Q", QUARTERLY_RELEASE_OFFSETS), ("M", MONTHLY_RELEASE_OFFSETS)):
        valid_periods = {d.to_period(freq): d for d in _period_ends(freq)}

        for variable in offsets_by_var:
            for source in SOURCES:
                for vintage in calendar:
                    v_period = vintage.to_period(freq)
                    for target_offset in (-1, 0, 1):
                        target_period = v_period + target_offset
                        target = valid_periods.get(target_period)
                        if target is None:
                            continue
                        rows.append(
                            {
                                "date": target,
                                "variable": variable,
                                "vintage_date": vintage,
                                "source": source,
                                "frequency": freq,
                                "forecast_horizon": max(target_offset, 0),
                                "value": round(_BASE[variable] + rng.normal(0, 1), 4),
                            }
                        )

    return pd.DataFrame(rows)

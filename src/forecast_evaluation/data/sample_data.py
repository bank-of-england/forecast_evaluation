import numpy as np
import pandas as pd


def compute_days_in_period(vintage_dates: pd.Series, frequencies: pd.Series) -> pd.Series:
    """Compute the number of days from the start of the vintage date's own period.

    For quarterly frequency, the period is the calendar quarter.
    For monthly frequency, the period is the calendar month.

    Parameters
    ----------
    vintage_dates : pd.Series
        Series of vintage dates (datetime-like).
    frequencies : pd.Series
        Series of frequency strings ('Q' or 'M'), aligned with vintage_dates.

    Returns
    -------
    pd.Series
        Integer series with the number of days elapsed since the period start.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.Series([pd.Timestamp("2024-02-14"), pd.Timestamp("2024-05-10")])
    >>> freqs = pd.Series(["Q", "Q"])
    >>> compute_days_in_period(dates, freqs)
    0    44
    1    39
    dtype: int64
    """
    vintage_dates = pd.to_datetime(vintage_dates)
    result = pd.Series(index=vintage_dates.index, dtype=int)

    for freq in frequencies.unique():
        mask = frequencies == freq
        dates = vintage_dates[mask]

        if freq == "Q":
            period_starts = dates.dt.to_period("Q").dt.start_time
        elif freq == "M":
            period_starts = dates.dt.to_period("M").dt.start_time
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

        result[mask] = (dates - period_starts).dt.days

    return result


def create_sample_outturns() -> pd.DataFrame:
    """Create sample outturns DataFrame for testing and examples."""
    df_outturn_1 = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=13, freq="QE"),
            "variable": ["cpisa"] * 13,
            "vintage_date": pd.to_datetime("2025-09-30"),
            "frequency": ["Q"] * 13,
            "value": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        }
    )

    df_outturn_2 = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=13, freq="QE"),
            "variable": ["gdpkp"] * 13,
            "vintage_date": pd.to_datetime("2025-09-30"),
            "frequency": ["Q"] * 13,
            "value": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        }
    )

    df_outturn = pd.concat([df_outturn_1, df_outturn_2], ignore_index=True)

    df_outturn["forecast_horizon"] = (
        df_outturn["date"].dt.to_period("Q") - df_outturn["vintage_date"].dt.to_period("Q")
    ).apply(lambda x: x.n)

    return df_outturn


def create_sample_forecasts() -> pd.DataFrame:
    """Create sample forecasts DataFrame for testing and examples."""
    df_forecasts = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=13, freq="QE"),
            "variable": ["gdpkp"] * 13,
            "vintage_date": pd.to_datetime("2025-03-31"),
            "source": ["mpr2"] * 13,
            "frequency": ["Q"] * 13,
            "value": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        }
    )

    df_forecasts["forecast_horizon"] = (
        df_forecasts["date"].dt.to_period("Q") - df_forecasts["vintage_date"].dt.to_period("Q")
    ).apply(lambda x: x.n)

    return df_forecasts


def create_sample_nowcast_outturns() -> pd.DataFrame:
    """Create sample outturns DataFrame for nowcasting tests and examples.

    Creates quarterly outturns for two variables (``gdp`` and ``cpi``) with
    multiple outturn vintages, covering 2020Q1 to 2025Q4 (24 quarters).

    Publication lags are realistic:

    - **gdp**: first release 6 weeks (42 days) after the end of the target quarter.
    - **cpi**: first release 1 week (7 days) after the end of the target quarter.

    For each target quarter, the outturn data has:

    - A **first-release vintage** at the publication date.
    - **Quarterly revision vintages** from the first subsequent quarter-end up to
      2026-03-31, with white noise shrinking as the vintage matures.

    The realistic first-release dates mean that intra-quarter nowcast vintages
    before the publication date have no backcast (h=-1) outturn available for
    pairing, while later vintages within that quarter do.

    Returns
    -------
    pd.DataFrame
        DataFrame with outturn data suitable for nowcasting evaluation.
    """
    np.random.seed(42)
    target_dates = pd.date_range("2020-03-31", periods=24, freq="QE")

    # True outturn values
    truth = {
        "gdp": {d: 99.0 + (i * 0.3 + np.sin(i / 4) * 0.5) for i, d in enumerate(target_dates)},
        "cpi": {d: 98.0 + (i * 0.45 + np.cos(i / 4) * 0.8) for i, d in enumerate(target_dates)},
    }

    # Publication lag in days after quarter-end:
    #   gdp: 6 weeks = 42 days; cpi: 1 week = 7 days
    pub_lag = {"gdp": 42, "cpi": 7}

    rows = []
    for variable, values in truth.items():
        lag = pub_lag[variable]
        for target_date, true_value in values.items():
            # First-release vintage: publication lag days after quarter-end
            first_release = target_date + pd.Timedelta(days=lag)

            # Quarterly revision vintages: quarter-ends from the one that
            # contains/follows first_release up to 2026-03-31
            revision_start = first_release + pd.offsets.QuarterEnd(0)
            if revision_start <= first_release:
                revision_start = first_release + pd.offsets.QuarterEnd(1)
            revision_vintages = pd.date_range(revision_start, "2026-03-31", freq="QE")

            # First release (k=0 in quarter terms: vintage is ~ same quarter as release)
            noise_fr = np.random.normal(0, 0.4)  # larger first-release noise
            rows.append(
                {
                    "date": target_date,
                    "variable": variable,
                    "vintage_date": first_release,
                    "frequency": "Q",
                    "value": round(true_value + noise_fr, 4),
                }
            )

            # Subsequent quarterly revision vintages with shrinking noise
            for vintage_date in revision_vintages:
                k = (vintage_date.to_period("Q") - target_date.to_period("Q")).n
                noise = np.random.normal(0, 0.5 / (1 + k))
                rows.append(
                    {
                        "date": target_date,
                        "variable": variable,
                        "vintage_date": vintage_date,
                        "frequency": "Q",
                        "value": round(true_value + noise, 4),
                    }
                )

    df_outturn = pd.DataFrame(rows)
    df_outturn["forecast_horizon"] = (
        df_outturn["date"].dt.to_period("Q") - df_outturn["vintage_date"].dt.to_period("Q")
    ).apply(lambda x: x.n)

    return df_outturn


def create_sample_nowcast_forecasts() -> pd.DataFrame:
    """Create sample nowcasting forecasts with intra-quarter vintage dates.

    Generates 6 years of nowcasts (2020-2025) from two models for two variables
    (``gdp`` and ``cpi``). Each quarter has 5 evenly-spaced vintage dates.
    Forecasts are produced at three horizons:

    - **h = -1** (backcast): the quarter that just ended, whose official data
      may not yet be published (gdp: 6 weeks, cpi: 1 week after quarter-end).
    - **h = 0** (nowcast): the current quarter.
    - **h = 1** (nearcast): one quarter ahead.

    Forecast errors converge monotonically to zero as the vintage approaches
    the end of the target quarter. The convergence is based on the total
    number of days remaining until target quarter-end, so later h=1 vintages
    are always less accurate than earlier h=0 vintages for the same target.

    For backcasts (h=-1), the error is driven by how many days before the
    official publication date the forecast is made. After publication the
    backcast error is near-zero; before it, a residual uncertainty remains.

    Returns
    -------
    pd.DataFrame
        DataFrame with nowcast data featuring intra-quarter vintage dates and
        columns: date, variable, vintage_date, source, frequency,
        forecast_horizon, value, days_in_period.

    Examples
    --------
    >>> df = create_sample_nowcast_forecasts()
    >>> sorted(df["forecast_horizon"].unique())
    [-1, 0, 1]
    >>> df["source"].unique()
    array(['nowcast_dfm', 'nowcast_bridge'], dtype=object)
    """
    # 24 quarters: 2020Q1 to 2025Q4
    target_dates = pd.date_range("2020-03-31", periods=24, freq="QE")
    # 5 evenly spaced vintage dates per quarter
    quarter_starts = pd.date_range("2020-01-01", periods=24, freq="QS")
    vintage_dates = pd.DatetimeIndex(
        [start + pd.Timedelta(days=int(i * 91 / 5)) for start in quarter_starts for i in range(5)]
    )

    # True outturn values (consistent with create_sample_nowcast_outturns)
    truth = {
        "gdp": {d: 99.0 + (i * 0.3 + np.sin(i / 4) * 0.5) for i, d in enumerate(target_dates)},
        "cpi": {d: 98.0 + (i * 0.45 + np.cos(i / 4) * 0.8) for i, d in enumerate(target_dates)},
    }

    # Initial bias per (model, variable) — drives the convergence error
    initial_bias = {
        "nowcast_dfm": {"gdp": 2.0, "cpi": -2.5},
        "nowcast_bridge": {"gdp": -1.5, "cpi": 3.0},
    }

    # Publication lag in days after quarter-end used to compute backcast error
    pub_lag = {"gdp": 42, "cpi": 7}

    # Reference scale for days_remaining (approx 2 quarters)
    max_days = 182.0

    rows = []
    for source, biases in initial_bias.items():
        for variable, bias in biases.items():
            lag = pub_lag[variable]
            for vintage in vintage_dates:
                for target in target_dates:
                    horizon = (target.to_period("Q") - vintage.to_period("Q")).n

                    if horizon < -1 or horizon > 1:
                        continue

                    if horizon >= 0:
                        # h=0 and h=1: error shrinks as days_remaining → 0, ensuring
                        # monotonic convergence across the h=1/h=0 boundary.
                        days_remaining = max(0, (target - vintage).days)
                        remaining_fraction = days_remaining / max_days
                        error = bias * remaining_fraction
                    else:
                        # h=-1 (backcast): target quarter has ended.
                        # Error is driven by how far before the publication date
                        # the forecast is made.  Once the official data is out,
                        # the backcast error is near-zero.
                        pub_date = target + pd.Timedelta(days=lag)
                        days_before_pub = max(0, (pub_date - vintage).days)
                        pre_pub_fraction = days_before_pub / max_days
                        # Small irreducible error (data revision uncertainty)
                        # plus a pre-publication component
                        error = bias * (0.05 + 0.15 * pre_pub_fraction)

                    value = truth[variable][target] + error

                    rows.append(
                        {
                            "date": target,
                            "variable": variable,
                            "vintage_date": vintage,
                            "source": source,
                            "frequency": "Q",
                            "forecast_horizon": horizon,
                            "value": round(value, 2),
                        }
                    )

    df = pd.DataFrame(rows)
    df["days_in_period"] = compute_days_in_period(df["vintage_date"], df["frequency"])

    return df


def create_sample_density_forecasts() -> pd.DataFrame:
    """Create sample density forecasts DataFrame with multiple quantiles.

    Generates forecasts by sampling from a normal distribution for each horizon,
    then extracts 100 quantiles from the samples. This ensures quantiles don't cross.

    Returns
    -------
    pd.DataFrame
        DataFrame with density forecast data including a 'quantile' column.
        Contains forecasts for 100 quantiles from 0.01 to 0.99.

    Examples
    --------
    >>> df = create_sample_density_forecasts()
    >>> len(df['quantile'].unique())
    100
    >>> # Verify quantiles don't cross for a given date
    >>> sample = df[df['date'] == df['date'].iloc[0]].sort_values('quantile')
    >>> assert all(sample['value'].diff().dropna() > 0)  # Values strictly increasing
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define 100 quantiles from 0.01 to 0.99
    quantiles = np.linspace(0.01, 0.99, 50)

    # Create base structure
    dates = pd.date_range(start="2025-12-31", periods=13, freq="QE")
    vintage_date = pd.to_datetime("2025-12-31")

    density_forecasts = []

    # For each forecast horizon
    for i, date in enumerate(dates):
        # Mean value increases with time (trend)
        mean = i + 1

        # Standard deviation proportional to mean (5% uncertainty)
        std = 0.5

        # Generate 10000 samples from normal distribution for this horizon
        samples = np.random.normal(mean, std, 10000)

        # Extract quantiles from the samples
        for quantile in quantiles:
            quantile_value = np.quantile(samples, quantile)

            forecast_horizon = (pd.Period(date, freq="Q") - pd.Period(vintage_date, freq="Q")).n

            density_forecasts.append(
                {
                    "date": date,
                    "variable": "gdpkp",
                    "vintage_date": vintage_date,
                    "source": "mpr2",
                    "frequency": "Q",
                    "quantile": quantile,
                    "value": quantile_value,
                    "forecast_horizon": forecast_horizon,
                }
            )

    result = pd.DataFrame(density_forecasts)

    # Sort by date and quantile for clarity
    result = result.sort_values(["date", "quantile"]).reset_index(drop=True)

    return result

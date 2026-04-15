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

    Creates quarterly outturns for two variables (gdp and cpi) with a single
    outturn vintage, covering 2022Q1 to 2025Q1.

    Returns
    -------
    pd.DataFrame
        DataFrame with outturn data suitable for nowcasting evaluation.
    """
    dates = pd.date_range(start="2022-01-01", periods=13, freq="QE")

    df_gdp = pd.DataFrame(
        {
            "date": dates,
            "variable": "gdp",
            "vintage_date": pd.Timestamp("2025-06-30"),
            "frequency": "Q",
            "value": [100, 101.2, 100.8, 102.0, 103.1, 102.5, 104.0, 105.2, 104.8, 106.1, 107.0, 106.5, 108.0],
        }
    )

    df_cpi = pd.DataFrame(
        {
            "date": dates,
            "variable": "cpi",
            "vintage_date": pd.Timestamp("2025-06-30"),
            "frequency": "Q",
            "value": [100, 101.5, 103.0, 104.2, 105.8, 107.0, 108.5, 110.0, 111.2, 112.8, 114.0, 115.5, 117.0],
        }
    )

    df_outturn = pd.concat([df_gdp, df_cpi], ignore_index=True)
    df_outturn["forecast_horizon"] = (
        df_outturn["date"].dt.to_period("Q") - df_outturn["vintage_date"].dt.to_period("Q")
    ).apply(lambda x: x.n)

    return df_outturn


def create_sample_nowcast_forecasts() -> pd.DataFrame:
    """Create sample nowcasting forecasts with weekly vintage dates.

    Generates one year of weekly nowcasts (2024-01-01 to 2024-12-31) from two
    models for two variables (gdp and cpi), each targeting the 4 quarters of 2024.
    Each weekly vintage produces forecasts for all target quarters that have
    ``forecast_horizon >= 0`` at that point in time.

    Returns
    -------
    pd.DataFrame
        DataFrame with nowcast data featuring weekly vintage dates and quarterly targets.

    Examples
    --------
    >>> df = create_sample_nowcast_forecasts()
    >>> df["source"].unique()
    array(['nowcast_dfm', 'nowcast_bridge'], dtype=object)
    >>> df["variable"].unique()
    array(['gdp', 'cpi'], dtype=object)
    """
    target_dates = pd.date_range("2024-03-31", periods=4, freq="QE")
    vintage_dates = pd.date_range("2024-01-01", "2024-12-31", freq="W-MON")

    # True outturn values for 2024 Q1-Q4 (must match create_sample_nowcast_outturns)
    truth = {
        "gdp": {target_dates[0]: 104.8, target_dates[1]: 106.1, target_dates[2]: 107.0, target_dates[3]: 106.5},
        "cpi": {target_dates[0]: 111.2, target_dates[1]: 112.8, target_dates[2]: 114.0, target_dates[3]: 115.5},
    }

    # Initial bias per (model, variable) — forecasts start here and linearly converge to truth
    initial_bias = {
        "nowcast_dfm": {"gdp": 2.0, "cpi": -2.5},
        "nowcast_bridge": {"gdp": -1.5, "cpi": 3.0},
    }

    rows = []
    for source, biases in initial_bias.items():
        for variable, bias in biases.items():
            for vintage in vintage_dates:
                for target in target_dates:
                    horizon = (target.to_period("Q") - vintage.to_period("Q")).n
                    if horizon < 0:
                        continue

                    # Deterministic convergence: error shrinks linearly with days_in_period.
                    # At day 0 the forecast equals truth + bias * (1 + horizon * 0.3),
                    # at day 91 (end of quarter) it equals truth exactly (for h=0).
                    days_into_quarter = (vintage - vintage.to_period("Q").start_time).days
                    remaining_fraction = 1 - days_into_quarter / 91
                    error = bias * remaining_fraction * (1 + horizon * 0.3)
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

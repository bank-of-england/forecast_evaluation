from collections.abc import Iterable
from typing import Literal

import pandas as pd

from forecast_evaluation.core.transformations import transform_forecast_to_levels, transform_series
from forecast_evaluation.data import ForecastData


def build_random_walk_model(
    data: ForecastData,
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Literal["Q", "M"] = "Q",
    forecast_periods: int = 13,
):
    """
    Build a random walk forecast model for the specified variable, metric, and frequency.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing outturn data.
    variable : str
        The variable to build the model for (e.g., 'gdpkp').
    metric : str
        The metric to build the model for (e.g., 'levels').
    frequency : Literal["Q", "M"], optional
        The frequency of the data ('Q' for quarterly, 'M' for monthly). Default is 'Q'.
    forecast_periods : int, optional
        Number of periods to forecast ahead. Default is 13.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the random walk forecasts.

    Notes
    -----
    If the random walk is built in levels, the forecasted value for each period ahead is equal to the latest
    observed value.
    If the random walk is built in growth rates (pop or yoy), the forecasted growth rates are assumed to be
    constant and equal to the latest observed growth rate.
    """
    # Filter data for the specified variable and frequency
    df = data._raw_outturns[
        (data._raw_outturns["variable"] == variable) & (data._raw_outturns["frequency"] == frequency)
    ].copy()

    # Check data availability after filtering
    if df.empty:
        raise ValueError(f"No outturn data available for variable '{variable}' and frequency '{frequency}'.")

    # Transform outturns depending on metric
    df = transform_series(df, transform=metric, frequency=frequency)

    # Ensure date and vintage_date is in datetime format
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if "vintage_date" in df.columns:
        df["vintage_date"] = pd.to_datetime(df["vintage_date"])

    # Generate random walk forecasts
    forecasts = []

    # Group by unique combinations
    grouped = df.groupby(["variable", "metric", "frequency", "vintage_date"])

    for (variable, metric, frequency, vintage_date), group in grouped:
        group = group.sort_values("date")

        try:
            # Get the latest date and value
            latest_date = group["date"].max()
            latest_value = group[group["date"] == latest_date]["value"].iloc[0]

            # Include the latest value in the forecast
            forecasts.append(
                {
                    "date": latest_date,
                    "variable": variable,
                    "metric": metric,
                    "vintage_date": vintage_date,
                    "value": latest_value,
                    "frequency": frequency,
                    "source": "baseline random walk model",
                    "forecast_horizon": -1,
                }
            )

            # Generate forecast dates
            if frequency == "M":
                forecast_dates = pd.date_range(
                    start=latest_date + pd.offsets.MonthEnd(), periods=forecast_periods, freq="ME"
                )
            elif frequency == "Q":
                forecast_dates = pd.date_range(
                    start=latest_date + pd.offsets.QuarterEnd(), periods=forecast_periods, freq="QE"
                )

            # Create forecast entries (random walk - same value for all forecasts)
            for period, date in enumerate(forecast_dates):
                forecasts.append(
                    {
                        "date": date,
                        "variable": variable,
                        "metric": metric,
                        "vintage_date": vintage_date,
                        "value": latest_value,
                        "frequency": frequency,
                        "source": "baseline random walk model",
                        "forecast_horizon": period,
                    }
                )

        except Exception as e:
            print(f"Skipping group {variable}, {vintage_date} due to error: {e}")

    forecast_df = pd.DataFrame(forecasts)

    # Order the results by variable, frequency, vintage_date, and date
    if not forecast_df.empty:
        forecast_df = forecast_df.sort_values(["variable", "frequency", "vintage_date", "date"])

    return forecast_df


def add_random_walk_forecasts(
    data: ForecastData,
    variable: str | Iterable[str] | None = None,
    metric: Literal["levels", "pop", "yoy"] = "levels",
    frequency: Literal["Q", "M"] | Iterable[Literal["Q", "M"]] | None = None,
    forecast_periods: int = 13,
) -> None:
    """
    Add random walk forecasts to the ForecastData object.

    Parameters
    ----------
    data : ForecastData
        ForecastData object to which random walk forecasts will be added.
    variable : str | Iterable[str] | None
        Variable name (e.g., 'gdpkp') or multiple variable names.
        If None, builds forecasts for all variable x frequency combinations present in the outturns.
    metric : str, optional
        The metric to build the model for. Default is 'levels'.
    frequency : Literal["Q", "M"] | Iterable[Literal["Q", "M"]] | None, optional
        Frequency (or multiple frequencies) to build forecasts for.
        If None, frequencies are inferred from the outturns' `frequency` column for the requested variable(s).
    forecast_periods : int, optional
        Number of periods to forecast ahead. Default is 13.

    Returns
    -------
    None
        The function modifies the ForecastData object in place.
    """
    if variable is None:
        variables: list[str] | None = None
    else:
        variables = [variable] if isinstance(variable, str) else list(variable)
        if len(variables) == 0:
            raise ValueError("variable must be None, a non-empty string, or a non-empty iterable of strings")

    if frequency is None:
        frequencies: list[str] | None = None
    else:
        frequencies = [frequency] if isinstance(frequency, str) else list(frequency)
        if len(frequencies) == 0:
            raise ValueError("frequency must be None, a frequency string, or a non-empty iterable of frequencies")
        invalid = sorted(set(frequencies) - {"Q", "M"})
        if invalid:
            raise ValueError(f"Unsupported frequency value(s): {invalid}. Supported: 'Q', 'M'.")

    # Filter variable and frequency combinations from outturns
    outturns = data._raw_outturns
    subset = outturns[["variable", "frequency"]].dropna()
    if variables is not None:
        subset = subset[subset["variable"].isin(variables)]
    if frequencies is not None:
        subset = subset[subset["frequency"].isin(frequencies)]

    pairs = list(subset.drop_duplicates().itertuples(index=False, name=None))
    if len(pairs) == 0:
        raise ValueError("No (variable, frequency) combinations available in outturns for the requested filters")

    forecast_frames = [
        build_random_walk_model(
            data=data,
            variable=var,
            metric=metric,
            frequency=freq,
            forecast_periods=forecast_periods,
        )
        for (var, freq) in pairs
    ]

    rw_forecasts = pd.concat(forecast_frames, ignore_index=True) if len(forecast_frames) > 1 else forecast_frames[0]

    # Transform forecasts to levels
    rw_forecasts_in_levels = transform_forecast_to_levels(
        outturns=data._raw_outturns,
        forecasts=rw_forecasts,
    )

    # Append random walk forecasts to existing forecasts
    data.add_forecasts(rw_forecasts_in_levels)

    return None

from collections.abc import Iterable
from typing import Literal, Optional, Union

import pandas as pd
from tqdm import tqdm

from forecast_evaluation.core.transformations import transform_forecast_to_levels, transform_series
from forecast_evaluation.data import ForecastData


def build_random_walk_model(
    data: ForecastData,
    variable: str,
    metric: Literal["levels", "pop", "yoy"],
    frequency: Optional[Literal["Q", "M"]] = None,
    forecast_periods: int = 13,
    first_forecast_horizon: Union[int, dict[str, int]] = 0,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Build a random walk forecast model for the specified variable, metric, and frequency.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing outturn data.
    variable : str
        The variable to build the model for (e.g., 'gdpkp').
    metric : {"levels", "pop", "yoy"}
        The metric to build the model for.
    frequency : {"Q", "M"} or None, optional
        The frequency of the data ('Q' for quarterly, 'M' for monthly). If None,
        inferred from the data. Default is None.
    forecast_periods : int, optional
        Number of periods to forecast ahead. Default is 13.
    first_forecast_horizon : int or dict[str, int], optional
        The minimum forecast horizon to produce. Training data is restricted to periods
        strictly before this horizon, so the benchmark never uses data it is supposed to
        predict. Default is 0.
    show_progress : bool, optional
        Whether to show a progress bar. Default is False.

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
    if frequency is None:
        inferred = data._raw_outturns[data._raw_outturns["variable"] == variable]["frequency"].unique()
        if len(inferred) != 1:
            raise ValueError(
                f"Could not infer a unique frequency from data; found: {list(inferred)}. "
                "Please specify the 'frequency' argument explicitly."
            )
        frequency = inferred[0]

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

    if isinstance(first_forecast_horizon, dict):
        first_forecast_horizon = first_forecast_horizon.get(variable, 0)

    date_offset = pd.offsets.QuarterEnd() if frequency == "Q" else pd.offsets.MonthEnd()

    # Generate random walk forecasts
    forecasts = []

    if not data.outturn_vintages:
        # --- No outturn vintages: create benchmark forecasts for each forecast vintage ---
        if data._raw_forecasts is None or data._raw_forecasts.empty:
            raise ValueError(
                "Cannot build benchmark forecasts when outturn_vintages=False and no forecasts have been added. "
                "Add forecasts first so that forecast vintage dates are available."
            )

        # Get unique forecast vintage dates for this variable/frequency
        raw_fc = data._raw_forecasts
        relevant = raw_fc[(raw_fc["variable"] == variable) & (raw_fc["frequency"] == frequency)]
        vintage_dates = sorted(relevant["vintage_date"].dropna().unique())
        if len(vintage_dates) == 0:
            raise ValueError(f"No forecast vintage dates found for variable '{variable}' and frequency '{frequency}'.")

        # Group by variable/metric/frequency only (no vintage_date in outturns)
        grouped = df.groupby(["variable", "metric", "frequency"])

        for (grp_variable, grp_metric, grp_frequency), group in tqdm(
            grouped, desc=f"Building random walk model for {variable} ({frequency})", disable=not show_progress
        ):
            group = group.sort_values("date")

            for vintage_date in vintage_dates:
                try:
                    cutoff_date = vintage_date + first_forecast_horizon * date_offset
                    available = group[group["date"] < cutoff_date]
                    if available.empty:
                        continue

                    latest_date = available["date"].max()
                    latest_value = available[available["date"] == latest_date]["value"].iloc[0]

                    forecasts.append(
                        {
                            "date": latest_date,
                            "variable": grp_variable,
                            "metric": grp_metric,
                            "vintage_date": vintage_date,
                            "value": latest_value,
                            "frequency": grp_frequency,
                            "source": "baseline random walk model",
                            "forecast_horizon": first_forecast_horizon - 1,
                        }
                    )

                    if frequency == "M":
                        forecast_dates = pd.date_range(
                            start=latest_date + pd.offsets.MonthEnd(), periods=forecast_periods, freq="ME"
                        )
                    elif frequency == "Q":
                        forecast_dates = pd.date_range(
                            start=latest_date + pd.offsets.QuarterEnd(), periods=forecast_periods, freq="QE"
                        )

                    for period, date in enumerate(forecast_dates, start=first_forecast_horizon):
                        forecasts.append(
                            {
                                "date": date,
                                "variable": grp_variable,
                                "metric": grp_metric,
                                "vintage_date": vintage_date,
                                "value": latest_value,
                                "frequency": grp_frequency,
                                "source": "baseline random walk model",
                                "forecast_horizon": period,
                            }
                        )

                except Exception as e:
                    print(f"Skipping vintage {vintage_date} for {grp_variable} due to error: {e}")
    else:
        # --- Standard path: outturn vintages available ---
        grouped = df.groupby(["variable", "metric", "frequency", "vintage_date"])

        for (variable, metric, frequency, vintage_date), group in tqdm(
            grouped, desc=f"Building random walk model for {variable} ({frequency})", disable=not show_progress
        ):
            group = group.sort_values("date")

            try:
                cutoff_date = vintage_date + first_forecast_horizon * date_offset
                training_data = group[group["date"] < cutoff_date]
                if training_data.empty:
                    continue

                latest_date = training_data["date"].max()
                latest_value = training_data[training_data["date"] == latest_date]["value"].iloc[0]

                forecasts.append(
                    {
                        "date": latest_date,
                        "variable": variable,
                        "metric": metric,
                        "vintage_date": vintage_date,
                        "value": latest_value,
                        "frequency": frequency,
                        "source": "baseline random walk model",
                        "forecast_horizon": first_forecast_horizon - 1,
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
                for period, date in enumerate(forecast_dates, start=first_forecast_horizon):
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
    show_progress: bool = False,
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
    show_progress : bool, optional
        Whether to show progress bars. Default is False.

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
            first_forecast_horizon=data.first_forecast_horizon,
            show_progress=show_progress,
        )
        for (var, freq) in pairs
    ]

    rw_forecasts = pd.concat(forecast_frames, ignore_index=True) if len(forecast_frames) > 1 else forecast_frames[0]

    # Transform forecasts to levels
    rw_forecasts_in_levels = transform_forecast_to_levels(
        outturns=data._raw_outturns,
        forecasts=rw_forecasts,
        first_forecast_horizon=data.first_forecast_horizon,
    )

    # Append random walk forecasts to existing forecasts
    data.add_forecasts(rw_forecasts_in_levels)

    return None

from typing import Literal

import pandas as pd

from forecast_evaluation.utils import reconstruct_id_cols_from_unique_id


def transform_series(
    df: pd.DataFrame, transform: Literal["levels", "diff", "pop", "yoy"], frequency: Literal["Q", "M"]
) -> pd.DataFrame:
    """Transform a time series based on specified transformation type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data.
    transform : str
        Type of transformation to apply ('levels', 'diff', 'pop', 'yoy').
    frequency : str
        Frequency of the data ('Q' for quarterly, 'M' for monthly).

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with updated values.

    """

    # Grouping columns
    grouping_cols = ["variable", "vintage_date", "frequency"]
    if "unique_id" in df.columns:
        grouping_cols.append("unique_id")

    # Sorting columns
    sorting_cols = grouping_cols + ["date"]

    df = df.sort_values(by=sorting_cols)

    # Don't transform forecasts that are already in levels
    if transform == "levels":
        df["metric"] = "levels"
        return df

    if transform == "diff":
        df["value"] = df.groupby(grouping_cols)["value"].diff(periods=1)
        df["metric"] = "diff"
    elif transform == "pop":
        df["value"] = df.groupby(grouping_cols)["value"].pct_change(periods=1)
        df["metric"] = "pop"
    elif transform == "yoy":
        n_periods = {"Q": 4, "M": 12}[frequency]
        df["value"] = df.groupby(grouping_cols)["value"].pct_change(periods=n_periods)
        df["metric"] = "yoy"

    df = df[df["value"].notna()]

    return df


def prepare_forecasts(forecasts: pd.DataFrame, outturns: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    """Prepare forecast data for evaluation by combining with outturns and applying transformations.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Validated DataFrame containing forecast data.
    outturns : pd.DataFrame
        Validated DataFrame containing outturns data.
    id_columns : list of str
        List of columns that uniquely identify a forecast.

    Returns
    -------
    pd.DataFrame
        Prepared forecast data with levels, period-on-period, and year-on-year transformations.
    """

    # Handle empty inputs
    if forecasts is None or forecasts.empty:
        return pd.DataFrame() if forecasts is None else forecasts.copy()

    if outturns is None or outturns.empty:
        # Keep only valid forecast horizons; do not attempt transformations.
        df = forecasts.copy()
        if "metric" not in df.columns:
            df["metric"] = "levels"
        return df[df["forecast_horizon"] >= 0].copy()

    forecasts = forecasts.drop(columns=id_columns)

    frequencies = forecasts["frequency"].unique()
    forecast_id_list = forecasts["unique_id"].unique()

    forecasts_all = []

    for frequency in frequencies:
        forecasts_freq = forecasts[(forecasts["frequency"] == frequency) & (forecasts["forecast_horizon"] >= 0)].copy()
        outturns_freq = outturns[outturns["frequency"] == frequency].copy()

        # YoY or MoM transform for the forecast mean it needs appending to outturns for that vintage first
        # Note we filter outturns first on the latest 4 values
        # as no point brining in everything for change space metrics
        outturns_filtered = outturns_freq[outturns_freq["forecast_horizon"] >= -5].copy()
        # We need to loop through each id and concat to the outturns
        forecast_dfs = []
        for f_id in forecast_id_list:
            outturns_id = outturns_filtered.copy()
            outturns_id["unique_id"] = f_id
            forecast_id = forecasts_freq[forecasts_freq["unique_id"] == f_id]
            forecast_dfs.append(pd.concat([outturns_id, forecast_id], ignore_index=True))

        # We compute all transformations now (levels, pop (period-on-period), yoy (year-on-year)
        forecasts_with_outturns = pd.concat(forecast_dfs, ignore_index=True)

        # Levels
        forecast_levels = transform_series(forecasts_with_outturns, transform="levels", frequency=frequency)
        forecast_levels = forecast_levels[forecast_levels["forecast_horizon"] >= 0]
        # pop Change
        forecast_pop_change = transform_series(forecasts_with_outturns, transform="pop", frequency=frequency)
        forecast_pop_change = forecast_pop_change[forecast_pop_change["forecast_horizon"] >= 0]
        # YoY Change
        forecast_yoy_change = transform_series(forecasts_with_outturns, transform="yoy", frequency=frequency)
        forecast_yoy_change = forecast_yoy_change[forecast_yoy_change["forecast_horizon"] >= 0]

        # Append them all
        forecasts_freq = pd.concat([forecast_levels, forecast_pop_change, forecast_yoy_change], ignore_index=True)

        forecasts_all.append(forecasts_freq)

    df_forecasts = pd.concat(forecasts_all, ignore_index=True)

    # reconstruct individual id columns from unique_id
    df_forecasts = reconstruct_id_cols_from_unique_id(df_forecasts, id_columns)

    return df_forecasts


def prepare_outturns(outturns: pd.DataFrame) -> pd.DataFrame:
    """Prepare outturn data by applying transformations across different frequencies.

    Parameters
    ----------
    outturns : pd.DataFrame
        Validated DataFrame containing outturn data.

    Returns
    -------
    pd.DataFrame
        Prepared outturn data with levels, period-on-period, and year-on-year transformations.
    """

    frequencies = outturns["frequency"].unique()
    outturns_all = []

    for frequency in frequencies:
        df = outturns[outturns["frequency"] == frequency].copy()

        # Transform outturns data
        outturns_levels = transform_series(df, transform="levels", frequency=frequency)
        outturns_pop_change = transform_series(df, transform="pop", frequency=frequency)
        outturns_yoy_change = transform_series(df, transform="yoy", frequency=frequency)

        # Append them all
        outturns_freq = pd.concat([outturns_levels, outturns_pop_change, outturns_yoy_change], ignore_index=True)

        outturns_all.append(outturns_freq)

    return pd.concat(outturns_all, ignore_index=True)


def transform_forecast_to_levels(
    outturns: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Transform forecast back to levels.

    Parameters
    ----------
    outturns : pd.DataFrame
        Validated DataFrame containing outturn data.
    forecasts : pd.DataFrame
        Validated DataFrame containing forecast data.

    Returns
    -------
    pd.DataFrame
        Transformed forecast in levels (forecast_horizon >= 0 only).
    """

    # Keep only forecast rows
    if "forecast_horizon" in forecasts.columns:
        forecasts = forecasts[forecasts["forecast_horizon"] >= 0].copy()

    # Separate forecasts that are already in levels and those that need transformation
    forecasts_levels = forecasts[forecasts["metric"] == "levels"].copy()
    forecasts_to_transform = forecasts[forecasts["metric"] != "levels"].copy()

    transformed_forecasts: list[pd.DataFrame] = []

    group_cols = ["source", "variable", "metric", "frequency", "vintage_date"]
    if "unique_id" in forecasts_to_transform.columns:
        group_cols.append("unique_id")

    for keys, group in forecasts_to_transform.groupby(group_cols):
        try:
            key_dict = dict(zip(group_cols, keys))
            variable = key_dict["variable"]
            metric = key_dict["metric"]
            vintage_date = key_dict["vintage_date"]

            group = group.sort_values("date").copy()
            group["date"] = pd.to_datetime(group["date"]).dt.normalize()
            group["vintage_date"] = pd.to_datetime(group["vintage_date"]).dt.normalize()

            outturns_subset = (
                outturns[(outturns["variable"] == variable) & (outturns["vintage_date"] == vintage_date)]
                .sort_values("date")
                .copy()
            )

            if outturns_subset.empty:
                raise ValueError(f"No outturn data available for variable '{variable}', vintage_date '{vintage_date}'.")

            outturns_subset["date"] = pd.to_datetime(outturns_subset["date"]).dt.normalize()

            latest_level = float(outturns_subset.iloc[-1]["value"])
            known_levels = pd.Series(
                outturns_subset["value"].to_numpy(),
                index=outturns_subset["date"],
            ).to_dict()

            if metric == "diff":
                base_level = latest_level
                reconstructed = []
                for _, row in group.iterrows():
                    base_level = base_level + float(row["value"])
                    reconstructed.append(base_level)
                group["value"] = reconstructed

            elif metric == "pop":
                base_level = latest_level
                reconstructed = []
                for _, row in group.iterrows():
                    base_level = base_level * (1.0 + float(row["value"]))
                    reconstructed.append(base_level)
                group["value"] = reconstructed

            elif metric == "yoy":
                reconstructed = []
                for _, row in group.iterrows():
                    date_t = row["date"]
                    base_date = date_t - pd.DateOffset(years=1)
                    if base_date not in known_levels:
                        raise ValueError(
                            f"Cannot reconstruct YoY level for date '{date_t}': missing base level at '{base_date}'."
                        )
                    level_t = (1.0 + float(row["value"])) * float(known_levels[base_date])
                    reconstructed.append(level_t)
                    known_levels[date_t] = level_t  # allow chaining beyond 1 year
                group["value"] = reconstructed

            else:
                raise ValueError(f"Unsupported metric '{metric}' for level reconstruction")

            group["metric"] = "levels"
            transformed_forecasts.append(group)

        except Exception as e:
            print(f"Skipping group {keys} due to error: {e}")

    if transformed_forecasts:
        transformed_forecasts_df = pd.concat(transformed_forecasts, ignore_index=True)
        final_forecasts = pd.concat([forecasts_levels, transformed_forecasts_df], ignore_index=True)
    else:
        final_forecasts = forecasts_levels

    return final_forecasts

from typing import Literal, Union

import pandas as pd

from forecast_evaluation.utils import reconstruct_id_cols_from_unique_id


def compute_k(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Compute the forecast horizon k for each row in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing vintage_date_outturn and date columns.
    frequency : str
        Data frequency, either 'Q' for quarterly or 'M' for monthly.

    Returns
    -------
    pd.DataFrame
        Input dataframe with additional 'k' column representing the forecast horizon.
    """
    # Compute k as number of quarters or months between forecast date and outturns vintage date
    if frequency == "Q":
        # -- Optimal K (diagonal method): we take -1 so it can be 0, 4 and 12 (which is 1,5 and 13)
        df["k"] = (
            (df["vintage_date_outturn"].dt.year - df["date"].dt.year) * 4
            + (df["vintage_date_outturn"].dt.month // 3 - df["date"].dt.month // 3)
            - 1
        )
    elif frequency == "M":
        df["k"] = (
            (df["vintage_date_outturn"].dt.year - df["date"].dt.year) * 12
            + (df["vintage_date_outturn"].dt.month - df["date"].dt.month)
            - 1
        )
    else:
        raise ValueError("Unsupported frequency")

    df = df[df["k"].notna() & (df["k"] >= -1)]
    # -1 instead of 0 because some surveys can be released before the end of the period
    # so we get the data the same period. Plus can cause issue when constructing artificial vintages
    df["k"] = df["k"].astype(int)
    return df


def build_main_table(
    forecasts: pd.DataFrame,
    outturns: pd.DataFrame,
    id_columns: list[str],
    variables: Union[str, list[str]] = None,
    forecast_ids: Union[str, list[str]] = None,
    frequency: Literal["Q", "M"] = "Q",
) -> pd.DataFrame:
    """Calculate the k-diagonal table for forecast evaluation.

    Parameters
    ----------
    forecasts : pd.DataFrame
        DataFrame containing forecast data.
    outturns : pd.DataFrame
        DataFrame containing outturn data.
    id_columns : list of str
        List of columns that uniquely identify a forecast.
    variables : str or list of str, optional
        Name of the variable to analyze, or list of variable names.
    forecast_ids : str or list of str, optional
        Single identifier or list of forecast identifier to include.
        Can be elements of column 'source' or extra_ids columns.
    frequency : {"Q", "M"}, default "Q"
        Frequency of the data, either quarterly or monthly.

    Returns
    -------
    pd.DataFrame
        Table containing forecast evaluation metrics with forecast errors and vintage information.
    """

    forecasts = forecasts.copy()

    # remove individual id columns
    forecasts = forecasts.drop(columns=id_columns)

    if variables is None:
        variables = forecasts["variable"].unique().tolist()

    if forecast_ids is None:
        forecast_ids = forecasts["unique_id"].unique().tolist()

    # switch to list if single string provided
    if not isinstance(forecast_ids, list):
        forecast_ids = [forecast_ids]

    if not isinstance(variables, list):
        variables = [variables]

    forecasts_filtered = forecasts[forecasts["variable"].isin(variables) & forecasts["unique_id"].isin(forecast_ids)]
    outturns_filtered = outturns[outturns["variable"].isin(variables)]

    # Set multi-index for faster merge on large datasets
    merge_cols = ["date", "variable", "frequency", "metric"]
    forecasts_indexed = forecasts_filtered.set_index(merge_cols)
    outturns_indexed = outturns_filtered.set_index(merge_cols)

    # Join using index (faster than merge for large data)
    merged = forecasts_indexed.join(outturns_indexed, lsuffix="_forecast", rsuffix="_outturn").reset_index()

    # Keep only outturns fro; the forecasted date
    # >= and not > because some data can be released before the end of the period
    # e.g. some surveys. Plus this line can be problematic when constructing artificial vintages
    merged = merged[merged["vintage_date_outturn"] >= merged["date"]]
    merged = merged.rename(columns={"id_forecast": "unique_id", "forecast_horizon_forecast": "forecast_horizon"})

    merged = merged[
        [
            "date",
            "variable",
            "vintage_date_forecast",
            "vintage_date_outturn",
            "unique_id",
            "metric",
            "frequency",
            "forecast_horizon",
            "value_forecast",
            "value_outturn",
        ]
    ].copy()

    merged = compute_k(merged, frequency)

    merged["latest_vintage"] = merged.groupby(["variable", "metric", "frequency", "unique_id", "date"])[
        "vintage_date_outturn"
    ].transform("max")

    # Compute forecast errors
    merged["forecast_error"] = merged["value_outturn"] - merged["value_forecast"]

    # reconstruct individual id columns from unique_id and remove unique id
    merged = reconstruct_id_cols_from_unique_id(merged, id_columns)

    return merged

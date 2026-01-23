from typing import Callable, Optional, Union

import pandas as pd

from forecast_evaluation.utils import filter_sources


def filter_fer_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataset to only include the variables from the FER (Forecast Evaluation Report).

    Filters for specific variable-metric combinations:
    - unemp (unemployment) in levels
    - cpisa (CPI inflation) in year-over-year terms
    - gdpkp (GDP) in year-over-year terms
    - aweagg (average weekly earnings) in year-over-year terms

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast data with 'variable' and 'metric' columns

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only FER variables
    """
    df = df[
        (
            ((df["variable"] == "unemp") & (df["metric"] == "levels"))
            | ((df["variable"] == "cpisa") & (df["metric"] == "yoy"))
            | ((df["variable"] == "gdpkp") & (df["metric"] == "yoy"))
            | ((df["variable"] == "aweagg") & (df["metric"] == "yoy"))
        )
    ].copy()

    return df


def filter_fer_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataset to only include the forecast models from the FER (Forecast Evaluation Report).

    Includes the following models:
    - Baseline AR(p) model
    - Random walk model
    - MPR (Monetary Policy Report)
    - COMPASS conditional
    - COMPASS unconditional
    - BVAR conditional
    - BVAR unconditional

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast data with 'source' column

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only FER models
    """
    allowed_models = [
        "baseline ar(p) model",
        "random walk model",
        "mpr",
        "compass conditional",
        "compass unconditional",
        "bvar conditional",
        "bvar unconditional",
    ]
    df = df[df["unique_id"].str.lower().isin(allowed_models)].copy()

    return df


def filter_tables(
    df,
    start_date: str = None,
    end_date: str = None,
    start_vintage: str = None,
    end_vintage: str = None,
    variables: Optional[Union[str, list[str]]] = None,
    metrics: Optional[list[str]] = None,
    sources: Optional[Union[list[str], str]] = None,
    frequencies: Optional[Union[str, list[str]]] = None,
    custom_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
):
    """Filter the dataset based on date and vintage ranges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast data with 'date' and 'vintage_date' columns
    start_date : str, optional
        Start date to filter forecasts (inclusive). Format 'YYYY-MM-DD'.
        Default is None in which case the analysis start with the initial date.
    end_date : str, optional
        End date to filter forecasts (inclusive). Format 'YYYY-MM-DD'. Default is None.
        Default is None in which case the analysis ends with the initial date.
    start_vintage : str, optional
        Start vintage date to filter forecasts (inclusive). Format 'YYYY-MM-DD'.
        Default is None in which case the analysis start with the initial vintage.
    end_vintage : str, optional
        End vintage date to filter forecasts (inclusive). Format 'YYYY-MM-DD'.
        Default is None in which case the analysis ends with the initial vintage.
    variables: Optional[Union[list[str], str]] = None
        List of variable identifiers to filter. Default is None (no filtering).
    metrics: Optional[list[str]] = None
        List of metric identifiers to filter. Default is None (no filtering).
    sources: Optional[Union[list[str], str]] = None
        List of source identifiers to filter, or a single source string. Default is None (no filtering).
    frequencies: Optional[Union[list[str], str]] = None
        List of frequency identifiers to filter, or a single frequency string. Default is None (no
    custom_filter : Callable[[pd.DataFrame], pd.DataFrame], optional
        A custom filtering function that takes a DataFrame as input and returns a filtered DataFrame.
        Default is None.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame based on specified date and vintage ranges
    """

    # Validate and convert date strings to pandas datetime
    if start_date is not None:
        try:
            start_date = pd.to_datetime(start_date, format="%Y-%m-%d")
        except ValueError:
            raise ValueError(f"start_date must be in format 'YYYY-MM-DD', got: {start_date}")
        df = df[df["date"] >= start_date]

    if end_date is not None:
        try:
            end_date = pd.to_datetime(end_date, format="%Y-%m-%d")
        except ValueError:
            raise ValueError(f"end_date must be in format 'YYYY-MM-DD', got: {end_date}")
        df = df[df["date"] <= end_date]

    if start_vintage is not None:
        try:
            start_vintage = pd.to_datetime(start_vintage, format="%Y-%m-%d")
        except ValueError:
            raise ValueError(f"start_vintage must be in format 'YYYY-MM-DD', got: {start_vintage}")
        df = df[df["vintage_date_forecast"] >= start_vintage]

    if end_vintage is not None:
        try:
            end_vintage = pd.to_datetime(end_vintage, format="%Y-%m-%d")
        except ValueError:
            raise ValueError(f"end_vintage must be in format 'YYYY-MM-DD', got: {end_vintage}")
        df = df[df["vintage_date_forecast"] <= end_vintage]

    if variables is not None:
        if isinstance(variables, str):
            variables = [variables]
        df = df[df["variable"].isin(variables)]

    if metrics is not None:
        if isinstance(metrics, str):
            metrics = [metrics]
        df = df[df["metric"].isin(metrics)]

    if frequencies is not None:
        if isinstance(frequencies, str):
            frequencies = [frequencies]
        df = df[df["frequency"].isin(frequencies)]

    if sources is not None:
        if isinstance(sources, str):
            sources = [sources]

        df_source = []
        sources_without_plus = sources.copy()
        for source in sources:
            if "+" in source:
                # Sources contain '+', treat as concatenated unique_ids for exact match
                df_source.append(df[df["unique_id"] == source])
                sources_without_plus.remove(source)

        if len(sources_without_plus) > 0:
            # Sources using individual id matching
            df_filtered = filter_sources(df, sources_without_plus)
            df_source.append(df_filtered)

        df = pd.concat(df_source).drop_duplicates().reset_index(drop=True)

    if custom_filter is not None:
        df = custom_filter(df)

    return df


def construct_unique_id(df: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    """
    Construct the 'unique_id' column by concatenating specified identifier columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing individual identifier columns.
    id_columns : list of str
        List of column names to concatenate into 'unique_id'.

    Returns
    -------
    pd.DataFrame
        DataFrame with constructed 'unique_id' column.
    """

    unique_id = df[id_columns].fillna("").astype(str).agg(" + ".join, axis=1)

    return unique_id

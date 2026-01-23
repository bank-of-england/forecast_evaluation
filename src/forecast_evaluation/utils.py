import pandas as pd


def filter_k(df: pd.DataFrame, k: int = 12, fill_k: bool = True) -> pd.DataFrame:
    """Filter the dataset for a particular k, replacing unreleased outturn vintages with the latest vintage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast data with 'k', 'latest_vintage', and 'vintage_date_outturn' columns
    k : int, default=12
        Number of revisions to filter by
    fill_k : bool, default=True
        If True, substitutes unreleased outturns from the latest vintage

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows where k matches or latest vintage is used
    """
    # Subset the data
    if fill_k:
        df = df[(df["k"] == k) | ((df["k"] < k) & (df["latest_vintage"] == df["vintage_date_outturn"]))]
    else:
        df = df[df["k"] == k]

    return df


def covid_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to exclude COVID-affected periods.

    For 'gdpkp' variable: excludes dates from 2020-01-01 to 2022-03-31 unless forecast vintage is from
    2022-01-01 onwards.
    For other variables: removes all 2020 and 2021 dates for pre-2020Q4 vintages.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast data with 'variable', 'date', and 'vintage_date_forecast' columns

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with COVID periods removed based on variable type
    """
    # Apply GDP-specific filter for gdpkp rows
    gdpkp_mask = df["variable"] == "gdpkp"
    gdpkp_filter = (
        (df["date"] < "2020-01-01") | (df["date"] >= "2022-04-01") | (df["vintage_date_forecast"] >= "2022-01-01")
    )

    # Apply default filter for non-gdpkp rows
    default_filter = (
        (df["date"] < "2020-01-01") | (df["date"] > "2021-12-31") | (df["vintage_date_forecast"] >= "2020-10-01")
    )

    # Combine filters: use GDP filter for gdpkp rows, default filter for others
    df = df[(gdpkp_mask & gdpkp_filter) | (~gdpkp_mask & default_filter)]

    return df


def ensure_consistent_date_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to ensure consistent vintage_date_forecast range across all sources for each variable.

    This function addresses the problem where different forecast sources may have different
    availability periods for the same variable. It finds the overlapping time period where
    ALL sources have data for each variable, ensuring fair comparison across models.

    The function uses the latest start date and earliest end date across all sources for
    each variable, creating a "common denominator" time period.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing forecast accuracy data with required columns:

        - 'variable' : str - Variable identifier (e.g., 'gdpkp', 'cpisa', 'unemp')
        - 'source' : str - Forecast source identifier (e.g., 'compass conditional', 'mpr')
        - 'vintage_date_forecast' : datetime - Forecast vintage date

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only data within the consistent date range for each variable.
    """
    # For each variable, find the date range that all sources have in common
    if "vintage_date_forecast" in df.columns:
        date_ranges = df.groupby(["variable", "unique_id"])["vintage_date_forecast"].agg(["min", "max"]).reset_index()
    elif "vintage_date" in df.columns:
        date_ranges = df.groupby(["variable", "unique_id"])["vintage_date"].agg(["min", "max"]).reset_index()
    else:
        raise ValueError("DataFrame must contain either 'vintage_date_forecast' or 'vintage_date' column.")

    # For each variable, get the consistent date range (latest start, earliest end)
    consistent_ranges = (
        date_ranges.groupby("variable")
        .agg(
            {
                "min": "max",  # Latest start date across all sources
                "max": "min",  # Earliest end date across all sources
            }
        )
        .reset_index()
    )
    consistent_ranges.columns = ["variable", "start_date", "end_date"]

    # Merge back and filter
    df = df.merge(consistent_ranges, on="variable")
    if "vintage_date_forecast" in df.columns:
        df = df[(df["vintage_date_forecast"] >= df["start_date"]) & (df["vintage_date_forecast"] <= df["end_date"])]
    else:
        df = df[(df["vintage_date"] >= df["start_date"]) & (df["vintage_date"] <= df["end_date"])]

    return df.drop(["start_date", "end_date"], axis=1)


def flatten_col_name(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Convert tuple column names to strings

    Parameters
    ----------
    obj : pd.DataFrame or pd.Series
        DataFrame or Series with potentially multi-level column names

    Returns
    -------
    pd.DataFrame or pd.Series
        DataFrame or Series with flattened column names
    """
    if isinstance(obj, pd.DataFrame):
        obj.columns = ["_".join(map(str, col)) if isinstance(col, tuple) else col for col in obj.columns]
    elif isinstance(obj, pd.Series):
        obj.name = "_".join(map(str, obj.name)) if isinstance(obj.name, tuple) else obj.name
    else:
        raise ValueError("Input must be a DataFrame or Series")

    return obj


def reconstruct_id_cols_from_unique_id(df: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    """
    Reconstruct individual identifier columns from the 'unique_id' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'unique_id' column with concatenated identifiers.
    id_columns : list of str
        List of column names to assign to the reconstructed identifier parts.

    Returns
    -------
    pd.DataFrame
        DataFrame with reconstructed identifier columns.
    """
    id_components = df["unique_id"].str.split("+", expand=True)
    id_components = id_components.apply(lambda col: col.str.strip())  # remove white spaces
    id_components.columns = [id_columns[i] for i in range(id_components.shape[1])]
    df = pd.concat([df, id_components], axis=1)

    return df


def find_ids_to_exclude(df: pd.DataFrame, sources: list[str]) -> list[str]:
    """
    Work by exclusion (which should be the most efficient approach).
    Let's say we have id1 = ["A", "B"] and id2 = ["big", "small"].
    If a user selects sources = ["A"], all unique_id with "B" should be excluded.
    If a user selects sources = ["C"], all unique_id with "A" and "B" should be excluded.
    So first we have to find the ids to exclude.

    Parameters
    ----------
    unique_id : str
        Concatenated identifier string with parts separated by '+'
    sources : list[str]
        List of source identifiers to check against

    Returns
    -------
    list[str]
        List of identifier parts to exclude
    """

    # create matrix of unique id
    id_components = df["unique_id"].str.split("+", expand=True)

    # drop duplicates
    id_components = id_components.drop_duplicates()

    # Replace empty strings with NaN
    id_components = id_components.replace("", pd.NA)

    # remove white spaces
    id_components = id_components.apply(lambda col: col.str.strip())

    # for each column, if a source is present, store the sources that are not present
    ids_to_exclude = set()
    sources_match = 0
    for col in id_components.columns:
        unique_ids = id_components[col].dropna().unique()

        # drop empty strings (they correspond to forecasts with no id in that column)
        unique_ids = [uid for uid in unique_ids if uid != ""]

        all_ids = set(unique_ids)
        wanted_ids = set([source.strip() for source in sources])
        missing_ids = all_ids - wanted_ids

        if len(missing_ids) < len(all_ids):
            ids_to_exclude.update(missing_ids)
            sources_match += 1

    if sources_match == 0:
        # if there is no match exclude all
        ids_to_exclude = set(id_components.stack().dropna().unique())

    return list(ids_to_exclude)


def filter_sources(df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    """
    Filter the dataset based on source identifiers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast data with 'unique_id' column
    sources : list[str]
        List of source identifiers to filter by

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows matching the specified sources
    """
    ids_to_exclude = find_ids_to_exclude(df, sources)

    # Filter out rows where any part of unique_id is in ids_to_exclude
    mask = df["unique_id"].apply(lambda x: not any(part.strip() in ids_to_exclude for part in x.split("+")))
    df = df[mask]

    return df

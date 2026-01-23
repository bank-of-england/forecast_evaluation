import pandas as pd

from forecast_evaluation.utils import filter_k


def create_revision_dataframe(main_df: pd.DataFrame, forecasts: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """Create a merged DataFrame containing forecast revisions and forecast errors.

    This function processes raw forecast data to calculate revisions (changes between
    consecutive vintage forecasts) and merges them with forecast error data for
    correlation analysis.

    Parameters
    ----------
    main_df : pd.DataFrame
        DataFrame containing forecasts, outturns and forecast errors.
    forecasts : pd.DataFrame
        DataFrame containing forecast data with vintage information.
    k : int, optional, default=12
        Number of revisions used to define the outturns.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with forecast revisions and errors.
    """

    main_df = filter_k(main_df, k)

    # Calculate revisions and data processing
    df_revisions = forecasts.copy()
    df_revisions = df_revisions.sort_values(
        by=["variable", "unique_id", "metric", "frequency", "date", "vintage_date"], ascending=True
    ).reset_index(drop=True)
    df_revisions["revision"] = df_revisions.groupby(["variable", "unique_id", "metric", "frequency", "date"])[
        "value"
    ].diff()
    df_revisions["revision_number"] = df_revisions.groupby(
        ["variable", "unique_id", "metric", "frequency", "date"]
    ).cumcount()
    df_revisions = df_revisions.rename(columns={"vintage_date": "vintage_date_forecast"})

    # Merge revisions and errors dataframes
    df_merged = main_df.merge(
        df_revisions[
            [
                "date",
                "unique_id",
                "variable",
                "metric",
                "frequency",
                "vintage_date_forecast",
                "revision",
                "revision_number",
            ]
        ],
        on=["variable", "unique_id", "metric", "frequency", "date", "vintage_date_forecast"],
        how="left",
    )

    # Drop NAs of revision or error
    df_merged = df_merged.dropna(subset=["revision"])

    # Select columns
    df_merged = df_merged[
        [
            "date",
            "variable",
            "unique_id",
            "metric",
            "frequency",
            "vintage_date_forecast",
            "forecast_horizon",
            "value_forecast",
            "value_outturn",
            "forecast_error",
            "revision",
            "revision_number",
        ]
    ]

    return df_merged

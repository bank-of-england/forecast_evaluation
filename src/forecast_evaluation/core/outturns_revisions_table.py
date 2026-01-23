from forecast_evaluation.data import ForecastData


def create_outturn_revisions(data: ForecastData):
    """
    Create outturn revisions dataframe.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing forecast and outturn data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing outturn revisions.
    """
    outturns = data.outturns

    # Get first release of data
    outturns_first_release = outturns[outturns["forecast_horizon"] == -1]
    # Rename columns
    outturns_first_release = outturns_first_release.rename(
        columns={"value": "value_original", "vintage_date": "vintage_date_original"}
    )
    outturns_first_release = outturns_first_release.drop(columns=["forecast_horizon"])

    # Get outturns after k revisions
    outturns_revised = outturns[outturns["forecast_horizon"] <= -1]
    outturns_revised = outturns_revised.rename(
        columns={"value": "value_outturn", "vintage_date": "vintage_date_outturn"}
    )

    # Merge first release and revised data
    # Set multi-index for faster merge on large datasets
    merge_cols = ["date", "variable", "frequency", "metric"]
    outturns_first_release = outturns_first_release.set_index(merge_cols)
    outturns_revised = outturns_revised.set_index(merge_cols)

    # Join using index (faster than merge for large data)
    merged = outturns_first_release.join(outturns_revised).dropna().reset_index()

    # Add latest_vintage column
    merged["latest_vintage"] = merged.groupby(["date", "variable", "metric", "frequency"])[
        "vintage_date_outturn"
    ].transform("max")

    # Create k and revision column
    merged["k"] = (-1 - merged["forecast_horizon"]).astype(int)
    merged = merged.drop(columns=["forecast_horizon"])
    merged["revision"] = merged["value_outturn"] - merged["value_original"]

    return merged

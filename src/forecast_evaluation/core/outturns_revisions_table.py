from forecast_evaluation.data import ForecastData


def create_outturn_revisions(data: ForecastData):
    """
    Create outturn revisions dataframe.

    ``k`` is defined by release order: k=0 is the first release, k=1 the
    second, and so on.  This is more natural than horizon-based k for
    nowcasting data where different variables have different revision
    frequencies (e.g. GDP quarterly, CPI monthly).

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

    if outturns.empty or "forecast_horizon" not in outturns.columns:
        return outturns

    # NowcastData expands outturns to match weekly forecast vintages (tagged
    # with ``_aligned=True``).  These forward-filled copies are needed for
    # forecast matching but are not real releases — exclude them here.
    if "_aligned" in outturns.columns:
        outturns = outturns[~outturns["_aligned"]].drop(columns=["_aligned"])

    group_cols = ["date", "variable", "frequency", "metric"]

    # Assign k by release order (earliest vintage = k=0, next = k=1, ...)
    outturns = outturns.sort_values(group_cols + ["vintage_date"])
    outturns["k"] = outturns.groupby(group_cols).cumcount()

    # Split: first release (k=0) vs all releases
    first_release = outturns[outturns["k"] == 0].copy()
    first_release = first_release.rename(columns={"value": "value_original", "vintage_date": "vintage_date_original"})
    first_release = first_release.drop(columns=["forecast_horizon", "k"])

    revised = outturns.rename(columns={"value": "value_outturn", "vintage_date": "vintage_date_outturn"})

    # Merge first release with all releases
    first_release = first_release.set_index(group_cols)
    revised = revised.set_index(group_cols)
    merged = first_release.join(revised).dropna().reset_index()

    # Add latest_vintage column
    merged["latest_vintage"] = merged.groupby(group_cols)["vintage_date_outturn"].transform("max")

    merged["revision"] = merged["value_outturn"] - merged["value_original"]
    merged = merged.drop(columns=["forecast_horizon"])

    return merged

from typing import Literal, Union

import pandas as pd


def _horizon_threshold(df: pd.DataFrame, first_forecast_horizon: Union[int, dict[str, int]]) -> Union[int, pd.Series]:
    """Return the threshold to compare forecast_horizon against.

    Returns a scalar int when first_forecast_horizon is an int, or a per-row Series
    aligned to df when first_forecast_horizon is a dict (variables absent from the
    dict default to 0).
    """
    if isinstance(first_forecast_horizon, dict):
        return df["variable"].map(first_forecast_horizon).fillna(0).astype(int)
    return first_forecast_horizon


def _validate_first_forecast_horizon(
    first_forecast_horizon: Union[int, dict[str, int]],
    known_variables: set[str],
) -> None:
    """Raise if a dict references variable names that are not in the data."""
    if not isinstance(first_forecast_horizon, dict):
        return
    unknown = set(first_forecast_horizon) - known_variables
    if unknown:
        raise ValueError(
            f"first_forecast_horizon contains unknown variable(s): {sorted(unknown)}. "
            f"Known variables: {sorted(known_variables)}."
        )


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
        df["value"] = df.groupby(grouping_cols, dropna=False)["value"].diff(periods=1)
        df["metric"] = "diff"
    elif transform == "pop":
        df["value"] = df.groupby(grouping_cols, dropna=False)["value"].pct_change(periods=1)
        df["metric"] = "pop"
    elif transform == "yoy":
        n_periods = {"Q": 4, "M": 12}[frequency]
        df["value"] = df.groupby(grouping_cols, dropna=False)["value"].pct_change(periods=n_periods)
        df["metric"] = "yoy"

    df = df[df["value"].notna()]

    return df


def prepare_forecasts(
    forecasts: pd.DataFrame,
    outturns: pd.DataFrame,
    id_columns: list[str],
    compute_levels: bool = True,
    first_forecast_horizon: Union[int, dict[str, int]] = 0,
) -> pd.DataFrame:
    """Prepare forecast data for evaluation by combining with outturns and applying transformations.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Validated DataFrame containing forecast data.
    outturns : pd.DataFrame
        Validated DataFrame containing outturns data.
    id_columns : list of str
        List of columns that uniquely identify a forecast.
    compute_levels : bool, optional
        Whether to automatically transform non-levels forecasts to levels if outturns data is available.
        When True, forecasts in 'pop' and 'yoy' metrics will be converted to levels
        using the available outturns data.
        Useful if you add 'pop' and want to analyse 'yoy' forecasts and vice versa.
        If the transformation fails for specific groups (e.g., due to insufficient
        historical data), those groups will be skipped with a warning message.
        Default is True.
    first_forecast_horizon : int or dict[str, int], optional
        The minimum forecast horizon to retain. Pass an int to apply the same threshold to all
        variables, or a dict mapping variable names to per-variable thresholds (variables not in
        the dict default to 0). Set to a negative value to include backcasts. Default is 0.

    Returns
    -------
    pd.DataFrame
        Prepared forecast data with levels, period-on-period, and year-on-year transformations.
    """

    # Handle empty inputs
    if forecasts is None or forecasts.empty:
        return pd.DataFrame() if forecasts is None else forecasts.copy()

    forecasts = forecasts.copy()

    # Validate dict keys against the variables present in forecasts + outturns.
    known_variables = set(forecasts["variable"].dropna().unique())
    if outturns is not None and not outturns.empty:
        known_variables |= set(outturns["variable"].dropna().unique())
    _validate_first_forecast_horizon(first_forecast_horizon, known_variables)

    if outturns is None or outturns.empty:
        # Keep only valid forecast horizons; do not attempt transformations.
        df = forecasts.copy()
        if "metric" not in df.columns:
            df["metric"] = "levels"
        return df[df["forecast_horizon"] >= _horizon_threshold(df, first_forecast_horizon)].copy()

    # Auto-transform non-levels forecasts to levels if requested
    if compute_levels:
        non_levels_forecasts = forecasts[forecasts["metric"] != "levels"].copy()

        # Add levels forecasts
        updated_level_forecasts = transform_forecast_to_levels(
            outturns,
            forecasts,
            first_forecast_horizon=first_forecast_horizon,
        )

        # add back non-levels forecasts
        forecasts = pd.concat([updated_level_forecasts, non_levels_forecasts], ignore_index=True)

    # Split forecasts by metric type
    non_levels_forecasts = forecasts[forecasts["metric"] != "levels"].copy()
    levels_forecasts = forecasts[forecasts["metric"] == "levels"].copy()

    # Only transform 'levels' forecasts
    if levels_forecasts.empty:
        # reconstruct individual id columns from unique_id
        # non_levels_forecasts = reconstruct_id_cols_from_unique_id(non_levels_forecasts, id_columns)
        return non_levels_forecasts

    # Build a lookup table: unique_id -> id columns, to avoid expensive string splitting later
    id_lookup = levels_forecasts[["unique_id"] + id_columns].drop_duplicates(subset=["unique_id"])

    levels_forecasts = levels_forecasts.drop(columns=id_columns)

    frequencies = levels_forecasts["frequency"].unique()
    forecast_id_list = levels_forecasts["unique_id"].unique()

    forecasts_all = []

    for frequency in frequencies:
        _thresholds = _horizon_threshold(levels_forecasts, first_forecast_horizon)
        forecasts_freq = levels_forecasts[
            (levels_forecasts["frequency"] == frequency) & (levels_forecasts["forecast_horizon"] >= _thresholds)
        ].copy()
        outturns_freq = outturns[outturns["frequency"] == frequency].copy()

        # YoY/MoM transforms require prepending enough outturn history so that
        # pct_change(n_periods) has a valid base at first_forecast_horizon.
        # We need n_periods+1 outturn rows before first_forecast_horizon per vintage.
        n_periods = {"Q": 4, "M": 12}[frequency]
        if isinstance(first_forecast_horizon, dict):
            min_ffh = min(first_forecast_horizon.values(), default=0)
        else:
            min_ffh = first_forecast_horizon
        min_outturn_horizon = min_ffh - (n_periods + 1)
        outturns_filtered = outturns_freq[outturns_freq["forecast_horizon"] >= min_outturn_horizon].copy()
        outturns_filtered = outturns_filtered[outturns_filtered["metric"] == "levels"]
        # Mark outturn rows so they can be stripped out after transformations.
        outturns_filtered["_helper_outturn"] = True

        # When outturns have NaT vintage_dates (outturn_vintages=False),
        # create synthetic vintage copies for each forecast vintage_date
        # so that transform_series can group outturns and forecasts together.
        if not outturns_filtered.empty and outturns_filtered["vintage_date"].isna().all():
            forecast_vintage_dates = forecasts_freq["vintage_date"].dropna().unique()
            synthetic_frames = []
            for v_date in forecast_vintage_dates:
                outturn_copy = outturns_filtered[outturns_filtered["date"] < v_date].copy()
                # Keep the last n_periods+1 observations per variable for YoY computation
                outturn_copy = outturn_copy.sort_values("date").groupby("variable").tail(n_periods + 1)
                outturn_copy["vintage_date"] = v_date
                synthetic_frames.append(outturn_copy)
            if synthetic_frames:
                outturns_filtered = pd.concat(synthetic_frames, ignore_index=True)

        # We need to loop through each id and concat to the outturns
        forecast_dfs = []
        for f_id in forecast_id_list:
            outturns_id = outturns_filtered.copy()
            outturns_id["unique_id"] = f_id
            forecast_id = forecasts_freq[forecasts_freq["unique_id"] == f_id]
            # forecast_id rows do not have _helper_outturn; it becomes NaN after concat
            forecast_dfs.append(pd.concat([outturns_id, forecast_id], ignore_index=True))

        # We compute all transformations now (levels, pop (period-on-period), yoy (year-on-year)
        forecasts_with_outturns = pd.concat(forecast_dfs, ignore_index=True)

        # pop Change
        forecast_pop_change = transform_series(forecasts_with_outturns, transform="pop", frequency=frequency)

        # YoY Change
        forecast_yoy_change = transform_series(forecasts_with_outturns, transform="yoy", frequency=frequency)

        # Append them all
        forecasts_freq = pd.concat(
            [forecasts_with_outturns, forecast_pop_change, forecast_yoy_change], ignore_index=True
        )

        forecasts_all.append(forecasts_freq)

    df_forecasts = pd.concat(forecasts_all, ignore_index=True)

    # Restore id columns from lookup instead of parsing unique_id strings
    df_forecasts = df_forecasts.merge(id_lookup, on="unique_id", how="left")

    # Combine transformed levels forecasts with pass-through non-levels forecasts
    if not non_levels_forecasts.empty:
        df_forecasts = pd.concat([df_forecasts, non_levels_forecasts], ignore_index=True)

    df_forecasts = df_forecasts[
        df_forecasts["forecast_horizon"] >= _horizon_threshold(df_forecasts, first_forecast_horizon)
    ]

    # Remove the helper outturn rows that were prepended for transformation purposes only.
    if "_helper_outturn" in df_forecasts.columns:
        df_forecasts = df_forecasts[df_forecasts["_helper_outturn"].isna()]
        df_forecasts = df_forecasts.drop(columns=["_helper_outturn"])

    # Ensure forecast_horizon stays integer (concat with _helper_outturn NaN column
    # can upcast int to float)
    if "forecast_horizon" in df_forecasts.columns:
        df_forecasts["forecast_horizon"] = df_forecasts["forecast_horizon"].astype(int)

    # drop duplicates TODO: ideally we shouldn't have any duplicates
    # these are introduced because pop and yoy are always computed, even if
    # they are already present
    df_forecasts = df_forecasts.drop_duplicates(subset=[col for col in df_forecasts.columns if col != "value"])

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

    # Split outturns by metric type
    levels_outturns = outturns[outturns["metric"] == "levels"].copy()
    non_levels_outturns = outturns[outturns["metric"] != "levels"].copy()

    # Only transform 'levels' outturns
    if levels_outturns.empty:
        return non_levels_outturns

    frequencies = levels_outturns["frequency"].unique()
    outturns_all = []

    for frequency in frequencies:
        df = levels_outturns[levels_outturns["frequency"] == frequency].copy()

        # Transform outturns data
        outturns_pop_change = transform_series(df, transform="pop", frequency=frequency)
        outturns_yoy_change = transform_series(df, transform="yoy", frequency=frequency)

        # Combine transformed outturns — use `df` (this frequency's slice), not
        # `levels_outturns` (all frequencies), otherwise level rows are duplicated
        # once per extra frequency in the final concat.
        outturns_freq = pd.concat([df, outturns_pop_change, outturns_yoy_change], ignore_index=True)

        outturns_all.append(outturns_freq)

    df_outturns = pd.concat(outturns_all, ignore_index=True)

    # Combine transformed levels outturns with pass-through non-levels outturns
    if not non_levels_outturns.empty:
        df_outturns = pd.concat([df_outturns, non_levels_outturns], ignore_index=True)

    return df_outturns


def transform_forecast_to_levels(
    outturns: pd.DataFrame,
    forecasts: pd.DataFrame,
    first_forecast_horizon: Union[int, dict[str, int]] = 0,
) -> pd.DataFrame:
    """Transform forecast back to levels.

    Parameters
    ----------
    outturns : pd.DataFrame
        Validated DataFrame containing outturn data.
    forecasts : pd.DataFrame
        Validated DataFrame containing forecast data.
    first_forecast_horizon : int or dict[str, int], optional
        The minimum forecast horizon to retain. Pass an int to apply the same threshold to all
        variables, or a dict mapping variable names to per-variable thresholds (variables not in
        the dict default to 0). Default is 0.

    Returns
    -------
    pd.DataFrame
        Transformed forecast in levels (forecast_horizon >= first_forecast_horizon only).
    """

    # Keep only forecast rows
    if "forecast_horizon" in forecasts.columns:
        forecasts = forecasts[
            forecasts["forecast_horizon"] >= _horizon_threshold(forecasts, first_forecast_horizon)
        ].copy()

    # Separate forecasts that are already in levels and those that need transformation
    forecasts_levels = forecasts[forecasts["metric"] == "levels"].copy()
    forecasts_to_transform = forecasts[forecasts["metric"] != "levels"].copy()

    transformed_forecasts: list[pd.DataFrame] = []

    group_cols = ["source", "variable", "metric", "frequency", "vintage_date"]
    if "unique_id" in forecasts_to_transform.columns:
        group_cols.append("unique_id")

    # Pre-compute set of existing level groups for O(1) lookup inside the loop
    level_key_cols = group_cols.copy()
    level_key_cols.remove("metric")

    existing_level_groups = (
        set(forecasts_levels[level_key_cols].itertuples(index=False, name=None))
        if not forecasts_levels.empty
        else set()
    )

    for keys, group in forecasts_to_transform.groupby(group_cols):
        # keys order: source(0), variable(1), metric(2), frequency(3), vintage_date(4), [unique_id(5)]
        lookup_key = (keys[0], keys[1], keys[3], keys[4])
        if "unique_id" in group_cols:
            lookup_key = lookup_key + (keys[5],)

        if lookup_key in existing_level_groups:
            # If we already have levels for this group, skip transformation
            continue

        try:
            key_dict = dict(zip(group_cols, keys))
            variable = key_dict["variable"]
            metric = key_dict["metric"]
            vintage_date = key_dict["vintage_date"]

            group = group.sort_values("date").copy()
            group["date"] = pd.to_datetime(group["date"]).dt.normalize()
            group["vintage_date"] = pd.to_datetime(group["vintage_date"]).dt.normalize()

            outturns_subset = (
                outturns[
                    (outturns["variable"] == variable)
                    & (outturns["vintage_date"] == vintage_date)
                    & (outturns["metric"] == "levels")
                ]
                .sort_values("date")
                .copy()
            )

            # When outturns have NaT vintage_dates (outturn_vintages=False),
            # fall back to using all outturns for this variable with dates
            # before the forecast vintage_date.
            if outturns_subset.empty:
                outturns_subset = (
                    outturns[
                        (outturns["variable"] == variable)
                        & (outturns["vintage_date"].isna())
                        & (outturns["metric"] == "levels")
                        & (outturns["date"] < vintage_date)
                    ]
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

            # Mark this (source, variable, frequency, vintage_date[, unique_id])
            # as having levels so that a second metric (e.g. yoy after pop) does
            # not produce duplicate level rows for the same keys.
            existing_level_groups.add(lookup_key)

        except Exception as e:
            print(f"Skipping group {keys} due to error: {e}")

    if transformed_forecasts:
        transformed_forecasts_df = pd.concat(transformed_forecasts, ignore_index=True)
        final_forecasts = pd.concat([forecasts_levels, transformed_forecasts_df], ignore_index=True)
    else:
        final_forecasts = forecasts_levels

    return final_forecasts

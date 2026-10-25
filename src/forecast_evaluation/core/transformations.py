import warnings
from typing import Literal

import numpy as np
import pandas as pd

_OUTTURN_KEY = ["variable", "frequency", "date", "vintage_date", "metric"]
# Supplied pop/yoy values within this tolerance of the re-derived value are treated as equal.
_DERIVED_RTOL = 1e-6
_DERIVED_ATOL = 1e-9


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
    Returns
    -------
    pd.DataFrame
        Prepared forecast data with levels, period-on-period, and year-on-year transformations.
    """

    # Handle empty inputs
    if forecasts is None or forecasts.empty:
        return pd.DataFrame() if forecasts is None else forecasts.copy()

    forecasts = forecasts.copy()

    if outturns is None or outturns.empty:
        df = forecasts.copy()
        if "metric" not in df.columns:
            df["metric"] = "levels"
        return df

    # Auto-transform non-levels forecasts to levels if requested
    if compute_levels:
        non_levels_forecasts = forecasts[forecasts["metric"] != "levels"].copy()

        # Add levels forecasts
        updated_level_forecasts = transform_forecast_to_levels(
            outturns,
            forecasts,
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
        forecasts_freq = levels_forecasts[levels_forecasts["frequency"] == frequency].copy()
        outturns_freq = outturns[outturns["frequency"] == frequency].copy()

        # YoY/MoM transforms require prepending enough outturn history so that
        # pct_change(n_periods) has a valid base at the earliest forecast target.
        n_periods = {"Q": 4, "M": 12}[frequency]
        min_distance = forecasts_freq["target_minus_vintage"].min() - (n_periods + 1)
        outturns_filtered = outturns_freq[
            outturns_freq["target_minus_vintage"].isna() | (outturns_freq["target_minus_vintage"] >= min_distance)
        ].copy()
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
                from forecast_evaluation.data.utils import compute_target_minus_vintage

                outturns_filtered = compute_target_minus_vintage(outturns_filtered)

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

    # Remove the helper outturn rows that were prepended for transformation purposes only.
    if "_helper_outturn" in df_forecasts.columns:
        df_forecasts = df_forecasts[df_forecasts["_helper_outturn"].isna()]
        df_forecasts = df_forecasts.drop(columns=["_helper_outturn"])

    # drop duplicates TODO: ideally we shouldn't have any duplicates
    # these are introduced because pop and yoy are always computed, even if
    # they are already present
    df_forecasts = df_forecasts.drop_duplicates(subset=[col for col in df_forecasts.columns if col != "value"])

    if "forecast_horizon" in df_forecasts.columns:
        df_forecasts["forecast_horizon"] = df_forecasts["forecast_horizon"].astype(int)

    return df_forecasts


def drop_superseded_outturns(derived: pd.DataFrame, supplied: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows of ``derived`` that ``supplied`` already provides.

    Rows match on (variable, frequency, date, vintage_date, metric); null vintage dates match each other.
    A single aggregated ``UserWarning`` is emitted if any supplied value differs from the matching
    ``derived`` value by more than a small tolerance. Supplied rows flagged by a boolean ``_aligned``
    column (nowcast snapshots, NaN counts as False) still supersede derived rows but are exempt from
    that warning, since they may legitimately differ from values re-derived from the latest levels.

    Parameters
    ----------
    derived : pd.DataFrame
        Outturn rows (for example pop/yoy derived from levels) that yield to ``supplied`` rows.
    supplied : pd.DataFrame
        Outturn rows to prefer, with unique keys.

    Returns
    -------
    pd.DataFrame
        ``derived`` without the rows that have a match in ``supplied``, in the original order.
    """
    if derived.empty or supplied.empty:
        return derived

    has_aligned = "_aligned" in supplied.columns
    supplied_keys = supplied[_OUTTURN_KEY + ["value"] + (["_aligned"] if has_aligned else [])].drop_duplicates(
        subset=_OUTTURN_KEY
    )
    merged = derived[_OUTTURN_KEY + ["value"]].merge(
        supplied_keys, on=_OUTTURN_KEY, how="left", suffixes=("", "_supplied"), indicator=True
    )
    matched = (merged["_merge"] == "both").to_numpy()
    if not matched.any():
        return derived

    derived_values = merged.loc[matched, "value"].to_numpy(dtype=float)
    supplied_values = merged.loc[matched, "value_supplied"].to_numpy(dtype=float)
    abs_diff = np.abs(derived_values - supplied_values)
    differs = abs_diff > _DERIVED_ATOL + _DERIVED_RTOL * np.abs(supplied_values)
    if has_aligned:
        aligned = merged.loc[matched, "_aligned"].eq(True).to_numpy()  # NaN -> False without a downcast warning
        differs &= ~aligned
    if differs.any():
        warnings.warn(
            f"{int(differs.sum())} supplied pop/yoy outturn rows differ from the values re-derived from levels "
            f"(largest absolute difference {abs_diff[differs].max():.6g}); keeping the supplied values.",
            UserWarning,
            stacklevel=3,
        )

    return derived[~matched]


def prepare_outturns(outturns: pd.DataFrame) -> pd.DataFrame:
    """Prepare outturn data by applying transformations across different frequencies.

    ``pop`` and ``yoy`` rows are derived from the ``levels`` rows. Rows of those metrics that are
    already present in ``outturns`` (for example when re-wrapping ``ForecastData.outturns``) are
    kept as supplied and the derived row with the same (variable, frequency, date, vintage_date,
    metric) is dropped, so no row is duplicated. If a supplied value differs from the re-derived
    value by more than a small tolerance, the supplied value is kept and one ``UserWarning`` reports
    how many rows differ and the largest absolute difference. Rows flagged ``_aligned`` (nowcast
    snapshots) supersede derived rows like any other but are exempt from that warning.

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
        derived = df_outturns[df_outturns["metric"] != "levels"]
        kept = drop_superseded_outturns(derived, non_levels_outturns)
        df_outturns = df_outturns.drop(index=derived.index.difference(kept.index))
        df_outturns = pd.concat([df_outturns, non_levels_outturns], ignore_index=True)

    return df_outturns


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
        Transformed forecast in levels.
    """

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

            from forecast_evaluation.data.utils import infer_last_observation_dates

            last_observation_date = infer_last_observation_dates(group, variable, group["frequency"].iloc[0]).loc[
                vintage_date
            ]

            outturns_subset = (
                outturns[
                    (outturns["variable"] == variable)
                    & (outturns["vintage_date"] == vintage_date)
                    & (outturns["metric"] == "levels")
                    & (outturns["date"] <= last_observation_date)
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
                        & (outturns["date"] <= last_observation_date)
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

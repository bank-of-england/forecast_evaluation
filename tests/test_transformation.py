import pandas as pd

import forecast_evaluation as fe


def test_transformations_levels_to_pop_and_yoy(fer_minimal_fd, snapshot):
    """Check add_forecasts compute transformation correctly."""
    # extract forecasts in levels
    forecasts_in_levels = fer_minimal_fd._raw_forecasts

    # extract forecasts with transformations applied
    forecast_with_transformations = fer_minimal_fd.forecasts

    # -----------------------------------------------------------------------
    # Compute transformation manually
    # ------------------------------------------------------------------------

    grouping_cols = ["variable", "source", "vintage_date"]

    # Sorting columns
    sorting_cols = grouping_cols + ["date"]
    forecasts_in_levels = forecasts_in_levels.sort_values(by=sorting_cols)

    # pop
    df_pop = forecasts_in_levels.copy()
    df_pop["value"] = df_pop.groupby(grouping_cols)["value"].pct_change(periods=1)
    df_pop["metric"] = "pop"

    # yoy
    df_yoy = forecasts_in_levels.copy()
    df_yoy["value"] = df_yoy.groupby(grouping_cols)["value"].pct_change(periods=4)
    df_yoy["metric"] = "yoy"

    # filter out nans
    df_pop = df_pop[df_pop["value"].notna()]
    df_yoy = df_yoy[df_yoy["value"].notna()]

    # ------------------------------------------------------------------------

    # check that the transformed forecasts match the ones in fer_minimal_fd
    # yoy
    df_package = forecast_with_transformations[forecast_with_transformations["metric"] == "yoy"].copy()
    df_yoy_merged = pd.merge(
        df_package[grouping_cols + ["date", "value"]],
        df_yoy[grouping_cols + ["date", "value"]],
        on=grouping_cols + ["date"],
        suffixes=("_manual", "_package"),
    )

    # check that manual and package transformation match, allowing for some numerical tolerance
    diff = df_yoy_merged["value_manual"] - df_yoy_merged["value_package"]
    max_diff = diff.abs().max()
    assert max_diff < 1e-6, f"Problem with YoY transformation, max difference is {max_diff}"

    # pop
    df_package = forecast_with_transformations[forecast_with_transformations["metric"] == "pop"].copy()
    df_pop_merged = pd.merge(
        df_package[grouping_cols + ["date", "value"]],
        df_pop[grouping_cols + ["date", "value"]],
        on=grouping_cols + ["date"],
        suffixes=("_manual", "_package"),
    )

    # check that manual and package transformation match, allowing for some numerical tolerance
    diff = df_pop_merged["value_manual"] - df_pop_merged["value_package"]
    max_diff = diff.abs().max()
    assert max_diff < 1e-6, f"Problem with PoP transformation, max difference is {max_diff}"


def test_transformations_pop_and_yoy_to_levels(fer_minimal_fd, snapshot):
    """Check add_forecasts compute transformation correctly."""
    # extract forecasts in levels
    forecasts = fer_minimal_fd.forecasts
    outturns = fer_minimal_fd.outturns

    # filter out forecasts in levels
    forecasts_in_pop = forecasts[forecasts["metric"] == "pop"].copy()
    forecasts_in_yoy = forecasts[forecasts["metric"] == "yoy"].copy()

    # create ForecastData objects
    fd_pop = fe.ForecastData(outturns_data=outturns, forecasts_data=forecasts_in_pop, compute_levels=True)
    fd_yoy = fe.ForecastData(outturns_data=outturns, forecasts_data=forecasts_in_yoy, compute_levels=True)

    # get levels
    levels_from_pop = fd_pop.forecasts[fd_pop.forecasts["metric"] == "levels"].copy()
    levels_from_yoy = fd_yoy.forecasts[fd_yoy.forecasts["metric"] == "levels"].copy()

    # check that the levels computed from pop and yoy match the original levels in fer_minimal_fd
    original_levels = forecasts[forecasts["metric"] == "levels"].copy()
    # pop
    df_merged = pd.merge(
        original_levels[["variable", "source", "vintage_date", "date", "value"]],
        levels_from_pop[["variable", "source", "vintage_date", "date", "value"]],
        on=["variable", "source", "vintage_date", "date"],
        suffixes=("_original", "_from_pop"),
    )
    diff = df_merged["value_original"] - df_merged["value_from_pop"]
    max_diff = diff.abs().max()
    assert max_diff < 1e-6, f"Problem with levels from PoP transformation, max difference is {max_diff}"

    # yoy
    df_merged = pd.merge(
        original_levels[["variable", "source", "vintage_date", "date", "value"]],
        levels_from_yoy[["variable", "source", "vintage_date", "date", "value"]],
        on=["variable", "source", "vintage_date", "date"],
        suffixes=("_original", "_from_yoy"),
    )
    diff = df_merged["value_original"] - df_merged["value_from_yoy"]
    max_diff = diff.abs().max()
    assert max_diff < 1e-6, f"Problem with levels from YoY transformation, max difference is {max_diff}"

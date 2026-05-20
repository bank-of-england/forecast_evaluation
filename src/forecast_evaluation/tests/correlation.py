from typing import Union

import numpy as np
import pandas as pd

from forecast_evaluation.data import ForecastData
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.utils import ensure_consistent_date_range, filter_k


def forecast_errors_correlation_analysis(
    data: ForecastData,
    source: Union[None, str, list[str]] = None,
    variable: Union[None, str, list[str]] = None,
    k: int = 12,
    same_date_range: bool = True,
    min_observations: int = 5,
) -> TestResult:
    """
    Compute pairwise Pearson correlations of forecast errors across sources.

    For every unique combination of (variable, metric, frequency, forecast_horizon),
    forecast errors are pivoted into a (date, vintage_date) by source matrix and the
    correlation is computed for every ordered pair of sources (including the
    diagonal, which equals 1). The output is a long-format DataFrame with one row
    per pair, suitable for both heatmap and rolling visualisations.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing the main table with forecast errors.
    source : None, str, or list of str, default=None
        Filter for specific forecast source(s). If None, includes all sources.
    variable : None, str, or list of str, default=None
        Filter for specific variable(s). If None, includes all variables.
    k : int, default=12
        Number of revisions used to define the outturns (passed to filter_k).
    same_date_range : bool, default=True
        If True, restricts all sources to the common vintage range before
        computing correlations (fairer cross-pair comparisons). If False, each
        pair uses its own overlapping observations.
    min_observations : int, default=5
        Minimum number of overlapping observations required for a pair to be
        included in the output.

    Returns
    -------
    TestResult
        TestResult holding a DataFrame with columns:

        - 'variable', 'metric', 'frequency', 'forecast_horizon'
        - 'unique_id'      : anchor source (column 'a')
        - 'unique_id_b'    : partner source (column 'b')
        - 'correlation'    : Pearson correlation of the paired forecast errors
        - 'n_observations' : number of overlapping observations
        - 'start_date'     : earliest paired observation date
        - 'end_date'       : latest paired observation date
    """
    if data._main_table is None or data._main_table.empty:
        raise ValueError("ForecastData main table is not available. Please ensure data has been added and processed.")

    df = data._main_table.copy()

    df = filter_k(df, k)

    if source is not None:
        if isinstance(source, str):
            df = df[df["unique_id"] == source]
        else:
            df = df[df["unique_id"].isin(source)]

    if same_date_range and df["unique_id"].nunique() > 1:
        df = ensure_consistent_date_range(df)

    if variable is not None:
        if isinstance(variable, str):
            df = df[df["variable"] == variable]
        else:
            df = df[df["variable"].isin(variable)]

    group_cols = ["variable", "metric", "frequency", "forecast_horizon"]
    results: list[pd.DataFrame] = []

    for keys, group in df.groupby(group_cols, sort=False):
        var, metric, frequency, horizon = keys

        wide = group.pivot_table(
            index=["date", "vintage_date_forecast"],
            columns="unique_id",
            values="forecast_error",
            aggfunc="mean",
        )

        if wide.shape[1] < 2:
            # Need at least two sources to form any informative pair.
            continue

        # Correlation across sources, plus pair-wise observation counts and date ranges.
        corr = wide.corr(min_periods=min_observations)
        notna = wide.notna().astype(np.int64)
        counts = notna.T.dot(notna)

        # Per-pair date range from the dates index
        date_index = wide.index.get_level_values("date")
        # Cast to int64 ns for fast min/max via masked arrays
        date_arr = date_index.values.astype("datetime64[ns]")

        sources = list(wide.columns)
        rows = []
        for a in sources:
            mask_a = wide[a].notna().to_numpy()
            for b in sources:
                n_obs = int(counts.loc[a, b])
                if n_obs < min_observations:
                    continue
                mask_b = wide[b].notna().to_numpy()
                overlap = mask_a & mask_b
                if not overlap.any():
                    continue
                rows.append(
                    {
                        "variable": var,
                        "metric": metric,
                        "frequency": frequency,
                        "forecast_horizon": horizon,
                        "unique_id": a,
                        "unique_id_b": b,
                        "correlation": corr.loc[a, b],
                        "n_observations": n_obs,
                        "start_date": pd.Timestamp(date_arr[overlap].min()),
                        "end_date": pd.Timestamp(date_arr[overlap].max()),
                    }
                )

        if rows:
            results.append(pd.DataFrame(rows))

    if results:
        results_df = pd.concat(results, ignore_index=True)
    else:
        results_df = pd.DataFrame(
            columns=[
                "variable",
                "metric",
                "frequency",
                "forecast_horizon",
                "unique_id",
                "unique_id_b",
                "correlation",
                "n_observations",
                "start_date",
                "end_date",
            ]
        )

    metadata = {
        "test_name": "forecast_errors_correlation_analysis",
        "parameters": {
            "k": k,
            "same_date_range": same_date_range,
            "min_observations": min_observations,
        },
        "filters": {
            "unique_id": source,
            "variable": variable,
        },
        "date_range": (
            (results_df["start_date"].min(), results_df["end_date"].max()) if len(results_df) > 0 else (None, None)
        ),
    }

    return TestResult(results_df, data.id_columns, metadata)


# Example usage:
if __name__ == "__main__":
    import forecast_evaluation as fe

    forecast_data = fe.ForecastData(load_fer=True)
    res = forecast_errors_correlation_analysis(forecast_data, k=12)
    print(res.to_df().head())

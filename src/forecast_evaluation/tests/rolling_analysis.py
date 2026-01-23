import warnings
from typing import Optional

import pandas as pd
from tqdm import tqdm

from forecast_evaluation.data import ForecastData
from forecast_evaluation.tests.results import TestResult


def rolling_analysis(
    data: ForecastData,
    window_size: int,
    analysis_func: callable,
    analysis_args: dict,
    start_vintage: Optional[str] = None,
    end_vintage: Optional[str] = None,
):
    """
    Perform rolling window analysis using any analysis function.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing the main table
    window_size : int (>0)
        Number of periods to include in each window
    analysis_func : callable
        Analysis function to run on each window (e.g., compute_accuracy_statistics,
        blanchard_leigh_efficiency_test)
    analysis_args : dict
        Additional keyword arguments to pass to the analysis function.
        Should NOT include the 'ForecastData' instance as it will be added automatically.
    start_vintage : str, optional
        Start vintage date (format 'YYYY-MM-DD')
    end_vintage : str, optional
        End vintage date (format 'YYYY-MM-DD')

    Returns
    -------
    TestResult
        TestResult object containing the concatenated results from all windows.
        The DataFrame includes columns from the analysis function plus:

        - 'window_start': Start vintage of the window
        - 'window_end': End vintage of the window
    """

    if window_size <= 0:
        raise ValueError("window_size must be at least one.\n")

    main_table = data._main_table.copy()

    # Set default start and end vintage if not provided
    if start_vintage is None:
        start_vintage = main_table["vintage_date_forecast"].min()
    else:
        if start_vintage not in main_table["vintage_date_forecast"].values:
            raise ValueError(f"start_vintage {start_vintage} not found in data.")

    if end_vintage is None:
        end_vintage = main_table["vintage_date_forecast"].max()
    else:
        if end_vintage not in main_table["vintage_date_forecast"].values:
            raise ValueError(f"end_vintage {end_vintage} not found in data.")

    # Get all unique sorted vintages in the range
    vintages = main_table["vintage_date_forecast"].sort_values().unique()
    vintages = [v for v in vintages if (v >= start_vintage) and (v <= end_vintage)]

    last_vintage = len(vintages) - window_size - 1

    if last_vintage < 0:
        raise ValueError("Window size too large for the available vintages and starting vintage.")

    starting_vintages = vintages[: last_vintage + 1]

    results = []

    for starting_vintage in tqdm(range(len(starting_vintages)), desc="Rolling analysis"):
        data_vintage = data.copy()

        window_start = vintages[starting_vintage]
        window_end = vintages[starting_vintage + window_size - 1]

        data_vintage.filter(start_vintage=window_start, end_vintage=window_end)

        # Combine data with user-provided kwargs
        all_kwargs = {"data": data_vintage, **analysis_args}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = analysis_func(**all_kwargs).to_df()
                result["window_start"] = window_start
                result["window_end"] = window_end
                results.append(result)
        except Exception as e:
            print(f"Error in window {window_start} to {window_end}: {e}")
            continue

    # Concatenate all results into a single DataFrame
    df_results = pd.concat(results, ignore_index=True)

    # rename test col
    if analysis_func.__name__ == "diebold_mariano_table":
        df_results = df_results.rename(columns={"dm_statistic": "test_statistic"})

    if analysis_func.__name__ == "bias_analysis":
        df_results = df_results.rename(columns={"t_statistic": "test_statistic"})

    if analysis_func.__name__ == "weak_efficiency_analysis":
        df_results = df_results.rename(columns={"joint_test_fstat": "test_statistic"})

    # Create metadata
    metadata = {
        "test_name": "rolling_analysis",
        "parameters": {
            "window_size": window_size,
            "analysis_func": analysis_func.__name__,
            "analysis_args": analysis_args,
            "start_vintage": start_vintage,
            "end_vintage": end_vintage,
        },
        "date_range": (df_results["window_start"].min(), df_results["window_end"].max()),
    }

    return TestResult(df_results, data.id_columns, metadata)


# Example usage:
if __name__ == "__main__":
    import pandas as pd

    import forecast_evaluation as fe

    # Initialise with fer ---------
    data = fe.ForecastData(load_fer=True)

    data.filter(variables=["gdpkp"], sources=["mpr"], metrics=["yoy"])

    # Run rolling bias test
    accuracy_results = rolling_analysis(
        data=data, window_size=50, analysis_func=fe.compute_accuracy_statistics, analysis_args={"k": 12}
    )

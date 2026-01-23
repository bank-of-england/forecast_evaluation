from typing import Literal, Optional

import numpy as np
import pandas as pd

from forecast_evaluation.data import ForecastData
from forecast_evaluation.tests.results import TestResult
from forecast_evaluation.tests.rolling_analysis import rolling_analysis

# Giacomini-Rossi (2010) Fluctuation Test Critical Values
# Table I from "Forecast comparisons in unstable environments"
# Journal of Applied Econometrics, 2010

GR_CRITICAL_VALUES = np.array(
    [
        # mu, two-sided 0.05, two-sided 0.10, one-sided 0.05, one-sided 0.10
        [0.1, 3.393, 3.170, 3.176, 2.928],
        [0.2, 3.179, 2.948, 2.938, 2.676],
        [0.3, 3.012, 2.766, 2.770, 2.482],
        [0.4, 2.890, 2.626, 2.624, 2.334],
        [0.5, 2.779, 2.500, 2.475, 2.168],
        [0.6, 2.634, 2.356, 2.352, 2.030],
        [0.7, 2.560, 2.252, 2.248, 1.904],
        [0.8, 2.433, 2.130, 2.080, 1.740],
        [0.9, 2.248, 1.950, 1.975, 1.600],
    ]
)

# Create DataFrame for easier lookup
GR_CRITICAL_VALUES_DF = pd.DataFrame(
    GR_CRITICAL_VALUES, columns=["mu", "two_sided_0.05", "two_sided_0.1", "one_sided_0.05", "one_sided_0.1"]
)


def get_gr_critical_value(
    mu: float, alpha: Literal[0.05, 0.1] = 0.05, test_type: Literal["two-sided", "one-sided"] = "two-sided"
) -> float:
    """
    Get Giacomini-Rossi fluctuation test critical value.

    Table I from Giacomini & Rossi (2010).

    Parameters:
    -----------
    mu : float
        Ratio of rolling window size to out-of-sample size (m/P).
        Must be between 0.1 and 0.9.
    alpha : float
        Significance level. Must be 0.05 or 0.10.
    test_type : str
        Either "two-sided" or "one-sided"

    Returns:
    --------
    float
        Critical value for the fluctuation test (or interpolated value).
    """

    if not 0.1 <= mu <= 0.9:
        error_message = "Can't compute fluctuation tests.\n"
        error_message = f"The ratio between window size and overall sample must be between 0.1 and 0.9, got {mu}."
        raise ValueError(error_message)

    # Construct column name
    col_name = f"{test_type.replace('-', '_')}_{alpha}"

    # Use numpy interpolation
    critical_value = np.interp(mu, GR_CRITICAL_VALUES_DF["mu"].values, GR_CRITICAL_VALUES_DF[col_name].values)

    return critical_value


def fluctuation_tests(
    data: ForecastData,
    window_size: int,
    test_func: callable,
    test_args: dict = {},
    start_vintage: Optional[str] = None,
    end_vintage: Optional[str] = None,
):
    """
    Perform fluctuation tests. A fluctuation test in practice is a
    test performed on a rolling window with adjusted critical values.
    The fluctuation H0 is that the original test statistic is
    not rejected in all windows.

    Parameters
    ----------
    data : ForecastData
        ForecastData object containing the main table
    window_size : int (>0)
        Number of vintages to include in each window
    test_func : callable
        Test function to run on each window. Must be one of:
        'diebold_mariano_table', 'bias_analysis', or 'weak_efficiency_analysis'
    test_args : dict, default={}
        Additional keyword arguments to pass to the test function.
        Should NOT include the 'data' parameter as it will be added automatically.
    start_vintage : str, optional
        Start vintage date (format 'YYYY-MM-DD'). If None, uses the earliest vintage.
    end_vintage : str, optional
        End vintage date (format 'YYYY-MM-DD'). If None, uses the latest vintage.

    Returns
    -------
    TestResult
        TestResult object with rolling test results, including:
        - 'window_start': Start vintage of each window
        - 'window_end': End vintage of each window
        - 'test_statistic': Test statistic for each window
        - 'critical_value_05': Critical value at 5% significance level
        - 'critical_value_10': Critical value at 10% significance level
        - 'reject_05': Boolean indicating rejection at 5% level
        - 'reject_10': Boolean indicating rejection at 10% level
        - 'max_test_statistic': Maximum test statistic across all windows for each group
        - 'reject_max_05': Boolean indicating if max statistic rejects at 5% level
        - 'reject_max_10': Boolean indicating if max statistic rejects at 10% level
    """

    if test_func.__name__ not in ["diebold_mariano_table", "bias_analysis", "weak_efficiency_analysis"]:
        raise ValueError(
            "test_func must be one of 'diebold_mariano_table', 'bias_analysis', or 'weak_efficiency_analysis'"
        )

    if window_size <= 0:
        raise ValueError("window_size must be at least one.\n")

    # Running rolling tests
    tests = rolling_analysis(
        data=data,
        window_size=window_size,
        analysis_func=test_func,
        analysis_args=test_args,
        start_vintage=start_vintage,
        end_vintage=end_vintage,
    ).to_df()

    # Getting new fluctuation test critical values
    # Computate out-of-sample size P
    out_of_sample_size = len(tests["window_start"].sort_values().unique()) + window_size - 1

    # Adjusting critical values for each test
    mu = window_size / out_of_sample_size

    # get critical values
    critical_value_05 = get_gr_critical_value(mu=mu, alpha=0.05, test_type="two-sided")
    critical_value_10 = get_gr_critical_value(mu=mu, alpha=0.10, test_type="two-sided")

    # add critical values to the results
    tests["critical_value_05"] = critical_value_05
    tests["critical_value_10"] = critical_value_10

    # Convert test statistic to absolute value
    tests["test_statistic"] = tests["test_statistic"].abs()

    # add reject/fail cols for 5% and 10%
    tests["reject_05"] = tests["test_statistic"] > tests["critical_value_05"]
    tests["reject_10"] = tests["test_statistic"] > tests["critical_value_10"]

    # get the max stat for each group (variable, metric, frequency, source, forecast_horizon)
    group_cols = ["variable", "metric", "frequency", "unique_id", "forecast_horizon"]
    tests["max_test_statistic"] = tests.groupby(group_cols)["test_statistic"].transform("max")

    tests["reject_max_05"] = tests["max_test_statistic"] > tests["critical_value_05"]
    tests["reject_max_10"] = tests["max_test_statistic"] > tests["critical_value_10"]

    # Create metadata
    metadata = {
        "test_name": "fluctuation_tests",
        "parameters": {
            "window_size": window_size,
            "test_func": test_func.__name__,
            "test_args": test_args,
            "start_vintage": start_vintage,
            "end_vintage": end_vintage,
            "mu": mu,
            "out_of_sample_size": out_of_sample_size,
        },
        "filters": {},
        "date_range": (
            tests["window_start"].min() if len(tests) > 0 else None,
            tests["window_end"].max() if len(tests) > 0 else None,
        ),
    }

    return TestResult(tests, data.id_columns, metadata)


# Example usage:
if __name__ == "__main__":
    import pandas as pd

    import forecast_evaluation as fe

    # Initialise with fer ---------
    data = fe.ForecastData(load_fer=True)

    data.filter(variables=["gdpkp"], metrics=["yoy"], sources=["mpr", "compass unconditional"])

    # Run rolling DM test
    # accuracy_results = fluctuation_tests(
    #     data=data, window_size=50, test_func=fe.diebold_mariano_table, test_args={"benchmark_model": "mpr", "k": 12}
    # )
    # # Use the built-in plot method
    # accuracy_results.plot(variable="gdpkp", horizons=[1, 4, 8])

    # Run bias test
    rolling_bias = fluctuation_tests(data=data, window_size=50, test_func=fe.bias_analysis)
    # Use the built-in plot method
    rolling_bias.plot(horizons=[0, 4, 8, 12])

    # weak efficiency with MZ regression
    # bias_results = fluctuation_tests(data=data, window_size=50, test_func=fe.weak_efficiency_analysis)

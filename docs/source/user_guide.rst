User Guide
==========

This guide introduces the main workflow for the ``forecast_evaluation`` package.
It complements the API reference and follows the same usage patterns shown in the
example notebook.

You can browse the full worked example in the
`example notebook <https://github.com/bank-of-england/forecast_evaluation/blob/main/notebooks/example_notebook.ipynb>`_.

The package is designed around a small number of core tasks:

* loading forecast and outturn data
* filtering data to a consistent evaluation sample
* visualising forecasts, errors, and revisions
* computing accuracy, bias, and efficiency diagnostics
* comparing models with benchmark forecasts
* exploring results in an interactive dashboard

Quick Start
-----------

.. code-block:: python

   import forecast_evaluation as fe
   import pandas as pd

You can either load the built-in Forecast Evaluation Report data,
or create a dataset from your own forecasts and outturns.

Load the built-in dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   data = fe.ForecastData(load_fer=True)

This is the quickest way to start exploring the package and reproducing the examples.

Create a dataset from your own data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   forecast_data = fe.ForecastData(
       forecasts_data=forecasts_dataframe,
       outturns_data=outturns_dataframe,
   )

The central object is :class:`forecast_evaluation.data.ForecastData`. It stores raw
forecasts, outturns, transformed series, and the main evaluation table used by the
analysis and plotting functions.

Data Requirements
-----------------

Forecasts and outturns are provided as ``pandas`` DataFrames.

Forecasts must include the standard identification columns together with a value:

.. code-block:: text

   date, vintage_date, variable, source, frequency, forecast_horizon, value

Outturns use the same structure but do not require a ``source`` column.

An example forecast table looks like this:

.. code-block:: text

              date vintage_date variable source frequency  forecast_horizon  value
   0    2014-12-31   2015-03-31      gdp   BVAR         Q                -1    100
   1    2015-03-31   2015-03-31      gdp   BVAR         Q                 0    101
   2    2015-06-30   2015-03-31      gdp   BVAR         Q                 1    102
   3    2015-09-30   2015-03-31      gdp   BVAR         Q                 2    103

The package supports different forecast metrics such as ``levels``, ``pop`` (period-on-period), and
``yoy`` (year-on-year). When required, transformations between these representations are computed
internally when enough outturn history is available.

Working With ForecastData
-------------------------

Load data and filter the sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example notebook starts by loading FER data and filtering the variables used in
subsequent analysis:

.. code-block:: python

   data = fe.ForecastData(load_fer=True)
   data.filter(variables=["gdpkp", "cpisa", "aweagg"])

The :meth:`forecast_evaluation.data.ForecastData.filter` method can restrict the
sample by:

* forecast dates via ``start_date`` and ``end_date``
* vintage dates via ``start_vintage`` and ``end_vintage``
* variables, metrics, sources, and frequencies
* a custom filtering function through ``custom_filter``

For example, you can exclude the COVID period from the analysis using the built-in
``covid_filter`` helper:

.. code-block:: python

   data_covid_filtered = data.copy()
   data_covid_filtered.filter(custom_filter=fe.covid_filter)

If you need to reset to the original unfiltered data, use:

.. code-block:: python

   data.clear_filter()

Inspect the stored tables
~~~~~~~~~~~~~~~~~~~~~~~~~

Useful accessors on a ``ForecastData`` object are:

* ``data.df`` for the main evaluation table
* ``data.forecasts`` for the transformed forecast table
* ``data.outturns`` for the transformed outturn table
* ``data.id_columns`` for the identification columns used to distinguish models

The ``summary()`` method prints a compact overview of the loaded variables, date
range, vintages, and horizons.

Adding forecasts and labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can extend an existing dataset with additional forecasts.

The notebook demonstrates adding a new label column that is treated as part of the
forecast identifier:

.. code-block:: python

   sample_forecasts = fe.create_sample_forecasts()
   sample_forecasts["extra label"] = "Model family A"

   data_example_extra_columns = fe.ForecastData(load_fer=True)
   data_example_extra_columns.add_forecasts(
       sample_forecasts,
       extra_ids=["extra label"],
   )

This is useful when you want to separate forecasts by model family, conditioning
assumption, scenario, or other metadata beyond ``source``.

Visualisation Workflow
----------------------

The package includes plotting functions for vintages, forecast errors, outturns,
revisions, and rolling diagnostics. The examples below mirror the notebook.

Recent forecast errors against their historical distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dates_to_highlight = pd.date_range(
       start="2022-01-01",
       end="2024-12-31",
       freq="QE",
   )

   fe.plot_forecast_error_density(
       data=data,
       horizon=4,
       variable="cpisa",
       metric="yoy",
       frequency="Q",
       source="mpr",
       k=12,
       highlight_dates=dates_to_highlight,
   )

Vintage plots
~~~~~~~~~~~~~

.. code-block:: python

   fe.plot_vintage(
       data=data,
       variable="cpisa",
       forecast_source=["mpr", "compass conditional", "bvar conditional"],
       frequency="Q",
       vintage_date="2020-03-31",
       metric="yoy",
   )

Hedgehog charts
~~~~~~~~~~~~~~~

.. code-block:: python

   fe.plot_hedgehog(
       data=data,
       variable="cpisa",
       forecast_source="mpr",
       metric="yoy",
       frequency="Q",
       k=12,
       convert_to_percentage=True,
   )

Forecast errors over time
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fe.plot_errors_across_time(
       data_covid_filtered,
       variable="gdpkp",
       metric="yoy",
       ma_window=4,
       error="raw",
       sources=["mpr", "baseline ar(p) model"],
       k=12,
       horizons=[0, 4],
   )

Forecast errors by vintage or horizon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fe.plot_forecast_errors(
       data=data,
       variable="cpisa",
       metric="yoy",
       frequency="Q",
       source="mpr",
       vintage_date_forecast="2022-03-31",
       k=12,
       convert_to_percentage=True,
   )

   fe.plot_forecast_errors_by_horizon(
       data=data,
       variable="cpisa",
       metric="yoy",
       frequency="Q",
       source="mpr",
       k=12,
       convert_to_percentage=True,
   )

Outturns and revisions
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fe.plot_outturns(
       data=data,
       variable="gdpkp",
       metric="yoy",
       frequency="Q",
       k=[0, 12],
       fill_k=True,
       convert_to_percentage=True,
   )

   fe.plot_outturn_revisions(
       data=data,
       variable="gdpkp",
       metric="yoy",
       frequency="Q",
       k=[4, 12],
       ma_window=4,
       fill_k=True,
       convert_to_percentage=True,
   )

Forecast Evaluation Methods
---------------------------

Most analytical functions return a
:class:`forecast_evaluation.tests.results.TestResult` object. These results can be
converted to a DataFrame and, in many cases, plotted directly with ``.plot()``.

Accuracy statistics
~~~~~~~~~~~~~~~~~~~

Use :func:`forecast_evaluation.compute_accuracy_statistics` to calculate statistics
such as RMSE, mean absolute error, root median square error, and observation counts.

.. code-block:: python

   accuracy_results = fe.compute_accuracy_statistics(data=data, k=12)

   accuracy_results.plot(
       variable="cpisa",
       metric="yoy",
       frequency="Q",
       statistic="rmse",
       convert_to_percentage=True,
   )

Comparing to a benchmark
~~~~~~~~~~~~~~~~~~~~~~~~

You can compare model performance relative to a benchmark using summary functions
and companion plots.

.. code-block:: python

   accuracy_comparison = fe.compare_to_benchmark(
       df=accuracy_results,
       benchmark_model="baseline ar(p) model",
       statistic="rmse",
   )

   fe.plot_compare_to_benchmark(
       df=accuracy_results,
       variable="cpisa",
       metric="yoy",
       frequency="Q",
       benchmark_model="baseline ar(p) model",
       statistic="rmse",
   )

To create a compact table for selected horizons:

.. code-block:: python

   comparison_table = fe.create_comparison_table(
       df=accuracy_results.to_df(),
       variable="cpisa",
       metric="yoy",
       frequency="Q",
       benchmark_model="baseline ar(p) model",
       statistic="rmse",
       horizons=[0, 1, 2, 4, 8, 12],
   )

Relative accuracy tests
~~~~~~~~~~~~~~~~~~~~~~~

The package includes Diebold-Mariano testing and rolling-window extensions.

.. code-block:: python

   diebold_mariano_results = fe.diebold_mariano_table(
       data=data,
       benchmark_model="mpr",
   )

For rolling analysis, first create a focused dataset and then pass the test function
to :func:`forecast_evaluation.rolling_analysis`.

.. code-block:: python

   forecast_data_dm_rolling = data.copy()
   forecast_data_dm_rolling.filter(
       variables=["gdpkp"],
       metrics=["yoy"],
       sources=["mpr", "baseline random walk model"],
   )

   rolling_dm = fe.rolling_analysis(
       data=forecast_data_dm_rolling,
       window_size=40,
       analysis_func=fe.diebold_mariano_table,
       analysis_args={"benchmark_model": "mpr"},
   )

   rolling_dm.plot(variable="gdpkp", horizons=[0, 4])

Fluctuation tests
~~~~~~~~~~~~~~~~~

Fluctuation tests provide a multiple-window diagnostic that is robust to repeated
rolling analysis.

.. code-block:: python

   rolling_dm_fluctuation = fe.fluctuation_tests(
       data=forecast_data_dm_rolling,
       window_size=40,
       test_func=fe.diebold_mariano_table,
       test_args={"benchmark_model": "mpr"},
   )

   rolling_dm_fluctuation.plot(variable="gdpkp", horizons=[0, 4])

Bias and efficiency analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The notebook also demonstrates the main econometric diagnostics provided by the
package.

.. code-block:: python

   bias_results = fe.bias_analysis(data=data, source="mpr", k=12, verbose=False)
   bias_results.plot(variable="aweagg", source="mpr", metric="yoy", frequency="Q")

   rolling_bias = fe.rolling_analysis(
       data=data_gdp,
       window_size=40,
       analysis_func=fe.bias_analysis,
       analysis_args={"k": 12},
   )

   bl_results = fe.blanchard_leigh_horizon_analysis(
       data=data,
       source="mpr",
       outcome_variable="cpisa",
       outcome_metric="yoy",
       instrument_variable="gdpkp",
       instrument_metric="yoy",
   )

   weak_efficiency_results = fe.weak_efficiency_analysis(
       data=data,
       source="mpr",
       k=12,
       verbose=False,
   )

Revisions analysis
~~~~~~~~~~~~~~~~~~

Revisions can be analysed directly through dedicated tests and plots.

.. code-block:: python

   revisions_correlation_results = fe.revisions_errors_correlation_analysis(
       data=data,
       source="mpr",
       k=12,
   )

   revisions_predictable_results = fe.revision_predictability_analysis(
       data=data,
       frequency="Q",
       n_revisions=5,
   )

   fe.plot_average_revision_by_period(
       data=data,
       source="mpr",
       variable="gdpkp",
       metric="yoy",
       frequency="Q",
   )

Adding Benchmark Forecasts
--------------------------

You can augment a ``ForecastData`` object with simple benchmark models using
:meth:`forecast_evaluation.data.ForecastData.add_benchmarks`.

Supported benchmark families are ``AR`` (autoregressive) and ``random_walk``.

.. code-block:: python

   data.add_benchmarks(metric="pop", models=["AR", "random_walk"])

Optional arguments let you restrict the benchmark generation to selected variables
or frequencies, control the number of forecast periods, and supply an estimation
start date.

Density Forecasts
-----------------

For probabilistic forecasts with quantiles, use
:class:`forecast_evaluation.data.DensityForecastData`, which extends
``ForecastData``.

Density forecast input must include a ``quantile`` column with values between 0 and 1.

.. code-block:: python

   density_df = fe.create_sample_density_forecasts()
   density_data = fe.DensityForecastData(forecasts_data=density_df)

You can also add density forecasts to an existing object:

.. code-block:: python

   density_data = fe.DensityForecastData()
   density_data.add_density_forecasts(density_df)

Density forecast objects retain the standard forecast and outturn workflow while also
exposing a ``density_forecasts`` table for quantile-level analysis.

Dashboard
---------

The package includes an interactive dashboard for exploring forecasts, errors, and
analysis outputs.

Run it from a ``ForecastData`` object:

.. code-block:: python

   data.run_dashboard()

When working inside a notebook, you can embed the dashboard in the notebook output:

.. code-block:: python

   data.run_dashboard(from_jupyter=True)

Further Reading
---------------

For a worked example covering the main plotting and testing functions, see the
`example notebook <https://github.com/bank-of-england/forecast_evaluation/blob/main/notebooks/example_notebook.ipynb>`_
in the repository. For function signatures and parameter-level details, refer to
the API reference.

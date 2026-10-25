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

Outturns use the same ``date``, ``vintage_date``, ``variable``, ``frequency``, and ``value`` columns,
but do not require ``source`` or ``forecast_horizon``.

An example forecast table looks like this:

.. code-block:: text

              date vintage_date variable source frequency  forecast_horizon  value
   0    2015-03-31   2015-03-31      gdp   BVAR         Q                 0    101
   1    2015-06-30   2015-03-31      gdp   BVAR         Q                 1    102
   2    2015-09-30   2015-03-31      gdp   BVAR         Q                 2    103

``forecast_horizon`` is the information horizon supplied by the forecaster. It is based on the
forecast target date (``date``) and the last target observation used for estimation:
``forecast_horizon = forecast_target_date - date_last_target_obs_used_for_estimation - 1``,
measured in periods at the stated frequency. Therefore, horizon ``0`` is the first period after
the last observation used for estimation; it is not inferred from ``vintage_date``.
Negative horizons, including ``-1``, may be supplied for forecast/outturn consistency checks, but
are removed before forecasts are stored for evaluation.

The two horizons
~~~~~~~~~~~~~~~~

Results tables and plots use a different horizon from the one you supply, so it is worth keeping
the two apart:

``horizon``
   The calendar distance from the forecast vintage to the target date, ``date - vintage_date``,
   in periods. This is the dimension every analysis groups, filters and reports on, and the one
   ``TestResult.filter(horizon=...)`` selects. It is ``0`` for a same-period forecast, ``1`` for
   one period ahead, and ``-1`` for a backcast. Because it depends only on when a forecast was
   made, it is comparable across sources that conditioned on different amounts of data.

``forecast_horizon``
   The information horizon you supply, as described above. It is not a reporting dimension: its
   only role is to set the lag length of the HAC standard errors in the regression-based tests,
   because it is what bounds the serial correlation in forecast errors.

The two coincide when forecasts are published as soon as the previous period's outturn is
released, and differ when there is a publication lag. Where the lag changes over a sample, one
calendar horizon pools observations made at several information horizons; the lag length is then
taken from the largest information horizon in the group and reported in the ``hac_maxlags``
column of the results. See :doc:`methodology` for what pooling implies for interpretation.

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

Nowcasting
----------

:class:`forecast_evaluation.data.NowcastData` extends ``ForecastData`` for
evaluating forecasts that are released multiple times *within* a single target
period, e.g. weekly nowcast vintages of a quarterly variable. It uses
integer-period forecast horizons (``h=-1`` backcast, ``h=0`` nowcast, ``h=1``
one-period-ahead) so that many intra-period vintages sharing the same horizon
can be pooled for accuracy statistics.

Key differences from ``ForecastData``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``first_forecast_horizon`` defaults to ``-1`` so that backcasts are
  retained by default.
* ``default_k`` defaults to ``0``, i.e. the first outturn release, rather
  than the fully revised outturn.
* The ``load_fer`` and ``outturn_vintages`` constructor arguments used by
  ``ForecastData`` are not exposed.

Creating a ``NowcastData`` instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package ships sample nowcast data with weekly forecast vintages and
realistic outturn publication lags and revisions:

.. code-block:: python

   nowcast_outturns = fe.create_sample_nowcast_outturns()
   nowcast_forecasts = fe.create_sample_nowcast_forecasts()

   nowcast_data = fe.NowcastData(
       outturns_data=nowcast_outturns,
       forecasts_data=nowcast_forecasts,
   )

Plotting nowcast vintages
~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`forecast_evaluation.plot_nowcasts` shows how forecasts for a single
target quarter evolved over successive vintages, with one line per forecast
identity (``source`` plus any ``extra_ids``) and the selected outturn
maturity drawn as a horizontal reference. Pass ``k=0`` to reference the first
outturn release, which is usually the relevant comparison for nowcasts:

.. code-block:: python

   fe.plot_nowcasts(
       data=nowcast_data,
       variable="gdp",
       target_date="2024-06-30",
       frequency="Q",
       metric="levels",
       k=0,
   )

Intra-period accuracy and bias
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`forecast_evaluation.compute_intra_period_accuracy` and
:func:`forecast_evaluation.compute_intra_period_bias` group forecast errors by
a within-period time axis rather than by calendar vintage or forecast
horizon, which suits densely-spaced intra-period vintages. Their plotting
counterparts, :func:`forecast_evaluation.plot_intra_period_accuracy` and
:func:`forecast_evaluation.plot_intra_period_bias`, share the same arguments.

The ``axis`` argument selects the time axis:

* ``axis="period_end"`` (default) measures days from the forecast vintage to
  the end of the target period.
* ``axis="publication"`` measures days from the forecast vintage to the
  release of the selected outturn vintage.

If ``k`` is left as ``None`` it resolves to ``data.default_k`` (``0`` for
``NowcastData``).

.. code-block:: python

   intra_period_accuracy = fe.compute_intra_period_accuracy(
       data=nowcast_data,
       variable="gdp",
       metric="levels",
       frequency="Q",
       statistic="rmse",
       axis="period_end",
   )

   fe.plot_intra_period_accuracy(
       data=nowcast_data,
       variable="gdp",
       metric="levels",
       frequency="Q",
       statistic="rmse",
       axis="period_end",
   )

   intra_period_bias = fe.compute_intra_period_bias(
       data=nowcast_data,
       variable="gdp",
       metric="levels",
       frequency="Q",
       axis="publication",
   )

   fe.plot_intra_period_bias(
       data=nowcast_data,
       variable="gdp",
       metric="levels",
       frequency="Q",
       axis="publication",
   )

Unsupported ``ForecastData`` functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some ``ForecastData`` methods and analyses do not apply to intra-period
nowcast vintages and are unsupported on ``NowcastData``. The following raise
``NotImplementedError``:

* FER data loading and filtering (``add_fer_data``, ``add_fer_outturns``,
  ``add_fer_forecasts``, ``filter_fer``)
* pseudo-vintage creation (``create_pseudo_vintages``)
* benchmark generation (``add_benchmarks``)

The following raise ``ValueError``:

* efficiency, Blanchard-Leigh, revision-predictability, and
  revision-error-correlation analyses

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

Density forecast input uses the point forecast columns plus a ``quantile`` column.
Quantile levels must be finite and in the inclusive range [0, 1]; density ``value``
entries must also be finite and non-null. Add outturns before density forecasts.
The sample density generator is available
from ``forecast_evaluation.data.sample_data``:

.. code-block:: python

   from forecast_evaluation.data.sample_data import create_sample_density_forecasts

   density_df = create_sample_density_forecasts()
   density_data = fe.DensityForecastData(
       outturns_data=fe.create_sample_outturns(),
       forecasts_data=density_df,
   )

You can also add density forecasts to an existing object:

.. code-block:: python

   density_data = fe.DensityForecastData(outturns_data=fe.create_sample_outturns())
   density_data.add_density_forecasts(density_df)

In a density-only object, ``density_forecasts`` and ``outturns`` are populated,
while ``forecasts`` and the point-analysis table ``df`` remain empty until explicit
conversion. ``add_density_forecasts`` stores the supplied density metrics; it does
not automatically derive additional ``pop`` or ``yoy`` densities, even when the
constructor uses ``compute_levels=True``.

Outturn preparation derives growth from levels where needed, but retains supplied
``pop`` and ``yoy`` outturn values. A mismatch against levels-derived growth emits
a warning rather than replacing the supplied value. Library-created nowcast
snapshot rows marked ``_aligned`` are exempt from that mismatch warning.

Quantile crossing and filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The constructor and ``add_density_forecasts`` accept ``crossing``. The default,
``"rearrange"``, sorts values within each crossed density and warns once per call.
``"error"`` rejects crossings; ``"ignore"`` keeps supplied values. Equal adjacent
quantile values are valid. These policies do not relax finite-value or
quantile-range validation.

``filter`` applies to density forecasts, point forecasts, outturns and ``df`` by
default, including when no point forecasts exist. Use
``filter_point_forecasts=False`` or ``filter_density_forecasts=False`` to leave
the corresponding views unchanged. Filtering preserves the backing density
data; ``clear_filter()`` restores all views.

Converting to point forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conversion adds a selected quantile to the same object's point forecast tables,
using the currently filtered densities:

.. code-block:: python

   density_data.to_point_forecast(method="median")

``method`` defaults to ``"median"`` (0.5), or accepts a numeric quantile level or
its string representation. ``"mean"`` is not implemented. If the requested level
is missing, ``on_missing="nearest"`` (the default) warns and selects the nearest
stored level common to all dates of each forecast path. Equally near levels
resolve to the lower one. Use ``on_missing="raise"`` to reject missing levels
during conversion; this option does not change ingestion or plotting.
Missing-quantile interpolation for conversion remains a deferred follow-up and
is not implemented.

Converted rows retain their source unless ``source_suffix`` is provided. For
example, distinct suffixes allow several selected quantiles to coexist as point
forecasts. ``compute_levels`` defaults to the constructor's setting. Any derived
``pop`` or ``yoy`` point forecasts are growth rates of the selected level-quantile
path, not quantiles of the marginal growth distribution.

Sampling densities
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   samples = density_data.sample_from_density(n_samples=10000, random_state=42)

Sampling uses the currently filtered densities and a private random generator.
Pass ``random_state`` for reproducibility. Draws are independent across densities:
``sample_id`` is an index, not a joint path across dates or variables.

The inverse quantile function is linearly interpolated between stored levels.
Outside that range it is clamped to the endpoint values, not extrapolated, so
tail mass accumulates at those endpoints and can narrow the sampled distribution.
A warning is issued if an outer quantile is farther than 0.02 from 0 or 1.
Sampling raises ``ValueError`` if no densities remain or any density has fewer
than two distinct quantile levels.

Plotting densities
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig, ax = density_data.plot_density_vintage(
       variable="gdpkp", vintage_date="2025-12-31", return_plot=True
   )

``quantiles`` and ``return_plot`` are keyword-only. Quantiles default to
``(0.16, 0.5, 0.84)`` and are sorted; if a requested level is not stored, the
nearest stored level is used with a warning, not interpolation. The plot shades
one outer fan band for two or more levels, with other levels shown as faint
dashed lines. The requested median is solid; otherwise an odd-length list uses
its middle level as the solid line. An even-length list without the median has
no solid central line.

Density forecasts must already contain the requested ``metric``. Outturns use
the latest available outturn vintage: dates before the forecast vintage are
solid and dates at or after it are dashed. Prior outturns are included by default;
``outturn_start_date`` sets their first displayed date. Invalid or duplicate
quantile requests, or requests resolving to the same stored level, are rejected.

Existing forecast error distribution plots describe point forecast errors,
including errors after explicit conversion, not predictive density calibration.
Density calibration and scoring are covered next.

Fitting densities to quantiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluation works on a continuous density fitted to each stored set of quantiles:

.. code-block:: python

   fitted = density_data.fit_parametric_density("skew_t")
   fitted.params        # one row per density: keys plus loc, scale, shape, df
   fitted.diagnostics   # levels used, fit errors, success flag
   fitted.moments()     # mean, variance, skewness

``method`` is ``"skew_t"`` (Azzalini skew-t), ``"two_piece_normal"``, ``"normal"``
or ``"mpz"`` (the exact semi-parametric distribution through the quantiles, with
Gaussian tails). Requested levels map to the nearest stored level without
interpolation, warning if more than 0.01 away. A warning is also issued when the
fit misses a stored quantile by more than ``approx_tol`` times the fitted
interquartile range.

Densities with known parameters, such as published fan charts, can be evaluated
without fitting:

.. code-block:: python

   params = keys.assign(mode=..., sigma1=..., sigma2=...)  # keys: density_forecasts key columns
   fitted = fe.FittedDensityForecasts.from_params(params, "two_piece_normal", data=density_data)

Calibration, scoring and sharpness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Outturns are matched exactly as for point forecasts: ``k`` selects the outturn
revision (default ``data.default_k``; ``"latest"`` uses the latest vintage), and
results are grouped by variable, source, metric, frequency and horizon
(``target_minus_vintage``). Every function returns a ``TestResult``.

.. code-block:: python

   pit = fe.compute_pit(fitted, k="latest")
   fe.knuppel_test(fitted)                  # PIT moments, HAC
   fe.mitchell_weale_test(fitted)           # censored moments + coverage, alpha=0.10
   rs = fe.rossi_sekhposyan_test(fitted, seed=0)
   fe.galvao_mantoan_mitchell_test(fitted, seed=0)   # joint test across horizons
   crps = fe.crps_analysis(fitted)                   # mean CRPS by horizon
   fe.crps_analysis(fitted, weight="left")           # downside-weighted CRPS
   fe.sharpness_analysis(fitted)                     # interval widths

   pit.plot()                                        # PIT histogram
   pit.plot(kind="ecdf", rs_result=rs)               # PIT ECDF with RS bands
   crps.plot()

``weight`` is ``"none"``, ``"centre"``, ``"tails"``, ``"left"``, ``"right"``,
``"heavy_tails"`` or a callable of the quantile level. Rossi-Sekhposyan and
Galvao-Mantoan-Mitchell critical values come from a block bootstrap; pass
``seed`` for reproducible results. The path test needs PITs for every
horizon from the same forecast origins and treats the first horizon as one step
ahead. See :doc:`methodology` for definitions.

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

``DensityForecastData.run_dashboard()`` supports density plots and outturn views
without point forecasts. Point-analysis tabs display placeholders until point
forecasts are added or ``to_point_forecast()`` is called. Converting a quantile
enables point analysis; it does not evaluate the full density.

Further Reading
---------------

For a worked example covering the main plotting and testing functions, see the
`example notebook <https://github.com/bank-of-england/forecast_evaluation/blob/main/notebooks/example_notebook.ipynb>`_
in the repository. For function signatures and parameter-level details, refer to
the API reference.

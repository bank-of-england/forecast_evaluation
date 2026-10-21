.. _methodology:

Forecast Evaluation Methodology
================================

Overview
--------

This document provides technical detail on the Bank of England's enhanced forecast evaluation approach. The methodologies described here form the foundation of the forecast evaluation toolkit and are used to assess the Bank's forecast performance across multiple dimensions, enabling continuous learning from forecast errors.

**Source Documentation**

This documentation is derived from the Bank of England's Macro Technical Paper: `Learning from forecast errors: the Bank's enhanced approach to forecast evaluation <https://www.bankofengland.co.uk/macro-technical-paper/2026/learning-from-forecast-errors-the-Banks-enhanced-approach-to-forecast-evaluation>`_

Evaluation Approaches
---------------------

The evaluation framework employs three complementary approaches:

**Approach 1: Long-term Statistical Evaluation**

Characterizes forecast performance over extended historical periods using standard statistical tests:

- Accuracy metrics (RMSE, MAE)
- Unbiasedness tests
- Efficiency tests
- Benchmarking against model alternatives

**Approach 2: Benchmark Comparisons**

Compares Bank forecasts against a rich set of benchmark models to control for economic conditions and shock effects:

- AR(p) autoregressive models
- Bayesian VAR models
- COMPASS DSGE model
- Random walk models

**Approach 3: Targeted Analysis of Recent Errors**

Interrogates specific recent forecast errors and their drivers through complementary techniques:

- Distributional analysis of errors
- Rolling-window fluctuation tests
- Data revision analysis

Density Forecast Interpretation
-------------------------------

``DensityForecastData`` stores predictive quantiles for individual target dates.
Loading densities does not itself populate point forecasts or the point-error
analysis table, and does not derive additional metric densities. Explicit
``to_point_forecast`` conversion selects a quantile path for the point evaluation
methods described below. It defaults to the nearest stored quantile with a
warning when a requested level is absent; ``on_missing="raise"`` makes this a
conversion error. Interpolation of missing conversion quantiles is deferred,
not implemented.

Growth rates calculated from a selected level-quantile path are not generally
quantiles of the marginal growth distribution. Matching quantile ranks across
dates implicitly assumes perfect dependence for that interpretation; the stored
marginal densities do not specify such a joint distribution.

Likewise, ``sample_from_density`` draws independently across densities, not joint
paths. It linearly interpolates the inverse quantile function within the stored
range and clamps tails to the endpoint values. This places probability mass at
the endpoints and can understate dispersion; outer levels farther than 0.02 from
0 or 1 trigger a warning. Such draws do not recover temporal or cross-variable
dependence for growth rates or aggregates.

Density-specific scoring and calibration diagnostics, including CRPS and PIT,
are not implemented. The existing forecast error distribution plots concern
point forecast errors, not predictive density calibration. Accuracy, bias and
efficiency results after conversion therefore assess the selected point path,
not the entire density forecast.

Forecast Error Definition and Data
-----------------------------------

Forecast errors are computed as the difference between outturns and forecasts:

.. math::

    \varepsilon(y)_{t|t-h} = y_t - \hat{y}_{t|t-h}

where:

- :math:`y_t` is the realized value (outturn)
- :math:`\hat{y}_{t|t-h}` is the forecast made :math:`h` quarters ahead
- :math:`h` is ``horizon``, the calendar distance from the forecast vintage to the target date
  (``date - vintage_date``, in periods). Every analysis groups and reports on this horizon:
  :math:`h=0` is a same-period forecast, :math:`h=1` is one period ahead, and :math:`h=-1` is a
  backcast. It depends only on when a forecast was made, so it is comparable across sources that
  conditioned on different amounts of data.

**Horizons and serial correlation**

A second horizon, ``forecast_horizon``, is supplied by the forecaster and measures the
*information* distance: the periods between the last outturn used for estimation and the target
date, zero-based (:math:`0` is the first period after that last observation). The two horizons
coincide only when forecasts are published as soon as the previous period's outturn is released;
a publication lag makes the information horizon the longer of the two.

``forecast_horizon`` plays no part in grouping. Its sole role is to set the bandwidth of the HAC
(Newey-West) standard errors in the regression-based tests — bias, weak and strong efficiency,
revisions-errors correlation, and Diebold-Mariano — because it is the information horizon, not the
calendar one, that governs the overlap in forecast errors: an :math:`f`-period information horizon
produces errors following an MA(:math:`f`) process.

When the publication lag changes over a sample, one calendar horizon pools observations made at
several information horizons. A concatenation of MA(:math:`f`) segments has autocovariances
bounded by the largest :math:`f` present, so the bandwidth is set to the maximum
``forecast_horizon`` in the pooled sample. This keeps the estimator consistent and is merely
conservative in the rows where the lag shifted. The bandwidth actually used is reported in the
``hac_maxlags`` column of each results table (``hac_bandwidth`` for Blanchard-Leigh).

For the Diebold-Mariano test, the loss differential mixes both forecasters' errors, so the
bandwidth follows the longer information horizon of the pair. Forecasts are compared only when
made at the same vintage for the same target; two sources that used different amounts of outturn
data at that vintage are still compared, since the calendar horizon is equal by construction.

.. note::

   Pooling estimates an average. A bias test at calendar horizon :math:`h` tests whether the
   *average* error is zero, so offsetting biases either side of a publication-lag change (say
   :math:`+0.5` before and :math:`-0.5` after) can read as unbiased. Use
   :func:`~forecast_evaluation.fluctuation_tests` to detect instability over time. Relatedly, a
   horizon present in only one lag regime covers only part of the sample, so its rolling series
   starts or stops mid-sample and the globally computed window ratio makes the Giacomini-Rossi
   critical values approximate for that horizon.

**Data Vintages**

The toolkit accounts for data revisions by comparing forecasts against outturns published :math:`k` quarters after initial release. By default, :math:`k=12` is used, ensuring:

- GDP data has been fully "balanced" at least twice in the ONS Blue Book
- Sufficient time has elapsed for material revisions
- Comparability with original forecast conditions

When the final vintage is unavailable, the latest published data is used.

Statistical Evaluation Metrics
------------------------------

Accuracy Assessment
^^^^^^^^^^^^^^^^^^^

Forecast accuracy is evaluated using two complementary metrics:

**Root Mean Squared Error (RMSE)**

.. math::

    \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\varepsilon_i^2}

where :math:`\varepsilon_i` represents the :math:`i`-th forecast error and :math:`N` is the number of observations.

RMSE penalises larger errors more heavily, consistent with a quadratic loss function.

**Mean Absolute Error (MAE)**

.. math::

    \text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\varepsilon_i|

MAE is less sensitive to outliers and equals the average absolute error size.

**Relative Accuracy: RMSE Ratio**

.. math::

    \text{RMSE Ratio} = \frac{\text{RMSE}_{\text{Forecast A}}}{\text{RMSE}_{\text{Forecast B}}}

A ratio greater than 1.0 indicates that the denominator forecast performs better.

**Diebold-Mariano Test**

The Diebold-Mariano (DM) test evaluates whether differences in forecast accuracy between two models are statistically significant. The test assesses:

- **Null Hypothesis**: Expected difference in forecast loss is zero (forecasts have equal accuracy)
- **Test Statistic**: Computed using HAC variance estimators to account for autocorrelation at multi-step horizons
- **Interpretation**: Positive values indicate the base model performs better; negative values indicate worse performance

The Harvey (1997) correction is applied to account for small sample sizes.

Unbiasedness Assessment
^^^^^^^^^^^^^^^^^^^^^^^

A forecast is unbiased if it does not systematically over- or under-predict outcomes. Bias testing uses the following regression:

.. math::

    \varepsilon_{t|t-h} = \beta + u_t

where:

- :math:`\varepsilon_{t|t-h}` is the forecast error (outturn minus forecast)
- :math:`\beta` is the bias coefficient (sample mean of errors)
- :math:`u_t` is the error term

**Interpretation**:

- :math:`\beta > 0`: Forecasts systematically underestimate outcomes
- :math:`\beta < 0`: Forecasts systematically overestimate outcomes
- :math:`\beta \approx 0`: Forecasts are unbiased

Statistical significance is tested using a t-test with HAC standard errors, with lag order set to :math:`h` (forecast horizon in quarters).

Efficiency Assessment
^^^^^^^^^^^^^^^^^^^^^

Efficient forecasts utilize all available information optimally. The toolkit implements both weak and strong efficiency tests.

**Weak Efficiency: Mincer-Zarnowitz Test**

The Mincer-Zarnowitz regression tests whether anticipated changes are fully incorporated:

.. math::

    y_t = \alpha + \beta \hat{y}_{t|t-h} + u_t

Under the null hypothesis of efficiency:

- :math:`\alpha = 0` (no constant term)
- :math:`\beta = 1` (unit coefficient)

**Interpretation**:

- :math:`\alpha \neq 0`: Systematic bias in forecasts
- :math:`\beta \neq 1`: Over- or under-reaction to available information

**Strong Efficiency: Blanchard-Leigh regressions**

This approach examines whether cross-variable relationships are correctly calibrated in forecasts. It tests whether misforecasts in one variable predict misforecasts in related variables, indicating potential miscalibration of economic relationships.

Benchmark Models
^^^^^^^^^^^^^^^^

The toolkit benchmarks forecasts against several model classes:

**AR(p) Models**

- Univariate autoregressive models estimated on real-time data
- Provides a mechanical statistical baseline

**Bayesian VAR**

- Multivariate model capturing key economic relationships

**COMPASS DSGE Model**

- Bank's workhorse dynamic stochastic general equilibrium model

**Random Walk**

- Model assuming no change from current value
- Most naive baseline available

Benchmarking Strategy
^^^^^^^^^^^^^^^^^^^^^

Benchmarks are constructed on a **real-time basis**, using only information available when each forecast was made. This approach ensures fair comparison (no look-ahead bias).


Density Forecast Evaluation
---------------------------

Density forecasts are evaluated as continuous distributions fitted to their stored quantiles, following the forecast evaluation in Aikman et al. (2026). Fitted parameters are the single source of truth: PITs, scores and interval widths use each family's exact cdf and quantile function.

Fitting
^^^^^^^

Each family is fitted by least squares to the stored quantiles at the matched levels. Location and scale have a closed form given the shape, which is found by a grid search refined with the exact quantile function. The skew-t degrees of freedom are searched over an integer grid (default 5 to 30). The defaults match the skew-t at levels 0.05, 0.25, 0.5, 0.75 and 0.95, and the two-piece normal at 0.25, 0.5 and 0.75. The MPZ distribution is not fitted: it is uniform between adjacent stored quantiles, with Gaussian tails set by the outermost pairs. Bank of England fan charts are two-piece normals built from their mean, mode and uncertainty.

Calibration
^^^^^^^^^^^

For outturn :math:`y_t` and fitted cdf :math:`F_t`, the PIT is :math:`z_t = F_t(y_t)`, which is i.i.d. uniform for a correctly specified one-step forecast.

- **Knüppel (2015)**: with :math:`s_t = \sqrt{12}(z_t - 1/2)`, the moments :math:`(s_t, s_t^2 - 1, s_t^3, s_t^4 - 1.8)` have zero mean under uniformity. Odd and even pairs give separate Wald statistics with Newey-West covariances (automatic lag, Andrews 1991), summed and compared with :math:`\chi^2_4`.
- **Mitchell and Weale (2023)**: the Knüppel moments of PITs inside :math:`[\alpha/2, 1-\alpha/2]`, rescaled to that region, plus a HAC test that the share inside equals :math:`1-\alpha` (default :math:`\alpha = 0.10`); :math:`\chi^2_5`.
- **Rossi and Sekhposyan (2019)**: Kolmogorov-Smirnov and Cramér-von Mises distances between the empirical PIT cdf and the 45-degree line, with block-bootstrap critical values (block length :math:`\lfloor T^{1/3} \rfloor`). This allows for serial correlation at multi-step horizons.
- **Galvão, Mantoan and Mitchell**: a joint test across horizons from the same forecast origins. The statistic is the maximum (unweighted or with decreasing or increasing horizon weights) of the per-horizon Rossi-Sekhposyan statistics. Bootstrap draws share multipliers across horizons, and the first horizon uses the Brownian-bridge limit. Sup-test and Bonferroni critical values are reported.

Accuracy and sharpness
^^^^^^^^^^^^^^^^^^^^^^

The (weighted) continuous ranked probability score is the quantile-weighted integral of quantile scores (Gneiting and Ranjan, 2011):

.. math::

   \text{wCRPS}(F, y) = 2 \int_0^1 v(\tau)\,\big(\mathbb{1}\{y < F^{-1}(\tau)\} - \tau\big)\big(F^{-1}(\tau) - y\big)\, d\tau,

with :math:`v(\tau) = 1` for the CRPS, :math:`\tau^2` for the right tail (e.g. upside inflation risk) and :math:`(1-\tau)^2` for the left tail (e.g. downside GDP risk). It is integrated over a 2000-point value grid between the :math:`10^{-6}` and :math:`1-10^{-6}` quantiles, with the outturn added as a node. Lower is better. Sharpness is summarised by interval widths :math:`F^{-1}(u) - F^{-1}(l)`: by default the central 30%, 50% and 90% intervals and the 5-25% and 75-95% tail bands.

Replication
^^^^^^^^^^^

An opt-in parity test tier (``pytest -m parity``, local only) checks each statistic against the original MATLAB code. It also reproduces the published evaluation tables from the MATLAB forecast parameters. Two differences are by design. Path-test decisions come from a single bootstrap rather than rejection frequencies over repeated bootstraps. The skew-t CRPS uses the exact cdf rather than a numerically integrated one, so it differs from the MATLAB value by at most the MATLAB grid spacing. The inflation fan-chart inputs store the 2020-06-30 origin (no published fan chart) as zero mean, mode and uncertainty. MATLAB scores these 13 forecasts as a point mass at zero, with PIT 1. The package rejects a zero scale, so the replication keeps MATLAB's values for those forecasts only.


Practical Guidance for Users
----------------------------

When to Use Different Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**RMSE and Accuracy Metrics**:

- Use when absolute closeness to outcomes matters
- Appropriate for general-purpose forecast assessment

**Unbiasedness Tests**:

- Use when systematic over/under-prediction would be problematic
- Important for policy-relevant variables where consistent bias could lead to persistent policy errors
- Requires moderate sample sizes for power

**Efficiency Tests**:

- Use when assessing whether forecasters are fully utilizing available information
- Mincer-Zarnowitz for single-variable efficiency
- Blanchard-Leigh for cross-variable relationship calibration

**Benchmark Comparisons**:

- Always use when evaluating forecast value-added
- Helps control for underlying economic volatility
- Essential for context (is performance good or bad relative to realistic alternatives?)

Interpreting Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^

Several sources of uncertainty affect forecast evaluation:

1. **Small Sample Sizes**: Historical evaluation periods may be limited, reducing statistical power
2. **Non-stationary Economic Relationships**: Structural breaks or regime changes complicate inference
3. **Data Revisions**: Outturns change over time, affecting computed errors
4. **Forecast Vintage Correlations**: Multiple horizons from the same forecast are correlated

The toolkit mitigates these through HAC standard errors, Harvey corrections for small samples, sensitivity analysis over vintage choices, and fluctuation tests for detecting instabilities.

Technical References
--------------------

**Key Literature**

- Diebold, F. and Mariano, R. (1995). `Comparing Predictive Accuracy <https://EconPapers.repec.org/RePEc:bes:jnlbes:v:13:y:1995:i:3:p:253-63>`_. *Journal of Business & Economic Statistics*, 13(3), 253–263.

- Mincer, J. A. and Zarnowitz, V. (1969). `The Evaluation of Economic Forecasts <https://www.nber.org/system/files/chapters/c1214/c1214.pdf>`_. In *Economic Forecasts and Expectations*, NBER, pp. 3–46.

- Nordhaus, W. D. (1987). `Forecasting Efficiency: Concepts and Applications <http://www.jstor.org/stable/1935962>`_. *Review of Economics and Statistics*, 69(4), 667–674.

- Blanchard, O. J. and Leigh, D. (2013). `Growth Forecast Errors and Fiscal Multipliers <https://www.aeaweb.org/articles?id=10.1257/aer.103.3.117>`_. *American Economic Review*, 103(3), 117–120.

- Harvey, D., Leybourne, S. and Newbold, P. (1997). `Testing the equality of prediction mean squared errors <https://www.sciencedirect.com/science/article/pii/S0169207096007194>`_. *International Journal of Forecasting*, 13(2), 281–291.

- Harvey, D. I., Leybourne, S. J. and Whitehouse, E. J. (2017). `Forecast evaluation tests and negative long-run variance estimates in small samples <https://www.sciencedirect.com/science/article/pii/S0169207017300559>`_. *International Journal of Forecasting*, 33(4), 833–847.

- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817–858.

- Gneiting, T. and Ranjan, R. (2011). Comparing density forecasts using threshold- and quantile-weighted scoring rules. *Journal of Business & Economic Statistics*, 29(3), 411–422.

- Knüppel, M. (2015). Evaluating the calibration of multi-step-ahead density forecasts using raw moments. *Journal of Business & Economic Statistics*, 33(2), 270–281.

- Mitchell, J. and Weale, M. (2023). Censored density forecasts: production and evaluation. *Journal of Applied Econometrics*.

- Rossi, B. and Sekhposyan, T. (2019). Alternative tests for correct specification of conditional predictive densities. *Journal of Econometrics*, 208(2), 638–657.

**Bank of England Resources**

- Bank of England (2026). `Forecast Evaluation Report: January 2026 <https://www.bankofengland.co.uk/paper/2026/forecast-evaluation-report-january-2026>`_.

- Kanngiesser, D. and Willems, T. (2024). `Forecast accuracy and efficiency at the Bank of England <https://www.bankofengland.co.uk/-/media/boe/files/working-paper/2024/forecast-accuracy-and-efficiency-at-boe-how-errors-leveraged-to-do-better.pdf>`_. Bank of England Staff Working Paper.

- Independent Evaluation Office (2015). `Evaluating forecast performance <https://www.bankofengland.co.uk/-/media/boe/files/independent-evaluation-office/2015/evaluating-forecast-performance-november-2015.pdf>`_. Bank of England.

**Package Documentation**

See the :doc:`user_guide` and :doc:`api` for details on installation, usage, and API reference.

- GitHub Repository: `bank-of-england/forecast_evaluation <https://github.com/bank-of-england/forecast_evaluation>`_

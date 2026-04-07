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

Forecast Error Definition and Data
-----------------------------------

Forecast errors are computed as the difference between outturns and forecasts:

.. math::

    \varepsilon(y)_{t|t-h} = y_t - \hat{y}_{t|t-h}

where:

- :math:`y_t` is the realized value (outturn)
- :math:`\hat{y}_{t|t-h}` is the forecast made :math:`h` quarters ahead
- :math:`h` is the forecast horizon (0 to 12 quarters)

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

**Bank of England Resources**

- Bank of England (2026). `Forecast Evaluation Report: January 2026 <https://www.bankofengland.co.uk/paper/2026/forecast-evaluation-report-january-2026>`_.

- Kanngiesser, D. and Willems, T. (2024). `Forecast accuracy and efficiency at the Bank of England <https://www.bankofengland.co.uk/-/media/boe/files/working-paper/2024/forecast-accuracy-and-efficiency-at-boe-how-errors-leveraged-to-do-better.pdf>`_. Bank of England Staff Working Paper.

- Independent Evaluation Office (2015). `Evaluating forecast performance <https://www.bankofengland.co.uk/-/media/boe/files/independent-evaluation-office/2015/evaluating-forecast-performance-november-2015.pdf>`_. Bank of England.

**Package Documentation**

See the :doc:`user_guide` and :doc:`api` for details on installation, usage, and API reference.

- GitHub Repository: `bank-of-england/forecast_evaluation <https://github.com/bank-of-england/forecast_evaluation>`_

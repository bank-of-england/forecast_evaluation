---
name: forecast-evaluation
description: Use this skill when working with the `forecast_evaluation` Python package.
---

# forecast_evaluation

A Python package (Bank of England, MIT-licensed) for evaluating economic forecasts — both point and density forecasts — across real-time vintages. PyPI: `forecast_evaluation`. Repo: https://github.com/bank-of-england/forecast_evaluation.

Built around `ForecastData` (holds raw forecasts + outturns) with methods for data-cleaning, statistical tests (returning uniform `TestResult` objects), visualisations, and an interactive Shiny dashboard. A `NowcastData` subclass extends the workflow to intra-period (e.g. weekly) vintages for nowcasting.

---

## Installation & quick start

```sh
pip install forecast_evaluation
import forecast_evaluation as fe

data = fe.ForecastData(load_fer=True)  # Load embedded BoE data
data.summary()
```

---

## Data format

Required DataFrame columns:

| Column             | Forecasts | Outturns | Type      | Notes                                  |
|--------------------|-----------|----------|-----------|----------------------------------------|
| `date`             | ✓         | ✓        | Timestamp | Period end-date                        |
| `vintage_date`     | ✓         | ✓*       | Timestamp | When published / known                 |
| `variable`         | ✓         | ✓        | str       | e.g. `"gdpkp"`, `"cpisa"`, `"unemp"`   |
| `frequency`        | ✓         | ✓        | str       | `"Q"` (quarterly) or `"M"` (monthly)   |
| `forecast_horizon` | ✓         | ✓*       | int       | 0=nowcast, ≥1=future, -1=backcast      |
| `value`            | ✓         | ✓        | float     | The value itself                       |
| `source`           | ✓         | —        | str       | Forecaster ID (forecasts only)         |
| `metric`           | optional  | optional | str       | `"levels"`, `"pop"`, or `"yoy"`        |

*Not required if `outturn_vintages=False`.

---

## ForecastData — the main class

```python
data = fe.ForecastData(
    outturns_data=df,              # optional
    forecasts_data=df,             # optional (outturns first!)
    load_fer=False,                # True = load embedded BoE 2026 FER data
    extra_ids=None,                # extra label cols beyond 'source'
    metric="levels",               # default metric if column missing
    compute_levels=True,           # auto-derive levels from pop/yoy
    data_check=True,               # warn on scale mismatches
    outturn_vintages=True,         # False = outturns have no vintage info
    first_forecast_horizon=None,   # int, dict[var, int], or None (auto = min non-negative h per variable)
)

# Key methods (in typical order)
data.add_outturns(df, *, metric="levels")
data.add_forecasts(df, *, extra_ids=None, metric="levels", compute_levels=True, data_check=True)
data.add_benchmarks(models=["AR", "random_walk"], metric="levels")
data.filter(variables=None, sources=None, start_vintage=None, end_vintage=None, ...)
data.clear_filter()
data.merge(other)
data.copy()
data.summary()
data.run_dashboard(from_jupyter=False, host="127.0.0.1", port=8000)

# Properties
data.df               # main table (forecasts + outturns)
data.forecasts        # transformed forecasts
data.outturns         # transformed outturns
data.id_columns       # list of identifier columns
data.outturn_vintages # bool
```

**Key constraint:** Outturns MUST be added before forecasts. Each instance = one frequency only.

`add_outturns` and `add_forecasts` copy the input DataFrame upfront — they do not mutate the caller's frame.

---

## NowcastData — intra-period (e.g. weekly) vintages

Subclass of `ForecastData` for nowcasting workflows where multiple forecast vintages (and possibly outturn vintages) occur within a single target period.

```python
now = fe.NowcastData(
    outturns_data=df,
    forecasts_data=df,
    extra_ids=None,
    metric="levels",
    compute_levels=True,
    data_check=True,
    first_forecast_horizon=-1,     # default -1 to include backcasts
)
```

Differences vs `ForecastData`:

- Uses **integer-period horizons** (`h=-1` backcast, `h=0` nowcast, `h=1` one-period-ahead). Multiple weekly vintages per horizon give many obs per group.
- `add_forecasts` automatically calls `_align_outturn_vintages` to build a point-in-time outturn snapshot for every forecast vintage (so transformations have gap-free history), then `_set_revision_index_k` and `_add_days_to_publication`.
- Main table gains a `days_to_publication` column = `(vintage_date_outturn - vintage_date_forecast).days`.
- `k` is a **dense revision index** over outturn vintages per `(variable, metric, frequency, date)`: post-release vintages get `k = 0, 1, 2, ...` (first release, first revision, ...); pre-release vintages get `k = -1, -2, -3, ...` (latest backcast snapshot first).
- **Not supported on `NowcastData`:** `add_fer_*`, `filter_fer`, `create_pseudo_vintages`, `add_benchmarks`, and the efficiency analyses (weak/strong efficiency, Blanchard-Leigh, revision predictability, revisions-errors correlation) — these raise `NotImplementedError` or are simply not meaningful.

Sample data:

```python
out = fe.create_sample_nowcast_outturns()
fcs = fe.create_sample_nowcast_forecasts()
now = fe.NowcastData(out, fcs)
```

---

## DensityForecastData — for quantile/probabilistic forecasts

```python
density = fe.DensityForecastData(outturns_data=df, forecasts_data=df)
# forecasts_data MUST include a 'quantile' column ∈ [0, 1]

density.add_density_forecasts(df, extra_ids=None)
density.sample_from_density(n_samples=10_000, random_state=None)
density.to_point_forecast(method="median")  # extract point from quantiles
density.plot_density_vintage(variable, vintage_date, quantiles=[.16, .5, .84], ...)
```

---

## Statistical tests — all return TestResult

Every test returns a `TestResult` object that wraps results and metadata:

```python
res = fe.compute_accuracy_statistics(data, source=None, variable=None, k=12, same_date_range=True)
res = fe.diebold_mariano_table(data, benchmark_model="mpr", k=12, loss_function="mse")
res = fe.bias_analysis(data, source=None, variable=None, k=12, same_date_range=True, verbose=False)
res = fe.weak_efficiency_analysis(data, source=None, variable=None, k=12, same_date_range=True, verbose=False)
res = fe.strong_efficiency_analysis(data, source, outcome_variable, outcome_metric, 
                                    instrument_variable, instrument_metric, horizons=np.arange(13), j=2, k=12, alpha=0.05)
res = fe.blanchard_leigh_horizon_analysis(data, source, outcome_variable, outcome_metric,
                                          instrument_variable, instrument_metric, horizons=np.arange(13), j=2, k=12, alpha=0.05)
res = fe.revision_predictability_analysis(data, variable=None, source=None, frequency=None, n_revisions=5, same_date_range=True)
res = fe.revisions_errors_correlation_analysis(data, source=None, variable=None, k=12, same_date_range=True)
res = fe.forecast_errors_correlation_analysis(data, variable=None, source=None, k=12, same_date_range=True)

# Nowcasting (NowcastData only): accuracy / bias as a function of days-to-target
res = fe.compute_intra_period_accuracy(data, variable, metric="levels", frequency="Q",
                                       forecast_horizon=None, statistic="rmse")
res = fe.compute_intra_period_bias(data, variable, metric="levels", frequency="Q",
                                   forecast_horizon=None)

# TestResult methods
res.to_df()            # underlying DataFrame
res.plot(**kwargs)     # auto-routes to matching visualization; specify variable/source/horizon as needed
res.filter(variable=, source=, horizon=, **extra)
res.summary()          # text summary
res.describe()         # pd.DataFrame.describe()
res.to_csv(path=None)  # export
```

`res.plot()` is implemented for: `bias_analysis`, `compute_accuracy_statistics`, `strong_efficiency_analysis`, `blanchard_leigh_horizon_analysis`, `rolling_analysis` (bias/DM only), `fluctuation_tests` (bias/DM only).

### Rolling-window and fluctuation tests

```python
roll = fe.rolling_analysis(
    data, window_size=40,
    analysis_func=fe.diebold_mariano_table,  # or bias_analysis, weak_efficiency_analysis
    analysis_args={"benchmark_model": "mpr", "k": 12},
    start_vintage=None, end_vintage=None,
)
# Runs analysis_func on each rolling window; adds 'window_start', 'window_end' cols.

fluct = fe.fluctuation_tests(
    data, window_size=40,
    test_func=fe.diebold_mariano_table,  # or bias_analysis, weak_efficiency_analysis
    test_args={"benchmark_model": "mpr"},
    start_vintage=None, end_vintage=None,
)
# Same as rolling_analysis but adds Giacomini-Rossi critical values (5%/10%)
# and reject_05, reject_10, max_test_statistic, reject_max_05/10 columns.
```

---

## Visualisations

Most are **methods on `TestResult`** via `.plot()`, which auto-detects the test type.

**Methods on ForecastData** (via `PlottingMixin`):
```python
data.plot_vintage(variable, vintage_date, forecast_source=None, frequency=None, metric="levels", k=12, ...)
data.plot_hedgehog(variable, forecast_source, metric, frequency=None, k=12, ...)
data.plot_forecast_errors(variable, metric, frequency, source, vintage_date_forecast, k=12, ...)
data.plot_forecast_errors_by_horizon(variable, source, metric, frequency, k=12, ...)
data.plot_errors_across_time(variable, metric, horizons=None, sources=None, frequency=None, k=12, ...)
data.plot_forecast_error_density(variable, horizon, metric, frequency, source, k=12, ...)
data.plot_outturns(variable, metric, frequency, k=12, fill_k=True, ...)
data.plot_outturn_revisions(variable, metric, frequency, k=12, fill_k=False, ma_window=1, ...)
data.plot_average_revision_by_period(source, variable, metric, frequency, ...)
data.plot_density_vintage(variable, vintage_date, quantiles=[.16,.5,.84], ...)  # DensityForecastData only
```

**Nowcasting plots (use with `NowcastData`):**
```python
fe.plot_nowcasts(data, variable, target_date, forecast_source=None, frequency="Q",
                 metric="levels", k=12, ...)  # evolution of nowcasts for one target period
fe.plot_intra_period_accuracy(data, variable, metric="levels", frequency="Q",
                              forecast_horizon=None, statistic="rmse",
                              confidence_level=None, ...)
fe.plot_intra_period_bias(data, variable, metric="levels", frequency="Q",
                          forecast_horizon=None, confidence_level=None, ...)
```

All return `None` by default; pass `return_plot=True` to get `(fig, ax)`. Most accept `convert_to_percentage=False`.

**Standalone plotting functions:**
```python
fe.plot_accuracy(df, variable, metric, frequency, statistic="rmse", ...)
fe.plot_compare_to_benchmark(df, variable, metric, frequency, benchmark_model, statistic="rmse", ...)
fe.plot_rolling_relative_accuracy(df, variable, horizons, ...)
fe.plot_bias_by_horizon(df, variable, source, metric, frequency, ...)
fe.plot_rolling_bias(df, horizons, ...)
fe.plot_blanchard_leigh_ratios(results, ...)
fe.plot_strong_efficiency(results, ...)
fe.plot_radar(df, mode, variable=None, variables=None, metric=None, horizon=None, ...)
  # modes: "metrics" (RMSE/MAE/RMedSE by source), "variables" (variables by source), "tests" (accuracy/bias/efficiency by source)
fe.plot_correlation_heatmap(df, ...)
fe.plot_rolling_correlation(df, ...)
```

---

## Bundled FER (Forecast Evaluation Report) data

```python
data = fe.ForecastData(load_fer=True)
data.filter_fer()  # restrict to canonical variable-metric and source combos
```

Variables: `gdpkp`, `cpisa`, `unemp`, `aweagg`, etc. Sources: `mpr`, `compass conditional`, `compass unconditional`, `bvar conditional`, `baseline ar(p) model`, `baseline random walk model`. Frequency: quarterly (`Q`). Embedded in parquet files; no download needed.

---

## Benchmarks

```python
data.add_benchmarks(models=["AR", "random_walk"], variables=None, metric="levels", 
                    frequency=None, forecast_periods=13, estimation_start_date=None)
```

Adds AR(p) and/or random-walk rolling-origin forecasts as new sources (`baseline ar(p) model`, `baseline random walk model`). Useful for DM comparisons.

---

## Utilities

```python
fe.create_outturn_revisions(data)          # outturn revision table
fe.create_sample_outturns(); fe.create_sample_forecasts()
fe.create_sample_nowcast_outturns(); fe.create_sample_nowcast_forecasts()
```

---

## Common patterns

### Quick accuracy + DM + bias
```python
data = fe.ForecastData(load_fer=True)
data.filter(variables=["gdpkp", "cpisa"], frequencies=["Q"])

acc = fe.compute_accuracy_statistics(data, k=12)
acc.plot(variable="cpisa", metric="yoy", statistic="rmse")

dm = fe.diebold_mariano_table(data, benchmark_model="mpr", k=12)

bias = fe.bias_analysis(data, source="mpr", k=12)
bias.plot(variable="aweagg", source="mpr", metric="yoy")
```

### Custom data
```python
data = fe.ForecastData()
data.add_outturns(outturns_df)
data.add_forecasts(forecasts_df, extra_ids=["model_family"])
data.add_benchmarks(models=["AR"], metric="pop")
```

### Rolling DM with fluctuation test
```python
sub = data.copy()
sub.filter(variables=["gdpkp"], sources=["mpr", "baseline random walk model"])

fluct = fe.fluctuation_tests(sub, window_size=40,
                             test_func=fe.diebold_mariano_table,
                             test_args={"benchmark_model": "mpr"})
fluct.plot(variable="gdpkp", horizons=[0, 4])
```

### Density forecasts
```python
density = fe.DensityForecastData(outturns_data=outturns_df, forecasts_data=density_df)
density.plot_density_vintage(variable="gdp", vintage_date="2023-03-31", quantiles=[.1, .5, .9])
samples = density.sample_from_density(n_samples=10_000)
point = density.to_point_forecast("median")
```

### Nowcasting (intra-period analysis)
```python
now = fe.NowcastData(fe.create_sample_nowcast_outturns(),
                     fe.create_sample_nowcast_forecasts())

# Evolution of nowcasts for a single target quarter
fe.plot_nowcasts(now, variable="gdp", target_date="2024-03-31")

# Accuracy / bias as a function of days-to-target (all horizons combined)
fe.plot_intra_period_accuracy(now, variable="gdp", statistic="rmse")
fe.plot_intra_period_bias(now, variable="gdp", confidence_level=90)
```

### Dashboard
```python
data.run_dashboard()                  # standalone app at 127.0.0.1:8000
data.run_dashboard(from_jupyter=True) # embed in notebook
```

---

## Key gotchas

- **Outturns before forecasts** — must call `.add_outturns()` first.
- **One frequency per instance** — for multi-frequency, use separate instances or `.merge()` (which requires same `outturn_vintages`).
- **`NowcastData` is a `ForecastData`** — `isinstance(now, ForecastData)` is `True`. Intra-period analyses (`compute_intra_period_*`, `plot_intra_period_*`) require a `NowcastData` instance.
- `forecast_horizon=0` is nowcast; `≥1` is future; `-1` is backcast. DM internally adds `+1`.
- `source` vs `unique_id`: `source` is a single column; `unique_id` is the concatenated string of all id columns.
- `data_check=True` warns only (never errors). Set `False` only if you understand the scale differences.
- `same_date_range=True` (default) restricts to the intersection of vintage dates across sources.
- `filter()` mutates in place; use `.copy()` first to keep the original.
- `clear_filter()` is the canonical way to undo filters (rebuilds from raw data).
- Weak efficiency and most regressions need ≥10 observations per (variable, source, metric, frequency, horizon) group.
- `fluctuation_tests` requires `mu = window_size / out_of_sample_size ∈ [0.1, 0.9]`.

---

## Complete public API

```text
Classes:      ForecastData, NowcastData, DensityForecastData, TestResult
Data:         create_sample_forecasts, create_sample_outturns,
              create_sample_nowcast_forecasts, create_sample_nowcast_outturns
Tests:        compute_accuracy_statistics, diebold_mariano_table, bias_analysis,
              weak_efficiency_analysis, strong_efficiency_analysis, blanchard_leigh_horizon_analysis,
              revision_predictability_analysis, revisions_errors_correlation_analysis,
              forecast_errors_correlation_analysis,
              compute_intra_period_accuracy, compute_intra_period_bias,
              rolling_analysis, fluctuation_tests,
              compare_to_benchmark, create_comparison_table
Benchmarks:   add_ar_p_forecasts, add_random_walk_forecasts, create_outturn_revisions
Utilities:    filter_k, covid_filter, filter_fer_variables, reconstruct_id_cols_from_unique_id
Plots:        plot_vintage, plot_hedgehog, plot_nowcasts,
              plot_forecast_errors, plot_forecast_errors_by_horizon,
              plot_forecast_error_density, plot_errors_across_time, plot_outturns, plot_outturn_revisions,
              plot_average_revision_by_period, plot_accuracy, plot_compare_to_benchmark,
              plot_rolling_relative_accuracy, plot_bias_by_horizon, plot_rolling_bias,
              plot_intra_period_accuracy, plot_intra_period_bias,
              plot_correlation_heatmap, plot_rolling_correlation,
              plot_blanchard_leigh_ratios, plot_strong_efficiency, plot_radar
```

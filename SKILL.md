---
name: forecast-evaluation
description: Use this skill when working with the `forecast_evaluation` Python package.
---

# Commit: fe116342e2dc946ef96667752449e2dec70506ba
# forecast_evaluation

BoE Python pkg (MIT). Evaluate point + density econ forecasts across real-time vintages. PyPI: `forecast_evaluation`.

```sh
pip install forecast_evaluation
import forecast_evaluation as fe
data = fe.ForecastData(load_fer=True)
data.summary()
```

---

## Data format

Required cols:

| Col | Forecasts | Outturns | Type | Notes |
|-----|-----------|----------|------|-------|
| `date` | ✓ | ✓ | Timestamp | Period end-date |
| `vintage_date` | ✓ | ✓* | Timestamp | When published |
| `variable` | ✓ | ✓ | str | e.g. `"gdpkp"` |
| `frequency` | ✓ | ✓ | str | `"Q"` or `"M"` |
| `forecast_horizon` | ✓ | ✓* | int | 0=nowcast, ≥1=future, -1=backcast |
| `value` | ✓ | ✓ | float | |
| `source` | ✓ | — | str | Forecaster ID |
| `metric` | optional | optional | str | `"levels"`, `"pop"`, `"yoy"` |

*Not required if `outturn_vintages=False`.

---

## ForecastData

```python
data = fe.ForecastData(
    outturns_data=df,
    forecasts_data=df,
    load_fer=False,           # True = embedded BoE 2026 FER data
    extra_ids=None,
    metric="levels",
    compute_levels=True,
    data_check=True,
    outturn_vintages=True,
    first_forecast_horizon=None,  # int | dict[var,int] | None (auto = min non-neg h per var)
)

data.add_outturns(df, *, metric="levels")
data.add_forecasts(df, *, extra_ids=None, metric="levels", compute_levels=True, data_check=True)
data.add_benchmarks(models=["AR", "random_walk"], variables=None, metric="levels",
                    frequency=None, forecast_periods=13, estimation_start_date=None,
                    max_lag=2)   # max_lag: Literal[1,2]; skips BIC if 1
data.filter(variables=None, sources=None, start_vintage=None, end_vintage=None, ...)
data.clear_filter()
data.merge(other)
data.copy()
data.summary()
data.run_dashboard(from_jupyter=False, host="127.0.0.1", port=8000)

# Props
data.df               # main table
data.forecasts        # transformed forecasts
data.outturns         # transformed outturns
data.id_columns
data.outturn_vintages
```

**Constraints:** Outturns before forecasts. One frequency per instance.

`add_outturns`/`add_forecasts` copy input — no mutation of caller frame.

---

## NowcastData

Subclass of `ForecastData` for intra-period (e.g. weekly) vintages.

```python
now = fe.NowcastData(outturns_data=df, forecasts_data=df,
                     extra_ids=None, metric="levels", compute_levels=True,
                     data_check=True, first_forecast_horizon=-1)

out = fe.create_sample_nowcast_outturns()
fcs = fe.create_sample_nowcast_forecasts()
now = fe.NowcastData(out, fcs)
```

Differences vs `ForecastData`:
- Integer-period horizons; many weekly vintages per horizon.
- `add_forecasts` auto-calls `_align_outturn_vintages` + `_set_revision_index_k` + `_add_days_to_publication`.
- Main table gains `days_to_publication` = `(vintage_date_outturn - vintage_date_forecast).days`.
- `k` = dense revision index per `(variable, metric, frequency, date)`: post-release → `k=0,1,2,...`; pre-release → `k=-1,-2,-3,...`.
- Not supported: `add_fer_*`, `filter_fer`, `create_pseudo_vintages`, `add_benchmarks`, efficiency analyses → `NotImplementedError`.

---

## DensityForecastData

```python
density = fe.DensityForecastData(outturns_data=df, forecasts_data=df)
# forecasts_data must include 'quantile' col ∈ [0,1]

density.add_density_forecasts(df, extra_ids=None)
density.sample_from_density(n_samples=10_000, random_state=None)
density.to_point_forecast(method="median")
density.plot_density_vintage(variable, vintage_date, quantiles=[.16,.5,.84], ...)
```

---

## Statistical tests — all return TestResult

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

# NowcastData only
res = fe.compute_intra_period_accuracy(data, variable, metric="levels", frequency="Q",
                                       forecast_horizon=None, statistic="rmse")
res = fe.compute_intra_period_bias(data, variable, metric="levels", frequency="Q",
                                   forecast_horizon=None)

# TestResult methods
res.to_df()
res.plot(**kwargs)     # auto-routes; specify variable/source/horizon as needed
res.filter(variable=, source=, horizon=, **extra)
res.summary()
res.describe()
res.to_csv(path=None)
```

`res.plot()` works for: `bias_analysis`, `compute_accuracy_statistics`, `strong_efficiency_analysis`, `blanchard_leigh_horizon_analysis`, `rolling_analysis` (bias/DM only), `fluctuation_tests` (bias/DM only).

### Rolling + fluctuation tests

```python
roll = fe.rolling_analysis(
    data, window_size=40,
    analysis_func=fe.diebold_mariano_table,
    analysis_args={"benchmark_model": "mpr", "k": 12},
    start_vintage=None, end_vintage=None,
)

fluct = fe.fluctuation_tests(
    data, window_size=40,
    test_func=fe.diebold_mariano_table,
    test_args={"benchmark_model": "mpr"},
    start_vintage=None, end_vintage=None,
)
# Adds Giacomini-Rossi critical values (5%/10%) + reject_05/10, max_test_statistic, reject_max_05/10 cols.
```

---

## Visualisations

**On `ForecastData` via `PlottingMixin`:**
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
data.plot_density_vintage(...)  # DensityForecastData only
```

**Nowcasting (NowcastData):**
```python
fe.plot_nowcasts(data, variable, target_date, forecast_source=None, frequency="Q", metric="levels", k=12, ...)
fe.plot_intra_period_accuracy(data, variable, metric="levels", frequency="Q",
                              forecast_horizon=None, statistic="rmse", confidence_level=None, ...)
fe.plot_intra_period_bias(data, variable, metric="levels", frequency="Q",
                          forecast_horizon=None, confidence_level=None, ...)
```

All return `None`; pass `return_plot=True` for `(fig, ax)`. Most accept `convert_to_percentage=False`.

**Standalone:**
```python
fe.plot_accuracy(df, variable, metric, frequency, statistic="rmse", ...)
fe.plot_compare_to_benchmark(df, variable, metric, frequency, benchmark_model, statistic="rmse", ...)
fe.plot_rolling_relative_accuracy(df, variable, horizons, ...)
fe.plot_bias_by_horizon(df, variable, source, metric, frequency, ...)
fe.plot_rolling_bias(df, horizons, ...)
fe.plot_blanchard_leigh_ratios(results, ...)
fe.plot_strong_efficiency(results, ...)
fe.plot_radar(df, mode, variable=None, variables=None, metric=None, horizon=None, ...)
  # modes: "metrics" (RMSE/MAE/RMedSE by source), "variables" (by source), "tests" (accuracy/bias/efficiency by source)
fe.plot_correlation_heatmap(df, ...)
fe.plot_rolling_correlation(df, ...)
```

---

## FER data

```python
data = fe.ForecastData(load_fer=True)
data.filter_fer()  # canonical var-metric + source combos
```

Variables: `gdpkp`, `cpisa`, `unemp`, `aweagg`. Sources: `mpr`, `compass conditional`, `compass unconditional`, `bvar conditional`, `baseline ar(p) model`, `baseline random walk model`. Frequency: `Q`. Parquet-embedded; no download.

---

## Benchmarks

```python
data.add_benchmarks(models=["AR", "random_walk"], variables=None, metric="levels",
                    frequency=None, forecast_periods=13, estimation_start_date=None,
                    max_lag=2)   # Literal[1,2]; max_lag=1 skips BIC selection
```

Adds rolling-origin AR(p) / random-walk as new sources (`baseline ar(p) model`, `baseline random walk model`).

---

## Utilities

```python
fe.create_outturn_revisions(data)
fe.create_sample_forecasts(); fe.create_sample_outturns()
fe.create_sample_nowcast_forecasts(); fe.create_sample_nowcast_outturns()
fe.filter_k(data, k); fe.covid_filter(...); fe.filter_fer_variables(...)
fe.reconstruct_id_cols_from_unique_id(...)
```

---

## Common patterns

### Accuracy + DM + bias
```python
data = fe.ForecastData(load_fer=True)
data.filter(variables=["gdpkp", "cpisa"])
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
data.add_benchmarks(models=["AR"], metric="pop", max_lag=1)
```

### Rolling DM + fluctuation
```python
sub = data.copy()
sub.filter(variables=["gdpkp"], sources=["mpr", "baseline random walk model"])
fluct = fe.fluctuation_tests(sub, window_size=40,
                             test_func=fe.diebold_mariano_table,
                             test_args={"benchmark_model": "mpr"})
fluct.plot(variable="gdpkp", horizons=[0, 4])
```

### Density
```python
density = fe.DensityForecastData(outturns_data=outturns_df, forecasts_data=density_df)
density.plot_density_vintage(variable="gdp", vintage_date="2023-03-31", quantiles=[.1,.5,.9])
samples = density.sample_from_density(n_samples=10_000)
point = density.to_point_forecast("median")
```

### Nowcasting
```python
now = fe.NowcastData(fe.create_sample_nowcast_outturns(),
                     fe.create_sample_nowcast_forecasts())
fe.plot_nowcasts(now, variable="gdp", target_date="2024-03-31")
fe.plot_intra_period_accuracy(now, variable="gdp", statistic="rmse")
fe.plot_intra_period_bias(now, variable="gdp", confidence_level=90)
```

### Dashboard
```python
data.run_dashboard()                   # standalone 127.0.0.1:8000
data.run_dashboard(from_jupyter=True)  # embed in notebook
```

---

## Gotchas

- Outturns before forecasts — must call `.add_outturns()` first.
- One frequency per instance — multi-freq: separate instances or `.merge()` (same `outturn_vintages` required).
- `NowcastData` is `ForecastData` — `isinstance(now, ForecastData)` True. Intra-period fns need `NowcastData`.
- `forecast_horizon=0` nowcast; `≥1` future; `-1` backcast. DM adds `+1` internally.
- `source` = single col; `unique_id` = concatenated all id cols.
- `data_check=True` warns only, never errors.
- `same_date_range=True` restricts to intersection of vintage dates across sources.
- `filter()` mutates in place — `.copy()` first to keep original.
- `clear_filter()` canonical undo — rebuilds from raw data.
- Weak efficiency + regressions need ≥10 obs per `(variable, source, metric, frequency, horizon)`.
- `fluctuation_tests` requires `mu = window_size / out_of_sample_size ∈ [0.1, 0.9]`.

---

## Public API

```text
Classes:    ForecastData, NowcastData, DensityForecastData, TestResult
Data:       create_sample_forecasts, create_sample_outturns,
            create_sample_nowcast_forecasts, create_sample_nowcast_outturns
Tests:      compute_accuracy_statistics, diebold_mariano_table, bias_analysis,
            weak_efficiency_analysis, strong_efficiency_analysis, blanchard_leigh_horizon_analysis,
            revision_predictability_analysis, revisions_errors_correlation_analysis,
            forecast_errors_correlation_analysis,
            compute_intra_period_accuracy, compute_intra_period_bias,
            rolling_analysis, fluctuation_tests,
            compare_to_benchmark, create_comparison_table
Benchmarks: add_ar_p_forecasts, add_random_walk_forecasts, create_outturn_revisions
Utilities:  filter_k, covid_filter, filter_fer_variables, reconstruct_id_cols_from_unique_id
Plots:      plot_vintage, plot_hedgehog, plot_nowcasts,
            plot_forecast_errors, plot_forecast_errors_by_horizon,
            plot_forecast_error_density, plot_errors_across_time, plot_outturns, plot_outturn_revisions,
            plot_average_revision_by_period, plot_accuracy, plot_compare_to_benchmark,
            plot_rolling_relative_accuracy, plot_bias_by_horizon, plot_rolling_bias,
            plot_intra_period_accuracy, plot_intra_period_bias,
            plot_correlation_heatmap, plot_rolling_correlation,
            plot_blanchard_leigh_ratios, plot_strong_efficiency, plot_radar
```

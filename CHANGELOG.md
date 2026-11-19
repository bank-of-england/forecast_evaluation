# Changelog

## Unreleased
### Added
- Normal, two-piece normal, Azzalini skew-t and semi-parametric MPZ distributions, with PDF, CDF, quantile and moment calculations. Two-piece normals support Bank of England fan-chart parameter conversion.
- `to_distribution()` and `ForecastDistributions.from_params()` for fitted or known-parameter densities, with diagnostics. MPZ uses stored quantiles directly without parametric fitting.
- `compute_pit()` and density evaluation tables with `k` revision selection, including `"latest"`.
- Knuppel, Mitchell-Weale, Rossi-Sekhposyan and Galvao-Mantoan-Mitchell calibration tests, including joint-horizon calibration.
- CRPS, quantile-weighted CRPS and sharpness analysis, with PIT, CRPS and sharpness plots through `TestResult.plot()`.
- `plot_density_vintage()` options: `k=12` and keyword-only `convert_to_percentage` for growth metrics.
- Density summaries, top-level API exports including `create_sample_density_forecasts()`, a worked notebook, dashboard example and methodology documentation.

### Adjustments
- Aligned density constructors and validation with `ForecastData`; supplied density metrics remain unchanged, with point conversion explicit.
- Finite quantiles in [0, 1] and finite values required. By default, `crossing="rearrange"` sorts crossings with warnings; `"error"` rejects and `"ignore"` preserves them. Negative information horizons are dropped with warnings.
- `to_point_forecast()` preserves sources, supports `source_suffix` and `on_missing="raise"`, and otherwise selects the nearest quantile common across each path, breaking ties downwards. Numeric and string levels are accepted. Repeated conversions are idempotent; conflicts raise errors.
- Sampling uses filtered densities, requires two distinct quantiles and warns about clamped tails. Draws are independent, not joint paths. Sampling and sample generators use private random generators; example density levels match point examples.
- Vintage plots use keyword-only `quantiles` and `return_plot`; positional quantiles are deprecated. Positional support remains for one release; defaults are `(0.16, 0.5, 0.84)`. Missing quantiles warn and use nearest levels; invalid or duplicate selections raise errors.
- Historical outturns use the forecast vintage; subsequent outturns use `k`, falling back to the furthest maturity.

### Fixed
- Density identifiers exclude `quantile` and retain extra IDs; failed additions and merges restore state. Merges into plain `ForecastData` warn about discarded densities.
- Density-only filtering, dashboards and notebook representations. Skipped custom filters warn; point-only analyses require conversion. `clear_filter()` restores all stored views. Dashboards show placeholders until point forecasts are available.
- Supplied growth outturns override derived duplicates, warning on discrepancies except for aligned nowcast snapshots.
- Rossi-Sekhposyan and Galvao-Mantoan-Mitchell bootstrap tests default to `seed=42`; fitting warnings identify callers.

### Development
- Density regression tests, numerical snapshots and shared fixtures.
- Package quality and OPERA ecosystem checks; post-publication dependency-pin and generated-API updates through pull requests after PyPI availability.
- Local Ruff, pytest and doctest hooks, plus protected-branch contribution guidance.

## 0.1.13
### Added
- New `NowcastData` class to handle intra-period vintages.

### Adjustments
- Specify metric in `add_fer_outturns` to be 'levels'.
- Redefined `forecast_horizon` as "forecast target period" minus "last target observation used for forecasting" minus 1. Analyses now group and report on `horizon`, currently set to the calendar distance `target_minus_vintage`. `forecast_horizon` sets the HAC bandwidth, reported in the new `hac_maxlags`.

## 0.1.12
### Adjustments
- Moved `pyarrow` back to required dependencies as it's required when initiating the `ForecastData` class with `load_fer=True`.

## 0.1.11
### Added
- `max_lag` argument (1 or 2, default 2) on `add_benchmarks`, `add_ar_p_forecasts`, and `build_ar_p_model` to control the maximum AR order considered during BIC lag selection.

### Adjustments
- For the AR model, when `max_lag=1`, the BIC lag-selection loop is skipped and `optimal_lag` is set directly to 1, avoiding a redundant model fit per vintage.
- Moved pyarray and Ipython to dev deps.

## 0.1.10
### Adjustments
- **Bugfix** in `add_ar_p_forecasts` where the AR(2) forecasts were forecasting with the coefficients the wrong way around.
- fer_forecasts.parquet updated with the bugfix with the AR(2) forecasts.

## [0.1.9] - 2026-06-04 
### Added
- `first_forecast_horizon` argument on `ForecastData` and `add_forecasts`; accepts an int or a dict mapping variable names to per-variable thresholds (variables not in the dict default to 0). Computed from the data if not given. Allows backtesting and forecasting in addition to "nowcasting".
- `publication_lag` argument on `create_pseudo_vintages`; the method now also works on outturns without vintages.
- Individual scale per edge in the radar plot.

### Adjustments
- Validation errors raised by `add_forecasts` / `add_outturns` are now reported as a single human-readable message that lists every offending column with example values and row indices, instead of a raw pandera traceback.
- `add_forecasts` / `add_outturns` now emit a `UserWarning` when the `metric` column is missing and is auto-filled with the default (`"levels"` unless overridden via the `metric=` argument).
- `add_forecasts` / `add_outturns` now copy the input DataFrame upfront and never modify it in place.
- The `frequency=` keyword argument of plot and analysis methods (e.g. `plot_accuracy`, `plot_hedgehog`, `weak_efficiency`, `blanchard_leigh_horizon_analysis`, ...) is now optional and inferred from the `ForecastData` instance. Passing it explicitly is deprecated and emits a `DeprecationWarning`. (This refers to the function argument, not the `frequency` column of the input data, which is still required.)
- `consistent_date_range` now enforces consistency across vintages **and** target dates (previously only across one axis).

### Fixed
- Removed hardcoded quarterly frequency in some of the dashboard tabs.
- Fixed type hints and docstrings across the codebase.

## [0.1.8] - 2026-05-08
- Added forecast error correlation analysis and plots.

## [0.1.7] - 2026-05-07
### Fixed
- Clean the `unique_id` when using multi id cols.

## [0.1.6] - 2026-04-27
### Added
- Added plotting methods directly to `ForecastData` via `PlottingMixin`: `plot_hedgehog()`, `plot_forecast_errors()`, `plot_forecast_errors_by_horizon()`, `plot_forecast_error_density()`, `plot_outturn_revisions()`, `plot_outturns()`, `plot_average_revision_by_period()`, `plot_vintage()`, `plot_errors_across_time()`
- Added `outturn_revisions` argument to `ForecastData` to support forecast evaluation without outturn vintages. Users don't need to provide the "vintage_date" or "forecast_horizon" columns in their outturns if the argument is False. 

### Adjustments
- Made `frequency` argument optional (default `None`) and inferred from the data in: `plot_hedgehog()`, `build_ar_p_model`, `build_random_walk_model`, `plot_density_vintage`, `plot_vintage`, `plot_radar`, `plot_accuracy`, `strong_efficiency_analysis`, `revision_predictability_analysis`, `blanchard_leigh_horizon_analysis`, `plot_accuracy`

## [0.1.5] - 2026-04-14
### Added
- Radar charts

## [0.1.4] - 2026-04-08
### Fixed
- Fixed duplicate level rows when both pop and yoy forecasts present with `compute_levels=True`
- Fixed insufficient outturn history window for YoY transformations (now frequency-aware)
- Fixed multi-frequency outturn duplication in `prepare_outturns`
- Fixed hedgehog chart not displaying single-horizon forecasts

### Added
- Enhanced data validation and diagnostics
- Improved duplicate detection logic to compare against raw input only

## [0.1.3] - 
- Added option to plot multiple sources in `plot_forecast_errors_by_horizon()`
- Added optional argument `convert_to_percentage` to `plot_vintage()`, similar to other plotting functions

## [0.1.2] - 2026-03-19
- Added `construct_pseudo_vintages()` method to `ForecastData`
- Added `add_benchmark()` method to `ForecastData` for computing benchmark forecasts, with a progress bar
- Added `compute_levels` argument to forecast transformation methods; existing pop and yoy columns are no longer overwritten
- Optimised merging inside `build_main_table` function

## [0.1.1] - 2026-02-26
- Updated documentation.
- Fixed legend issue with rolling plots when showing only one horizon.
- Added filtering of outturns in the filtering methods.
- Removed method-chaining with filter() and merge() methods.

## [0.1.0] - 2026-01-22
- Initial release.

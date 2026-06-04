# Changelog

## [0.1.0] - 2026-01-22
- Initial release.

## [0.1.1] - 2026-02-26
- Updated documentation.
- Fixed legend issue with rolling plots when showing only one horizon.
- Added filtering of outturns in the filtering methods.
- Removed method-chaining with filter() and merge() methods.

## [0.1.2] - 2026-03-19
- Added `construct_pseudo_vintages()` method to `ForecastData`
- Added `add_benchmark()` method to `ForecastData` for computing benchmark forecasts, with a progress bar
- Added `compute_levels` argument to forecast transformation methods; existing pop and yoy columns are no longer overwritten
- Optimised merging inside `build_main_table` function

## [0.1.3] - 
- Added option to plot multiple sources in `plot_forecast_errors_by_horizon()`
- Added optional argument `convert_to_percentage` to `plot_vintage()`, similar to other plotting functions

## [0.1.4] - 2026-04-08
### Fixed
- Fixed duplicate level rows when both pop and yoy forecasts present with `compute_levels=True`
- Fixed insufficient outturn history window for YoY transformations (now frequency-aware)
- Fixed multi-frequency outturn duplication in `prepare_outturns`
- Fixed hedgehog chart not displaying single-horizon forecasts

### Added
- Enhanced data validation and diagnostics
- Improved duplicate detection logic to compare against raw input only

## [0.1.5] - 2026-04-14
### Added
- Radar charts

## [0.1.6] - 2026-04-27
### Added
- Added plotting methods directly to `ForecastData` via `PlottingMixin`: `plot_hedgehog()`, `plot_forecast_errors()`, `plot_forecast_errors_by_horizon()`, `plot_forecast_error_density()`, `plot_outturn_revisions()`, `plot_outturns()`, `plot_average_revision_by_period()`, `plot_vintage()`, `plot_errors_across_time()`
- Added `outturn_revisions` argument to `ForecastData` to support forecast evaluation without outturn vintages. Users don't need to provide the "vintage_date" or "forecast_horizon" columns in their outturns if the argument is False. 

### Adjustments
- Made `frequency` argument optional (default `None`) and inferred from the data in: `plot_hedgehog()`, `build_ar_p_model`, `build_random_walk_model`, `plot_density_vintage`, `plot_vintage`, `plot_radar`, `plot_accuracy`, `strong_efficiency_analysis`, `revision_predictability_analysis`, `blanchard_leigh_horizon_analysis`, `plot_accuracy`

## [0.1.7] - 2026-05-07
### Fixed
- Clean the `unique_id` when using multi id cols.

## [0.1.8] - 2026-05-08
- Added forecast error correlation analysis and plots.

## [0.1.8.dev] - 
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
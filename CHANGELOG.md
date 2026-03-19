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

## [0.1.3.dev0] - Unreleased
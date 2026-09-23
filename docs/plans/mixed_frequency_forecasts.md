# Plan: mixed-frequency forecasts in one `ForecastData`

## Goal

Store quarterly and monthly forecasts in one `ForecastData` instance, so that
forecast routines can validate and condition on both.

## Rules

1. **One frequency per variable.** A variable has a single frequency across all
   forecasts and outturns in an instance.
2. **Forecasts follow their outturns.** A forecast's frequency must equal the
   frequency of the outturns for the same variable.
3. **Joint analyses share one frequency.** Any analysis or plot that combines
   several series must receive series of a single frequency.

Rules 1 and 2 make `variable` sufficient to identify a frequency. Every
existing grouping by `variable` therefore stays correct, and the pooling risks
found in the audit (same variable, same `unique_id`, both frequencies) cannot
arise.

## Design

No new state. The variable-to-frequency mapping is read from the stored data
whenever it is needed.

### 1. Validation (`data/ForecastData.py`)

- Add one helper, `_check_variable_frequencies(new, *existing)`. It combines
  the `(variable, frequency)` pairs of the new batch with those of the existing
  tables and raises if any variable maps to more than one frequency. The error
  names each offending variable and its frequencies.
- Call it from `add_outturns` with the raw outturns. This enforces rule 1 for
  outturns.
- In `_validate_new_forecasts`, replace the single-frequency block
  ([L442-L466](../../src/forecast_evaluation/data/ForecastData.py#L442-L466))
  with a call against the raw outturns and every table in
  `self._forecast_tables`. This enforces rules 1 and 2 for point and density
  forecasts alike.
- Leave `_check_missing_outturns` as it is: a variable without outturns still
  only warns.
- Update the class and `add_forecasts` docstrings to state rules 1 and 2.

### 2. Main table (`core/main_table.py`)

- `compute_k` uses one frequency for the whole table, and callers pass the
  first forecast row's frequency. Monthly `k` would be counted in quarters.
- Make `build_main_table` compute `k` per frequency present in `merged` and
  remove its `frequency` parameter.
- Remove `frequency=...iloc[0]` from the three callers in `ForecastData`
  ([L364](../../src/forecast_evaluation/data/ForecastData.py#L364),
  [L619](../../src/forecast_evaluation/data/ForecastData.py#L619),
  [L790](../../src/forecast_evaluation/data/ForecastData.py#L790)).
  Check `NowcastData` and `DensityForecastData` for further callers.

### 3. Joint analyses (rule 3)

- Add one helper in `utils.py`, `require_single_frequency(df, context)`. It
  returns the frequency or raises with a message that suggests
  `filter(frequencies=...)`.
- Apply it where different variables are combined:
  - `strong_efficiency_analysis` and `blanchard_leigh_horizon_analysis`: check
    the outcome and instrument rows after filtering, rather than reading the
    first row of the whole table
    ([strong_efficiency.py L223](../../src/forecast_evaluation/tests/strong_efficiency.py#L223),
    [blanchard_leigh.py L247](../../src/forecast_evaluation/tests/blanchard_leigh.py#L247)).
  - Radar plots already require or accept a selected frequency. The dashboard
    now exposes the frequency selector and passes it to the radar plot, so only
    variables at that frequency are compared.
- Analyses of one variable at a time need no change: bias, weak efficiency,
  revisions, accuracy, Diebold–Mariano, correlation, fluctuation and intra-period.
  The same holds for `ensure_consistent_date_range`.

### 4. Presentation

- Dashboard ([ui.py L36](../../src/forecast_evaluation/dashboard/ui.py#L36)):
  derive the period label from the selected variable's frequency, or use
  "periods" when the data contain both. Ensure the radar tab passes a single
  frequency.
- `plot_forecast`
  ([forecast.py L62](../../src/forecast_evaluation/visualisations/forecast.py#L62)):
  take the frequency from the plotted variable, not the first row.
- Accuracy plots should format date ranges using the variable's frequency;
  monthly ranges must not be labelled as quarters.
- `summary()` already prints the frequency per variable, so it needs no change.

### 5. Documentation

- Remove the "one frequency per instance" statements from the user guide and
  the docstrings. Document the three rules instead, with a short quarterly plus
  monthly example.

## Out of scope

- One variable in two frequencies, which is case B in the audit.
- Converting between frequencies, such as aggregating monthly to quarterly.

## Tests

The fixture `tests/realtime_dataset.py` already has distinct quarterly and
monthly variables, so it can provide a mixed instance.

New tests:

- A mixed instance built in one `add_forecasts` call and in two calls stores
  both frequencies; point and density forecasts may also use different
  frequencies when they belong to different variables.
- In a mixed instance, monthly `k` matches a monthly-only instance, and
  quarterly `k` matches a quarterly-only instance. Repeat this check after
  `clear_filter()` and `create_pseudo_vintages()`.
- A variable given two frequencies raises, in outturns and in forecasts.
- A forecast whose frequency differs from its outturns raises.
- `strong_efficiency_analysis` and `blanchard_leigh_horizon_analysis` raise
  when the outcome and instrument differ in frequency.
- `add_benchmarks(models="random_walk")` builds forecasts for quarterly and
  monthly variables in one call.
- Accuracy for one variable in a mixed instance matches the result from a
  single-frequency instance.
- Accuracy plots display monthly date ranges as months.

**Existing rejection tests updated with approval:**

- `test_add_forecasts_mixed_frequencies_raises` in `tests/test_data_class.py`:
  it adds `gdpkp` in both frequencies. It should still raise, but with the new
  message.
- `test_add_forecasts_different_frequency_from_existing_raises` in
  `tests/test_data_class.py`: it adds monthly `gdpkp` against quarterly
  outturns. It should still raise, but with the new message.
- `test_point_and_density_forecasts_must_share_frequency` in
  `tests/test_density_data_class.py`: same situation; the `match=` string
  changes.

## Order of work

1. Main table `k` per frequency (step 2). This is safe on its own and fixes the
   silent error first.
2. Validation rules (step 1), with the test updates approved.
3. Joint-analysis guard (step 3).
4. Presentation and documentation (steps 4 and 5).
5. `ruff format`, `ruff check --fix`, `pytest -n auto`.

## Suggested commit message

```
feat: allow quarterly and monthly forecasts in one ForecastData

Each variable keeps a single frequency, shared by its forecasts and
outturns; joint analyses require series of one frequency.
```

## Plan: Bound outturn alignment snapshots to needed history window

Trim `_align_outturn_vintages` in [NowcastData.py](src/forecast_evaluation/data/NowcastData.py) so each synthetic snapshot only keeps the outturn history actually consumed downstream (`first_forecast_horizon - (n_periods + 1)` onward), instead of the full historical series, cutting memory/row count while preserving identical `main_table`/transformation output. Also clarify in the method's docstring that these synthetic rows exist purely to feed the `pop`/`yoy` transformation pipeline (`transform_forecast_to_levels`, `transform_series`) and are explicitly excluded from `build_main_table` before the actual forecast-vs-outturn evaluation — they never appear in the final evaluated data.

**Phases**
1. **Phase 1: Bound snapshot history in `_align_outturn_vintages` and clarify its purpose**
    - **Objective:** Limit each per-`(variable, metric)` snapshot to rows with horizon (relative to the forecast `vintage`) `>= self.first_forecast_horizon - (n_periods + 1)`, where `n_periods` is 4 for `Q` / 12 for `M` frequency, matching the bound already applied later in `prepare_forecasts`. Only apply the bound when `self.first_forecast_horizon` is a concrete `int` at call time (not `None`/dict); otherwise keep current unbounded behaviour to avoid correctness risk. Update the method's docstring to explicitly state the snapshot is transformation-only scaffolding, not used in evaluation (`build_main_table` strips `_aligned` rows before comparing forecasts to outturns).
    - **Files/Functions to Modify/Create:**
      - [NowcastData.py](src/forecast_evaluation/data/NowcastData.py) — `_align_outturn_vintages` (add horizon-based filtering of `available`/`snapshot` per variable's `frequency`, guarded by an `isinstance(self.first_forecast_horizon, int)` check; update docstring).
      - [tests/test_nowcasting.py](tests/test_nowcasting.py) — new test(s).
    - **Tests to Write:**
      - `test_align_outturn_vintages_trims_old_history`: build outturns spanning many more quarters than `n_periods+1` before a forecast vintage's first horizon, call `add_forecasts`, assert the synthetic snapshot rows (`_aligned=True`) for that vintage only span the expected bounded window (oldest retained date matches the computed cutoff, nothing older present).
      - `test_align_outturn_vintages_trim_preserves_yoy_computation`: construct a case that specifically needs the full `n_periods+1` lookback for YoY (e.g. `first_forecast_horizon=-1`, quarterly data), and assert the resulting `main_table`/`df` YoY values are numerically unchanged versus current (untrimmed) behaviour — i.e. trimming doesn't drop a row that's actually needed.
      - `test_align_outturn_vintages_no_bound_when_horizon_not_int`: confirm behaviour is unchanged (full history) when `first_forecast_horizon` isn't resolved to a concrete int at alignment time (if easily triggerable via public API; otherwise test the guard directly).
    - **Steps:**
        1. Write the three tests above against the *current* unbounded implementation — the trimming-specific assertions should fail (red), the YoY-preservation test should already pass (baseline safety net).
        2. Run tests, confirm the expected failures.
        3. Implement the horizon-bound filtering in `_align_outturn_vintages`: compute `n_periods` from each variable's `frequency`, compute horizon via period-ordinal arithmetic relative to `vintage`, filter `available` (or the built `snapshot`) to `horizon >= first_forecast_horizon - (n_periods + 1)` when `self.first_forecast_horizon` is a concrete int. Update the docstring to state the transformation-only purpose and exclusion from evaluation.
        4. Run tests again, confirm all pass (green).
        5. Run `ruff format` and `ruff check --fix`.
        6. Run the full test suite (`pytest`) to confirm no regressions elsewhere (e.g. `test_no_outturn_vintages.py`, snapshot/`.ambr` tests, `test_summary.py`).

**Open Questions**
1. Should the fallback for non-int `first_forecast_horizon` (dict/None) also attempt trimming using `min(dict.values())`, matching `prepare_forecasts`'s own fallback — or is leaving it fully unbounded in that case acceptable for now?
2. Is a small buffer beyond `n_periods+1` (e.g. +1 extra period) warranted for safety margin against off-by-one errors between horizon computed here vs. in `prepare_forecasts`, or should we match exactly?

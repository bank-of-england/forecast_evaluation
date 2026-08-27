# Simulation Support Design
## 1. Executive summary
This document proposes simulation support for `forecast_evaluation` and
`forecast-realtime`. The workflow is to generate many simulated outturn histories
under one or more scenarios, expose each history through real-time vintages, pass
each history to `forecast-realtime` for estimation and backtesting, return forecasts
to `forecast_evaluation`, evaluate each path independently, and aggregate the
resulting statistics across draws or scenarios.
The public simulation object is a simulation façade. It owns one ordinary
`ForecastData` instance per simulation path and adds methods for panel ingestion,
path-level evaluation, and aggregation. `ForecastData` itself is not modified.
```python
class SimulationData:
    simulation_ids = ["draw", "scenario"]
    panels: dict[tuple, ForecastData]
```
## 2. Architectural decision
Keep `ForecastData` as the stable single-panel implementation. `SimulationData` is a
separate composition façade; it must not send a combined draw/scenario frame through
`ForecastData` or inherit its single-panel storage semantics.

`SimulationData` partitions input by simulation key, delegates each partition to an
ordinary `ForecastData`, and combines only tagged result frames. This reuses the
existing validation, transformations, benchmarks, main-table construction, and
analysis functions without changing their behavior. The façade adds the small amount
of orchestration that cannot belong in the single-panel class.

This is composition at both the implementation and public boundaries. The internal
children are an implementation detail; callers use `SimulationData` methods rather
than managing a list themselves.
## 3. Scope and goals
The implementation supports `draw` as a mandatory identifier, `scenario` as the
default second identifier, future configurable identifiers, simulated outturns with
multiple vintages, independent realtime estimation per path, existing point-forecast
evaluation functions, draw-level result rows, and explicit aggregation across IDs.
It preserves the existing meanings of `source`, `unique_id`, `date`,
`vintage_date`, `forecast_horizon`, `metric`, and `frequency`.
The first version does not require a simulation dashboard, separate result class,
balanced release calendars, pooled inference, or disk-backed results. Streaming and
specialized statistical summaries can follow after correctness is established.
## 4. Terminology
A **draw** is one Monte Carlo replication. A **scenario** describes the
data-generating process or shock, such as `baseline`, `shock_a`, or `shock_b`.
A **simulation path** is one combination of all configured simulation identifiers;
with the default configuration it is `(draw, scenario)`.
A **forecast identity** describes a forecasting model and its labels. It uses
`source` and existing forecast-side `extra_ids`. A **simulation identity** describes
the simulated world and uses `draw`, `scenario`, and future simulation dimensions.
Simulation identity must not be folded into forecast identity: `Ridge` remains the
source for every draw.
## 5. Public API
```python
sim_data = fe.SimulationData(
    outturns_data=simulated_outturns,
    simulation_ids=["draw", "scenario"],
    outturn_vintages=True,
)
```
The default is `simulation_ids=("draw", "scenario")`. A single-scenario experiment
may use `scenario="baseline"`; a caller may explicitly use `simulation_ids=["draw"]`.
Keeping a scenario column even for one scenario gives downstream code a stable shape.
The proposed constructor is:
```python
SimulationData(
    outturns_data=None,
    forecasts_data=None,
    *,
    simulation_ids=("draw", "scenario"),
    metric="levels",
    compute_levels=True,
    data_check=True,
    outturn_vintages=True,
    default_k=None,
)
```
The ordinary configuration is stored by `SimulationData` and passed to each child
`ForecastData`. The combined input frames are partitioned by
`SimulationData.add_outturns` and `.add_forecasts` before delegation.
```python
class SimulationData:
    def __init__(
        self,
        outturns_data=None,
        forecasts_data=None,
        *,
        simulation_ids=("draw", "scenario"),
        **forecast_data_options,
    ):
        self._simulation_ids = self._validate_simulation_ids(simulation_ids)
        self._panels = {}
        if outturns_data is not None:
            self.add_outturns(outturns_data, **forecast_data_options)
        if forecasts_data is not None:
            self.add_forecasts(forecasts_data, **forecast_data_options)
    @property
    def simulation_ids(self) -> list[str]:
        return self._simulation_ids.copy()
```
Returning a copy prevents mutation of the identity contract after ingestion.
## 6. SimulationData methods
`ForecastData` remains a single-panel class and is not given simulation-ID hooks.
`SimulationData` owns the panel-aware methods instead:
```python
class SimulationData:
    def add_outturns(self, df, *, metric="levels"): ...
    def add_forecasts(self, df, *, extra_ids=None, metric="levels", ...): ...
    def iter_panels(self): ...
    def evaluate(self, analysis, **kwargs): ...
    def evaluate_accuracy(self, **kwargs): ...
    def aggregate_accuracy(self, result, *, across=("draw",)): ...
    def aggregate_results(self, result, *, across=("draw",)): ...
```
`add_outturns` and `add_forecasts` partition by `simulation_ids` and delegate each
partition to ordinary `ForecastData` children. `iter_panels` yields
`(simulation_key, data)` so realtime and analysis orchestration can reuse the
unchanged single-panel API.

`evaluate` runs an existing analysis callable once per child, adds path IDs to each
returned result, and concatenates tagged rows. `evaluate_accuracy` is the convenience
wrapper around `compute_accuracy_statistics`; `aggregate_accuracy` computes means and
Monte Carlo SEs from its draw-level MSEs. `aggregate_results` handles other
path-level result types. Any helper needing simulation semantics belongs here or in a
new simulation utility module, not in `ForecastData`.
## 7. Identifier validation
`simulation_ids` must be a non-empty sequence, must include `draw`, and must contain
unique string names. Configured order is preserved because it determines composite-key
and output-column order.
Names cannot collide with core columns:
```text
date, vintage_date, variable, frequency, value, source,
forecast_horizon, metric, unique_id, target_minus_vintage
```
The constructor rejects collisions immediately and names the invalid identifier.
Future dimensions such as `parameter_set` use the same rules.
`draw` is a non-negative integer. Unambiguous integer-like values such as `1.0` may
be coerced; fractional values, negatives, nulls, and labels such as `"draw_1"` are
invalid. Draws need not be contiguous or start at zero or one.
Scenario values are non-null, non-empty labels stored consistently as strings. There
is no fixed scenario vocabulary. The same rules apply to future label-style IDs.
## 8. Outturn and forecast contracts
Simulation outturns retain the existing outturn columns and add configured IDs:
```text
draw  scenario  date        vintage_date  variable  frequency  value
1     baseline  2010-03-31  2010-03-31    gdp       Q          101.2
2     baseline  2010-03-31  2010-03-31    gdp       Q          100.8
1     shock_a   2010-03-31  2010-03-31    gdp       Q          104.5
```
`date` is the target period and `vintage_date` is the release date. Equal dates,
vintages, variables, and frequencies with different paths are valid.
```python
outturn_key = [
    *simulation_ids, "date", "vintage_date", "variable", "frequency", "metric"
]
```
Simulation forecasts carry the same IDs as their outturn path:
```text
draw  scenario  date        vintage_date  source  variable  frequency  horizon  value
1     baseline  2010-06-30  2010-03-31    Ridge   gdp       Q          0        102.0
2     baseline  2010-06-30  2010-03-31    Ridge   gdp       Q          0        101.5
1     shock_a   2010-06-30  2010-03-31    Ridge   gdp       Q          0        103.2
```
`source` remains the model identity. Forecast `extra_ids` remain separate. An
unknown forecast path is rejected.
## 9. Schema and duplicate handling
`ForecastData` continues to use its existing required columns, Pandera schemas, and
duplicate checks. `SimulationData` validates the configured simulation columns before
partitioning; those columns are not passed to the child schema.

The outer validation requires every configured ID, checks nulls and types, and rejects
duplicate complete path records. Each child then receives a normal single-panel frame
and needs no schema change. Different paths are distinct records; the same complete
path plus core metadata with different `value` raises a simulation-level error.

Combined tagged views may be exposed for inspection, but must never be fed back
through a child `ForecastData` validator.
The child `ForecastData` objects retain the normal tables `_raw_outturns`,
`_outturns`, `_raw_forecasts`, `_forecasts`, and `_main_table`. `SimulationData` keeps
those children in `_panels`; they are the source of truth. Combined tagged views may
be exposed for inspection, but are never passed back through `ForecastData`
validation or transformation methods.
## 10. Transformation rules
`prepare_outturns` creates levels, period-on-period changes, and year-on-year changes
inside each child. Because every child contains exactly one path, the existing
single-panel grouping remains unchanged:
```python
["variable", "vintage_date", "frequency"]
```
There is no cross-path grouping because cross-path rows never enter a child. The
façade reattaches path IDs to transformed and forecast rows after delegation.
Missing history in one path remains missing rather than falling back to another path.

Level reconstruction from `pop`, `yoy`, or another metric uses the existing child
grouping:
```python
[
    "source", "variable", "metric", "frequency", "vintage_date",
]
```
The historical base level necessarily comes from the same child and path. When
`outturn_vintages=False`, each child generates synthetic outturn copies using the
unchanged behavior; `SimulationData` reattaches path IDs to the resulting views.
## 11. Main table and revisions
`build_main_table` remains the critical forecast/outturn join inside each child.
Its existing merge dimensions are approximately
`["date", "variable", "frequency", "metric"]`; no simulation columns are needed
because a child contains one path. `SimulationData` concatenates completed child
main tables after adding path IDs:
```python
[
    "draw", "scenario", "date", "variable", "frequency", "metric",
]
```
Forecast and outturn vintages remain separate after the merge. The join produces one
row for each complete identity, never a cross-path Cartesian product.
For example:
```text
forecast: draw=1, shock_a, value=100       outturn: draw=1, shock_a, value=101
forecast: draw=2, shock_a, value=200       outturn: draw=2, shock_a, value=198
```
must produce errors `1` and `-2`, not four matches.
The main table retains `draw`, `scenario`, dates, variable, vintages, `unique_id`,
metric, frequency, horizon, `target_minus_vintage`, forecast value, outturn value,
and forecast error.
Revision grouping is also unchanged inside each child. The combined tagged revision
view has:
```python
[
    "draw", "scenario", "date", "variable", "frequency", "metric",
]
```
Within each path, the earliest vintage is `k=0`, the next is `k=1`, and
`latest_vintage` is path-specific. A release in one scenario cannot revise another.
## 12. SimulationData methods, filters, and benchmarks
`SimulationData` implements its own panel methods; `ForecastData` is unchanged. The
essential methods are `add_outturns`, `add_forecasts`, `iter_panels`, `evaluate`,
`evaluate_accuracy`, `aggregate_accuracy`, `aggregate_results`, `clear_filter`, and
`merge`.

`add_outturns` and `add_forecasts` validate simulation columns, partition by the
composite path key, create missing children, and delegate each partition to ordinary
`ForecastData`. `clear_filter` reselects panel children. `merge` requires matching
simulation-ID names and order; conflicting complete keys raise. Disjoint draws or
scenarios may be combined.

`evaluate` accepts an existing single-panel analysis callable, invokes it once per
child, adds path IDs to each result, and concatenates the rows. This keeps accuracy,
bias, correlation, and other existing analysis implementations unchanged.
Existing filters for dates, vintages, variables, metrics, sources, and frequencies
remain available. A custom filter can select a scenario:
```python
sim_data.filter(
    custom_filter=lambda df: df[df["scenario"] == "shock_a"]
)
```
An explicit simulation-filter argument can follow later. `clear_filter` restores all
raw paths.
Benchmark methods on `SimulationData` iterate over children. An AR model for draw 1
sees only draw 1 history; a random walk for `shock_a` sees only `shock_a` history.
Generated benchmark forecasts are tagged with path IDs and retain the existing model
`source` label.
## 13. Analysis compatibility
Existing analysis functions continue to accept an ordinary `ForecastData` object.
`SimulationData.evaluate` calls each function once per child and tags the returned
rows; the analysis functions themselves need no simulation changes. Accuracy grouping
inside a child remains:
```python
["variable", "unique_id", "metric", "frequency", "horizon"]
```
The combined accuracy result may contain:
```text
draw, scenario, variable, unique_id, metric, frequency, horizon,
mse, mean_abs_error, rmse, rmedse, n_observations, start_date, end_date
```
The existing `TestResult` can carry these columns initially; a `SimulationResult` is
not required for the first release.
Bias, correlation, efficiency regressions, rolling analysis, and revision analysis
are also called once per child and then tagged. A statistical test's statistic and
p-value belong to one path. P-values are not averaged by default; rejection rates
are not pooled p-values.
## 14. Realtime integration
`RealTimeModel` inspects `data.simulation_ids`; no new public simulation arguments
are required. Ordinary data uses the existing execution path. For simulation data it
iterates over `SimulationData.iter_panels`; the existing vintage loop runs once per
child path.
```python
if data.simulation_ids:
    return self._forecast_simulations(...)
return self._forecast_single_dataset(...)
```
The simulation branch uses the façade's panel iterator:
```python
for simulation_key, panel_data in data.iter_panels():
    run_one_panel(panel_data, simulation_key)
```
A tuple key may be normalized to `{"draw": 17, "scenario": "shock_a"}`. The panel
loop selects data, calls the existing vintage loop, and tags returned rows; it owns
no model-specific fitting logic.
Because each loop receives one path, existing internal selectors such as
`drop_duplicates(["date", "variable"])` remain valid inside the panel.
Conditioning forecasts and future regressor forecasts are filtered to the active path
before vintage selection. Draw 17 cannot use draw 18 conditioning data. The same
rule applies inside workers.
Paths may have different release calendars. The first version uses each path's
available vintages bounded by global `first_vintage` and `last_vintage`, so output
may be unbalanced. A balanced-vintage option can be added later.
Every generated forecast is tagged before ingestion:
```python
forecast_df["draw"] = draw
forecast_df["scenario"] = scenario
```
The model label remains in `source`; decomposition rows receive the same tags.
## 15. Model isolation and parallelism
Each simulation path is an independent estimation experiment. Every model is fitted
separately for every path and vintage. Fitted state must not leak between paths;
existing per-vintage model copying remains in place.
The first implementation may process paths sequentially and reuse current parallel
execution inside each path. This provides a clear correctness baseline. The eventual
process-pool task is conceptually:
```text
simulation_key + model + vintage_batch
```
Each task receives only its panel data and outputs are concatenated once. Revision
decomposition may remain sequential because it depends on ordered vintage state;
independent panels can later be parallelized while preserving within-panel order.
## 16. Aggregation API
Add an aggregation method to `SimulationData`:
```python
summary = sim_data.aggregate_results(
    accuracy,
    across=["draw"],
)
```
The method accepts a `TestResult` or DataFrame containing path-level results. It
validates that `across` contains configured IDs, retains IDs not listed in `across`,
groups by the remaining analysis dimensions, calculates supported summaries, and
reports the number of contributing paths. For `across=["draw"]`, `scenario` remains;
for `across=["draw", "scenario"]`, both dimensions are removed. Source, variable,
metric, frequency, and horizon remain, so unrelated rows are never joined. Unknown
IDs raise a clear error; empty results retain their schema where possible.

## 17. MSE aggregation and standard error
Accuracy is calculated first, independently for every path. For one fixed scenario,
model, variable, metric, frequency, and horizon, let `mse_d` be the MSE returned for
draw `d`. The cross-draw estimate is the mean of those draw-level MSEs:

```python
mse_mean = np.mean(mse_by_draw)
```

The Monte Carlo standard error of that mean is:

```python
mse_se = np.std(mse_by_draw, ddof=1) / np.sqrt(n_draws)
```

This measures uncertainty from using a finite number of simulated draws. It is not
the standard error of individual forecast errors and it is not a pooled MSE. The
same rule applies to draw-level RMSE, MAE, bias coefficients, and other path-level
estimators when a mean across draws is meaningful.

The output should make the distinction visible:

```text
scenario, unique_id, variable, metric, frequency, horizon,
mse_mean, mse_se, mse_sd, n_draws
```

For a 95% Monte Carlo interval, use an explicitly documented normal or Student-t
multiplier, for example `mse_mean +/- t_(n_draws-1, .975) * mse_se`. Percentiles of
draw-level MSEs describe their simulated distribution; they are not interchangeable
with the standard error of the mean.

The formula assumes independent draws. If common random numbers or another design
induces dependence, the implementation must document the limitation or use a
dependence-aware estimate. A bootstrap over draw IDs is an optional robustness check,
not a substitute for defining the estimand.

When draws have different numbers of forecast observations, the default estimate is
the unweighted mean of draw-level MSEs: every path receives equal weight. An
observation-weighted pooled MSE is a separate estimand and must be requested
explicitly.

## 18. Distribution and pooled metrics
The initial method summarizes draw-level result statistics. For a numeric metric such
as `mse`, output may include:
```text
mse_mean, mse_se, mse_sd, mse_p05, mse_median, mse_p95, n_draws
```
and tests may use rejection rates.

The percentile set should become configurable. The method must not summarize every
numeric column blindly; counts and dates need explicit rules. Accuracy uses MSE, RMSE,
and MAE distributions, bias uses coefficient distributions, model comparisons use win
rates, and tests may use rejection rates.

Draw-level MSE and pooled MSE are different:
```python
mse_draw = np.mean(error_draw**2)
mse_pooled = np.mean(all_errors**2)
```
The first release prioritizes draw-level means and Monte Carlo standard errors.
Pooled metrics require an explicit weighting policy and must not appear as an
accidental groupby mean. Pooled tests require an explicit statistical method.
## 19. Copy, visualization, and scale
`SimulationData.copy()` preserves configuration, panel children, simulation columns,
and child raw and derived tables, while remaining independent of its source.
Plots should require a path filter or explicit aggregation; overlaying 1,000 paths by
default is unreadable. Existing ordinary plots remain unchanged, and simulation
columns remain available to custom plotting code. A later API may provide scenario
small multiples and uncertainty bands.
The dominant cost is independent model fitting, not two extra columns. Partition raw
data by path once, avoid full-panel copies in inner loops, concatenate forecasts once,
and benchmark realistic 1,000-draw workloads before optimizing.
If memory becomes a problem, partitioned Parquet output or a streaming result sink is
preferable to creating a second data model. A natural partition key is scenario and
draw.
## 20. Test fixtures and schema tests
The smallest useful fixture has two draws and two scenarios with deliberately
different histories. Forecast values should expose cross-path contamination; one path
should have a distinctive level and one scenario a distinctive revision.
Constructor and schema tests cover default IDs, missing draw, empty IDs, duplicate IDs,
reserved names, custom order, null identifiers, invalid fractional draws, and strict
schema retention.
Ingestion tests verify that rows differing only by draw or scenario survive, complete
metadata duplicates are detected, simulation columns survive all stored tables, and
unknown forecast paths are rejected.
## 21. Isolation and core tests
Transformation tests verify period-on-period changes, year-on-year changes, level
reconstruction, and synthetic vintages independently per path.
Main-table tests create distinct forecasts and outturns for two paths, assert one row
per expected path and forecast identity, assert matching forecast errors, and assert
that no Cartesian cross-path rows exist.
Revision tests create different histories, then assert that each path has its own
`k=0`, `k=1`, and latest-vintage values.
Analysis tests run accuracy and bias on two paths, assert retained identifiers, and
verify that coefficients and statistics are not pooled accidentally.
## 22. Realtime and aggregation tests
Realtime tests run one model over deliberately different outturn paths and assert that
each forecast depends only on its own history. Conditioning tests verify that paths
do not borrow conditioning or regressor forecasts. Vintage tests cover path-specific
release calendars, and decomposition tests verify simulation tags.
Aggregation tests cover retaining scenario while removing draw, removing both IDs,
hand-calculated means and quantiles, correct `n_draws`, and unknown IDs.
The complete ordinary `ForecastData` suite must continue to pass. Ordinary output,
source IDs, transformations, benchmark behavior, and realtime behavior must remain
unchanged, with no simulation columns appearing on ordinary data.
## 23. Documentation and exports
Export `SimulationData` from `forecast_evaluation.data` and the package root. Add it
to API documentation. Document outturn and forecast schemas, draw and scenario
semantics, independent model fitting, and draw-level versus pooled metrics.
Add an end-to-end example:
```python
sim_data = fe.SimulationData(outturns_data=simulated_outturns)
realtime = rt.RealTimeModel(data=sim_data, models=model)
realtime.forecast(y_variables=["gdp"], frequency="Q", steps=8)
accuracy = sim_data.evaluate_accuracy(k=0)
summary = sim_data.aggregate_accuracy(accuracy, across=["draw"])
```
`evaluate_accuracy` is the convenience wrapper around
`sim_data.evaluate(fe.compute_accuracy_statistics, k=0)`, so the caller never writes
a draw loop.
## 24. Implementation phases
### Phase 1: simulation façade
Add `SimulationData`, identifier validation, panel storage, `iter_panels`, and explicit
`add_outturns`/`add_forecasts` delegation. Keep `ForecastData` unchanged and run its
ordinary suite.
### Phase 2: ingestion
Add outer-frame validation, path-key duplicate checks, and constructor/schema tests.
### Phase 3: core evaluation
Delegate transformations, level reconstruction, main-table joins, and revision
grouping to child `ForecastData` objects. Add path-isolation tests.
### Phase 4: analysis
Add `evaluate`, tag child results with path IDs, and verify accuracy, bias, and one
regression-based analysis. Verify ordinary output remains unchanged.
### Phase 5: aggregation
Add `aggregate_accuracy`, `aggregate_results`, distribution summaries, Monte Carlo
SEs, draw counts, retained-dimension tests, and aggregation documentation.
### Phase 6: realtime
Teach `RealTimeModel` to iterate over `SimulationData.iter_panels`, filter conditioning
data by path, reuse the existing vintage loop, tag outputs, and pass integration tests.
### Phase 7: optimization
Benchmark 1,000 draws, measure memory, add panel-level parallelism if required, and
consider partitioned output only after correctness is established.
## 25. Acceptance criteria
- `SimulationData` is importable from the package root.
- Default IDs are `draw` and `scenario`.
- A configuration without `draw` is rejected.
- Simulation columns are required and retained on outturns and forecasts.
- Ordinary `ForecastData` behavior is unchanged.
- Transformations, main-table joins, and revisions are isolated by path.
- Existing analyses produce draw-level rows with simulation identifiers.
- Aggregation can retain scenario while removing draw.
- Aggregation can remove both default IDs.
- `RealTimeModel` fits independently for every path.
- Conditioning data cannot cross paths.
- Generated forecasts retain simulation identifiers and model source.
- Existing and new simulation tests pass.
- Documentation includes the complete workflow.
## 26. Open decisions
The recommended default is `simulation_ids=["draw", "scenario"]`; a single scenario
uses `baseline`. Draws may be any non-negative integer. The first version permits
unbalanced panels and uses path-specific available vintages.
The first version prioritizes draw-level metric distributions. Pooled metrics require
explicit weighting, pooled tests require explicit statistical methods, and a dedicated
`SimulationResult` is deferred until raw errors or specialized inference require it.
## 27. Final architecture
```text
simulated outturns
        |
        v
SimulationData
        |
    +-- one unchanged ForecastData per path
        +-- IDs: draw, scenario
        |
        v
RealTimeModel
        |
        +-- loop over simulation paths
        +-- fit each model independently
        +-- return tagged forecasts
        |
        v
SimulationData tagged results
        |
        +-- existing evaluation functions
        +-- draw-level result rows
        |
        v
aggregate_results()
        |
        +-- distributions across draws
        +-- scenario comparisons
        +-- pooled metrics later
```
The governing rule is:
```text
evaluate each simulation path independently;
aggregate only after evaluation.
```
`ForecastData` remains unchanged and owns all single-path validation and evaluation
logic. `SimulationData` owns path partitioning, child orchestration, result tagging,
and cross-simulation aggregation. `forecast-realtime` reuses the same child execution
loop for every path. This provides simulation support without a second single-path
evaluation framework.

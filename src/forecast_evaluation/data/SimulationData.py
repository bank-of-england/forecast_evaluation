import copy
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd

from forecast_evaluation.core.main_table import build_main_table
from forecast_evaluation.core.transformations import prepare_forecasts, prepare_outturns
from forecast_evaluation.data.ForecastData import ForecastData, _fix_extra_columns, _validate_records

_RESERVED_SIMULATION_IDS = {
    "date",
    "vintage_date",
    "variable",
    "frequency",
    "value",
    "source",
    "horizon",
    "forecast_horizon",
    "metric",
    "unique_id",
    "target_minus_vintage",
    "vintage_date_forecast",
    "vintage_date_outturn",
    "value_forecast",
    "value_outturn",
    "k",
    "latest_vintage",
    "forecast_error",
}
_RESULT_DIMENSIONS = ["unique_id", "variable", "metric", "frequency", "horizon"]
_RESULT_CONTRACTS = {
    "compute_accuracy_statistics": {
        "statistics": ["mse", "rmse", "mean_abs_error", "rmedse"],
        "metadata": {"n_observations": "int64", "start_date": "datetime64[ns]", "end_date": "datetime64[ns]"},
    },
    "bias_analysis": {
        "statistics": ["bias_estimate"],
        "metadata": {
            "std_error": "float64",
            "t_statistic": "float64",
            "p_value": "float64",
            "bias_conclusion": "object",
            "n_observations": "float64",
            "hac_maxlags": "int64",
            "ci_lower": "float64",
            "ci_upper": "float64",
        },
    },
}
_UNSET = object()


class SimulationData:
    """A composition facade containing one :class:`ForecastData` per path."""

    def __init__(
        self,
        outturns_data: Optional[pd.DataFrame] = None,
        forecasts_data: Optional[pd.DataFrame] = None,
        *,
        simulation_ids: Sequence[str] = ("draw", "scenario"),
        metric: Literal["levels", "pop", "yoy"] = "levels",
        compute_levels: bool = True,
        data_check: bool = True,
        outturn_vintages: bool = True,
        default_k: Optional[int] = None,
        first_forecast_horizon: Optional[Union[int, dict[str, int]]] = None,
    ):
        self._simulation_ids = self._validate_simulation_ids(simulation_ids)
        self._panels: dict[tuple[Any, ...], ForecastData] = {}
        self._outturn_vintages = outturn_vintages
        self.default_k = ForecastData.default_k if default_k is None else default_k
        self._forecast_options = {
            "compute_levels": compute_levels,
            "data_check": data_check,
            "first_forecast_horizon": first_forecast_horizon,
        }
        self._compute_levels = compute_levels
        self._metric = metric
        self._extra_ids: Optional[list[str]] = None

        if outturns_data is not None:
            self.add_outturns(outturns_data, metric=metric)
        if forecasts_data is not None:
            self.add_forecasts(
                forecasts_data,
                metric=metric,
                compute_levels=compute_levels,
                data_check=data_check,
                first_forecast_horizon=first_forecast_horizon,
            )

    @staticmethod
    def _validate_simulation_ids(simulation_ids: Sequence[str]) -> list[str]:
        if isinstance(simulation_ids, str) or not isinstance(simulation_ids, Sequence):
            raise ValueError("simulation_ids must be a non-empty sequence of unique strings.")
        ids = list(simulation_ids)
        if not ids:
            raise ValueError("simulation_ids must be a non-empty sequence of unique strings.")
        if any(not isinstance(identifier, str) or not identifier for identifier in ids):
            raise ValueError("simulation_ids must contain only non-empty strings.")
        if len(ids) != len(set(ids)):
            raise ValueError("simulation_ids must contain unique names.")
        if "draw" not in ids:
            raise ValueError("simulation_ids must include 'draw'.")
        reserved = sorted(set(ids) & _RESERVED_SIMULATION_IDS)
        if reserved:
            raise ValueError(f"Simulation identifiers cannot use reserved column names: {reserved}.")
        return ids

    @property
    def simulation_ids(self) -> list[str]:
        """Return the configured simulation identifiers."""
        return self._simulation_ids.copy()

    @property
    def panels(self) -> dict[tuple[Any, ...], ForecastData]:
        """Return the path panels, keyed by the configured simulation IDs."""
        return self._panels.copy()

    @property
    def id_columns(self) -> Optional[list[str]]:
        """Return the forecast identity columns shared by the panels."""
        for panel in self._panels.values():
            if panel.id_columns is not None:
                return panel.id_columns.copy()
        return None

    @property
    def outturn_vintages(self) -> bool:
        """Whether outturn vintage information is available."""
        return self._outturn_vintages

    @property
    def outturn_required_columns(self) -> list[str]:
        """Return the ordinary outturn required columns."""
        return ForecastData().outturn_required_columns

    @property
    def forecast_required_columns(self) -> list[str]:
        """Return the ordinary forecast required columns."""
        return ForecastData().forecast_required_columns

    def _normalise_frame(self, df: pd.DataFrame, *, forecast: bool) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Simulation data must be supplied as a pandas DataFrame.")
        missing = [identifier for identifier in self._simulation_ids if identifier not in df.columns]
        if missing:
            raise ValueError(f"Simulation data is missing required identifier columns: {missing}.")
        if forecast and "forecast_horizon" not in df:
            raise ValueError("Simulation forecasts require 'forecast_horizon'.")

        result = df.copy()
        result["draw"] = self._normalise_draws(result["draw"])
        for identifier in self._simulation_ids:
            if identifier == "draw":
                continue
            values = result[identifier]
            if values.isna().any():
                raise ValueError(f"Simulation identifier '{identifier}' cannot contain null values.")
            values = values.astype(str)
            if values.str.strip().eq("").any():
                raise ValueError(f"Simulation identifier '{identifier}' cannot contain empty values.")
            result[identifier] = values

        return result

    def _validate_panel_records(self, frame, panel, *, forecast, metric="levels", extra_ids=None):
        frame = frame.copy()
        frame["metric"] = frame["metric"].fillna(metric) if "metric" in frame else metric
        if not forecast and not self._outturn_vintages and "vintage_date" not in frame:
            frame["vintage_date"] = pd.NaT
        optional = ["metric"]
        if forecast and extra_ids:
            frame, extra_ids = _fix_extra_columns(frame, extra_ids)
            if len(extra_ids) != len(set(extra_ids)) or set(extra_ids) & (
                _RESERVED_SIMULATION_IDS | set(self._simulation_ids)
            ):
                raise ValueError(
                    "Forecast extra_ids must be distinct from core and simulation columns after normalisation."
                )
            optional += extra_ids
        validated = _validate_records(
            frame,
            forecast=forecast,
            optional_columns=optional,
            nullable_vintage=not forecast and not self._outturn_vintages,
        )
        stored = panel._raw_forecasts if forecast else panel._raw_outturns
        if not stored.empty:
            _validate_records(
                pd.concat([stored.reindex(columns=validated.columns), validated], ignore_index=True),
                forecast=forecast,
                optional_columns=optional,
                nullable_vintage=not forecast and not self._outturn_vintages,
            )
        return validated

    @staticmethod
    def _normalise_draws(values: pd.Series) -> pd.Series:
        if values.isna().any() or values.map(lambda value: isinstance(value, bool)).any():
            raise ValueError("Simulation identifier 'draw' must contain non-negative integers.")
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric) & (numeric >= 0) & (numeric < 2**63) & (numeric % 1 == 0)
        if not valid.all():
            raise ValueError("Simulation identifier 'draw' must contain non-negative integers.")
        return numeric.astype("int64")

    def _partition(self, df: pd.DataFrame) -> Iterable[tuple[tuple[Any, ...], pd.DataFrame]]:
        grouped = df.groupby(self._simulation_ids, sort=False, dropna=False)
        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            yield tuple(key), group.drop(columns=self._simulation_ids).reset_index(drop=True)

    def _new_panel(self) -> ForecastData:
        return ForecastData(
            outturn_vintages=self._outturn_vintages,
            default_k=self.default_k,
            first_forecast_horizon=self._forecast_options["first_forecast_horizon"],
        )

    def add_outturns(self, df: pd.DataFrame, *, metric: Literal["levels", "pop", "yoy"] = "levels") -> None:
        """Validate, partition, and add simulation outturns."""
        normalised = self._normalise_frame(df, forecast=False)
        panels = {key: panel.copy() for key, panel in self._panels.items()}
        for key, panel_df in self._partition(normalised):
            panel = panels.setdefault(key, self._new_panel())
            panel_df = self._validate_panel_records(panel_df, panel, forecast=False, metric=metric)
            panel.add_outturns(panel_df, metric=metric)
            self._rebuild_panel(panel)
        self._panels = panels

    def add_forecasts(
        self,
        df: pd.DataFrame,
        *,
        extra_ids: Optional[list[str]] = None,
        metric: Literal["levels", "pop", "yoy"] = "levels",
        compute_levels: Optional[bool] = None,
        data_check: bool = True,
        first_forecast_horizon: Union[Optional[int], dict[str, int], object] = _UNSET,
    ) -> None:
        """Add forecasts using the collection's transformation policy and identity schema.

        Omitted ``compute_levels`` inherits the constructor setting; explicit values
        must match it. The first non-empty addition establishes the ordered,
        normalised ``extra_ids``. Later additions inherit these IDs when omitted.
        """
        if compute_levels is None:
            compute_levels = self._compute_levels
        elif compute_levels != self._compute_levels:
            raise ValueError("compute_levels must match the SimulationData constructor setting.")
        normalised = self._normalise_frame(df, forecast=True)
        if extra_ids is not None:
            if set(extra_ids) & set(self._simulation_ids):
                raise ValueError("Forecast extra_ids cannot contain simulation identifiers.")
            normalised, extra_ids = _fix_extra_columns(normalised, extra_ids)
        else:
            extra_ids = list(self._extra_ids or [])
        if self._extra_ids is not None and extra_ids != self._extra_ids:
            raise ValueError("Forecast extra_ids must match the existing simulation forecast identity columns.")

        keys = list(self._partition(normalised))
        unknown = [key for key, _ in keys if key not in self._panels]
        if unknown:
            details = [dict(zip(self._simulation_ids, key)) for key in unknown]
            raise ValueError(f"Forecast path has no outturns: {details}.")

        panels = {key: panel.copy() for key, panel in self._panels.items()}
        for key, panel_df in keys:
            panel_df = self._validate_panel_records(
                panel_df,
                panels[key],
                forecast=True,
                metric=metric,
                extra_ids=extra_ids,
            )
            forecast_options = {
                "extra_ids": extra_ids,
                "metric": metric,
                "compute_levels": compute_levels,
                "data_check": data_check,
            }
            if first_forecast_horizon is not _UNSET:
                forecast_options["first_forecast_horizon"] = first_forecast_horizon
            panels[key].add_forecasts(panel_df, **forecast_options)
        if keys:
            self._extra_ids = extra_ids.copy()
        self._panels = panels

    def iter_panels(self):
        """Yield ``(simulation_key, ForecastData)`` pairs in insertion order."""
        for key, panel in self._panels.items():
            if not panel.outturns.empty or not panel.forecasts.empty or not panel.df.empty:
                yield key, panel

    def _tag(self, frame: pd.DataFrame, key: tuple[Any, ...]) -> pd.DataFrame:
        tagged = frame.copy()
        for identifier, value in reversed(list(zip(self._simulation_ids, key))):
            tagged.insert(0, identifier, value)
        return tagged

    def _combined(self, attribute: str) -> pd.DataFrame:
        frames = [self._tag(getattr(panel, attribute), key) for key, panel in self._panels.items()]
        if not frames:
            return pd.DataFrame(columns=self._simulation_ids)
        return pd.concat(frames, ignore_index=True)

    @property
    def outturns(self) -> pd.DataFrame:
        """Return transformed outturns with simulation identifiers attached."""
        return self._combined("outturns")

    @property
    def forecasts(self) -> pd.DataFrame:
        """Return transformed forecasts with simulation identifiers attached."""
        return self._combined("forecasts")

    @property
    def df(self) -> pd.DataFrame:
        """Return the combined main table with simulation identifiers attached."""
        return self._combined("df")

    def evaluate(self, analysis: Callable[..., Any], **kwargs: Any) -> Any:
        """Run a single-panel analysis independently for every simulation path."""
        from forecast_evaluation.tests.results import TestResult

        results = []
        template = None
        path_metadata = []
        for key, panel in self.iter_panels():
            try:
                result = analysis(panel, **kwargs)
            except Exception as error:
                path = dict(zip(self._simulation_ids, key))
                raise RuntimeError(f"Analysis failed for simulation path {path}: {error}") from error
            if not isinstance(result, (TestResult, pd.DataFrame)):
                raise TypeError(f"Analysis must return a TestResult or DataFrame; path: {key}.")
            frame = result.to_df() if isinstance(result, TestResult) else result.copy()
            results.append(self._tag(frame, key))
            if isinstance(result, TestResult):
                template = result
                path_metadata.append({"path": dict(zip(self._simulation_ids, key)), **copy.deepcopy(result._metadata)})

        if not results:
            return self._empty_evaluation(analysis, kwargs)
        combined = pd.concat(results, ignore_index=True)
        metadata = {
            "simulation_ids": self.simulation_ids,
            "path_metadata": path_metadata,
            "evaluation": "independent paths",
        }
        combined.attrs.update(metadata)
        if template is not None:
            wrapped = copy.deepcopy(template)
            wrapped._df = combined
            wrapped._metadata.pop("date_range", None)
            wrapped._metadata.update(metadata)
            return wrapped
        return combined

    def _empty_evaluation(self, analysis: Callable[..., Any], parameters: dict) -> Any:
        from forecast_evaluation.tests.accuracy import compute_accuracy_statistics
        from forecast_evaluation.tests.bias import bias_analysis
        from forecast_evaluation.tests.results import TestResult

        schema = {identifier: "int64" if identifier == "draw" else "object" for identifier in self._simulation_ids}
        if analysis not in (compute_accuracy_statistics, bias_analysis):
            return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in schema.items()})
        contract = _RESULT_CONTRACTS[analysis.__name__]
        schema.update({column: "int64" if column == "horizon" else "object" for column in _RESULT_DIMENSIONS})
        schema.update({column: "object" for column in self.id_columns or ["source"]})
        schema.update({column: "float64" for column in contract["statistics"]})
        schema.update(contract["metadata"])
        frame = pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in schema.items()})
        result = TestResult(
            frame,
            metadata={
                "test_name": analysis.__name__,
                "parameters": parameters,
                "simulation_ids": self.simulation_ids,
                "path_metadata": [],
                "evaluation": "independent paths",
            },
        )
        result._id_columns = self.id_columns
        return result

    def evaluate_accuracy(self, **kwargs: Any) -> Any:
        """Run ``compute_accuracy_statistics`` independently for every path."""
        from forecast_evaluation.tests.accuracy import compute_accuracy_statistics

        return self.evaluate(compute_accuracy_statistics, **kwargs)

    def aggregate_accuracy(self, result: Any, *, across: Iterable[str] = ("draw",)) -> Any:
        """Aggregate draw-level accuracy statistics across configured IDs."""
        return self._aggregate(result, across=across, contract_name="compute_accuracy_statistics")

    def aggregate_results(
        self,
        result: Any,
        *,
        across: Iterable[str] = ("draw",),
        statistics: Optional[Sequence[str]] = None,
        group_by: Optional[Sequence[str]] = None,
    ) -> Any:
        """Summarise supported estimates or an explicit estimate/dimension contract.

        Paths receive equal weight. Monte Carlo standard errors assume independent
        draws and are available only when removing ``draw`` alone. Explicit frames
        must contain only simulation IDs, declared dimensions and estimates.
        """
        from forecast_evaluation.tests.results import TestResult

        contract_name = result._metadata.get("test_name") if isinstance(result, TestResult) else None
        return self._aggregate(
            result, across=across, contract_name=contract_name, statistics=statistics, group_by=group_by
        )

    def _aggregate(self, result, *, across, contract_name, statistics=None, group_by=None):
        from forecast_evaluation.tests.results import TestResult

        if not isinstance(result, (TestResult, pd.DataFrame)):
            raise TypeError("Results must be a TestResult or DataFrame.")
        frame = result.to_df() if isinstance(result, TestResult) else result.copy()
        if isinstance(across, str):
            raise ValueError("across must be a sequence of simulation identifiers, not a string.")
        across_ids = list(across)
        unknown = [identifier for identifier in across_ids if identifier not in self._simulation_ids]
        if unknown:
            raise ValueError(f"Cannot aggregate across unknown simulation identifiers: {unknown}.")
        if not across_ids:
            raise ValueError("across must contain at least one simulation identifier.")
        if len(across_ids) != len(set(across_ids)):
            raise ValueError("across must contain unique simulation identifiers.")
        missing = [identifier for identifier in self._simulation_ids if identifier not in frame.columns]
        if missing:
            raise ValueError(f"Result is missing simulation identifier columns: {missing}.")
        if not frame.columns.is_unique:
            raise ValueError("Result column names must be unique.")
        if statistics is not None or group_by is not None:
            if statistics is None or group_by is None:
                raise ValueError("Explicit contracts require both statistics and group_by.")
            statistics = self._column_names(statistics, "statistics", allow_empty=False)
            dimensions = self._column_names(group_by, "group_by", allow_empty=True)
            contract_name = "explicit"
            metadata_columns = []
        else:
            if contract_name not in _RESULT_CONTRACTS:
                raise ValueError("Unknown automatic result contract; supply statistics and group_by.")
            contract = _RESULT_CONTRACTS[contract_name]
            statistics = contract["statistics"]
            identity_columns = result._id_columns if isinstance(result, TestResult) else self.id_columns
            dimensions = list(dict.fromkeys(_RESULT_DIMENSIONS + (identity_columns or ["source"])))
            metadata_columns = list(contract["metadata"])
        declared = self._simulation_ids + dimensions + statistics + metadata_columns
        if len(declared) != len(set(declared)):
            raise ValueError("Simulation IDs, dimensions, estimates and metadata must not overlap.")
        missing = [column for column in declared if column not in frame]
        if missing:
            raise ValueError(f"Result is missing required columns: {missing}.")
        undeclared = [column for column in frame if column not in declared]
        if undeclared:
            raise ValueError(f"Result contains undeclared columns: {undeclared}.")
        frame["draw"] = self._normalise_draws(frame["draw"])
        for identifier in self._simulation_ids:
            if identifier != "draw":
                if frame[identifier].isna().any() or frame[identifier].astype(str).str.strip().eq("").any():
                    raise ValueError(f"Invalid simulation identifier '{identifier}'.")
                frame[identifier] = frame[identifier].astype(str)
        if frame.duplicated(self._simulation_ids + dimensions).any():
            raise ValueError("Duplicate result rows for a simulation path and analysis combination.")
        for statistic in statistics:
            frame[statistic] = pd.to_numeric(frame[statistic], errors="raise").astype("float64")
            if np.isinf(frame[statistic]).any():
                raise ValueError(f"Statistic '{statistic}' contains infinite values.")

        group_columns = [identifier for identifier in self._simulation_ids if identifier not in across_ids] + dimensions
        grouped = frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [((), frame)]
        independent_draws = across_ids == ["draw"]
        schema = {"n_paths": "int64", "n_draws": "int64"}
        for statistic in statistics:
            schema.update(
                {f"{statistic}_{suffix}": "float64" for suffix in ["mean", "se", "sd", "p05", "median", "p95"]}
            )
            schema.update({f"{statistic}_n": "int64", f"{statistic}_n_missing": "int64"})
        coverage = "n_observations" in metadata_columns
        dates = [column for column in ("start_date", "end_date") if column in metadata_columns]
        if coverage:
            schema.update({f"n_observations_{suffix}": "float64" for suffix in ("total", "min", "max")})
        schema.update({column: "datetime64[ns]" for column in dates})
        collisions = sorted(set(group_columns) & schema.keys())
        if collisions:
            raise ValueError(f"Grouping columns conflict with generated summary columns: {collisions}.")
        group_schema = {column: frame[column].dtype for column in group_columns}
        if contract_name != "explicit":
            group_schema.update({column: "int64" if column == "horizon" else "object" for column in dimensions})
        schema = {**group_schema, **schema}
        rows = []
        for group_key, group in grouped:
            if group.empty:
                continue
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = dict(zip(group_columns, group_key))
            row["n_paths"] = group[self._simulation_ids].drop_duplicates().shape[0]
            row["n_draws"] = group["draw"].nunique()
            for column in statistics:
                values = group[column].dropna().to_numpy(dtype=float)
                count = len(values)
                deviation = np.std(values, ddof=1) if count > 1 else np.nan
                row[f"{column}_mean"] = np.mean(values) if count else np.nan
                row[f"{column}_se"] = deviation / np.sqrt(count) if independent_draws and count > 1 else np.nan
                row[f"{column}_sd"] = deviation
                for suffix, percentile in [("p05", 5), ("median", 50), ("p95", 95)]:
                    row[f"{column}_{suffix}"] = np.percentile(values, percentile) if count else np.nan
                row[f"{column}_n"] = count
                row[f"{column}_n_missing"] = len(group) - count
            if coverage:
                counts = pd.to_numeric(group["n_observations"], errors="raise")
                row["n_observations_total"] = counts.sum(min_count=1)
                row["n_observations_min"] = counts.min()
                row["n_observations_max"] = counts.max()
            for column in dates:
                values = pd.to_datetime(group[column])
                row[column] = values.min() if column == "start_date" else values.max()
            rows.append(row)

        aggregated = pd.DataFrame(rows, columns=list(schema)).astype(schema)
        aggregated.attrs = {
            "contract": contract_name,
            "across": across_ids,
            "statistics": list(statistics),
            "group_by": group_columns,
            "weighting": "equal paths",
            "se_assumption": "independent draws"
            if independent_draws
            else "descriptive only; independence not established",
            "coverage": "represented paths only; dates describe a coverage envelope",
            "simulation_ids": self.simulation_ids,
        }
        return self._wrap_aggregate_result(result, aggregated)

    @staticmethod
    def _column_names(columns, argument, *, allow_empty):
        if isinstance(columns, str) or not isinstance(columns, Sequence):
            raise ValueError(f"{argument} must be a sequence of column names.")
        names = list(columns)
        if (not names and not allow_empty) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{argument} must contain column names.")
        if len(names) != len(set(names)):
            raise ValueError(f"{argument} must contain unique column names.")
        return names

    @staticmethod
    def _wrap_aggregate_result(original: Any, frame: pd.DataFrame) -> Any:
        from forecast_evaluation.tests.results import TestResult

        if isinstance(original, TestResult):
            wrapped = copy.deepcopy(original)
            wrapped._df = frame
            wrapped._metadata.pop("date_range", None)
            wrapped._metadata.update(frame.attrs)
            return wrapped
        return frame

    def filter(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_vintage: Optional[str] = None,
        end_vintage: Optional[str] = None,
        variables: Optional[Union[str, list[str]]] = None,
        metrics: Optional[list[str]] = None,
        sources: Optional[Union[str, list[str]]] = None,
        frequencies: Optional[Union[str, list[str]]] = None,
        custom_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> None:
        """Apply ordinary filters to every panel, including tagged custom filters."""
        for key, panel in self._panels.items():
            panel_filter = None
            if custom_filter is not None:

                def panel_filter(frame, key=key, custom_filter=custom_filter):
                    tagged = self._tag(frame, key)
                    filtered = custom_filter(tagged)
                    return filtered.drop(columns=self._simulation_ids, errors="ignore")

            panel.filter(
                start_date=start_date,
                end_date=end_date,
                start_vintage=start_vintage,
                end_vintage=end_vintage,
                variables=variables,
                metrics=metrics,
                sources=sources,
                frequencies=frequencies,
                custom_filter=panel_filter,
            )

    def clear_filter(self) -> None:
        """Restore every panel to its raw, unfiltered data."""
        for panel in self._panels.values():
            self._rebuild_panel(panel)

    def _rebuild_panel(self, panel: ForecastData) -> None:
        forecasts = prepare_forecasts(
            panel._raw_forecasts,
            panel._raw_outturns,
            panel._id_columns,
            compute_levels=self._compute_levels,
        )
        outturns = prepare_outturns(panel._raw_outturns)
        panel._forecasts = forecasts
        panel._outturns = outturns
        if forecasts.empty:
            panel._main_table = pd.DataFrame()
            return
        panel._main_table = build_main_table(
            forecasts,
            outturns,
            panel._id_columns,
            frequency=forecasts["frequency"].iloc[0] if not forecasts.empty else "Q",
            outturn_vintages=panel.outturn_vintages,
        )

    def add_benchmarks(self, *args: Any, **kwargs: Any) -> None:
        """Add benchmark forecasts independently to every path."""
        for panel in self._panels.values():
            panel.add_benchmarks(*args, **kwargs)

    def create_pseudo_vintages(self, *args: Any, **kwargs: Any) -> None:
        """Create pseudo outturn vintages independently for every path."""
        for panel in self._panels.values():
            panel.create_pseudo_vintages(*args, **kwargs)

    def merge(self, other: "SimulationData") -> None:
        """Merge a collection with matching simulation IDs, settings and forecast identity columns."""
        if not isinstance(other, SimulationData):
            raise TypeError("Can only merge another SimulationData instance.")
        if self._simulation_ids != other._simulation_ids:
            raise ValueError("Cannot merge SimulationData instances with different simulation IDs.")
        if self._outturn_vintages != other._outturn_vintages:
            raise ValueError("Cannot merge SimulationData instances with different outturn_vintages settings.")
        if self._compute_levels != other._compute_levels:
            raise ValueError("Cannot merge SimulationData instances with different compute_levels settings.")
        if self.default_k != other.default_k:
            raise ValueError("Cannot merge SimulationData instances with different default_k settings.")
        if self._extra_ids is not None and other._extra_ids is not None and self._extra_ids != other._extra_ids:
            raise ValueError("Cannot merge SimulationData instances with different forecast extra_ids.")
        next_extra_ids = self._extra_ids
        if next_extra_ids is None and other._extra_ids is not None:
            next_extra_ids = other._extra_ids.copy()
        panels = {key: panel.copy() for key, panel in self._panels.items()}
        for key, panel in other._panels.items():
            if key not in panels:
                panels[key] = panel.copy()
            else:
                self._validate_panel_records(panel._raw_outturns, panels[key], forecast=False)
                if not panel._raw_forecasts.empty:
                    self._validate_panel_records(
                        panel._raw_forecasts,
                        panels[key],
                        forecast=True,
                        extra_ids=other._extra_ids,
                    )
                panels[key].merge(panel, compute_levels=self._compute_levels)
                self._rebuild_panel(panels[key])
        self._extra_ids = next_extra_ids
        self._panels = panels

    def copy(self) -> "SimulationData":
        """Return an independent deep copy."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return self.df.__repr__()

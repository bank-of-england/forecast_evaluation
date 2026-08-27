import copy
import math
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd

from forecast_evaluation.core.main_table import build_main_table
from forecast_evaluation.core.transformations import prepare_forecasts, prepare_outturns
from forecast_evaluation.data.ForecastData import ForecastData


_RESERVED_SIMULATION_IDS = {
    "date",
    "vintage_date",
    "variable",
    "frequency",
    "value",
    "source",
    "forecast_horizon",
    "metric",
    "unique_id",
    "target_minus_vintage",
}
_AGGREGATABLE_RESULT_COLUMNS = {
    "mse",
    "rmse",
    "mean_abs_error",
    "rmedse",
    "bias",
    "coefficient",
    "estimate",
    "correlation",
    "slope",
    "intercept",
    "statistic",
    "score",
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

        identity_columns = [column for column in result.columns if column != "value"]
        duplicate_mask = result.duplicated(identity_columns, keep=False)
        if duplicate_mask.any():
            duplicates = result.loc[duplicate_mask]
            if duplicates.groupby(identity_columns, dropna=False)["value"].nunique(dropna=False).gt(1).any():
                kind = "forecast" if forecast else "outturn"
                raise ValueError(f"Duplicate {kind} records found with different values.")
            result = result.drop_duplicates().reset_index(drop=True)
        return result

    @staticmethod
    def _normalise_draws(values: pd.Series) -> pd.Series:
        if values.isna().any() or values.map(lambda value: isinstance(value, bool)).any():
            raise ValueError("Simulation identifier 'draw' must contain non-negative integers.")
        numeric = pd.to_numeric(values, errors="coerce")
        valid = (
            numeric.notna()
            & np.isfinite(numeric)
            & (numeric >= 0)
            & (numeric < 2**63)
            & (numeric % 1 == 0)
        )
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
            panel.add_outturns(panel_df, metric=metric)
            if not panel._forecasts.empty:
                self._rebuild_panel(panel)
        self._panels = panels

    def add_forecasts(
        self,
        df: pd.DataFrame,
        *,
        extra_ids: Optional[list[str]] = None,
        metric: Literal["levels", "pop", "yoy"] = "levels",
        compute_levels: bool = True,
        data_check: bool = True,
        first_forecast_horizon: Union[Optional[int], dict[str, int], object] = _UNSET,
    ) -> None:
        """Validate, partition, and add simulation forecasts."""
        normalised = self._normalise_frame(df, forecast=True)
        next_extra_ids = self._extra_ids
        if extra_ids is not None:
            if set(extra_ids) & set(self._simulation_ids):
                raise ValueError("Forecast extra_ids cannot contain simulation identifiers.")
            next_extra_ids = extra_ids.copy()
        elif self._extra_ids is not None:
            extra_ids = self._extra_ids.copy()

        keys = list(self._partition(normalised))
        unknown = [key for key, _ in keys if key not in self._panels]
        if unknown:
            details = [dict(zip(self._simulation_ids, key)) for key in unknown]
            raise ValueError(f"Forecast path has no outturns: {details}.")

        panels = {key: panel.copy() for key, panel in self._panels.items()}
        for key, panel_df in keys:
            forecast_options = {
                "extra_ids": extra_ids,
                "metric": metric,
                "compute_levels": compute_levels,
                "data_check": data_check,
            }
            if first_forecast_horizon is not _UNSET:
                forecast_options["first_forecast_horizon"] = first_forecast_horizon
            panels[key].add_forecasts(panel_df, **forecast_options)
        self._extra_ids = next_extra_ids
        self._compute_levels = compute_levels
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
        result_type = None
        for key, panel in self._panels.items():
            result = analysis(panel, **kwargs)
            result_type = type(result)
            frame = result.to_df() if isinstance(result, TestResult) else result.copy()
            tagged = self._tag(frame, key)
            if isinstance(result, TestResult):
                tagged_result = copy.deepcopy(result)
                tagged_result._df = tagged
                results.append(tagged_result)
            else:
                results.append(tagged)

        if not results:
            return pd.DataFrame(columns=self._simulation_ids)
        if result_type is not None and issubclass(result_type, TestResult):
            combined = copy.deepcopy(results[0])
            combined._df = pd.concat([result._df for result in results], ignore_index=True)
            return combined
        return pd.concat(results, ignore_index=True)

    def evaluate_accuracy(self, **kwargs: Any) -> Any:
        """Run ``compute_accuracy_statistics`` independently for every path."""
        from forecast_evaluation.tests.accuracy import compute_accuracy_statistics

        return self.evaluate(compute_accuracy_statistics, **kwargs)

    def aggregate_accuracy(self, result: Any, *, across: Iterable[str] = ("draw",)) -> Any:
        """Aggregate draw-level accuracy statistics across configured IDs."""
        return self.aggregate_results(result, across=across)

    def aggregate_results(self, result: Any, *, across: Iterable[str] = ("draw",)) -> Any:
        """Summarise numeric path-level result statistics across simulation IDs."""
        from forecast_evaluation.tests.results import TestResult

        frame = result.to_df() if isinstance(result, TestResult) else result.copy()
        across_ids = list(across)
        unknown = [identifier for identifier in across_ids if identifier not in self._simulation_ids]
        if unknown:
            raise ValueError(f"Cannot aggregate across unknown simulation identifiers: {unknown}.")
        if not across_ids:
            raise ValueError("across must contain at least one simulation identifier.")
        if len(across_ids) != len(set(across_ids)):
            raise ValueError("across must contain unique simulation identifiers.")
        missing = [identifier for identifier in self._simulation_ids if identifier not in frame.columns]
        if missing and not frame.empty:
            raise ValueError(f"Result is missing simulation identifier columns: {missing}.")

        if frame.empty:
            empty = frame.copy()
            return self._wrap_aggregate_result(result, empty)

        numeric_exclusions = set(self._simulation_ids) | {
            "horizon",
            "forecast_horizon",
            "n_observations",
            "p_value",
            "pvalue",
            "p_value_corrected",
        }
        numeric_columns = [
            column
            for column in frame.select_dtypes(include=np.number).columns
            if column not in numeric_exclusions and column in _AGGREGATABLE_RESULT_COLUMNS
        ]
        date_columns = [
            column
            for column in frame.columns
            if column in {"start_date", "end_date"}
            and pd.api.types.is_datetime64_any_dtype(frame[column])
        ]
        group_columns = [
            column
            for column in frame.columns
            if column not in numeric_columns + date_columns + list(set(across_ids)) and column != "n_observations"
        ]
        grouped = frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [((), frame)]
        rows = []
        for group_key, group in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = dict(zip(group_columns, group_key))
            path_count = group[across_ids].drop_duplicates().shape[0]
            row["n_draws"] = path_count
            for column in numeric_columns:
                values = group[column].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    row[f"{column}_mean"] = np.nan
                    row[f"{column}_se"] = np.nan
                    row[f"{column}_sd"] = np.nan
                    row[f"{column}_p05"] = np.nan
                    row[f"{column}_median"] = np.nan
                    row[f"{column}_p95"] = np.nan
                    continue
                row[f"{column}_mean"] = np.mean(values)
                row[f"{column}_se"] = np.std(values, ddof=1) / math.sqrt(len(values)) if len(values) > 1 else np.nan
                row[f"{column}_sd"] = np.std(values, ddof=1) if len(values) > 1 else np.nan
                row[f"{column}_p05"] = np.percentile(values, 5)
                row[f"{column}_median"] = np.median(values)
                row[f"{column}_p95"] = np.percentile(values, 95)
            if "n_observations" in group:
                row["n_observations"] = group["n_observations"].sum()
            for column in date_columns:
                row[column] = group[column].min() if column == "start_date" else group[column].max()
            rows.append(row)

        aggregated = pd.DataFrame(rows)
        ordered_columns = [column for column in group_columns if column in aggregated] + [
            column for column in aggregated.columns if column not in group_columns
        ]
        return self._wrap_aggregate_result(result, aggregated[ordered_columns])

    @staticmethod
    def _wrap_aggregate_result(original: Any, frame: pd.DataFrame) -> Any:
        from forecast_evaluation.tests.results import TestResult

        if isinstance(original, TestResult):
            wrapped = copy.deepcopy(original)
            wrapped._df = frame
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
        """Merge another simulation collection with matching ID configuration."""
        if not isinstance(other, SimulationData):
            raise TypeError("Can only merge another SimulationData instance.")
        if self._simulation_ids != other._simulation_ids:
            raise ValueError("Cannot merge SimulationData instances with different simulation IDs.")
        if self._outturn_vintages != other._outturn_vintages:
            raise ValueError("Cannot merge SimulationData instances with different outturn_vintages settings.")
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
                panels[key].merge(panel)
                self._rebuild_panel(panels[key])
        self._extra_ids = next_extra_ids
        self._panels = panels

    def copy(self) -> "SimulationData":
        """Return an independent deep copy."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return self.df.__repr__()
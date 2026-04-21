import copy
import re
import warnings
from collections.abc import Iterable
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd

from forecast_evaluation.core.main_table import build_main_table
from forecast_evaluation.core.transformations import prepare_forecasts, prepare_outturns
from forecast_evaluation.data.loader import load_fer_forecasts, load_fer_outturns
from forecast_evaluation.data.schema import FORECAST_REQUIRED_COLUMNS, OUTTURN_REQUIRED_COLUMNS, create_data_schema
from forecast_evaluation.data.utils import (
    compute_forecast_horizon,
    construct_unique_id,
    filter_fer_models,
    filter_fer_variables,
    filter_tables,
)

BENCHMARK_MODELS = ["AR", "random_walk"]


class ForecastData:
    """Class for validation and extending forecast data.

    The main method is .add_forecasts() which validates the input data and compute relevant dataframes.
    underscore indicates that the object only meant to be used internally.

    Notes
    -----
    Each ForecastData instance should only contain forecasts of a single frequency (e.g., all quarterly
    or all monthly). To work with multiple frequencies, create separate ForecastData instances for each frequency.
    """

    def __init__(
        self,
        outturns_data: Optional[pd.DataFrame] = None,
        forecasts_data: Optional[pd.DataFrame] = None,
        load_fer: Optional[bool] = False,
        *,
        extra_ids: Optional[list[str]] = None,
        metric: Literal["levels", "pop", "yoy"] = "levels",
        compute_levels: bool = True,
        data_check: bool = True,
        nowcasting: bool = False,
        first_forecast_horizon: int = 0,
    ):
        """Initialise with user data, FER data or null.

        Parameters
        ----------
        outturns_data : pd.DataFrame, optional
            DataFrame containing outturn records to add on initialisation. Default is None.
        forecasts_data : pd.DataFrame, optional
            DataFrame containing forecast records to add on initialisation. Default is None.
        load_fer : bool, optional
            Whether to load FER outturns and forecast data on initialisation. Default is False.
        extra_ids: Optional[list[str]], optional
            List of extra label columns (in addition to 'source') present in the forecasts data. Default is None.
        metric : str, optional
            Metric to assign to the forecasts if 'metric' column is not present or contains null values.
            Default is 'levels'. Options: 'levels', 'pop', 'yoy'.
        compute_levels : bool, optional
            Whether to automatically transform non-levels forecasts to levels if outturns data is available.
            When True, forecasts in 'pop' and 'yoy' metrics will be converted to levels
            using the available outturns data.
            Useful if you add 'pop' and want to analyse 'yoy' forecasts and vice versa.
            If the transformation fails for specific groups (e.g., due to insufficient
            historical data), those groups will be skipped with a warning message.
            Default is True.
        data_check : bool, optional
            Whether to run data checks when adding forecasts. See :meth:`add_forecasts` for details.
            Default is True.
        nowcasting : bool, optional
            Whether the data contains nowcasting forecasts with intra-period vintage dates
            (e.g., weekly or daily). When True, a ``days_in_period`` column is automatically
            computed and added as an extra label column. Default is False.
        first_forecast_horizon : int, optional
            The minimum forecast horizon to retain in processed forecasts.
            Set to a negative value (e.g., -1, -2) to include backcasts, i.e., forecasts
            for periods that have already ended but whose data has not yet been released.
            Default is 0 (only current-period and future forecasts).
        """
        self._raw_forecasts = pd.DataFrame()
        self._raw_outturns = pd.DataFrame()
        self._outturns = pd.DataFrame()
        self._forecasts = pd.DataFrame()
        self._main_table = pd.DataFrame()
        self._id_columns = None
        self.nowcasting = nowcasting
        self.first_forecast_horizon = first_forecast_horizon

        if load_fer:
            self.add_fer_data()

        if outturns_data is not None:
            self.add_outturns(outturns_data, metric=metric)

        if forecasts_data is not None:
            self.add_forecasts(
                forecasts_data,
                extra_ids=extra_ids,
                metric=metric,
                compute_levels=compute_levels,
                data_check=data_check,
            )

    def __repr__(self) -> str:
        """Return DataFrame representation when printing the class."""
        return self._raw_forecasts.__repr__()

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks."""
        return self._raw_forecasts._repr_html_()

    def copy(self):
        """Return a deep copy of the ForecastData object."""
        return copy.deepcopy(self)

    def add_outturns(self, df: pd.DataFrame, *, metric: Literal["levels", "pop", "yoy"] = "levels") -> None:
        """Validate new outturns and add them to the outturns dataset

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing new outturn records to add.
        metric : str, optional
            Metric to assign to the outturns if 'metric' column is not present or contains null values.
            Default is 'levels'. Options: 'levels', 'pop', 'yoy'.
        """
        # Compute forecast_horizon if missing
        if "forecast_horizon" not in df.columns:
            df = compute_forecast_horizon(df)

        # Handle metric column: use column values if present, otherwise use parameter
        if "metric" not in df.columns:
            df["metric"] = metric
        else:
            # Fill any null values in metric column with the parameter value
            df["metric"] = df["metric"].fillna(metric)

        # Validate that all metrics are valid
        valid_metrics = ["levels", "pop", "yoy"]
        invalid_metrics = df[~df["metric"].isin(valid_metrics)]["metric"].unique()
        if len(invalid_metrics) > 0:
            raise ValueError(f"Invalid metric values found: {invalid_metrics}. Valid options: {valid_metrics}")

        # Validate records using the ForecastRecord model
        # Include 'metric' as an optional column in validation
        df_validated = _validate_records(df, optional_columns=["metric"])

        # Check for duplicates if there are already some records stored
        if not self._raw_outturns.empty:
            df_validated_unique = _check_duplicates(df_validated, self._raw_outturns)
        else:
            df_validated_unique = df_validated

        # Transform outturns (prepare_outturns handles metric-specific logic)
        outturns = prepare_outturns(df_validated_unique)

        self._outturns = pd.concat([self._outturns, outturns], ignore_index=True)
        self._raw_outturns = pd.concat([self._raw_outturns, df_validated_unique], ignore_index=True)

    def add_forecasts(
        self,
        df: pd.DataFrame,
        *,
        extra_ids: Optional[list[str]] = None,
        metric: Literal["levels", "pop", "yoy"] = "levels",
        compute_levels: bool = True,
        data_check: bool = True,
    ) -> None:
        """Validate new forecasts, transform forecasts and outturns and compute main table and revisions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing new forecast records to add.
        extra_ids : list of str, optional
            List of extra label/identification columns (in addition to 'source') present in the forecasts data.
            Default is None.
        metric : str, optional
            Metric to assign to the forecasts if 'metric' column is not present or contains null values.
            Default is 'levels'. Options: 'levels', 'pop', 'yoy'.
        compute_levels : bool, optional
            Whether to automatically transform non-levels forecasts to levels if outturns data is available.
            When True, forecasts in 'pop' and 'yoy' metrics will be converted to levels
            using the available outturns data.
            Useful if you add 'pop' and want to analyse 'yoy' forecasts and vice versa.
            If the transformation fails for specific groups (e.g., due to insufficient
            historical data), those groups will be skipped with a warning message.
            Default is True.
        data_check : bool, optional
            Whether to run data checks on the forecasts against outturns. When True, two checks
            are performed per (source, variable, metric, frequency) group:

        data_check : bool, optional
            Whether to run data checks comparing forecast values to outturns
            per (source, variable, metric, frequency) group. When True:

            - **Horizon -1 check** (primary): if ``forecast_horizon == -1`` rows exist,
              each is compared to the outturn **from the same vintage** at the same date.
              Warns if the mean absolute deviation exceeds 0.5 std of the outturn series.
            - **IQR ratio check** (fallback): over all (date, vintage_date) pairs that
              overlap, warns if the forecast IQR differs from the outturn IQR by >5x.

            Detects common user errors: wrong ``metric`` column, scaling mistakes
            (e.g. ``pct*100`` instead of ``pct``), or non-real-time vintages.

            Warnings only; never raises errors. Set to ``False`` to disable.
            Default is True.
        Notes
        -----
        Outturns must be added before forecasts (call add_outturns first).
        All forecasts added to a ForecastData instance must have the same frequency. To work with forecasts of
        different frequencies, create separate ForecastData instances for each frequency.
        When compute_levels is True, sufficient historical outturn data is required for transformation,
        especially for 'yoy' metrics which need data from one year prior.
        """

        if self._raw_outturns is None or self._raw_outturns.empty:
            raise ValueError(
                "Outturns must be added before forecasts. Call add_outturns(outturns_df) before add_forecasts(...)."
            )

        # Compute forecast_horizon if missing
        if "forecast_horizon" not in df.columns:
            df = compute_forecast_horizon(df)

        # Handle metric column: use column values if present, otherwise use parameter
        if "metric" not in df.columns:
            df["metric"] = metric
        else:
            # Fill any null values in metric column with the parameter value
            df["metric"] = df["metric"].fillna(metric)

        # Validate that all metrics are valid
        valid_metrics = ["levels", "pop", "yoy"]
        invalid_metrics = df[~df["metric"].isin(valid_metrics)]["metric"].unique()
        if len(invalid_metrics) > 0:
            raise ValueError(f"Invalid metric values found: {invalid_metrics}. Valid options: {valid_metrics}")

        # Convert extra col names to contain only letters, numbers, and underscores
        if extra_ids is not None:
            df, extra_ids = _fix_extra_columns(df, extra_ids)

        # Nowcasting: compute days_in_period if enabled and not already present
        if self.nowcasting and "days_in_period" not in df.columns:
            from forecast_evaluation.data.utils import compute_days_in_period

            df = df.copy()
            df["days_in_period"] = compute_days_in_period(df["vintage_date"], df["frequency"])

        if "days_in_period" in df.columns:
            if extra_ids is None:
                extra_ids = ["days_in_period"]
            elif "days_in_period" not in extra_ids:
                extra_ids = extra_ids + ["days_in_period"]

        # Validate records using the ForecastRecord model
        # Include 'metric' as an optional column in validation
        optional_cols = ["metric"] if extra_ids is None else ["metric"] + extra_ids
        df = _validate_records(df, forecast=True, optional_columns=optional_cols)

        # Check frequency uniqueness and consistency
        new_frequencies = df["frequency"].unique()
        if len(new_frequencies) > 1:
            raise ValueError(
                f"Forecasts being added contain multiple frequencies: {new_frequencies.tolist()}. "
                f"Each ForecastData instance should only contain forecasts of a single frequency. "
                f"Please add forecasts with different frequencies separately using different ForecastData instances."
            )

        if not self._forecasts.empty:
            existing_frequencies = self._forecasts["frequency"].unique()
            existing_freq = existing_frequencies[0]
            new_freq = new_frequencies[0]
            if new_freq != existing_freq:
                raise ValueError(
                    f"New forecasts have frequency '{new_freq}' but existing data has frequency '{existing_freq}'. "
                    f"Each ForecastData instance should only contain forecasts of a single frequency. "
                    f"Please create a new ForecastData instance for forecasts with different frequencies."
                )

        # ID columns
        id_cols = ["source"] if extra_ids is None else ["source"] + extra_ids
        if self._id_columns is None:
            self._id_columns = id_cols
        else:
            # check that the id columns of the new forecasts match the existing ones
            # and if not adjust the datasets
            if set(self._id_columns) != set(id_cols):
                all_id_cols = list(set(self._id_columns).union(set(id_cols)))
                for col in all_id_cols:
                    if col not in self._id_columns:
                        # add missing columns to existing data
                        self._raw_forecasts[col] = pd.NA
                        self._forecasts[col] = pd.NA
                        self._id_columns += [col]
                    if col not in id_cols:
                        # add missing columns to new data
                        df[col] = pd.NA

        # Check for duplicates if there are already some records stored
        # Compare against raw forecasts (original input data), not self._forecasts
        # (which contains derived/transformed rows like levels computed from pop/yoy).
        # Comparing against transformed data causes false positives: e.g. adding levels
        # forecasts after pop forecasts were auto-converted to levels by compute_levels=True.
        if not self._raw_forecasts.empty:
            df = _check_duplicates(df, self._raw_forecasts)

        # Check if forecasts have corresponding outturns
        _check_missing_outturns(df, self._raw_outturns)

        # data-check forecast values against outturns
        if data_check:
            _check_forecast_data(df, self._outturns)

        # create a unique identifier for forecasts
        df["unique_id"] = construct_unique_id(df, self._id_columns)

        # Transform forecasts (prepare_forecasts handles metric-specific logic and auto-transformation)
        forecasts = prepare_forecasts(
            df,
            self._raw_outturns,
            self._id_columns,
            compute_levels=compute_levels,
            first_forecast_horizon=self.first_forecast_horizon,
        )

        # Compute main table
        main_table = build_main_table(forecasts, self._outturns, self._id_columns)

        # Filter out rows already present before appending (anti-join, O(n_new + n_existing)).
        # This is needed because derived/transformed rows (e.g. levels back-computed from
        # pop/yoy, or pop/yoy derived from levels) can slip through _check_duplicates when
        # add_forecasts is called repeatedly with already-transformed data.
        raw_id_cols = [c for c in df.columns if c != "value"]
        df = _exclude_existing_rows(df, self._raw_forecasts, raw_id_cols)
        self._raw_forecasts = pd.concat([self._raw_forecasts, df], ignore_index=True)

        forecast_id_cols = [c for c in forecasts.columns if c != "value"]
        forecasts = _exclude_existing_rows(forecasts, self._forecasts, forecast_id_cols)
        self._forecasts = pd.concat([self._forecasts, forecasts], ignore_index=True)

        main_table_id_cols = [
            c for c in main_table.columns if c not in ("value_forecast", "value_outturn", "forecast_error")
        ]
        main_table = _exclude_existing_rows(main_table, self._main_table, main_table_id_cols)
        self._main_table = pd.concat([self._main_table, main_table], ignore_index=True)

    def create_pseudo_vintages(
        self,
        fill_to: str,
        vintage_frequency: Literal["M", "Q"] = "Q",
    ) -> None:
        """Create pseudo vintages for outturns.

        Starts from the earliest available vintage in the data and fills backward to ``fill_to``.

        This method computes the publication lag from existing data and creates a full vintage
        structure where each vintage contains all data available at that point in time.
        A vintage at date X contains all data up to (X - publication_lag).

        Parameters
        ----------
        fill_to : str
            The earliest vintage date to create (i.e. how far back to fill). Format 'YYYY-MM-DD'.
            Vintages are generated from this date up to the earliest existing vintage in the data.
        vintage_frequency : str, optional
            Frequency at which to create vintages. Default is 'Q' (quarterly).
            Options: 'M' (monthly), 'Q' (quarterly).

        Notes
        -----
        - Computes publication lag per variable from existing data (max_vintage - max_date)
        - Expands the dataset by creating multiple vintage records for each data point
        - Each vintage V includes all data points D where (D + lag) <= V
        - Requires outturns to already have vintage_date values to compute the lag
        """
        if self._raw_outturns.empty:
            raise ValueError("No outturns data available. Add outturns before creating pseudo vintages.")

        # check that vintage_frequency is valid
        if vintage_frequency not in ["M", "Q"]:
            raise ValueError(
                f"Invalid vintage_frequency: {vintage_frequency}."
                f"Valid options are 'M' for monthly and 'Q' for quarterly."
            )

        df = self._raw_outturns.copy()

        # Create the dataframe that will be propagated to the new vintages
        # For each variable, select only the rows from the earliest available vintage
        df_propagate = df[df["vintage_date"] == df.groupby("variable")["vintage_date"].transform("min")]

        # Compute publication lag per variable in units of vintage_frequency
        lag_diff = (
            df_propagate["vintage_date"].dt.to_period(vintage_frequency)
            - df_propagate["date"].dt.to_period(vintage_frequency)
        ).apply(lambda x: x.n)
        df_propagate["publication_lag"] = lag_diff.groupby(df_propagate["variable"]).transform("min")

        # Precompute publication_date: earliest vintage at which each data point becomes available
        offset = pd.tseries.frequencies.to_offset(f"{vintage_frequency}E")
        df_propagate["publication_date"] = df_propagate.apply(
            lambda row: row["date"] + row["publication_lag"] * offset, axis=1
        )

        # Generate vintage dates from fill_to up to the earliest real vintage in the data.
        # The per-variable mask (vintage < earliest_vintage) handles each variable's own cutoff.
        start_vintage = pd.to_datetime(fill_to).normalize()
        end_vintage = df_propagate["vintage_date"].max()

        # Check the start_vintage is before the end_vintage
        if start_vintage >= end_vintage:
            raise ValueError(
                f"fill_to ({fill_to}) must be earlier than the latest per-variable earliest vintage "
                f"({end_vintage.date()})."
            )

        # Create range of vintage dates
        # replace with period end
        offset_freq = f"{vintage_frequency}E"
        vintage_dates = pd.date_range(
            start=start_vintage, end=end_vintage, freq=pd.tseries.frequencies.to_offset(offset_freq)
        )

        # Convert to match df["vintage_date"] type (datetime64[ns] without frequency)
        vintage_dates = pd.DatetimeIndex(vintage_dates)

        # Loop through vintages and filter data vectorized for each vintage
        expanded_rows = []
        for vintage in vintage_dates:
            # Keep rows where data was available at this vintage (publication_date <= vintage)
            mask = df_propagate["publication_date"] <= vintage
            mask = mask & (vintage < df_propagate["vintage_date"])  # only propagate to earlier vintages
            filtered_rows = df_propagate[mask].copy()
            filtered_rows["vintage_date"] = vintage
            expanded_rows.append(filtered_rows)

        # Create expanded dataframe
        expanded_df = pd.concat(expanded_rows, ignore_index=True).drop(columns=["publication_lag", "publication_date"])

        # recompute forecast_horizon using each row's actual frequency
        expanded_df = compute_forecast_horizon(expanded_df)

        # Update raw outturns
        self._raw_outturns = pd.concat([expanded_df, self._raw_outturns], ignore_index=True)

        # Recompute transformed outturns
        outturns = prepare_outturns(self._raw_outturns)
        self._outturns = outturns

        # Recompute main table if forecasts exist
        if not self._forecasts.empty:
            main_table = build_main_table(self._forecasts, self._outturns, self._id_columns)
            self._main_table = main_table

    def add_fer_outturns(self) -> None:
        """Load and add FER outturn data to existing records."""
        fer_outturns = load_fer_outturns()
        self.add_outturns(fer_outturns)

    def add_fer_forecasts(self) -> None:
        """Load and add FER forecast data to existing records."""
        fer_forecasts = load_fer_forecasts()
        self.add_forecasts(fer_forecasts)

    def add_fer_data(self) -> None:
        """Load and add FER outturns and forecast data to existing records."""
        self.add_fer_outturns()
        self.add_fer_forecasts()

    def filter_fer(self) -> None:
        """Filter the main dataset to only include specific variable-metric and model combinations"""
        self._main_table = filter_fer_variables(self._main_table)
        self._main_table = filter_fer_models(self._main_table)

    def _apply_filter_with_standardized_columns(
        self,
        df: pd.DataFrame,
        rename_cols: bool,
        start_date: str = None,
        end_date: str = None,
        start_vintage: str = None,
        end_vintage: str = None,
        variables: Optional[Union[str, list[str]]] = None,
        metrics: Optional[list[str]] = None,
        sources: Optional[Union[str, list[str]]] = None,
        frequencies: Optional[Union[str, list[str]]] = None,
        custom_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Apply filter_tables with standardized column names."""
        # Temporarily rename if needed
        if rename_cols:
            df = df.rename(columns={"vintage_date": "vintage_date_forecast"})

        # Apply filter
        df = filter_tables(
            df,
            start_date,
            end_date,
            start_vintage,
            end_vintage,
            variables,
            metrics,
            sources,
            frequencies,
            custom_filter,
        )

        # Rename back if needed
        if rename_cols:
            df = df.rename(columns={"vintage_date_forecast": "vintage_date"})

        return df

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
    ):
        """Filter the forecasts and main tables to only include data within specified
        date and vintage ranges, and optionally by variables, metrics, sources, or a custom filter.

        Parameters
        ----------
        start_date : str, optional
            Start date to filter forecasts (inclusive). Format 'YYYY-MM-DD'.
            Default is None in which case the analysis starts with the earliest date.
        end_date : str, optional
            End date to filter forecasts (inclusive). Format 'YYYY-MM-DD'.
            Default is None in which case the analysis ends with the latest date.
        start_vintage : str, optional
            Start vintage date to filter forecasts (inclusive). Format 'YYYY-MM-DD'.
            Default is None in which case the analysis starts with the earliest vintage.
        end_vintage : str, optional
            End vintage date to filter forecasts (inclusive). Format 'YYYY-MM-DD'.
            Default is None in which case the analysis ends with the latest vintage.
        variables: Optional[Union[str, list[str]]] = None
            List of variable identifiers to filter. Default is None (no filtering).
        metrics: Optional[list[str]] = None
            List of metric identifiers to filter. Default is None (no filtering).
        sources: Optional[Union[str, list[str]]] = None
            List of source identifiers to filter. Default is None (no filtering).
        frequencies: Optional[Union[str, list[str]]] = None
            List of frequency identifiers to filter. Default is None (no filtering).
        custom_filter : Callable[[pd.DataFrame], pd.DataFrame], optional
            A custom filtering function that takes a DataFrame as input and returns a filtered DataFrame.
            Default is None. Custom filters should use 'vintage_date_forecast' as the column name.
        """
        if not self._forecasts.empty:
            self._forecasts = self._apply_filter_with_standardized_columns(
                self._forecasts,
                rename_cols=True,
                start_date=start_date,
                end_date=end_date,
                start_vintage=start_vintage,
                end_vintage=end_vintage,
                variables=variables,
                metrics=metrics,
                sources=sources,
                frequencies=frequencies,
                custom_filter=custom_filter,
            )

        self._outturns = self._apply_filter_with_standardized_columns(
            self._outturns,
            rename_cols=True,
            start_date=start_date,
            end_date=end_date,
            start_vintage=start_vintage,
            end_vintage=end_vintage,
            variables=variables,
            metrics=metrics,
            frequencies=frequencies,
            custom_filter=custom_filter,
        )

        if not self._main_table.empty:
            self._main_table = self._apply_filter_with_standardized_columns(
                self._main_table,
                rename_cols=False,
                start_date=start_date,
                end_date=end_date,
                start_vintage=start_vintage,
                end_vintage=end_vintage,
                variables=variables,
                metrics=metrics,
                sources=sources,
                frequencies=frequencies,
                custom_filter=custom_filter,
            )

    def clear_filter(self) -> None:
        """Reset the forecasts, main and revisions tables to include all original data."""
        # Separate forecasts and outturns
        forecasts = prepare_forecasts(
            self._raw_forecasts,
            self._raw_outturns,
            self._id_columns,
            first_forecast_horizon=self.first_forecast_horizon,
        )
        outturns = prepare_outturns(self._raw_outturns)

        self._forecasts = forecasts
        self._outturns = outturns
        self._main_table = build_main_table(forecasts, outturns, self._id_columns)

    @property
    def df(self) -> pd.DataFrame:
        """Get the main DataFrame."""
        return self._main_table

    @property
    def forecasts(self) -> pd.DataFrame:
        """Get forecasts."""
        return self._forecasts

    @property
    def outturns(self) -> pd.DataFrame:
        """Get outturns."""
        return self._outturns

    @property
    def forecast_required_columns(self) -> list[str]:
        """Get the required columns list to help the user."""
        return FORECAST_REQUIRED_COLUMNS

    @property
    def outturn_required_columns(self) -> list[str]:
        """Get the required columns list to help the user."""
        return OUTTURN_REQUIRED_COLUMNS

    @property
    def id_columns(self) -> list[str]:
        """Get identification / labelling columns."""
        return self._id_columns

    def run_dashboard(self, from_jupyter: bool = False, host="127.0.0.1", port=8000):
        """Run the Shiny dashboard with the current data.

        Parameters
        ----------
        from_jupyter : bool, optional
            Whether to run the dashboard within a Jupyter notebook. Default is False.
        host : str, optional
            Host address for the dashboard server. Default is "127.0.0.1".
        port : int, optional
            Port number for the dashboard server. Default is 8000.
        """
        from forecast_evaluation.dashboard.create_app import dashboard_app

        app = dashboard_app(self)

        if from_jupyter:
            import threading

            import uvicorn
            from IPython.display import IFrame, display

            # Run server in a background thread
            def run_server():
                uvicorn.run(app, host=host, port=port, log_level="error")

            thread = threading.Thread(target=run_server, daemon=True)
            thread.start()

            # Display in iframe
            print(f"Dashboard running at http://{host}:{port}")
            display(IFrame(src=f"http://{host}:{port}", width=1000, height=600))

        else:
            app.run(host=host, port=port)

    def merge(self, other: "ForecastData", compute_levels: bool = True) -> "ForecastData":
        """Merge another ForecastData instance into this one.

        Parameters
        ----------
        other : ForecastData
            Another ForecastData instance to merge with this one.
        compute_levels : bool, optional
            Whether to automatically transform non-levels forecasts from `other`
            to levels if outturns data is available.
            When True, forecasts in 'pop' and 'yoy' metrics will be converted to levels
            using the available outturns data.
            Useful if you add 'pop' and want to analyse 'yoy' forecasts and vice versa.
            If the transformation fails for specific groups (e.g., due to insufficient
            historical data), those groups will be skipped with a warning message.
            Default is True.

        Returns
        -------
        ForecastData
           Updated ForecastData instance containing merged data from both instances.
        """

        if not other._raw_outturns.empty:
            self.add_outturns(other._raw_outturns)

        if not other._raw_forecasts.empty:
            # Filter out 'source' from id_columns to get only the extra_ids
            extra_ids = [col for col in other._id_columns if col != "source"] if other._id_columns else None
            extra_ids = extra_ids if extra_ids else None  # Convert empty list to None
            self.add_forecasts(
                other._raw_forecasts, extra_ids=extra_ids, compute_levels=compute_levels, data_check=False
            )

    def add_benchmarks(
        self,
        models: list[str] | str = BENCHMARK_MODELS,
        variables: str | Iterable[str] | None = None,
        metric: Literal["levels", "diff", "pop", "yoy"] = "levels",
        frequency: Literal["Q", "M"] | Iterable[Literal["Q", "M"]] | None = None,
        forecast_periods: int = 13,
        *,
        estimation_start_date: pd.Timestamp = None,
        show_progress: bool = False,
    ):
        """Add benchmark models to the ForecastData instance."""

        # validate models arg
        if isinstance(models, str):
            models = [models]

        if not all(model in BENCHMARK_MODELS for model in models):
            # which model is invalid
            invalid_models = [model for model in models if model not in BENCHMARK_MODELS]
            raise ValueError(
                f"Invalid model(s) specified in models argument."
                f"Valid options are {BENCHMARK_MODELS}. Got: {invalid_models}"
            )

        if "AR" in models:
            from forecast_evaluation.core.ar_p_model import add_ar_p_forecasts

            add_ar_p_forecasts(
                self,
                variable=variables,
                metric=metric,
                frequency=frequency,
                forecast_periods=forecast_periods,
                estimation_start_date=estimation_start_date,
                show_progress=show_progress,
            )

        if "random_walk" in models:
            from forecast_evaluation.core.random_walk_model import add_random_walk_forecasts

            add_random_walk_forecasts(
                self,
                variable=variables,
                metric=metric,
                frequency=frequency,
                forecast_periods=forecast_periods,
                show_progress=show_progress,
            )

    def summary(self):
        """Print a summary of the forecast and outturns datasets.

        For each dataset, displays:
        - Number of variables
        - List of variables with their properties
        - For each variable: frequency, date range, and first vintage date
        """
        print("\n" + "=" * 80)
        print("DATA SUMMARY")
        print("=" * 80)

        if not self._outturns.empty:
            self._print_outturn_summary()
        else:
            print("\n[OUTTURNS] No data loaded")

        if not self._forecasts.empty:
            self._print_forecast_summary()
        else:
            print("\n[FORECASTS] No data loaded")

        print("=" * 80 + "\n")

    def _print_variable_table(self, df, show_horizon=False, group_label=None):
        """Print a table of variables with date/vintage/horizon info."""
        if group_label:
            print(f"\n  {group_label}")

        variables = sorted(df["variable"].unique())
        max_var_len = max(len(var) for var in variables) if variables else 0
        var_col_width = max(max_var_len, 8)

        # Build header
        header_parts = [f"  {'Variable':<{var_col_width}}", "Freq", f"{'Date':<24}", f"{'Vintage':<24}"]
        if show_horizon:
            header_parts.append(f"{'Horizon':<8}")

        header = " | ".join(header_parts)
        separator = "  " + "-" * (len(header) - 2)

        print(header)
        print(separator)

        # Print rows for each variable
        for var in variables:
            var_data = df[df["variable"] == var]
            frequency = var_data["frequency"].iloc[0]
            date_range = (
                f"{var_data['date'].min().strftime('%Y-%m-%d')} to {var_data['date'].max().strftime('%Y-%m-%d')}"
            )
            vintage_range = f"{var_data['vintage_date'].min().strftime('%Y-%m-%d')} "
            vintage_range += f"to {var_data['vintage_date'].max().strftime('%Y-%m-%d')}"

            row_parts = [f"  {var:<{var_col_width}}", f"{frequency:4}", f"{date_range:<24}", f"{vintage_range:<24}"]
            if show_horizon:
                horizon_range = f"{var_data['forecast_horizon'].min()} to {var_data['forecast_horizon'].max()}"
                row_parts.append(f"{horizon_range:<8}")

            print(" | ".join(row_parts))

        print(separator)

    def _print_outturn_summary(self):
        """Print summary statistics for outturns data."""
        print("\n[OUTTURNS]")
        df = self._outturns

        variables = sorted(df["variable"].unique())
        print(f"  Number of variables: {len(variables)}")
        print(f"  Variables: {', '.join(variables)}\n")

        self._print_variable_table(df)

    def _print_forecast_summary(self):
        """Print summary statistics for forecasts data."""
        print("\n[FORECASTS]")
        df = self._forecasts

        variables = sorted(df["variable"].unique())
        print(f"  Number of variables: {len(variables)}")
        print(f"  Variables: {', '.join(variables)}\n")

        sources = sorted(df["source"].unique())
        print(f"  Number of sources: {len(sources)}")
        print(f"  Sources: {', '.join(sources)}")

        # Identify extra ID columns
        extra_id_cols = []
        if self._id_columns:
            extra_id_cols = [
                col
                for col in self._id_columns
                if col not in ["date", "vintage_date", "variable", "frequency", "forecast_horizon", "value", "source"]
            ]

        if extra_id_cols:
            for col in extra_id_cols:
                vals = sorted(df[col].astype(str).unique())
                print(f"  {col}: {', '.join(vals)}")

        # One table per unique (source, extra_id_cols) combination
        group_cols = ["source"] + extra_id_cols
        for group_key, group_data in df.groupby(group_cols, observed=True):
            keys = (group_key,) if not isinstance(group_key, tuple) else group_key
            source = keys[0]
            id_values = keys[1:]

            # Print group header
            group_label = source
            if extra_id_cols:
                id_label = "  ".join(f"{col}: {val}" for col, val in zip(extra_id_cols, id_values))
                group_label = f"{source}  [{id_label}]"

            self._print_variable_table(group_data, show_horizon=True, group_label=group_label)


def _validate_records(df: pd.DataFrame, forecast=False, optional_columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Validate a DataFrame of forecast records using the Pandera ForecastSchema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing forecast records to validate.
    forecast : bool, optional
        Whether the data is forecast data (True) or outturn data (False). Default is False.
    optional_columns : list of str, optional
        List of optional labelling columns to include in the schema validation. Default is empty.

    Returns
    -------
    pd.DataFrame
        DataFrame with validated and coerced records.
    """

    # Check for missing columns
    required_columns = FORECAST_REQUIRED_COLUMNS if forecast else OUTTURN_REQUIRED_COLUMNS
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        error_col = f"Attempting to add data but the following columns are missing: {sorted(missing_columns)}"
        raise ValueError(error_col)

    schema = create_data_schema(forecast, optional_columns)
    validated_df = schema.validate(df, lazy=False)

    # make sure that the dates are end-of-period dates according to frequency
    if not validated_df.empty:
        dates = validated_df["date"]
        for freq in validated_df["frequency"].unique():
            mask = validated_df["frequency"] == freq
            validated_df.loc[mask, "date"] = dates[mask].dt.to_period(freq).dt.end_time.dt.normalize()

    # make sure not to have hours attached to dates and consistent datetime64[ns] dtype
    validated_df["date"] = pd.to_datetime(validated_df["date"]).dt.normalize()
    validated_df["vintage_date"] = validated_df["vintage_date"].dt.normalize()

    # remove fully duplicated rows (same values and metadata)
    validated_df = validated_df.drop_duplicates().reset_index(drop=True)

    # check that there is only one record per unique combination of metadata
    metadata_cols = [col for col in validated_df.columns if col != "value"]

    if validated_df.duplicated(subset=metadata_cols).any():
        duplicate_rows = validated_df[validated_df.duplicated(subset=metadata_cols, keep=False)]
        raise ValueError(f"Duplicate records found with different values. Here are the duplicates:\n{duplicate_rows}")

    return validated_df


def _exclude_existing_rows(new_df: pd.DataFrame, existing_df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Return only rows from new_df whose id_cols key is not already present in existing_df.

    Uses a left-anti merge (hash join), O(n_new + n_existing), so the cost scales with
    the size of the batch being added rather than the full accumulated DataFrame.

    Parameters
    ----------
    new_df : pd.DataFrame
        Candidate rows to add.
    existing_df : pd.DataFrame
        Already-stored rows to check against.
    id_cols : list of str
        Columns that together form the unique key (all columns except value columns).

    Returns
    -------
    pd.DataFrame
        Subset of new_df containing only rows not already in existing_df.
    """
    if existing_df.empty or new_df.empty:
        return new_df
    # Only match on columns present in both DataFrames.
    # If new_df has extra id columns that existing_df doesn't have yet (e.g. a
    # newly introduced label column), no existing row could match on that key,
    # so those rows are definitively new. Using the intersection is safe because
    # unique_id (always present in both) already encodes all label columns.
    common_cols = [c for c in id_cols if c in existing_df.columns]
    if not common_cols:
        return new_df
    existing_keys = existing_df[common_cols].drop_duplicates()
    merged = new_df[common_cols].merge(existing_keys, on=common_cols, how="left", indicator=True)
    mask = (merged["_merge"] == "left_only").to_numpy()
    return new_df.loc[mask].reset_index(drop=True)


def _check_duplicates(new_df: pd.DataFrame, old_df: pd.DataFrame):
    """Check if there are duplicate records (same metadata) in the new data compared to existing data.

    Parameters
    ----------
    new_df : pd.DataFrame
        New forecast data to check for duplicates.
    old_df : pd.DataFrame
        Existing forecast data to compare against.

    Raises
    ------
    ValueError
        If duplicate forecast records are found.
    """

    # Define metadata columns (all except value)
    metadata_cols = [col for col in new_df.columns if col != "value"]

    # check for duplicates
    filtered_old_df = old_df[
        old_df[metadata_cols].apply(tuple, axis=1).isin(new_df[metadata_cols].apply(tuple, axis=1))
    ].copy()
    filtered_new_df = new_df[
        new_df[metadata_cols].apply(tuple, axis=1).isin(old_df[metadata_cols].apply(tuple, axis=1))
    ].copy()

    # order them both
    filtered_old_df = filtered_old_df.sort_values(by=metadata_cols)
    filtered_new_df = filtered_new_df.sort_values(by=metadata_cols)

    if not filtered_new_df.empty:
        diff = filtered_old_df["value"].values - filtered_new_df["value"].values

        if np.mean(np.abs(diff / np.std(filtered_old_df["value"].values))) > 1e-3:
            raise ValueError(
                f"Duplicate records found with different values. "
                f"Here are the duplicates from the existing data:\n{filtered_old_df}\n"
                f"And here are the corresponding records from the new data:\n{filtered_new_df}"
            )

        warnings.warn(
            f"Removed {len(filtered_old_df)} duplicate records with identical values.",
            UserWarning,
            stacklevel=3,
        )
        filtered_new_df_unique = new_df[~new_df.index.isin(filtered_new_df.index)]
    else:
        filtered_new_df_unique = new_df

    return filtered_new_df_unique


def _fix_extra_columns(df: pd.DataFrame, optional_columns: list[str]) -> pd.DataFrame:
    # Shiny requires id columns to have only letters, numbers, and underscores
    # we construct the id using the column names so we need to sanitise them here

    sanitized_optional_columns = []
    rename_dict = {}
    for col in optional_columns:
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", col)
        sanitized_optional_columns.append(sanitized)
        if sanitized != col:
            rename_dict[col] = sanitized
    if rename_dict:
        df = df.rename(columns=rename_dict)
    optional_columns = sanitized_optional_columns

    return df, optional_columns


def _check_forecast_data(forecasts_df: pd.DataFrame, outturns_df: pd.DataFrame) -> None:
    """Sanity-check forecasts against outturns. Warns on likely scale/metric errors.

    For each (source, variable, metric, frequency) group:

    - **Horizon -1 check** (primary): if ``forecast_horizon == -1`` rows exist,
      compares each forecast value to the outturn **from the same vintage** at the
      same date. Warns if the mean absolute deviation exceeds 0.5 standard deviations
      of the outturn series.
    - **IQR ratio check** (fallback): over all (date, vintage_date) pairs that overlap
      between forecasts and outturns, warns if the forecast IQR differs from the
      outturn IQR by more than 5x in either direction.

    Outturns are matched vintage-for-vintage so data revisions do not cause false positives.
    Issues are reported as UserWarnings, never errors.
    """
    _Y, _R = "\033[93m", "\033[0m"
    _TIP = (
        " Possible causes: wrong 'metric' column, incorrect scaling "
        "(e.g. pct*100 instead of pct), or non-real-time vintages."
    )

    if forecasts_df.empty or outturns_df.empty:
        return

    # Pair every forecast row with the outturn from the same vintage, variable,
    # metric, frequency and date.  This is the real-time outturn the forecaster
    # would have seen.
    join_keys = ["variable", "metric", "frequency", "date", "vintage_date"]
    outturns_keyed = outturns_df[join_keys + ["value"]].rename(columns={"value": "outturn_value"})
    paired_all = forecasts_df.merge(outturns_keyed, on=join_keys, how="inner")

    # Pre-compute per-(variable,metric,frequency) outturn std from all available
    # outturn data (used to normalise the h=-1 deviation).
    outturn_std_map = outturns_df.groupby(["variable", "metric", "frequency"])["value"].std().rename("outturn_std")

    group_keys = ["source", "variable", "metric", "frequency"]

    for keys, _group in forecasts_df.groupby(group_keys, sort=False):
        source, variable, metric, frequency = keys
        label = f"source='{source}', variable='{variable}', metric='{metric}', frequency='{frequency}'"

        paired = paired_all[
            (paired_all["source"] == source)
            & (paired_all["variable"] == variable)
            & (paired_all["metric"] == metric)
            & (paired_all["frequency"] == frequency)
        ]

        if paired.empty:
            continue

        # Check 1: horizon -1 rows — compared to same-vintage outturns
        horizon_minus1 = paired[paired["forecast_horizon"] == -1]
        if not horizon_minus1.empty:
            if len(horizon_minus1) >= 2:
                outturn_std = outturn_std_map.get((variable, metric, frequency), 0)
                if outturn_std > 0:
                    mean_abs_dev = (horizon_minus1["value"] - horizon_minus1["outturn_value"]).abs().mean()
                    if mean_abs_dev > 0.5 * outturn_std:
                        warnings.warn(
                            f"{_Y}[Data check] {label}: horizon=-1 forecasts deviate from "
                            f"same-vintage outturns by {mean_abs_dev / outturn_std:.2f} std "
                            f"deviations.{_TIP}{_R}",
                            UserWarning,
                        )
            continue

        # Check 2 (fallback): IQR ratio — same-vintage paired values
        if len(paired) < 3:
            continue

        outturn_iqr = paired["outturn_value"].quantile(0.75) - paired["outturn_value"].quantile(0.25)
        forecast_iqr = paired["value"].quantile(0.75) - paired["value"].quantile(0.25)
        if outturn_iqr > 0 and forecast_iqr > 0:
            spread_ratio = max(forecast_iqr / outturn_iqr, outturn_iqr / forecast_iqr)
            if spread_ratio > 5:
                warnings.warn(
                    f"{_Y}[Data check] {label}: forecast/outturn IQR ratio is {spread_ratio:.1f}x.{_TIP}{_R}",
                    UserWarning,
                )


def _check_missing_outturns(forecasts_df: pd.DataFrame, outturns_df: pd.DataFrame):
    """Check if all unique combinations of variable and frequency from forecasts
    have corresponding outturns.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Forecast data to check.
    outturns_df : pd.DataFrame
        Outturn data to compare against.

    Warnings
    --------
    UserWarning
        If forecast combinations are found that have no corresponding outturns.
    """
    # Return early if outturns dataframe is empty
    if outturns_df.empty:
        raise ValueError(
            "No outturns data available. All forecasts will have no corresponding outturns.", UserWarning, stacklevel=3
        )

    # Get unique combinations from forecasts
    forecast_combinations = forecasts_df[["variable", "frequency"]].drop_duplicates()

    # Get unique combinations from outturns
    outturn_combinations = outturns_df[["variable", "frequency"]].drop_duplicates()

    # Find forecasts without matching outturns using merge with indicator
    merged = forecast_combinations.merge(outturn_combinations, on=["variable", "frequency"], how="left", indicator=True)

    # Filter for combinations that only exist in forecasts (no outturns)
    missing_outturns = merged[merged["_merge"] == "left_only"][["variable", "frequency"]]

    if not missing_outturns.empty:
        warning_message = (
            f"Warning: {len(missing_outturns)} forecast combination(s) have no corresponding outturns:\n"
            f"{missing_outturns.to_string(index=False)}"
        )
        warnings.warn(warning_message, UserWarning, stacklevel=3)


# Example usage:
if __name__ == "__main__":
    import forecast_evaluation as fe

    # launch dashboard
    forecast_data = fe.ForecastData(load_fer=True)
    forecast_data.summary()

    forecast_data.run_dashboard()

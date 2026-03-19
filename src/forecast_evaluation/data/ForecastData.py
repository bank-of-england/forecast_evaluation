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
from forecast_evaluation.data.utils import construct_unique_id, filter_fer_models, filter_fer_variables, filter_tables

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
        """
        self._raw_forecasts = pd.DataFrame()
        self._raw_outturns = pd.DataFrame()
        self._outturns = pd.DataFrame()
        self._forecasts = pd.DataFrame()
        self._main_table = pd.DataFrame()
        self._id_columns = None

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
            per (source, variable, metric, frequency, vintage_date) group. When True:

            - **Horizon -1 check**: if ``forecast_horizon == -1`` rows exist, warns if
              mean deviation from outturns exceeds 0.5 standard deviations.
            - **Scale/mean check** (fallback): over overlapping dates, warns if forecast
              mean differs from outturns by >2 std, or spread ratio >5× (larger/smaller).

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
        if not self._forecasts.empty:
            df = _check_duplicates(df, self._forecasts)

        # Check if forecasts have corresponding outturns
        _check_missing_outturns(df, self._raw_outturns)

        # data-check forecast values against outturns
        if data_check:
            _check_forecast_data(df, self._outturns)

        # create a unique identifier for forecasts
        df["unique_id"] = construct_unique_id(df, self._id_columns)

        # Transform forecasts (prepare_forecasts handles metric-specific logic and auto-transformation)
        forecasts = prepare_forecasts(df, self._raw_outturns, self._id_columns, compute_levels=compute_levels)

        # Compute main table
        main_table = build_main_table(forecasts, self._outturns, self._id_columns)

        # Add to existing data
        self._raw_forecasts = pd.concat([self._raw_forecasts, df], ignore_index=True)
        self._forecasts = pd.concat([self._forecasts, forecasts], ignore_index=True)
        self._main_table = pd.concat([self._main_table, main_table], ignore_index=True)

    def create_pseudo_vintages(
        self,
        first_vintage_date: str,
        vintage_frequency: Literal["M", "Q"] = "Q",
    ) -> None:
        """Create pseudo vintage datasets for outturns when only latest values are available.

        This method computes the publication lag from existing data and creates a full vintage
        structure where each vintage contains all data available at that point in time.
        A vintage at date X contains all data up to (X - publication_lag).

        Parameters
        ----------
        first_vintage_date : str
            The first vintage date to create. Format 'YYYY-MM-DD'.
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

        # replace with period end
        vintage_frequency = f"{vintage_frequency}E"

        df = self._raw_outturns.copy()

        # Compute publication lag for each variable
        # lag = max(vintage_date) - max(date) for each variable
        publication_lags = {}
        for variable in df["variable"].unique():
            var_data = df[df["variable"] == variable]
            max_vintage = var_data["vintage_date"].max()
            max_date = var_data["date"].max()
            lag = max_vintage - max_date
            publication_lags[variable] = lag

        # Generate vintage dates from first_vintage_date to latest required
        first_vintage = pd.to_datetime(first_vintage_date).normalize()
        latest_vintage = df["vintage_date"].max()

        # Check the first_vintage_date is before the first available vintage date in the data
        if first_vintage >= latest_vintage:
            raise ValueError(f"first_vintage_date {first_vintage_date} must be before {latest_vintage.date()}. ")

        # Create range of vintage dates
        vintage_dates = pd.date_range(
            start=first_vintage, end=latest_vintage, freq=pd.tseries.frequencies.to_offset(vintage_frequency)
        )

        # Build expanded dataset: for each vintage, include all data available at that time
        # Add lags column for vectorized filtering
        df_with_lags = df.copy()
        df_with_lags["lag"] = df_with_lags["variable"].map(publication_lags)
        df_with_lags["first_appearance"] = df_with_lags["date"] + df_with_lags["lag"]

        # Loop through vintages and filter data vectorized for each vintage
        expanded_rows = []
        for vintage in vintage_dates:
            # Vectorized filtering: keep rows where vintage >= first_appearance
            mask = vintage >= df_with_lags["first_appearance"]
            filtered_rows = df_with_lags[mask].copy()
            filtered_rows["vintage_date"] = vintage
            expanded_rows.append(filtered_rows)

        # Create expanded dataframe
        expanded_df = pd.concat(expanded_rows, ignore_index=True).drop(columns=["lag", "first_appearance"])

        # recompute forecast_horizon
        expanded_df["forecast_horizon"] = (
            expanded_df["date"].dt.to_period("Q") - expanded_df["vintage_date"].dt.to_period("Q")
        ).apply(lambda x: x.n)

        # Update raw outturns
        self._raw_outturns = expanded_df

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
        forecasts = prepare_forecasts(self._raw_forecasts, self._raw_outturns, self._id_columns)
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
        validated_df["date"] = pd.to_datetime(
            validated_df.apply(
                lambda row: pd.Period(year=row["date"].year, month=row["date"].month, freq=row["frequency"]).end_time,
                axis=1,
            )
        )

    # make sure not to have hours attached to dates
    validated_df["date"] = validated_df["date"].dt.normalize()
    validated_df["vintage_date"] = validated_df["vintage_date"].dt.normalize()

    # remove fully duplicated rows (same values and metadata)
    validated_df = validated_df.drop_duplicates().reset_index(drop=True)

    # check that there is only one record per unique combination of metadata
    metadata_cols = [col for col in validated_df.columns if col != "value"]

    if validated_df.duplicated(subset=metadata_cols).any():
        duplicate_rows = validated_df[validated_df.duplicated(subset=metadata_cols, keep=False)]
        raise ValueError(f"Duplicate records found with different values. Here are the duplicates:\n{duplicate_rows}")

    return validated_df


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
    """data-check forecast values against outturns, per (source, variable, frequency, metric) group.

    Check 1 — horizon -1 (if available): mean absolute deviation vs outturns > 0.5 std.
    Check 2 — fallback: mean or spread of positive-horizon forecasts differs markedly from outturns
    (mean > 2 std away, or spread ratio > 5×) over overlapping dates.

    Issues are reported as UserWarnings, never errors.
    """
    _YELLOW = "\033[93m"
    _RESET = "\033[0m"
    _TIP = (
        " Possible causes: wrong 'metric' column, incorrect scaling "
        "(e.g. pct*100 instead of pct for 'pop'/'yoy'), "
        "or not using real-time vintages for level forecasts."
    )

    outturns = outturns_df.copy()
    forecasts = forecasts_df.copy()

    key_cols = ["variable", "source", "frequency", "metric", "vintage_date"]
    for (variable, source, frequency, metric, vintage_date), forecast_group in forecasts.groupby(key_cols):
        outturns_group = outturns[
            (outturns["variable"] == variable)
            & (outturns["frequency"] == frequency)
            & (outturns["metric"] == metric)
            & (outturns["vintage_date"] == vintage_date)
        ]

        if outturns_group.empty:
            # normalise shouldnt be allowed, but for now leave it here
            continue

        label = f"vintage_date='{vintage_date}', source='{source}', variable='{variable}', "
        label += f"metric='{metric}', frequency='{frequency}'"

        # Check 1: horizon -1 vs outturns
        forecast_h1 = forecast_group[forecast_group["forecast_horizon"] == -1]
        outturns_h1 = outturns_group[outturns_group["forecast_horizon"] == -1]

        if not forecast_h1.empty:
            discrepancy = forecast_h1["value"].values - outturns_h1["value"].values
            if discrepancy > 1e-4:
                warnings.warn(
                    f"{_YELLOW}[Data check] {label}: horizon -1 forecasts deviate from outturns "
                    f"by {discrepancy}" + _TIP + f"{_RESET}",
                    UserWarning,
                    stacklevel=4,
                )
            continue  # skip check 2 whether or not h=-1 matched

        # Check 2: relative difference between first forecast and last outturn
        if outturns_group.shape[0] > 10:  # need enough data points to compute a meaningful std
            first_forecast_id = min(forecast_group["forecast_horizon"].min(), 0)
            last_outturn_id = max(outturns_group["forecast_horizon"].max(), -1)

            last_outturn = outturns_group[outturns_group["forecast_horizon"] == last_outturn_id]["value"].values[0]
            first_forecast = forecast_group[forecast_group["forecast_horizon"] == first_forecast_id]["value"].values[0]

            outturn_std = outturns_group["value"].diff().dropna().std()
            diff = first_forecast - last_outturn
            relative_diff = abs(diff) / outturn_std

            if relative_diff > 10:
                warnings.warn(
                    f"{_YELLOW}[Data check] {label}: first forecast value deviates from last outturn by "
                    f"{relative_diff} times the outturn std." + _TIP + f"{_RESET}",
                    UserWarning,
                    stacklevel=4,
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
    forecast_data.run_dashboard()

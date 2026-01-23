"""
Result classes for forecast evaluation tests.

This module provides an object-oriented wrapper for test results that enables
rich data exploration, filtering, visualization, and export capabilities.
"""

from typing import Any, Literal, Optional, Union

import pandas as pd

from forecast_evaluation.utils import filter_sources, reconstruct_id_cols_from_unique_id


class TestResult:
    """
    Universal result object for all forecast evaluation tests.

    Provides common functionality for storing test results, metadata,
    and methods for data manipulation, visualization, and export.

    Attributes
    ----------
    _df : pd.DataFrame
        The underlying DataFrame containing test results
    _metadata : dict
        Metadata about the test including parameters, filters, and provenance
    """

    def __init__(self, df: pd.DataFrame, id_columns: list[str] = None, metadata: Optional[dict] = None):
        """
        Initialize a TestResult object.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing test results
        id_columns : list of str, optional
            List of identifier columns to reconstruct from unique_id
        metadata : dict, optional
            Dictionary containing test metadata including:

            - test_name : str - Name of the test function
            - parameters : dict - Test parameters (k, same_date_range, etc.)
            - filters : dict - Applied filters (source, variable, etc.)
            - date_range : tuple - (start_date, end_date) of the data
        """

        if id_columns is not None:
            df = reconstruct_id_cols_from_unique_id(df, id_columns)

        self._df = df.copy()
        self._metadata = metadata or {}
        self._id_columns = id_columns or None

    def to_df(self) -> pd.DataFrame:
        """
        Return the underlying DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of the test results DataFrame
        """
        return self._df.copy()

    def __repr__(self) -> str:
        """
        Rich console representation of the test results.

        Returns
        -------
        str
            Formatted string representation
        """
        test_name = self._metadata.get("test_name", "Test Results")
        metadata_str = "\n".join([f"  {k}: {v}" for k, v in self._metadata.items()])
        return f"TestResult: {test_name}\n{metadata_str}\n\nResults:\n{self._df.__repr__()}"

    def __len__(self) -> int:
        """
        Return the number of test results.

        Returns
        -------
        int
            Number of rows in the results DataFrame
        """
        return len(self._df)

    def __getitem__(self, key):
        """
        Enable DataFrame-like indexing and slicing.

        Parameters
        ----------
        key : str, int, slice, or array-like
            Index, column name, or slice for accessing data

        Returns
        -------
        Any
            Indexed data from the underlying DataFrame
        """
        return self._df[key]

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying DataFrame.

        This enables DataFrame methods to be called directly on the result object.

        Parameters
        ----------
        name : str
            Attribute name

        Returns
        -------
        Any
            Attribute from the underlying DataFrame

        Raises
        ------
        AttributeError
            If the attribute doesn't exist on the DataFrame
        """
        if name.startswith("_"):
            raise AttributeError(f"'TestResult' object has no attribute '{name}'")
        try:
            return getattr(self._df, name)
        except AttributeError:
            raise AttributeError(f"'TestResult' object has no attribute '{name}'")

    def __dataframe__(self, *args, **kwargs):
        """
        Implement pandas interchange protocol for DataFrame compatibility.

        This allows TestResult objects to be used directly with libraries
        that expect DataFrame-compatible objects (e.g., Shiny's render.DataGrid).

        Returns
        -------
        pandas.core.interchange.dataframe.DataFrameInterchange
            Interchange object for the underlying DataFrame
        """
        return self._df.__dataframe__(*args, **kwargs)

    def summary(self) -> str:
        """
        Generate a formatted statistical summary of the test results.

        Returns
        -------
        str
            Formatted summary of key findings
        """
        summary_parts = [f"\n{'=' * 70}", f"{self._metadata.get('test_name', 'Test Results').upper()}", f"{'=' * 70}"]

        # Add metadata information
        if "parameters" in self._metadata:
            summary_parts.append("\nTest Parameters:")
            for k, v in self._metadata["parameters"].items():
                summary_parts.append(f"  {k}: {v}")

        if "filters" in self._metadata:
            summary_parts.append("\nApplied Filters:")
            for k, v in self._metadata["filters"].items():
                summary_parts.append(f"  {k}: {v}")

        if "date_range" in self._metadata:
            start, end = self._metadata["date_range"]
            summary_parts.append(f"\nDate Range: {start} to {end}")

        summary_parts.append(f"\nNumber of Results: {len(self._df)}")

        # Add basic statistics
        summary_parts.append("\nDescriptive Statistics:")
        summary_parts.append(str(self._df.describe()))

        summary_parts.append(f"\n{'=' * 70}\n")

        return "\n".join(summary_parts)

    def describe(self) -> pd.DataFrame:
        """
        Generate descriptive statistics of test results.

        Returns
        -------
        pd.DataFrame
            Descriptive statistics (count, mean, std, min, max, etc.)
        """
        return self._df.describe()

    def filter(
        self,
        variable: Optional[Union[str, list[str]]] = None,
        source: Optional[Union[str, list[str]]] = None,
        horizon: Optional[Union[int, list[int]]] = None,
        **kwargs,
    ) -> "TestResult":
        """
        Filter results by specified criteria.

        Parameters
        ----------
        variable : str or list of str, optional
            Variable(s) to include
        source : str or list of str, optional
            Source(s) to include
        horizon : int or list of int, optional
            Forecast horizon(s) to include
        **kwargs
            Additional column-based filters

        Returns
        -------
        TestResult
            New TestResult object with filtered data
        """
        df_filtered = self._df.copy()

        # Apply variable filter
        if variable is not None:
            if isinstance(variable, str):
                df_filtered = df_filtered[df_filtered["variable"] == variable]
            else:
                df_filtered = df_filtered[df_filtered["variable"].isin(variable)]

        # Apply source filter
        if source is not None:
            if isinstance(source, str):
                source = [source]
            df_filtered = filter_sources(df_filtered, source)

        # Apply horizon filter
        if horizon is not None:
            if "forecast_horizon" in df_filtered.columns:
                if isinstance(horizon, int):
                    df_filtered = df_filtered[df_filtered["forecast_horizon"] == horizon]
                else:
                    df_filtered = df_filtered[df_filtered["forecast_horizon"].isin(horizon)]

        # Apply additional filters
        for col, value in kwargs.items():
            if col in df_filtered.columns:
                if isinstance(value, (list, tuple)):
                    df_filtered = df_filtered[df_filtered[col].isin(value)]
                else:
                    df_filtered = df_filtered[df_filtered[col] == value]

        # Update metadata
        new_metadata = self._metadata.copy()
        if "filters" not in new_metadata:
            new_metadata["filters"] = {}
        if variable is not None:
            new_metadata["filters"]["variable"] = variable
        if source is not None:
            new_metadata["filters"]["unique_id"] = source
        if horizon is not None:
            new_metadata["filters"]["horizon"] = horizon
        new_metadata["filters"].update(kwargs)

        return TestResult(df_filtered, self._id_columns, new_metadata)

    def to_csv(self, path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Export results to CSV file or return as string.

        Parameters
        ----------
        path : str, optional
            Output file path. If None, returns CSV as string.
        **kwargs
            Additional arguments passed to pd.DataFrame.to_csv()

        Returns
        -------
        str or None
            CSV string if path is None, otherwise None
        """
        if path is None:
            return self._df.to_csv(index=False, **kwargs)
        else:
            self._df.to_csv(path, **kwargs)
            return None

    def plot(self, **kwargs):
        """
        Generate appropriate visualization for this result type.

        Automatically detects the test type from metadata and routes to
        the appropriate visualization function.

        Parameters
        ----------
        **kwargs
            Visualization-specific parameters (vary by test type)

        Returns
        -------
        tuple or None
            (fig, ax) matplotlib figure and axes objects, or None

        Raises
        ------
        ValueError
            If test type cannot be determined or no visualization is available
        """
        test_name = self._metadata.get("test_name")

        if test_name == "bias_analysis":
            return self._plot_bias(**kwargs)
        elif test_name == "compute_accuracy_statistics":
            return self._plot_accuracy(**kwargs)
        elif test_name == "weak_efficiency_analysis":
            raise NotImplementedError("Weak efficiency plotting is not yet implemented.")
        elif test_name == "strong_efficiency_analysis":
            return self._plot_strong_efficiency(**kwargs)
        elif test_name == "blanchard_leigh_horizon_analysis":
            return self._plot_blanchard_leigh(**kwargs)
        elif test_name == "diebold_mariano_table":
            raise NotImplementedError("Diebold-Mariano plotting is not yet implemented.")
        elif test_name == "revisions_errors_correlation_analysis":
            raise NotImplementedError("Revisions correlation plotting is not yet implemented.")
        elif test_name == "revision_predictability_analysis":
            raise NotImplementedError(
                "RevisionsPredictabilityResults.plot() requires access to the original ForecastData object. "
                "Please use the plot_average_revision_by_period() function directly with your ForecastData object."
            )
        elif test_name == "rolling_analysis":
            return self._plot_rolling_analysis(**kwargs)
        elif test_name == "fluctuation_tests":
            return self._plot_fluctuation_tests(**kwargs)
        else:
            raise ValueError(
                f"Cannot determine visualization for test '{test_name}'. "
                "Please ensure the test_name metadata is set correctly."
            )

    def _plot_bias(
        self,
        variable: Optional[str] = None,
        source: Optional[str] = None,
        metric: Optional[Literal["levels", "pop", "yoy"]] = None,
        frequency: Optional[Literal["Q", "M"]] = None,
        convert_to_percentage: bool = False,
        return_plot: bool = False,
        **kwargs,
    ):
        """Plot bias estimates with confidence intervals by forecast horizon."""
        from forecast_evaluation.visualisations.bias import plot_bias_by_horizon

        # Auto-detect parameters if only one unique value exists
        if variable is None:
            unique_vars = self._df["variable"].unique()
            if len(unique_vars) == 1:
                variable = unique_vars[0]
            else:
                raise ValueError(f"Multiple variables found: {unique_vars}. Please specify 'variable' parameter.")

        if source is None:
            unique_sources = self._df["unique_id"].unique()
            if len(unique_sources) == 1:
                source = unique_sources[0]
            else:
                raise ValueError(f"Multiple sources found: {unique_sources}. Please specify 'source' parameter.")

        if metric is None:
            unique_metrics = self._df["metric"].unique()
            if len(unique_metrics) == 1:
                metric = unique_metrics[0]
            else:
                raise ValueError(f"Multiple metrics found: {unique_metrics}. Please specify 'metric' parameter.")

        if frequency is None:
            unique_freqs = self._df["frequency"].unique()
            if len(unique_freqs) == 1:
                frequency = unique_freqs[0]
            else:
                raise ValueError(f"Multiple frequencies found: {unique_freqs}. Please specify 'frequency' parameter.")

        return plot_bias_by_horizon(
            df=self._df,
            variable=variable,
            source=source,
            metric=metric,
            frequency=frequency,
            convert_to_percentage=convert_to_percentage,
            return_plot=return_plot,
            **kwargs,
        )

    def _plot_accuracy(
        self,
        variable: Optional[str] = None,
        metric: Optional[Literal["levels", "pop", "yoy"]] = None,
        frequency: Optional[Literal["Q", "M"]] = "Q",
        statistic: Literal["rmse", "rmedse", "mse", "mean_abs_error"] = "rmse",
        benchmark_model: str = None,
        convert_to_percentage: bool = False,
        return_plot: bool = False,
        **kwargs,
    ):
        """Plot accuracy statistics by forecast horizon."""
        from forecast_evaluation.visualisations.accuracy import plot_accuracy, plot_compare_to_benchmark

        # Auto-detect parameters if only one unique value exists
        if variable is None:
            unique_vars = self._df["variable"].unique()
            if len(unique_vars) == 1:
                variable = unique_vars[0]
            else:
                raise ValueError(f"Multiple variables found: {unique_vars}. Please specify 'variable' parameter.")

        if metric is None:
            unique_metrics = self._df["metric"].unique()
            if len(unique_metrics) == 1:
                metric = unique_metrics[0]
            else:
                raise ValueError(f"Multiple metrics found: {unique_metrics}. Please specify 'metric' parameter.")

        if benchmark_model is None:
            return plot_accuracy(
                df=self._df,
                variable=variable,
                metric=metric,
                frequency=frequency,
                statistic=statistic,
                convert_to_percentage=convert_to_percentage,
                return_plot=return_plot,
                **kwargs,
            )
        else:
            return plot_compare_to_benchmark(
                df=self._df,
                variable=variable,
                metric=metric,
                frequency=frequency,
                statistic=statistic,
                benchmark_model=benchmark_model,
                return_plot=return_plot,
                **kwargs,
            )

    def _plot_strong_efficiency(
        self,
        return_plot: bool = False,
        **kwargs,
    ):
        """Plot strong efficiency coefficients across forecast horizons."""
        from forecast_evaluation.visualisations.strong_efficiency import plot_strong_efficiency

        return plot_strong_efficiency(
            results=self._df,
            return_plot=return_plot,
            **kwargs,
        )

    def _plot_blanchard_leigh(
        self,
        return_plot: bool = False,
        **kwargs,
    ):
        """Plot Blanchard-Leigh ratios across forecast horizons."""
        from forecast_evaluation.visualisations.blanchard_leigh import plot_blanchard_leigh_ratios

        return plot_blanchard_leigh_ratios(
            results=self._df,
            return_plot=return_plot,
            **kwargs,
        )

    def _plot_rolling_analysis(
        self,
        **kwargs,
    ):
        """Plot rolling analysis results based on the analysis function used."""
        # Get the analysis function name from metadata
        analysis_func_name = self._metadata.get("parameters", {}).get("analysis_func")

        if analysis_func_name is None:
            raise ValueError("Cannot determine analysis function. Ensure metadata contains 'parameters.analysis_func'.")

        # Route to appropriate visualization based on analysis function
        if analysis_func_name == "bias_analysis":
            from forecast_evaluation.visualisations.bias import plot_rolling_bias

            return plot_rolling_bias(df=self._df, **kwargs)

        elif analysis_func_name == "diebold_mariano_table":
            from forecast_evaluation.visualisations.accuracy import plot_rolling_relative_accuracy

            return plot_rolling_relative_accuracy(df=self._df, **kwargs)

        elif analysis_func_name == "weak_efficiency_analysis":
            raise NotImplementedError("Weak efficiency rolling analysis plotting is not yet implemented.")

        else:
            raise NotImplementedError(
                f"Rolling analysis plotting for '{analysis_func_name}' is not yet implemented. "
                f"Supported analysis functions: bias_analysis, diebold_mariano_table."
            )

    def _plot_fluctuation_tests(
        self,
        **kwargs,
    ):
        """Plot fluctuation test results based on the test function used.

        Routes to the appropriate visualization based on the test_func parameter
        stored in metadata.

        Parameters
        ----------
        **kwargs
            Visualization-specific parameters (vary by test function)

        Returns
        -------
        tuple or None
            (fig, ax) matplotlib figure and axes objects, or None

        Raises
        ------
        ValueError
            If test function cannot be determined
        NotImplementedError
            If plotting for the test function is not yet implemented
        """
        # Get the test function name from metadata
        test_func_name = self._metadata.get("parameters", {}).get("test_func")

        if test_func_name is None:
            raise ValueError("Cannot determine test function. Ensure metadata contains 'parameters.test_func'.")

        # Route to appropriate visualization based on test function
        if test_func_name == "bias_analysis":
            from forecast_evaluation.visualisations.bias import plot_rolling_bias

            return plot_rolling_bias(df=self._df, **kwargs)

        elif test_func_name == "diebold_mariano_table":
            from forecast_evaluation.visualisations.accuracy import plot_rolling_relative_accuracy

            return plot_rolling_relative_accuracy(df=self._df, **kwargs)

        elif test_func_name == "weak_efficiency_analysis":
            raise NotImplementedError("Weak efficiency fluctuation test plotting is not yet implemented.")

        else:
            raise NotImplementedError(
                f"Fluctuation test plotting for '{test_func_name}' is not yet implemented. "
                f"Supported test functions: bias_analysis, diebold_mariano_table."
            )

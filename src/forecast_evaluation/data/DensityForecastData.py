"""Density forecast data class for handling probabilistic forecasts with quantiles."""

from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

from forecast_evaluation.core.transformations import prepare_forecasts
from forecast_evaluation.data.ForecastData import (
    ForecastData,
    _check_duplicates,
    _check_missing_outturns,
    _fix_extra_columns,
    _validate_records,
)
from forecast_evaluation.data.sample_data import (
    create_sample_density_forecasts,
)
from forecast_evaluation.data.utils import construct_unique_id


class DensityForecastData(ForecastData):
    """Class for density forecasts with quantile information.

    Extends the ForecastData class to handle density forecasts.
    DensityForecastData objects include a `density_forecasts` attribute that contains
    forecasts with quantile information (basically an extra column indicating quantiles).
    It can still handle point forecasts with the `forecasts` attribute.


    Parameters
    ----------
    outturns_data : pd.DataFrame, optional
        DataFrame containing outturn (actual) data.
    forecasts_data : pd.DataFrame, optional
        DataFrame containing point forecast records.
    density_forecasts_data : pd.DataFrame, optional
        DataFrame containing density forecast records. Must include 'quantile' column.
    point_estimator : str, optional
        Estimator for point forecasts; can be 'median', 'mean' or 'mode'.
        Default is 'median'.
    load_fer : bool, optional
        Whether to load FER (Forecast Evaluation Report) data. Default is False.
    extra_ids : list of str, optional
        Additional identification columns beyond 'source' and 'quantile'.

    Examples
    --------
    >>> import pandas as pd
    >>> from forecast_evaluation.data import DensityForecastData
    >>>
    >>> # Create sample density forecasts
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2023-01-01', periods=4, freq='QE'),
    ...     'vintage_date': pd.Timestamp('2023-01-01'),
    ...     'variable': 'gdp',
    ...     'frequency': 'Q',
    ...     'forecast_horizon': [1, 2, 3, 4],
    ...     'source': 'model_1',
    ...     'quantile': 0.5,
    ...     'value': [100, 101, 102, 103]
    ... })
    >>>
    >>> density_data = DensityForecastData(forecasts_data=df)
    >>> median = density_data.get_median_forecast()
    """

    def __init__(
        self,
        outturns_data: Optional[pd.DataFrame] = None,
        forecasts_data: Optional[pd.DataFrame] = None,
        load_fer: Optional[bool] = False,
        extra_ids: Optional[list[str]] = None,
    ):
        """Initialise DensityForecastData.

        Initialises the density forecast data object. If forecasts_data is provided,
        it will be validated and added. The 'quantile' column is automatically
        included as an identification column.
        """
        # Initialise parent class without forecasts
        super().__init__(outturns_data=outturns_data, load_fer=load_fer, extra_ids=extra_ids)

        # Add density-specific attribute
        self._density_forecasts = pd.DataFrame()
        self._density_df = pd.DataFrame()

        # Add forecasts if provided (using density-specific method)
        if forecasts_data is not None:
            self.add_density_forecasts(forecasts_data, extra_ids=extra_ids)

    def add_density_forecasts(self, df: pd.DataFrame, extra_ids: Optional[list[str]] = None) -> None:
        """Validate and add density forecasts with quantile column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing density forecast records. Must include 'quantile' column
            with values between 0 and 1.
        extra_ids : list of str, optional
            Additional identification columns beyond 'source' and 'quantile'.

        Raises
        ------
        ValueError
            If 'quantile' column is missing from the DataFrame.

        Examples
        --------
        >>> density_data = DensityForecastData()
        >>> df = pd.DataFrame({
        ...     'date': ['2023-01-01'],
        ...     'vintage_date': ['2023-01-01'],
        ...     'variable': ['gdp'],
        ...     'frequency': ['Q'],
        ...     'forecast_horizon': [1],
        ...     'source': ['model_1'],
        ...     'quantile': [0.5],
        ...     'value': [100]
        ... })
        >>> density_data.add_density_forecasts(df)
        """
        # Check for quantile column
        if "quantile" not in df.columns:
            raise ValueError("Density forecasts must include a 'quantile' column")

        # Add 'quantile' to extra_ids if not already there
        if extra_ids is None:
            extra_ids = ["quantile"]
        elif "quantile" not in extra_ids:
            extra_ids = ["quantile"] + list(extra_ids)

        # Convert extra col names to contain only letters, numbers, and underscores
        if extra_ids is not None:
            df, extra_ids = _fix_extra_columns(df, extra_ids)

        # Validate records using the ForecastRecord model
        df = _validate_records(df, forecast=True, optional_columns=extra_ids)

        # ID columns
        id_cols = ["source"] if extra_ids is None else ["source"] + extra_ids
        if self._id_columns is None:
            self._id_columns = id_cols
        else:
            # re-add "quantile" to id columns if missing
            if "quantile" not in self._id_columns:
                self._id_columns += ["quantile"]
            # check that the id columns of the new forecasts match the existing ones
            # and if not adjust the datasets
            if set(self._id_columns) != set(id_cols):
                all_id_cols = list(set(self._id_columns).union(set(id_cols)))
                for col in all_id_cols:
                    if col not in self._id_columns:
                        # add missing columns to existing data
                        self._raw_forecasts[col] = pd.NA
                        self._forecasts[col] = pd.NA
                        self._density_df[col] = pd.NA
                        self._density_forecasts[col] = pd.NA
                        self._id_columns += [col]
                    if col not in id_cols:
                        # add missing columns to new data
                        df[col] = pd.NA

        # Check for duplicates if there are already some records stored
        if not self._density_forecasts.empty:
            df = _check_duplicates(df, self._density_forecasts)

        # Check if forecasts have corresponding outturns
        _check_missing_outturns(df, self._outturns)

        # create a unique identifier for forecasts
        df["unique_id"] = construct_unique_id(df, self._id_columns)

        # Transform density forecasts
        # Unlike point forecasts, we don't match the data with outturns
        # It doesnt make sense to match the quantile 0.1 with the outturn
        # and take yoy, for instance
        forecasts_levels = df.copy()
        forecasts_levels["metric"] = "levels"
        forecasts_yoy = _prepare_density_forecasts(df, "yoy")
        forecasts_pop = _prepare_density_forecasts(df, "pop")
        forecasts = pd.concat([forecasts_levels, forecasts_yoy, forecasts_pop], ignore_index=True)

        # trim outturns from forecasts
        forecasts = forecasts[forecasts["forecast_horizon"] >= 0]

        # Ensure quantile remains float in forecasts
        df["quantile"] = df["quantile"].astype(float)
        forecasts["quantile"] = forecasts["quantile"].astype(float)

        # reconstruct id without quantiles
        self._id_columns = [col for col in self._id_columns if col != "quantile"]
        df["unique_id"] = construct_unique_id(df, self._id_columns)
        forecasts["unique_id"] = construct_unique_id(forecasts, self._id_columns)

        # Add to existing density-specific data
        self._density_df = pd.concat([self._density_df, df], ignore_index=True)
        self._density_forecasts = pd.concat([self._density_forecasts, forecasts], ignore_index=True)

    @property
    def density_forecasts(self) -> pd.DataFrame:
        """Get density forecasts with quantile information."""
        return self._density_forecasts

    def filter(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_vintage: Optional[str] = None,
        end_vintage: Optional[str] = None,
        variables: Optional[list[str]] = None,
        metrics: Optional[list[str]] = None,
        sources: Optional[list[str]] = None,
        frequencies: Optional[list[str]] = None,
        custom_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        filter_point_forecasts: Optional[bool] = True,
        filter_density_forecasts: Optional[bool] = True,
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
        filter_point_forecasts : bool, optional
            Whether to apply the filter to point forecasts. Default is True.
        filter_density_forecasts : bool, optional
            Whether to apply the filter to density forecasts. Default is True.

        Returns
        -------
        ForecastData
            The filtered ForecastData object (for method chaining).
        """
        # Call parent filter for regular forecasts (only if they exist)
        if not self._forecasts.empty and filter_point_forecasts:
            super().filter(
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

        # Also filter density forecasts (only if they exist)
        if not self._density_forecasts.empty and filter_density_forecasts:
            self._density_forecasts = self._apply_filter_with_standardized_columns(
                self._density_forecasts,
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

        return self

    def clear_filter(self) -> None:
        """Reset both parent forecasts and density forecasts to include all original data."""
        # Call parent clear_filter for regular forecasts
        super().clear_filter()

        # Also reset density forecasts (only if they exist)
        if not self._density_df.empty:
            forecasts = prepare_forecasts(self._density_df, self._outturns, self._id_columns)
            self._density_forecasts = forecasts

    def sample_from_density(self, n_samples: int = 10000, random_state: Optional[int] = None) -> pd.DataFrame:
        """Generate samples from the empirical distribution defined by quantiles.

        Uses inverse transform sampling to draw samples from the distribution
        defined by the quantile forecasts.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate per forecast group. Default is 10000.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with sampled values. Columns: [id_columns..., 'sample_id', 'value']

        Notes
        -----
        Works well when you have many quantiles (50+). With few quantiles,
        consider using parametric fitting instead.

        Examples
        --------
        >>> samples = density_data.sample_from_density(n_samples=10000, random_state=42)
        >>> mean = samples.groupby(['date', 'variable'])['value'].mean()
        """
        # Group by all columns except 'quantile' and 'value'
        group_cols = [col for col in self._density_forecasts.columns if col not in ["quantile", "value"]]

        # Sample for each group
        sampled_groups = []

        grouped = self._density_forecasts.groupby(group_cols)

        for group_vals, group in tqdm(grouped, desc="Sampling from density", total=len(grouped)):
            group = group.sort_values("quantile")

            # Create interpolated quantile function
            quantile_func = _interpolate_quantile_function(group["quantile"].values, group["value"].values)

            # Draw samples
            samples = _sample_from_quantile_function(quantile_func, n_samples, random_state)

            # Create DataFrame for this group - replicate group metadata
            group_df = group[group_cols].iloc[[0]].copy()
            group_df = pd.concat([group_df] * n_samples, ignore_index=True)
            group_df["sample_id"] = range(n_samples)
            group_df["value"] = samples

            sampled_groups.append(group_df)

        return pd.concat(sampled_groups, ignore_index=True)

    def to_point_forecast(self, method: str = "median") -> ForecastData:
        """Convert density forecasts to point forecasts.

        Parameters
        ----------
        method : str, optional
            Method to extract point forecast:
            - 'median': Use 0.5 quantile (default, most robust)
            - 'mean': Average via sampling from distribution
            - specific quantile: e.g., '0.5', '0.75'

        Returns
        -------
        ForecastData
            Point forecast data object.

        Examples
        --------
        >>> # Convert using median
        >>> point_data = density_data.to_point_forecast('median')
        >>>
        >>> # Convert using mean via sampling
        >>> point_data = density_data.to_point_forecast('mean')
        >>>
        >>> # Convert using specific quantile
        >>> point_data = density_data.to_point_forecast('0.75')
        """
        import warnings

        if method == "median":
            target_quantile = 0.5
        elif method == "mean":
            raise NotImplementedError("Mean estimation from density forecasts is not implemented yet.")
            return
        else:
            try:
                target_quantile = float(method)
            except ValueError:
                raise ValueError(f"Invalid method '{method}'. Use 'median', 'mean', or a quantile value (e.g., '0.5').")

        # Try to find exact match
        forecast_df = self._density_df[self._density_df["quantile"] == target_quantile].copy()

        # If no exact match, find closest quantile
        if forecast_df.empty:
            available_quantiles = self._density_df["quantile"].unique()

            # Find closest quantile
            closest_quantile = min(available_quantiles, key=lambda x: abs(x - target_quantile))

            warnings.warn(
                f"Quantile {target_quantile} not found. Using closest available quantile: {closest_quantile}",
                UserWarning,
                stacklevel=2,
            )

            forecast_df = self._density_df[self._density_df["quantile"] == closest_quantile].copy()

        super().add_forecasts(forecast_df)

    def merge(self, other: "ForecastData") -> "ForecastData":
        """Merge another ForecastData or DensityForecastData instance into this one.

        Parameters
        ----------
        other : ForecastData or DensityForecastData
            Another ForecastData or DensityForecastData instance to merge with this one.

        Returns
        -------
        DensityForecastData
           Updated DensityForecastData instance containing merged data from both instances.
        """

        if not other._raw_outturns.empty:
            self.add_outturns(other._raw_outturns)

        if not other._raw_forecasts.empty:
            self.add_forecasts(other._raw_forecasts, extra_ids=other._id_columns)

        if isinstance(other, DensityForecastData):
            if not other._density_df.empty:
                self.add_density_forecasts(other._density_df, extra_ids=other._id_columns)

        return self

    def __repr__(self) -> str:
        """Return DataFrame representation when printing the class."""
        return self._density_forecasts.__repr__()

    def plot_density_vintage(
        self,
        variable: str,
        vintage_date: str | pd.Timestamp,
        quantiles: Optional[list[float]] = [0.16, 0.5, 0.84],
        forecast_source: list[str] = None,
        outturn_start_date: str | pd.Timestamp = None,
        frequency: Literal["Q", "M"] = "Q",
        metric: Literal["levels", "pop", "yoy"] = "levels",
        return_plot: bool = False,
        **kwargs,
    ) -> None:
        """Plot forecast density plots.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the plotting function.

        Notes
        -----
        This method creates density plots for the density forecasts.
        """
        from forecast_evaluation.visualisations.density_plots import plot_density_vintage

        return plot_density_vintage(
            self,
            variable,
            vintage_date,
            forecast_source=forecast_source,
            quantiles=quantiles,
            outturn_start_date=outturn_start_date,
            frequency=frequency,
            metric=metric,
            return_plot=return_plot,
            **kwargs,
        )


def _prepare_density_forecasts(df: pd.DataFrame, transform: str) -> pd.DataFrame:
    df_transformed = []

    for frequency in df["frequency"].unique():
        df_freq = df[df["frequency"] == frequency].copy()

        # the sorting_cols are all cols but value and date
        grouping_cols = [col for col in df_freq.columns if col not in ["value", "date", "forecast_horizon"]]
        sorting_cols = grouping_cols + ["date"]
        df_freq = df_freq.sort_values(sorting_cols)

        if transform == "pop":
            df_freq["value"] = df_freq.groupby(grouping_cols)["value"].pct_change(periods=1)
            df_freq["metric"] = "pop"
        elif transform == "yoy":
            n_periods = {"Q": 4, "M": 12}[frequency]
            df_freq["value"] = df_freq.groupby(grouping_cols)["value"].pct_change(periods=n_periods)
            df_freq["metric"] = "yoy"

        df_freq = df_freq[df_freq["value"].notna()]
        df_transformed.append(df_freq)

    return pd.concat(df_transformed, ignore_index=True)


def _interpolate_quantile_function(quantiles: np.ndarray, values: np.ndarray) -> Callable:
    """Create interpolated quantile function (inverse CDF) from quantile data.

    Parameters
    ----------
    quantiles : np.ndarray
        Array of quantile levels (between 0 and 1).
    values : np.ndarray
        Array of values corresponding to each quantile.

    Returns
    -------
    Callable
        Interpolation function mapping probability → value.

    Notes
    -----
    Uses linear interpolation between quantiles. Values outside the range
    are extrapolated using the nearest quantile value.
    """
    return interp1d(
        quantiles, values, kind="linear", bounds_error=False, fill_value=(values[0], values[-1]), assume_sorted=True
    )


def _interpolate_cdf_function(quantiles: np.ndarray, values: np.ndarray) -> Callable:
    """Create interpolated CDF function from quantile data.

    Parameters
    ----------
    quantiles : np.ndarray
        Array of quantile levels (between 0 and 1).
    values : np.ndarray
        Array of values corresponding to each quantile.

    Returns
    -------
    Callable
        Interpolation function mapping value → cumulative probability.

    Notes
    -----
    This is the inverse of the quantile function. Uses linear interpolation
    between values. Probabilities outside the range are clipped to [0, 1].
    """
    return interp1d(values, quantiles, kind="linear", bounds_error=False, fill_value=(0.0, 1.0), assume_sorted=True)


def _sample_from_quantile_function(
    quantile_func: Callable, n_samples: int, random_state: Optional[int] = None
) -> np.ndarray:
    """Draw samples from distribution using inverse transform sampling.

    Parameters
    ----------
    quantile_func : Callable
        Quantile function (inverse CDF) that maps probability → value.
    n_samples : int
        Number of samples to draw.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of sampled values.

    Notes
    -----
    Uses inverse transform sampling:
    1. Draw uniform random samples U ~ Uniform(0, 1)
    2. Transform using inverse CDF: X = Q(U)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Draw uniform samples
    uniform_samples = np.random.uniform(0, 1, size=n_samples)

    # Transform using quantile function (inverse CDF)
    samples = quantile_func(uniform_samples)

    return samples


# Example usage:
if __name__ == "__main__":
    import forecast_evaluation as fe

    # load point forecasts
    forecast_data = fe.ForecastData(load_fer=True)

    # sample density forecasts
    density_forecast_df = create_sample_density_forecasts()
    outturns_df = forecast_data.outturns
    outturns_df = outturns_df[outturns_df["metric"] == "levels"]

    density_data = fe.DensityForecastData(outturns_data=outturns_df, forecasts_data=density_forecast_df)

    # merge
    density_data.merge(forecast_data)

    # run dashboard
    density_data.run_dashboard()

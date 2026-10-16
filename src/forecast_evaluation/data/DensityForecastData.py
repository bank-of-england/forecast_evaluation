"""Density forecast data class for handling probabilistic forecasts with quantiles."""

import warnings
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
from tqdm import tqdm

from forecast_evaluation.data.ForecastData import ForecastData, _atomic_state
from forecast_evaluation.data.sample_data import (
    create_sample_density_forecasts,
)

# Outer quantile levels farther than this from 0 or 1 trigger the tail-clamping warning in sample_from_density.
_TAIL_MASS_WARNING_THRESHOLD = 0.02


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
        DataFrame containing density forecast records. Must include a 'quantile' column.
    load_fer : bool, optional
        Whether to load FER (Forecast Evaluation Report) data. Default is False.
    extra_ids : list of str, optional
        Additional identification columns beyond 'source'. Quantile distinguishes
        observations within a density, not separate forecasts.

    Examples
    --------
    >>> from forecast_evaluation.data import DensityForecastData
    >>> from forecast_evaluation.data.sample_data import create_sample_density_forecasts, create_sample_outturns
    >>> density_data = DensityForecastData(
    ...     outturns_data=create_sample_outturns(), forecasts_data=create_sample_density_forecasts()
    ... )
    >>> density_data.density_forecasts["quantile"].nunique()
    50
    """

    _forecast_tables = (*ForecastData._forecast_tables, "_density_df", "_density_forecasts")

    def __init__(
        self,
        outturns_data: Optional[pd.DataFrame] = None,
        forecasts_data: Optional[pd.DataFrame] = None,
        load_fer: Optional[bool] = False,
        extra_ids: Optional[list[str]] = None,
        compute_levels: bool = True,
    ):
        """Initialise DensityForecastData.

        Initialises the density forecast data object. Outturns must be available
        before density forecasts are added. Quantiles share the source's forecast ID.
        """
        self._density_forecasts = pd.DataFrame()
        self._density_df = pd.DataFrame()

        # Initialise parent class without forecasts
        super().__init__(
            outturns_data=outturns_data, load_fer=load_fer, extra_ids=extra_ids, compute_levels=compute_levels
        )

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
            Additional identification columns beyond 'source'. The 'quantile'
            column is validated separately and never identifies a forecast.

        Raises
        ------
        ValueError
            If 'quantile' column is missing from the DataFrame.

        Examples
        --------
        >>> from forecast_evaluation.data.sample_data import create_sample_density_forecasts, create_sample_outturns
        >>> density_data = DensityForecastData(outturns_data=create_sample_outturns())
        >>> density_data.add_density_forecasts(create_sample_density_forecasts())
        >>> density_data.density_forecasts["quantile"].nunique()
        50
        """
        if "quantile" not in df.columns:
            raise ValueError("Density forecasts must include a 'quantile' column")

        if "metric" not in df.columns:
            df = df.assign(metric="levels")

        extra_ids = [column for column in extra_ids or [] if column != "quantile"]
        with _atomic_state(self, *self._forecast_state):
            # Pass the backing table name because validation may align its id columns.
            df = self._validate_new_forecasts(df, stored="_density_df", extra_ids=extra_ids, extra_columns=["quantile"])

            self._density_df = pd.concat([self._density_df, df], ignore_index=True)
            self._density_forecasts = pd.concat([self._density_forecasts, df], ignore_index=True)

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
        """Filter the outturns, forecasts, main and density tables to only include data within specified
        date and vintage ranges, and optionally by variables, metrics, sources, or a custom filter.

        The filter is atomic: if any table fails to filter, no table is changed.

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
            Whether to filter the outturns, point forecasts and main table. Default is True.
            Outturns and the main table are filtered even when there are no point forecasts.
        filter_density_forecasts : bool, optional
            Whether to apply the filter to density forecasts. Default is True.
        """
        criteria = {
            "start_date": start_date,
            "end_date": end_date,
            "start_vintage": start_vintage,
            "end_vintage": end_vintage,
            "variables": variables,
            "metrics": metrics,
            "sources": sources,
            "frequencies": frequencies,
            "custom_filter": custom_filter,
        }

        updates, skipped = self._filtered_tables(**criteria) if filter_point_forecasts else ({}, {})

        if filter_density_forecasts and not self._density_forecasts.empty:
            updates["_density_forecasts"] = self._apply_filter_with_standardized_columns(
                self._density_forecasts, rename_cols=True, table="_density_forecasts", skipped=skipped, **criteria
            )

        self._commit_filter(updates, skipped)

    def clear_filter(self) -> None:
        """Reset both parent forecasts and density forecasts to include all original data."""
        # Call parent clear_filter for regular forecasts
        super().clear_filter()

        # Also reset density forecasts (only if they exist)
        if not self._density_df.empty:
            self._density_forecasts = self._density_df.copy()

    def sample_from_density(
        self, n_samples: int = 10000, random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> pd.DataFrame:
        """Generate samples from the empirical distribution defined by quantiles.

        Uses inverse transform sampling to draw samples from the distribution
        defined by the quantile forecasts, as currently filtered.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate per forecast group. Default is 10000.
        random_state : int or numpy.random.Generator, optional
            Seed (or generator) for one private random generator used for the whole call.
            The global NumPy random state is neither read nor modified, so reproducibility
            requires passing ``random_state``.

        Returns
        -------
        pd.DataFrame
            One block of ``n_samples`` rows per density. Columns: all density-table columns
            except 'quantile' and 'value' (including 'unique_id' and 'target_minus_vintage'),
            then 'sample_id' (0 to ``n_samples - 1``) and 'value'.

        Raises
        ------
        ValueError
            If there are no density forecasts to sample from (none loaded, or all removed
            by a filter), or if any density has fewer than two distinct quantile levels.

        Warns
        -----
        UserWarning
            If any density has an outer quantile level farther than 0.02 from 0 or 1.

        Notes
        -----
        Outside the stored quantile range the quantile function is clamped to the
        end values (it is not extrapolated), so mass ``min(quantile)`` and
        ``1 - max(quantile)`` sits exactly on the two end values and the samples are too
        narrow. The effect is driven by how close the outer levels are to 0 and 1, not by
        the number of levels: a grid from 0.01 to 0.99 is nearly unaffected, while 0.1 to 0.9
        gives a sample standard deviation of roughly 82% of a normal distribution's.

        Draws are marginal only: they are independent across densities, and ``sample_id`` is
        an index, not a path, so sums or growth rates across dates treat them as independent.

        Examples
        --------
        >>> from forecast_evaluation.data.sample_data import create_sample_density_forecasts, create_sample_outturns
        >>> density_data = DensityForecastData(
        ...     outturns_data=create_sample_outturns(), forecasts_data=create_sample_density_forecasts()
        ... )
        >>> samples = density_data.sample_from_density(n_samples=10, random_state=42)
        >>> mean = samples.groupby(['date', 'variable'])['value'].mean()
        """
        if self._density_forecasts.empty:
            raise ValueError("No density forecasts to sample from (none loaded, or the filter removed them all).")

        # Positional index so each group's first row can be found without relying on labels
        density = self._density_forecasts.reset_index(drop=True)

        # Group by all columns except 'quantile' and 'value'
        group_cols = [col for col in density.columns if col not in ["quantile", "value"]]

        grouped = density.groupby(group_cols, sort=True, dropna=False)

        levels = grouped["quantile"].agg(["min", "max", "nunique"])
        short = levels[levels["nunique"] < 2]
        if not short.empty:
            names = "; ".join(str(key) for key in short.index[:5])
            raise ValueError(
                f"{len(short)} density(ies) have fewer than 2 distinct quantile levels and cannot be sampled: {names}"
            )
        _warn_if_tails_clamped(levels)

        rng = np.random.default_rng(random_state)

        draws = []
        first_rows = []

        for _, group in tqdm(grouped, desc="Sampling from density", total=len(grouped)):
            group = group.sort_values("quantile")

            # Create interpolated quantile function
            quantile_func = _interpolate_quantile_function(group["quantile"].values, group["value"].values)

            draws.append(_sample_from_quantile_function(quantile_func, n_samples, rng))
            first_rows.append(group.index[0])

        # One metadata row per group, repeated n_samples times
        metadata = density.loc[first_rows, group_cols].reset_index(drop=True)
        result = metadata.loc[metadata.index.repeat(n_samples)].reset_index(drop=True)
        result["sample_id"] = np.tile(np.arange(n_samples, dtype=np.int64), len(first_rows))
        result["value"] = np.concatenate(draws)

        return result

    def to_point_forecast(self, method: str = "median") -> None:
        """Add a selected quantile to this object's point forecasts.

        Reads the density forecasts as currently filtered. If the filter leaves no
        density forecasts, a warning is raised and nothing is added.

        Parameters
        ----------
        method : str, optional
            Method to extract point forecast:
            - 'median': Use 0.5 quantile (default, most robust)
            - 'mean': Not implemented
            - specific quantile: e.g., '0.5', '0.75'

        Returns
        -------
        None
            The point forecasts are stored on this object.

        Raises
        ------
        ValueError
            If no density forecasts have been loaded, or ``method`` is invalid.

        Examples
        --------
        >>> from forecast_evaluation.data.sample_data import create_sample_density_forecasts, create_sample_outturns
        >>> density_data = DensityForecastData(
        ...     outturns_data=create_sample_outturns(), forecasts_data=create_sample_density_forecasts()
        ... )
        >>> density_data.to_point_forecast(str(density_data.density_forecasts["quantile"].iloc[0]))
        >>> density_data.forecasts.empty
        False
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

        if self._density_df.empty:
            raise ValueError("No density forecasts loaded. Add density forecasts before calling to_point_forecast.")

        density = self._density_forecasts
        if density.empty:
            warnings.warn(
                "No density forecasts remain after filtering; no point forecasts were added.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Try to find exact match
        forecast_df = density[density["quantile"] == target_quantile].copy()

        # If no exact match, find closest quantile
        if forecast_df.empty:
            available_quantiles = density["quantile"].unique()

            # Find closest quantile
            closest_quantile = min(available_quantiles, key=lambda x: abs(x - target_quantile))

            warnings.warn(
                f"Quantile {target_quantile} not found. Using closest available quantile: {closest_quantile}",
                UserWarning,
                stacklevel=2,
            )

            forecast_df = density[density["quantile"] == closest_quantile].copy()

        super().add_forecasts(forecast_df)

    def merge(self, other: "ForecastData", compute_levels: bool = True) -> None:
        """Merge another ForecastData or DensityForecastData instance into this one.

        Parameters
        ----------
        other : ForecastData or DensityForecastData
            Another ForecastData or DensityForecastData instance to merge with this one.
        compute_levels : bool, optional
            Whether to automatically transform non-levels forecasts from `other`
            to levels if outturns data is available.
            When True, forecasts in 'pop' and 'yoy' metrics will be converted to
            levels using the available outturns data.
            Useful if you add 'pop' and want to analyse 'yoy' forecasts and vice versa.
            If the transformation fails for specific groups (e.g., due to insufficient
            historical data), those groups will be skipped with a warning message.
            Default is True.
        """

        super().merge(other, compute_levels=compute_levels)

        if isinstance(other, DensityForecastData):
            if not other._density_df.empty:
                self.add_density_forecasts(other._density_df, extra_ids=other._extra_ids)

    def __repr__(self) -> str:
        """Return DataFrame representation when printing the class."""
        return self._density_forecasts.__repr__()

    def _repr_html_(self) -> str | None:
        """Return the density table as HTML so notebooks match ``__repr__``."""
        return self._density_forecasts._repr_html_()

    def plot_density_vintage(
        self,
        variable: str,
        vintage_date: str | pd.Timestamp,
        quantiles: Optional[list[float]] = None,
        forecast_source: Optional[list[str]] = None,
        outturn_start_date: str | pd.Timestamp | None = None,
        frequency: Union[Literal["Q", "M"], None] = None,
        metric: Literal["levels", "pop", "yoy"] = "levels",
        return_plot: bool = False,
        **kwargs,
    ) -> tuple | None:
        """Plot forecast density plots.

        Parameters
        ----------
        quantiles : list of float, optional
            Quantile levels to plot. If None, uses [0.16, 0.5, 0.84]. The list passed in
            is not modified.
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
    are clamped to the end values (no extrapolation).
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


def _warn_if_tails_clamped(levels: pd.DataFrame) -> None:
    """Warn once if any density's outer quantile levels are far from 0 and 1.

    ``levels`` has one row per density with columns 'min' and 'max' (outer quantile levels).
    """
    outer_mass = np.maximum(levels["min"].to_numpy(), 1 - levels["max"].to_numpy())
    # Small tolerance because 1 - 0.99 is 0.010000000000000009 in floating point
    flagged = outer_mass > _TAIL_MASS_WARNING_THRESHOLD + 1e-9
    if not flagged.any():
        return

    worst = int(np.argmax(outer_mass))
    p = float(outer_mass[worst])
    c = norm.ppf(1 - p)
    # Standard deviation of a standard normal clamped at +/- c, relative to the unclamped value of 1
    sd_ratio = float(np.sqrt((1 - 2 * p) - 2 * c * norm.pdf(c) + 2 * p * c**2))

    warnings.warn(
        f"sample_from_density: {int(flagged.sum())} of {len(levels)} densities have outer quantile levels "
        f"farther than {_TAIL_MASS_WARNING_THRESHOLD} from 0 or 1 (worst: {levels['min'].iloc[worst]:g} and "
        f"{levels['max'].iloc[worst]:g}). Samples are clamped to the stored range, so about {p * 100:.3g}% of "
        f"draws sit exactly on each end value and the draws are too narrow: for a normal-like distribution the "
        f"sample standard deviation will be roughly {sd_ratio * 100:.0f}% of the true value. Supply quantiles "
        f"closer to 0 and 1, or treat the draws as truncated.",
        UserWarning,
        stacklevel=3,
    )


def _sample_from_quantile_function(
    quantile_func: Callable, n_samples: int, rng: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """Draw samples from distribution using inverse transform sampling.

    Parameters
    ----------
    quantile_func : Callable
        Quantile function (inverse CDF) that maps probability → value.
    n_samples : int
        Number of samples to draw.
    rng : int or numpy.random.Generator, optional
        Seed or generator for a private random stream. The global NumPy random state is not touched.

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
    rng = np.random.default_rng(rng)

    # Draw uniform samples
    uniform_samples = rng.random(n_samples)

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

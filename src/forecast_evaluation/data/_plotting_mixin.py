from datetime import date
from typing import Literal, Optional, Union

import pandas as pd


class PlottingMixin:
    """Mixin that adds visualisation methods to ForecastData."""

    def plot_hedgehog(
        self,
        variable: str,
        forecast_source: str,
        metric: Literal["levels", "pop", "yoy"],
        frequency: Optional[Literal["Q", "M"]] = None,
        k: int = 12,
        date_start: Union[str, date, None] = None,
        convert_to_percentage: bool = False,
        return_plot: bool = False,
    ):
        """Generate a hedgehog plot comparing forecasts with outturns.

        Parameters
        ----------
        variable : str
            Name of the variable to plot (e.g., 'cpisa', 'gdpkp').
        forecast_source : str
            Source of the forecasts.
        metric : {"levels", "pop", "yoy"}
            Type of transformation to apply to the data.
        k : int, default 12
            Number of revisions used to define the outturns.
        date_start : str, date, or None, default None
            Optional start date to filter the data.
        convert_to_percentage : bool, default False
            If True, multiplies values on the y-axis by 100.
        return_plot : bool, default False
            If True, returns the matplotlib figure and axis objects.

        Returns
        -------
        fig, ax : tuple or None
            If return_plot is True, returns a tuple (fig, ax).
            Otherwise, returns None.
        """
        from forecast_evaluation.visualisations.hedgehog import plot_hedgehog

        return plot_hedgehog(
            data=self,
            variable=variable,
            forecast_source=forecast_source,
            metric=metric,
            frequency=frequency,
            k=k,
            date_start=date_start,
            convert_to_percentage=convert_to_percentage,
            return_plot=return_plot,
        )

    def plot_forecast_errors(
        self,
        variable: str,
        metric: Literal["levels", "pop", "yoy"],
        source: str,
        vintage_date_forecast: str,
        frequency: Optional[Literal["Q", "M"]] = None,
        k: int = 12,
        convert_to_percentage: bool = False,
        return_plot: bool = False,
    ):
        """Plot average forecast errors for a specific variable/source/vintage combination.

        Parameters
        ----------
        variable : str
            The variable to analyse (e.g., 'gdpkp').
        metric : {"levels", "pop", "yoy"}
            The metric to analyse.
        source : str
            The source of the forecasts.
        vintage_date_forecast : str
            The vintage date of the forecasts (e.g., '2022-03-31').
        k : int, default 12
            Number of revisions used to define the outturns.
        convert_to_percentage : bool, default False
            If True, multiplies values on the y-axis by 100.
        return_plot : bool, default False
            If True, returns the matplotlib figure and axis objects.

        Returns
        -------
        fig, ax : tuple or None
            If return_plot is True, returns a tuple (fig, ax). Otherwise, returns None.
        """
        from forecast_evaluation.visualisations.forecast_errors import plot_forecast_errors

        return plot_forecast_errors(
            data=self,
            variable=variable,
            metric=metric,
            source=source,
            vintage_date_forecast=vintage_date_forecast,
            frequency=frequency,
            k=k,
            convert_to_percentage=convert_to_percentage,
            return_plot=return_plot,
        )

    def plot_forecast_errors_by_horizon(
        self,
        variable: str,
        source: Union[str, list[str]],
        metric: Literal["levels", "pop", "yoy"],
        frequency: Optional[Literal["Q", "M"]] = None,
        k: int = 12,
        convert_to_percentage: bool = False,
        return_plot: bool = False,
    ):
        """Plot average forecast errors by forecast horizon, averaged over all forecast vintages.

        Parameters
        ----------
        variable : str
            The variable to analyse (e.g., 'gdpkp').
        source : str or list of str
            The source(s) of the forecasts. When a list is provided, each source is
            plotted as a separate line on the same axes.
        metric : {"levels", "pop", "yoy"}
            The metric to analyse.
        k : int, default 12
            Number of revisions used to define the outturns.
        convert_to_percentage : bool, default False
            If True, multiplies values on the y-axis by 100.
        return_plot : bool, default False
            If True, returns the matplotlib figure and axis objects.

        Returns
        -------
        fig, ax : tuple or None
            If return_plot is True, returns a tuple (fig, ax). Otherwise, returns None.
        """
        from forecast_evaluation.visualisations.forecast_errors import plot_forecast_errors_by_horizon

        return plot_forecast_errors_by_horizon(
            data=self,
            variable=variable,
            source=source,
            metric=metric,
            frequency=frequency,
            k=k,
            convert_to_percentage=convert_to_percentage,
            return_plot=return_plot,
        )

    def plot_outturn_revisions(
        self,
        variable: str,
        metric: Literal["levels", "pop", "yoy"],
        frequency: Optional[Literal["Q", "M"]] = None,
        k: Union[int, list[int]] = 12,
        fill_k: bool = False,
        ma_window: int = 1,
        start_date: Union[date, str, None] = None,
        end_date: Union[date, str, None] = None,
        convert_to_percentage: bool = False,
        return_plot: bool = False,
    ):
        """Create outturn revisions plot.

        Parameters
        ----------
        variable : str
            The variable to analyse (e.g., 'gdpkp').
        metric : {"levels", "pop", "yoy"}
            The metric to analyse.
        k : int or list of int, default 12
            Number of revisions used to define the outturns. A list plots multiple
            revision horizons on the same axes.
        fill_k : bool, default False
            If True, uses only the latest vintage for each date when calculating revisions.
        ma_window : int, default 1
            Size of moving average window to smooth the revisions.
        start_date : date or str, default None
            The start date for the plot. If None, uses the earliest date in the data.
        end_date : date or str, default None
            The end date for the plot. If None, uses the latest date in the data.
        convert_to_percentage : bool, default False
            If True, multiplies values on the y-axis by 100.
        return_plot : bool, default False
            If True, returns the matplotlib figure and axis objects.

        Returns
        -------
        fig, ax : tuple or None
            If return_plot is True, returns a tuple (fig, ax). Otherwise, returns None.
        """
        from forecast_evaluation.visualisations.outturn_revisions import plot_outturn_revisions

        return plot_outturn_revisions(
            data=self,
            variable=variable,
            metric=metric,
            frequency=frequency,
            k=k,
            fill_k=fill_k,
            ma_window=ma_window,
            start_date=start_date,
            end_date=end_date,
            convert_to_percentage=convert_to_percentage,
            return_plot=return_plot,
        )

    def plot_outturns(
        self,
        variable: str,
        metric: Literal["levels", "pop", "yoy"],
        frequency: Optional[Literal["Q", "M"]] = None,
        k: Union[int, list[int]] = 12,
        fill_k: bool = True,
        start_date: Union[date, str, None] = None,
        end_date: Union[date, str, None] = None,
        convert_to_percentage: bool = False,
        return_plot: bool = False,
    ):
        """Create outturns plot.

        Parameters
        ----------
        variable : str
            The variable to analyse (e.g., 'gdpkp').
        metric : {"levels", "pop", "yoy"}
            The metric to analyse.
        k : int or list of int, default 12
            Number of revisions used to define the outturns.
        fill_k : bool, default True
            If True, uses only the latest vintage for each date.
        start_date : date or str, default None
            The start date for the plot. If None, uses the earliest date in the data.
        end_date : date or str, default None
            The end date for the plot. If None, uses the latest date in the data.
        convert_to_percentage : bool, default False
            If True, multiplies values on the y-axis by 100.
        return_plot : bool, default False
            If True, returns the matplotlib figure and axis objects.

        Returns
        -------
        fig, ax : tuple or None
            If return_plot is True, returns a tuple (fig, ax). Otherwise, returns None.
        """
        from forecast_evaluation.visualisations.outturn_revisions import plot_outturns

        return plot_outturns(
            data=self,
            variable=variable,
            metric=metric,
            frequency=frequency,
            k=k,
            fill_k=fill_k,
            start_date=start_date,
            end_date=end_date,
            convert_to_percentage=convert_to_percentage,
            return_plot=return_plot,
        )

    def plot_average_revision_by_period(
        self,
        source: str,
        variable: str,
        metric: str,
        frequency=None,
        return_plot: bool = False,
    ):
        """Plot the average revision grouped by forecast horizon.

        Parameters
        ----------
        source : str
            Forecast source identifier (e.g., 'mpr').
        variable : str
            Variable to analyse (e.g., 'gdpkp', 'cpisa').
        metric : str
            Metric to analyse ('levels', 'pop', or 'yoy').
        return_plot : bool, default False
            If True, returns (fig, ax) tuple instead of displaying the plot.

        Returns
        -------
        fig, ax : tuple or None
            If return_plot is True, returns a tuple (fig, ax). Otherwise, returns None.
        """
        from forecast_evaluation.visualisations.revisions_predictability import plot_average_revision_by_period

        return plot_average_revision_by_period(
            data=self,
            source=source,
            variable=variable,
            metric=metric,
            frequency=frequency,
            return_plot=return_plot,
        )

    def plot_vintage(
        self,
        variable: str,
        vintage_date: Union[str, pd.Timestamp],
        forecast_source: Optional[list[str]] = None,
        outturn_start_date: Union[str, pd.Timestamp, None] = None,
        frequency: Optional[Literal["Q", "M"]] = None,
        metric: Literal["levels", "pop", "yoy"] = "levels",
        k: int = 12,
        convert_to_percentage: bool = False,
        return_plot: bool = False,
    ):
        """Generate a plot comparing forecasts from different sources for a specific vintage.

        Parameters
        ----------
        variable : str
            Name of the variable to plot.
        vintage_date : str or pd.Timestamp
            The vintage date to plot.
        forecast_source : list of str, optional
            List of forecast sources to include. If None, all sources are used.
        outturn_start_date : str or pd.Timestamp, optional
            Start date for outturn data to display. If None, all available outturns are used.
        metric : {"levels", "pop", "yoy"}, default "levels"
            Type of transformation to apply to the data.
        k : int, default 12
            Number of revisions used to define the outturns.
        convert_to_percentage : bool, default False
            If True, multiplies values on the y-axis by 100.
        return_plot : bool, default False
            If True, returns the matplotlib figure and axis objects.

        Returns
        -------
        fig, ax : tuple or None
            If return_plot is True, returns a tuple (fig, ax). Otherwise, returns None.
        """
        from forecast_evaluation.visualisations.forecast import plot_vintage

        return plot_vintage(
            data=self,
            variable=variable,
            vintage_date=vintage_date,
            forecast_source=forecast_source,
            outturn_start_date=outturn_start_date,
            frequency=frequency,
            metric=metric,
            k=k,
            convert_to_percentage=convert_to_percentage,
            return_plot=return_plot,
        )

    def plot_errors_across_time(
        self,
        variable: str,
        metric: Literal["levels", "pop", "yoy"],
        error: Literal["raw", "absolute", "squared"] = "raw",
        horizons: Union[int, list[int], None] = None,
        sources: Union[str, list[str], None] = None,
        frequency: Optional[Literal["Q", "M"]] = None,
        k: int = 12,
        ma_window: int = 1,
        show_mean: bool = True,
        convert_to_percentage: bool = False,
        return_plot: bool = False,
        custom_labels: Optional[dict] = None,
        existing_plot: Optional[tuple] = None,
    ):
        """Plot forecast errors across time for one or more horizons and sources.

        Parameters
        ----------
        variable : str
            The variable to analyse (e.g., 'gdpkp', 'cpisa').
        metric : {"levels", "pop", "yoy"}
            The metric to analyse.
        error : {"raw", "absolute", "squared"}, default "raw"
            The type of error to plot.
        horizons : int or list of int, default None
            The forecast horizon(s) to analyse. If None, the minimum horizon in the
            data is used. A list creates faceted subplots by horizon.
        sources : str or list of str, default None
            The source(s) of the forecasts. If None, all sources in the data are used.
        k : int, default 12
            Number of revisions used to define the outturns.
        ma_window : int, default 1
            Size of moving average window to smooth the errors.
        show_mean : bool, default True
            If True, displays horizontal dashed lines showing the mean error per source.
        convert_to_percentage : bool, default False
            If True, multiplies values on the y-axis by 100.
        return_plot : bool, default False
            If True, returns the matplotlib figure and axis objects.
        custom_labels : dict, default None
            A dictionary mapping source names to custom labels for the legend.
        existing_plot : tuple, default None
            A (fig, axes) tuple from a previous call to add new data to existing axes.

        Returns
        -------
        fig, ax : tuple or None
            If return_plot is True, returns a tuple (fig, ax). Otherwise, returns None.
        """
        from forecast_evaluation.visualisations.errors import plot_errors_across_time

        return plot_errors_across_time(
            data=self,
            variable=variable,
            metric=metric,
            error=error,
            horizons=horizons,
            sources=sources,
            frequency=frequency,
            k=k,
            ma_window=ma_window,
            show_mean=show_mean,
            convert_to_percentage=convert_to_percentage,
            return_plot=return_plot,
            custom_labels=custom_labels,
            existing_plot=existing_plot,
        )

    def plot_forecast_error_density(
        self,
        variable: str,
        horizon: int,
        metric: Literal["levels", "pop", "yoy"],
        source: str,
        frequency: Optional[Literal["Q", "M"]] = None,
        k: int = 12,
        highlight_dates: Optional[Union[str, list[str]]] = None,
        highlight_vintages: Optional[Union[str, list[str]]] = None,
        return_plot: bool = False,
    ):
        """Plot density of forecast errors for a specific variable/source/horizon combination.

        Parameters
        ----------
        variable : str
            The variable to analyse (e.g., 'gdpkp').
        horizon : int
            The forecast horizon to analyse.
        metric : {"levels", "pop", "yoy"}
            The metric to analyse.
        source : str
            The source of the forecasts.
        k : int, default 12
            Number of revisions used to define the outturns.
        highlight_dates : str or list of str, optional
            Date(s) to highlight on the density plot (format: 'YYYY-MM-DD').
        highlight_vintages : str or list of str, optional
            Vintage date(s) to highlight (format: 'YYYY-MM-DD'). Takes precedence
            over highlight_dates.
        return_plot : bool, default False
            If True, returns the matplotlib figure and axis objects.

        Returns
        -------
        fig, ax : tuple or None
            If return_plot is True, returns a tuple (fig, ax). Otherwise, returns None.
        """
        from forecast_evaluation.visualisations.forecast_errors import plot_forecast_error_density

        return plot_forecast_error_density(
            data=self,
            variable=variable,
            horizon=horizon,
            metric=metric,
            source=source,
            frequency=frequency,
            k=k,
            highlight_dates=highlight_dates,
            highlight_vintages=highlight_vintages,
            return_plot=return_plot,
        )

"""Server logic for Time Machine and Hedgehog tabs."""

import io

from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info  # Adjust import path as needed
from forecast_evaluation.dashboard.utils import render_legend, remove_legend


def quantile_time_machine(input, output, session, data):
    """Register all server handlers for Time Machine."""

    # Time Machine
    @reactive.calc
    @reactive.event(input.update)
    def get_data():
        all_sources = list(input.sources())
        id_columns = [col for col in data.id_columns if col != "source"]
        for col in id_columns:
            _, _, id_multi = get_selector_info(col, data)
            all_sources += input[id_multi]()

        data_filtered = data.copy()
        data_filtered.filter(
            variables=[input.variable()],
            metrics=[input.transform()],
            sources=all_sources,
            filter_density_forecasts=True,
        )

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter, filter_density_forecasts=True)
        return data_filtered

    def get_plot():
        """Calculate and cache the plot."""
        data_filtered = get_data()

        fig, ax = data_filtered.plot_density_vintage(
            variable=input.variable(),
            frequency="Q",
            vintage_date=input.vintage(),
            outturn_start_date=input.start_date(),
            metric=input.transform(),
            quantiles=[float(input.lower_quantile()), 0.5, float(input.upper_quantile())],
            return_plot=True,
        )

        return fig, ax

    @render.plot
    def quantile_time_machine_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def quantile_time_machine_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def quantile_time_machine_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("quantile_time_machine_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def quantile_time_machine_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("quantile_time_machine_legend", height=f"{input.legend_height()}px")

    @render.download(filename="time_machine_data.csv")
    def quantile_time_machine_download():
        data_filtered = get_data()
        csv_bytes = data_filtered.density_forecasts.to_csv()
        return io.BytesIO(csv_bytes.encode())

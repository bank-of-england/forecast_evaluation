"""Server logic for Time Machine and Hedgehog tabs."""

import io

from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info
from forecast_evaluation.dashboard.utils import render_legend, remove_legend


def time_machine(input, output, session, data):
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
        )

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)
        return data_filtered

    def get_plot():
        """Calculate and cache the plot."""
        data_filtered = get_data()

        fig, ax = fe.plot_vintage(
            data=data_filtered,
            variable=input.variable(),
            frequency="Q",
            vintage_date=input.vintage(),
            outturn_start_date=input.start_date(),
            metric=input.transform(),
            k=int(input.k()),
            return_plot=True,
        )
        return fig, ax

    @render.plot
    def time_machine_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def time_machine_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def time_machine_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("time_machine_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def time_machine_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("time_machine_legend", height=f"{input.legend_height()}px")

    @render.download(filename="time_machine_data.csv")
    def download_time_machine():
        data_filtered = get_data()
        csv_bytes = data_filtered.forecasts.to_csv()
        return io.BytesIO(csv_bytes.encode())

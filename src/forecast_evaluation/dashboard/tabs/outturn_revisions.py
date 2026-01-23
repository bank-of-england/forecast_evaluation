"""Server logic for Outturn Revisions tab."""

import io

from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info
from forecast_evaluation.dashboard.utils import render_legend, remove_legend


def outturns(input, output, session, data):
    @reactive.calc
    @reactive.event(input.update)
    def get_data():
        return data

    def get_plot():
        data_filtered = get_data()

        # Handle k as either a single value or list
        k_values = input.k_multiple_outturns()
        if isinstance(k_values, (list, tuple)):
            k = [int(v) for v in k_values]
        else:
            k = int(k_values)

        fig, ax = fe.plot_outturns(
            data=data_filtered,
            variable=input.variable(),
            metric=input.transform(),
            frequency="Q",
            k=k,
            start_date=input.start_vintage(),
            end_date=input.end_vintage(),
            fill_k=True,
            convert_to_percentage=False,
            return_plot=True,
        )
        return fig, ax

    @render.plot
    def outturns_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def outturns_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def outturns_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("outturns_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def outturns_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("outturns_legend", height=f"{input.legend_height()}px")

    @render.download(filename="outturns_data.csv")
    def download_outturns():
        data = get_data()
        outturns = fe.create_outturn_revisions(data=data)
        csv_bytes = outturns.to_csv()
        return io.BytesIO(csv_bytes.encode())


def outturn_revisions(input, output, session, data):
    """Register all server handlers for Outturn Revisions."""

    @reactive.calc
    @reactive.event(input.update)
    def get_data():
        return data

    def get_plot():
        data_filtered = get_data()

        # Handle k as either a single value or list
        k_values = input.k_multiple()
        if isinstance(k_values, (list, tuple)):
            k = [int(v) for v in k_values]
        else:
            k = int(k_values)

        fig, ax = fe.plot_outturn_revisions(
            data=data_filtered,
            variable=input.variable(),
            metric=input.transform(),
            frequency="Q",
            k=k,
            ma_window=int(input.ma_window()),
            start_date=input.start_vintage(),
            end_date=input.end_vintage(),
            fill_k=True,
            convert_to_percentage=False,
            return_plot=True,
        )
        return fig, ax

    @render.plot
    def outturn_revisions_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def outturn_revisions_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def outturn_revisions_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("outturn_revisions_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def outturn_revisions_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("outturn_revisions_legend", height=f"{input.legend_height()}px")

    @render.download(filename="outturn_revisions_data.csv")
    def download_outturn_revisions():
        data = get_data()
        outturns = fe.create_outturn_revisions(data=data)
        csv_bytes = outturns.to_csv()
        return io.BytesIO(csv_bytes.encode())

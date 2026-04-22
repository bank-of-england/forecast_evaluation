"""Server logic for Intra-period tab."""

import io

from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info
from forecast_evaluation.dashboard.utils import render_legend, remove_legend


def _get_confidence_level(input):
    """Parse confidence_level input: 'None' -> None, else int."""
    val = input.confidence_level()
    return None if val == "None" else int(val)


def intra_period_accuracy(input, output, session, data):
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
            start_date=input.start_date(),
            end_date=input.end_date(),
            start_vintage=input.start_vintage(),
            end_vintage=input.end_vintage(),
            sources=all_sources,
            variables=[input.variable()],
            metrics=[input.transform()],
        )

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)

        return data_filtered

    def get_plot(confidence_level=None):
        data_filtered = get_data()

        fig, ax = fe.plot_intra_period_accuracy(
            data=data_filtered,
            variable=input.variable(),
            metric=input.transform(),
            statistic=input.intra_statistic(),
            convert_to_percentage=True,
            confidence_level=confidence_level,
            return_plot=True,
        )

        return fig, ax

    @render.plot
    def intra_accuracy_plot():
        fig, ax = get_plot(confidence_level=_get_confidence_level(input))

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def intra_accuracy_legend():
        new_plot = get_plot(confidence_level=_get_confidence_level(input))
        fig_legend = render_legend(new_plot, input.show_legend())
        return fig_legend

    @render.ui()
    def intra_accuracy_plot_ui():
        return ui.output_plot("intra_accuracy_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def intra_accuracy_legend_ui():
        return ui.output_plot("intra_accuracy_legend", height=f"{input.legend_height()}px")

    @render.download(filename="intra_period_accuracy.csv")
    def download_intra_accuracy():
        data_filtered = get_data()
        csv_bytes = data_filtered._main_table.to_csv()
        return io.BytesIO(csv_bytes.encode())


def intra_period_bias(input, output, session, data):
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
            start_date=input.start_date(),
            end_date=input.end_date(),
            start_vintage=input.start_vintage(),
            end_vintage=input.end_vintage(),
            sources=all_sources,
            variables=[input.variable()],
            metrics=[input.transform()],
        )

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)

        return data_filtered

    def get_plot(confidence_level=None):
        data_filtered = get_data()

        fig, ax = fe.plot_intra_period_bias(
            data=data_filtered,
            variable=input.variable(),
            metric=input.transform(),
            convert_to_percentage=True,
            confidence_level=confidence_level,
            return_plot=True,
        )

        return fig, ax

    @render.plot
    def intra_bias_plot():
        fig, ax = get_plot(confidence_level=_get_confidence_level(input))

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def intra_bias_legend():
        new_plot = get_plot(confidence_level=_get_confidence_level(input))
        fig_legend = render_legend(new_plot, input.show_legend())
        return fig_legend

    @render.ui()
    def intra_bias_plot_ui():
        return ui.output_plot("intra_bias_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def intra_bias_legend_ui():
        return ui.output_plot("intra_bias_legend", height=f"{input.legend_height()}px")

    @render.download(filename="intra_period_bias.csv")
    def download_intra_bias():
        data_filtered = get_data()
        csv_bytes = data_filtered._main_table.to_csv()
        return io.BytesIO(csv_bytes.encode())

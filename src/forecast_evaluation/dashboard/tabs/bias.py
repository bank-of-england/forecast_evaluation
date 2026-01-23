"""Server logic for Bias tab."""

import io

from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info  # Adjust import path as needed
from forecast_evaluation.dashboard.utils import render_legend, remove_legend


def errors(input, output, session, data):
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

    def get_plot():
        data_filtered = get_data()

        fig, ax = fe.plot_errors_across_time(
            data=data_filtered,
            variable=input.variable(),
            metric=input.transform(),
            error="raw",
            horizons=[int(h) for h in input.horizons()],
            k=int(input.k()),
            show_mean=True,
            convert_to_percentage=True,
            return_plot=True,
        )

        return fig, ax

    @render.plot
    def errors_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def errors_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def errors_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("errors_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def errors_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("errors_legend", height=f"{input.legend_height()}px")

    @render.download(filename="errors.csv")
    def download_errors():
        data_filtered = get_data()
        csv_bytes = data_filtered._main_table.to_csv()
        return io.BytesIO(csv_bytes.encode())


def rolling_errors(input, output, session, data):
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

    def get_plot():
        data_filtered = get_data()

        fig, ax = fe.plot_errors_across_time(
            data=data_filtered,
            variable=input.variable(),
            metric=input.transform(),
            error="raw",
            horizons=[int(h) for h in input.horizons()],
            k=int(input.k()),
            show_mean=True,
            ma_window=int(input.window_size()),
            convert_to_percentage=True,
            return_plot=True,
        )

        return fig, ax

    @render.plot
    def rolling_errors_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def rolling_errors_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def rolling_errors_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("rolling_errors_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def rolling_errors_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("rolling_errors_legend", height=f"{input.legend_height()}px")

    @render.download(filename="rolling_errors.csv")
    def download_rolling_errors():
        data_filtered = get_data()
        csv_bytes = data_filtered._main_table.to_csv()
        return io.BytesIO(csv_bytes.encode())


def bias(input, output, session, data):
    # Across horizons
    @reactive.calc
    @reactive.event(input.update)
    def get_data():
        data_filtered = data.copy()

        all_sources = [input.source()]
        id_columns = [col for col in data.id_columns if col != "source"]
        for col in id_columns:
            _, id_single, _ = get_selector_info(col, data)
            all_sources += [input[id_single]()]

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

        return fe.bias_analysis(data=data_filtered, source=all_sources, k=int(input.k()), verbose=False)

    def get_plot():
        bias_results = get_data()

        unique_id = bias_results["unique_id"].unique()[0]

        fig, ax = bias_results.plot(
            variable=input.variable(),
            source=unique_id,
            metric=input.transform(),
            frequency="Q",
            return_plot=True,
        )

        return fig, ax

    @render.plot
    def bias_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def bias_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def bias_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("bias_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def bias_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("bias_legend", height=f"{input.legend_height()}px")

    @render.download(filename="bias_data.csv")
    def download_bias():
        bias_results = get_data()
        csv_bytes = bias_results.to_csv()
        return io.BytesIO(csv_bytes.encode())


def rolling_bias(input, output, session, data):
    # Rolling Window
    @reactive.calc
    @reactive.event(input.update)
    def get_data():
        data_filtered = data.copy()

        all_sources = [input.source()]
        id_columns = [col for col in data.id_columns if col != "source"]
        for col in id_columns:
            _, id_single, _ = get_selector_info(col, data)
            all_sources += [input[id_single]()]

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

        if input.fluctuation_test() == "Yes":
            results = fe.fluctuation_tests(
                data=data_filtered,
                window_size=int(input.window_size()),
                test_func=fe.bias_analysis,
                test_args={"k": int(input.k())},
            )
        else:
            results = fe.rolling_analysis(
                data=data_filtered,
                window_size=int(input.window_size()),
                analysis_func=fe.bias_analysis,
                analysis_args={"k": int(input.k())},
            )

        return results

    def get_plot():
        bias_results = get_data()
        fig, ax = bias_results.plot(
            horizons=[int(h) for h in input.horizons()],
            return_plot=True,
        )

        return fig, ax

    @render.plot
    def rolling_bias_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def rolling_bias_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def rolling_bias_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("rolling_bias_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def rolling_bias_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("rolling_bias_legend", height=f"{input.legend_height()}px")

    @render.download(filename="rolling_bias_data.csv")
    def download_rolling_bias():
        bias_results = get_data()
        csv_bytes = bias_results.to_csv()
        return io.BytesIO(csv_bytes.encode())

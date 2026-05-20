"""Register server handlers for the Radar tab."""

import io

from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info
from forecast_evaluation.dashboard.utils import render_legend, remove_legend


def radar(input, output, session, data):
    """Server handler for the Radar tab."""

    @reactive.calc
    @reactive.event(input.update)
    def get_filtered_data():
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
        )

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)

        return data_filtered

    @reactive.calc
    @reactive.event(input.update)
    def get_accuracy_data():
        return fe.compute_accuracy_statistics(data=get_filtered_data(), k=int(input.k()))

    def get_plot():
        mode = input.radar_mode()

        # tests mode and variables mode with bias/efficiency/correlation need ForecastData
        if mode == "tests":
            df_arg = get_filtered_data()
        elif mode == "variables":
            test_type = input.radar_test_type()
            if test_type in ("bias", "efficiency", "correlation"):
                df_arg = get_filtered_data()
            else:
                df_arg = get_filtered_data()
        else:
            df_arg = get_accuracy_data()

        kwargs = dict(
            mode=mode,
            statistic=input.stat(),
            normalise=input.radar_normalise(),
            return_plot=True,
        )

        if mode == "metrics":
            kwargs["variable"] = input.variable()
            kwargs["horizon"] = int(input.radar_horizon())
        elif mode == "variables":
            kwargs["metric"] = input.transform()
            kwargs["horizon"] = int(input.radar_horizon())
            kwargs["k"] = int(input.k())
            kwargs["test_type"] = input.radar_test_type()
            kwargs["variables"] = list(input.radar_variables())
            if input.radar_test_type() == "bias":
                kwargs["bias_type"] = input.radar_bias_type()
            elif input.radar_test_type() == "efficiency":
                kwargs["efficiency_type"] = input.radar_efficiency_type()
            elif input.radar_test_type() == "correlation":
                kwargs["anchor_source"] = input.radar_anchor()
        elif mode == "tests":
            kwargs["variable"] = input.variable()
            kwargs["metric"] = input.transform()
            kwargs["horizon"] = int(input.radar_horizon())
            kwargs["k"] = int(input.k())
            kwargs["bias_type"] = input.radar_bias_type()
            kwargs["efficiency_type"] = input.radar_efficiency_type()

        fig, ax = fe.plot_radar(df_arg, **kwargs)
        return fig, ax

    @render.plot
    def radar_plot():
        fig, ax = get_plot()
        if not input.show_legend():
            remove_legend(ax)
        return fig

    @render.plot
    def radar_legend():
        new_plot = get_plot()
        fig_legend = render_legend(new_plot, input.show_legend())
        return fig_legend

    @render.ui()
    def radar_plot_ui():
        return ui.output_plot("radar_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def radar_legend_ui():
        return ui.output_plot("radar_legend", height=f"{input.legend_height()}px")

    @render.download(filename="radar_data.csv")
    def download_radar():
        mode = input.radar_mode()
        if mode == "tests":
            # export the underlying main table slice
            import pandas as pd

            main = get_filtered_data()._main_table
            csv_bytes = main.to_csv(index=False)
            return io.BytesIO(csv_bytes.encode())
        csv_bytes = get_accuracy_data().to_csv()
        return io.BytesIO(csv_bytes.encode())

"""Server logic for Time Machine and Hedgehog tabs."""

import io

from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info  # Adjust import path as needed


def hedgehog(input, output, session, data):
    """Register all server handlers for Hedgehog tabs."""

    @reactive.calc
    @reactive.event(input.update)
    def get_data():
        all_sources = [input.source()]
        id_columns = [col for col in data.id_columns if col != "source"]
        for col in id_columns:
            _, id_single, _ = get_selector_info(col, data)
            all_sources += [input[id_single]()]

        data_filtered = data.copy()
        data_filtered.filter(
            sources=all_sources,
            start_date=input.start_date(),
            end_date=input.end_date(),
            start_vintage=input.start_vintage(),
            end_vintage=input.end_vintage(),
        )
        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)
        return data_filtered

    def get_plot():
        data_filtered = get_data()
        unique_id = data_filtered.forecasts["unique_id"].unique()[0]

        fig, ax = fe.plot_hedgehog(
            data=data_filtered,
            variable=input.variable(),
            forecast_source=unique_id,
            frequency="Q",
            k=int(input.k()),
            metric=input.transform(),
            return_plot=True,
        )

        return fig, ax

    @render.plot
    def hedgehog_plot():
        fig, ax = get_plot()

        return fig

    @render.ui()
    def hedgehog_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("hedgehog_plot", height=f"{input.plot_height()}px")

    @render.download(filename="hedgehog_data.csv")
    def download_hedgehog():
        data_filtered = get_data()
        csv_bytes = data_filtered.forecasts.to_csv()
        return io.BytesIO(csv_bytes.encode())

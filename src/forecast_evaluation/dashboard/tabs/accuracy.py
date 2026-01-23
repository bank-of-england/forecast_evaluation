"""Register all server handlers for the Accuracy tab."""

import io
from shiny import reactive, render, ui

import forecast_evaluation as fe
import pandas as pd
from forecast_evaluation.dashboard.ui import get_selector_info  # Adjust import path as needed
from forecast_evaluation.dashboard.utils import render_legend, remove_legend


def compute_accuracy_summary(df, start_date="start_date", end_date="end_date"):
    # Group by horizon and aggregate start, end, and observations
    summary = (
        df.groupby("forecast_horizon")
        .agg(start_date=(start_date, "min"), end_date=(end_date, "max"), n_obs=("n_observations", "first"))
        .reset_index()
    )

    # reformat dates
    summary["start_date"] = summary["start_date"].dt.strftime("%Y-%m")
    summary["end_date"] = summary["end_date"].dt.strftime("%Y-%m")

    # rename columns for better display
    summary.columns = ["Forecast Horizon", "Start Date", "End Date", "Observations"]

    return summary


def average_accuracy(input, output, session, data):
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

        return fe.compute_accuracy_statistics(data=data_filtered, k=int(input.k()))

    def get_plot():
        accuracy_results = get_data()
        fig, ax = accuracy_results.plot(
            variable=input.variable(),
            metric=input.transform(),
            frequency="Q",
            statistic=input.stat(),
            convert_to_percentage=True,
            return_plot=True,
        )

        return fig, ax

    @render.plot
    def accuracy_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def accuracy_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def accuracy_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("accuracy_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def accuracy_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("accuracy_legend", height=f"{input.legend_height()}px")

    @render.data_frame
    def info_accuracy():
        accuracy_results = get_data()
        summary = compute_accuracy_summary(accuracy_results)

        return render.DataTable(summary)

    @render.download(filename="average_accuracy_data.csv")
    def download_accuracy():
        accuracy_results = get_data()
        csv_bytes = accuracy_results.to_csv()
        return io.BytesIO(csv_bytes.encode())


def relative_accuracy(input, output, session, data):
    # Relative Accuracy
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

        return fe.compute_accuracy_statistics(data=data_filtered, k=int(input.k()))

    def get_plot():
        accuracy_results = get_data()
        fig, ax = accuracy_results.plot(
            variable=input.variable(),
            metric=input.transform(),
            frequency="Q",
            statistic=input.stat(),
            benchmark_model=input.benchmark(),
            return_plot=True,
        )
        return fig, ax

    @render.plot
    def relative_accuracy_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def relative_accuracy_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def relative_accuracy_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("relative_accuracy_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def relative_accuracy_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("relative_accuracy_legend", height=f"{input.legend_height()}px")

    @render.data_frame
    def info_relative_accuracy():
        accuracy_results = get_data()
        summary = compute_accuracy_summary(accuracy_results)

        return render.DataTable(summary)

    @render.download(filename="relative_accuracy_data.csv")
    def download_relative_accuracy():
        accuracy_results = get_data()
        csv_bytes = accuracy_results.to_csv()
        return io.BytesIO(csv_bytes.encode())


def rolling_accuracy(input, output, session, data):
    # Rolling Accuracy
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
            error=input.error(),
            horizons=[int(h) for h in input.horizons()],
            k=int(input.k()),
            ma_window=int(input.window_size()),
            show_mean=False,
            convert_to_percentage=True,
            return_plot=True,
        )
        return fig, ax

    @render.plot()
    def rolling_accuracy_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot()
    def rolling_accuracy_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def rolling_accuracy_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("rolling_accuracy_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def rolling_accuracy_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("rolling_accuracy_legend", height=f"{input.legend_height()}px")

    @render.download(filename="rolling_accuracy.csv")
    def download_rolling_accuracy():
        data_filtered = get_data()
        csv_bytes = data_filtered._main_table.to_csv()
        return io.BytesIO(csv_bytes.encode())


def rolling_relative_accuracy(input, output, session, data):
    # Rolling Relative Accuracy
    @reactive.calc
    @reactive.event(input.update)
    def get_data():
        all_sources = [input.source()] + [input.benchmark()]
        id_columns = [col for col in data.id_columns if col != "source"]
        for col in id_columns:
            _, id_single, _ = get_selector_info(col, data)
            all_sources += [input[id_single]()]

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

        if input.fluctuation_test() == "Yes":
            results = fe.fluctuation_tests(
                data=data_filtered,
                window_size=int(input.window_size()),
                test_func=fe.diebold_mariano_table,
                test_args={
                    "k": int(input.k()),
                    "benchmark_model": input.benchmark(),
                    "loss_function": input.loss_function(),
                },
            )
        else:
            results = fe.rolling_analysis(
                data=data_filtered,
                window_size=int(input.window_size()),
                analysis_func=fe.diebold_mariano_table,
                analysis_args={
                    "k": int(input.k()),
                    "benchmark_model": input.benchmark(),
                    "loss_function": input.loss_function(),
                },
            )

        return results

    def get_plot():
        rolling_rmse_result = get_data()

        fig, ax = rolling_rmse_result.plot(
            variable=input.variable(),
            horizons=[int(h) for h in input.horizons()],
            return_plot=True,
        )
        return fig, ax

    @render.plot
    def rolling_relative_accuracy_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def rolling_relative_accuracy_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def rolling_relative_accuracy_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("rolling_relative_accuracy_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def rolling_relative_accuracy_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("rolling_relative_accuracy_legend", height=f"{input.legend_height()}px")

    @render.data_frame
    def info_rolling_relative_accuracy():
        rolling_rmse_result = get_data()

        data = {
            "Window": ["First window", "Last window"],
            "Start date": [rolling_rmse_result["window_start"].min(), rolling_rmse_result["window_start"].max()],
            "End date": [rolling_rmse_result["window_end"].min(), rolling_rmse_result["window_end"].max()],
        }

        summary = pd.DataFrame(data)

        # reformat dates
        summary["Start date"] = summary["Start date"].dt.strftime("%Y-%m")
        summary["End date"] = summary["End date"].dt.strftime("%Y-%m")

        return render.DataTable(summary)

    @render.download(filename="rolling_relative_accuracy_data.csv")
    def download_rolling_relative_accuracy():
        rolling_rmse_result = get_data()
        csv_bytes = rolling_rmse_result.to_csv()
        return io.BytesIO(csv_bytes.encode())


def diebold_mariano(input, output, session, data):
    # Diebold Mariano
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

        return fe.diebold_mariano_table(
            data=data_filtered, benchmark_model=input.benchmark(), loss_function=input.loss_function(), k=int(input.k())
        )

    @render.data_frame
    def DM_test():
        dm_result = get_data()
        return render.DataGrid(dm_result.to_df(), filters=True, width="100%")

    @render.download(filename="diebold_mariano_data.csv")
    def download_DM():
        dm_result = get_data()
        csv_bytes = dm_result.to_csv()
        return io.BytesIO(csv_bytes.encode())


def error_distribution(input, output, session, data):
    # Error Distribution
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
            start_date=input.start_date(),
            end_date=input.end_date(),
            start_vintage=input.start_vintage(),
            end_vintage=input.end_vintage(),
            variables=[input.variable()],
            metrics=[input.transform()],
            sources=all_sources,
        )

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)
        return data_filtered

    def get_plot():
        data_filtered = get_data()
        unique_id = data_filtered.forecasts["unique_id"].unique()[0]

        fig, ax = fe.plot_forecast_error_density(
            data=data_filtered,
            horizon=int(input.horizon()),
            variable=input.variable(),
            metric=input.transform(),
            frequency="Q",
            source=unique_id,
            k=int(input.k()),
            highlight_dates=input.dates_to_highlight(),
            return_plot=True,
        )
        return fig, ax

    @render.plot
    def error_density_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def error_density_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def error_density_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("error_density_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def error_density_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("error_density_legend", height=f"{input.legend_height()}px")

    @render.data_frame
    def info_error_density():
        data_filtered = get_data()

        main_table = data_filtered._main_table
        main_table = main_table[main_table["k"] == int(input.k())]
        main_table = main_table[main_table["forecast_horizon"] == int(input.horizon())]

        data = {
            "First observation": [main_table["date"].min()],
            "Last observation": [main_table["date"].max()],
            "Sample size": main_table.shape[0],
        }

        summary = pd.DataFrame(data)

        # reformat dates
        summary["First observation"] = summary["First observation"].dt.strftime("%Y-%m")
        summary["Last observation"] = summary["Last observation"].dt.strftime("%Y-%m")

        return render.DataTable(summary)

    @render.download(filename="error_density.csv")
    def download_error_density():
        data_filtered = get_data()
        csv_bytes = data_filtered.forecasts.to_csv()
        return io.BytesIO(csv_bytes.encode())

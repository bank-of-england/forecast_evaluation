"""Register all server handlers for the Correlation tab."""

import io

import pandas as pd
from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info
from forecast_evaluation.dashboard.utils import remove_legend, render_legend


def correlation_heatmap(input, output, session, data):
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

        return fe.forecast_errors_correlation_analysis(data=data_filtered, k=int(input.k()))

    def get_plot():
        corr_result = get_data()
        fig, ax = corr_result.plot(
            variable=input.variable(),
            metric=input.transform(),
            horizon=int(input.horizon()),
            return_plot=True,
        )
        return fig, ax

    @render.plot
    def correlation_heatmap_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def correlation_heatmap_legend():
        new_plot = get_plot()
        fig_legend = render_legend(new_plot, input.show_legend())
        return fig_legend

    @render.ui()
    def correlation_heatmap_plot_ui():
        return ui.output_plot("correlation_heatmap_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def correlation_heatmap_legend_ui():
        return ui.output_plot("correlation_heatmap_legend", height=f"{input.legend_height()}px")

    @render.data_frame
    def info_correlation_heatmap():
        corr_result = get_data()
        df = corr_result.to_df()
        if df.empty:
            return render.DataTable(pd.DataFrame({"info": ["No correlation data available."]}))

        summary = (
            df.groupby("forecast_horizon")
            .agg(start_date=("start_date", "min"), end_date=("end_date", "max"), n_obs=("n_observations", "min"))
            .reset_index()
        )
        summary["start_date"] = summary["start_date"].dt.strftime("%Y-%m")
        summary["end_date"] = summary["end_date"].dt.strftime("%Y-%m")
        summary.columns = ["Forecast Horizon", "Start Date", "End Date", "Min Observations"]
        return render.DataTable(summary)

    @render.download(filename="correlation_heatmap_data.csv")
    def download_correlation_heatmap():
        corr_result = get_data()
        csv_bytes = corr_result.to_csv()
        return io.BytesIO(csv_bytes.encode())


def rolling_correlation(input, output, session, data):
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

        return fe.rolling_analysis(
            data=data_filtered,
            window_size=int(input.window_size()),
            analysis_func=fe.forecast_errors_correlation_analysis,
            # Per-pair overlap inside each window: avoids dropping a whole window
            # when a single source has no data there (which produces gaps in the
            # rolling line). min_observations=2 (the floor for Pearson) so small
            # windows still yield a value; the analysis default of 5 would zero
            # out everything when window_size <= 4.
            analysis_args={"k": int(input.k()), "same_date_range": False, "min_observations": 2},
        )

    def get_plot():
        rolling_result = get_data()
        fig, axes = rolling_result.plot(
            variable=input.variable(),
            anchor_source=input.corr_anchor(),
            horizons=[int(h) for h in input.horizons()],
            metric=input.transform(),
            return_plot=True,
        )
        return fig, axes

    @render.plot
    def rolling_correlation_plot():
        fig, axes = get_plot()

        if not input.show_legend():
            remove_legend(axes)

        return fig

    @render.plot
    def rolling_correlation_legend():
        new_plot = get_plot()
        fig_legend = render_legend(new_plot, input.show_legend())
        return fig_legend

    @render.ui()
    def rolling_correlation_plot_ui():
        return ui.output_plot("rolling_correlation_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def rolling_correlation_legend_ui():
        return ui.output_plot("rolling_correlation_legend", height=f"{input.legend_height()}px")

    @render.data_frame
    def info_rolling_correlation():
        rolling_result = get_data()
        df = rolling_result.to_df()
        if df.empty:
            return render.DataTable(pd.DataFrame({"info": ["No rolling correlation data available."]}))

        summary = pd.DataFrame(
            {
                "Window": ["First window", "Last window"],
                "Start date": [df["window_start"].min(), df["window_start"].max()],
                "End date": [df["window_end"].min(), df["window_end"].max()],
            }
        )
        summary["Start date"] = summary["Start date"].dt.strftime("%Y-%m")
        summary["End date"] = summary["End date"].dt.strftime("%Y-%m")
        return render.DataTable(summary)

    @render.download(filename="rolling_correlation_data.csv")
    def download_rolling_correlation():
        rolling_result = get_data()
        csv_bytes = rolling_result.to_csv()
        return io.BytesIO(csv_bytes.encode())

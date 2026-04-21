"""Server logic for Intra-period tab."""

import io

import numpy as np
import pandas as pd
from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info
from forecast_evaluation.dashboard.utils import render_legend, remove_legend
from forecast_evaluation.visualisations.theme import create_themed_figure


def _compute_by_days_to_publication(df, variable, metric, statistic="rmse"):
    """Compute accuracy or bias grouped by days to publication.

    ``days_to_publication`` is computed as ``date - vintage_date_forecast``
    (days from the forecast vintage to the end of the target period).
    All horizons are included so the full range is visible on one axis.
    """
    mask = (df["variable"] == variable) & (df["metric"] == metric)
    sub = df.loc[mask].copy()

    if sub.empty:
        return pd.DataFrame(columns=["source", "days_to_publication", "value"])

    sub["days_to_publication"] = (pd.to_datetime(sub["date"]) - pd.to_datetime(sub["vintage_date_forecast"])).dt.days

    if statistic == "rmse":
        agg = sub.groupby(["source", "days_to_publication"])["forecast_error"].apply(lambda x: np.sqrt(np.mean(x**2)))
    elif statistic == "mae":
        agg = sub.groupby(["source", "days_to_publication"])["forecast_error"].apply(lambda x: np.mean(np.abs(x)))
    elif statistic == "mean_error":
        agg = sub.groupby(["source", "days_to_publication"])["forecast_error"].mean()
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    result = agg.reset_index()
    result.columns = ["source", "days_to_publication", "value"]
    return result.sort_values(["source", "days_to_publication"], ascending=[True, False]).reset_index(drop=True)


def _add_quarter_boundaries(ax, days_min, days_max):
    """Add dashed vertical lines at quarter boundaries (~91-day intervals)."""
    quarter_days = 91
    # Start from the first boundary that falls within the data range
    boundary = quarter_days * (int(days_min) // quarter_days)
    while boundary <= days_max:
        if days_min <= boundary <= days_max:
            ax.axvline(x=boundary, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        boundary += quarter_days


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

    def get_plot():
        data_filtered = get_data()
        statistic = input.intra_statistic()
        result = _compute_by_days_to_publication(
            data_filtered.df,
            input.variable(),
            input.transform(),
            statistic=statistic,
        )

        stat_labels = {"rmse": "RMSE", "mae": "MAE"}
        stat_label = stat_labels.get(statistic, statistic.upper())

        fig, ax = create_themed_figure()

        for source in sorted(result["source"].unique()):
            source_data = result[result["source"] == source]
            ax.plot(
                source_data["days_to_publication"],
                100 * source_data["value"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=source,
            )

        if not result.empty:
            _add_quarter_boundaries(ax, result["days_to_publication"].min(), result["days_to_publication"].max())

        ax.set_title(
            f"{stat_label} by Days to Publication\n{input.variable().upper()} - {input.transform()}",
            fontsize=14,
        )
        ax.set_xlabel("Days to Publication", fontsize=12)
        ax.set_ylabel(stat_label, fontsize=12)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(title="Source", loc="best")

        return fig, ax

    @render.plot
    def intra_accuracy_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def intra_accuracy_legend():
        new_plot = get_plot()
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

    def get_plot():
        data_filtered = get_data()
        result = _compute_by_days_to_publication(
            data_filtered.df,
            input.variable(),
            input.transform(),
            statistic="mean_error",
        )

        fig, ax = create_themed_figure()

        for source in sorted(result["source"].unique()):
            source_data = result[result["source"] == source]
            ax.plot(
                source_data["days_to_publication"],
                100 * source_data["value"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=source,
            )

        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        if not result.empty:
            _add_quarter_boundaries(ax, result["days_to_publication"].min(), result["days_to_publication"].max())

        ax.set_title(
            f"Bias by Days to Publication\n{input.variable().upper()} - {input.transform()}",
            fontsize=14,
        )
        ax.set_xlabel("Days to Publication", fontsize=12)
        ax.set_ylabel("Mean Error", fontsize=12)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(title="Source", loc="best")

        return fig, ax

    @render.plot
    def intra_bias_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def intra_bias_legend():
        new_plot = get_plot()
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

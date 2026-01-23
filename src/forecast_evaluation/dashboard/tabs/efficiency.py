"""Server logic for Efficiency tab."""

import io

import pandas as pd
from shiny import reactive, render, ui

import forecast_evaluation as fe
from forecast_evaluation.dashboard.ui import get_selector_info  # Adjust import path as needed
from forecast_evaluation.dashboard.utils import render_legend, remove_legend


def blanchard_leigh(input, output, session, data):
    """Register all server handlers for the Efficiency tab."""

    # Blanchard Leigh
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
            sources=all_sources,
        )

        unique_id = data_filtered.forecasts["unique_id"].unique()[0]

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)

        if input.correct_bias() == "Yes":
            return fe.blanchard_leigh_horizon_analysis(
                data=data_filtered,
                source=unique_id,
                outcome_variable=input.outcome_var(),
                outcome_metric=input.outcome_metric(),
                instrument_variable=input.instrument_var(),
                instrument_metric=input.instrument_metric(),
            )
        else:
            return fe.strong_efficiency_analysis(
                data=data_filtered,
                source=unique_id,
                outcome_variable=input.outcome_var(),
                outcome_metric=input.outcome_metric(),
                instrument_variable=input.instrument_var(),
                instrument_metric=input.instrument_metric(),
            )

    def get_plot():
        bl_results = get_data()
        fig, ax = bl_results.plot(return_plot=True)

        return fig, ax

    @render.plot
    def BL_plot():
        fig, ax = get_plot()

        if not input.show_legend():
            remove_legend(ax)

        return fig

    @render.plot
    def BL_legend():
        """Render legend separately from the main plot."""
        new_plot = get_plot()

        fig_legend = render_legend(new_plot, input.show_legend())

        return fig_legend

    @render.ui()
    def BL_plot_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("BL_plot", height=f"{input.plot_height()}px")

    @render.ui()
    def BL_legend_ui():
        # Return the UI container with dynamic height
        return ui.output_plot("BL_legend", height=f"{input.legend_height()}px")

    @render.download(filename="BL_data.csv")
    def download_bl_btn():
        bl_results = get_data()
        csv_bytes = bl_results.to_csv()
        return io.BytesIO(csv_bytes.encode())


def revisions_predictability(input, output, session, data):
    # Revisions predictability
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
            sources=all_sources,
        )

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)
        return fe.revision_predictability_analysis(data=data_filtered)

    @render.data_frame
    def revisions_reg():
        revision_pred = get_data()
        return render.DataGrid(revision_pred.to_df(), filters=True)

    @render.download(filename="revision_pred.csv")
    def download_revision_pred():
        revision_pred = get_data()
        csv_bytes = revision_pred.to_csv()
        return io.BytesIO(csv_bytes.encode())


def weak_efficiency(input, output, session, data):
    """Register server handlers for the Weak Efficiency (Optimal Scaling) subtab."""

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
            variables=[input.variable()],
            metrics=[input.transform()],
            sources=all_sources,
        )

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)
        results = fe.weak_efficiency_analysis(data=data_filtered, k=int(input.k()))
        # Remove ols_model column if it exists
        df = results.to_df()
        if "ols_model" in df.columns:
            df = df.drop(columns=["ols_model"])
            results._df = df
        return results

    @render.data_frame
    def weak_efficiency_table():
        weak_eff_results = get_data()
        return render.DataGrid(weak_eff_results.to_df(), filters=True)

    @render.download(filename="weak_efficiency.csv")
    def download_weak_efficiency():
        weak_eff_results = get_data()
        csv_bytes = weak_eff_results.to_csv()
        return io.BytesIO(csv_bytes.encode())


def revisions_errors_correlation(input, output, session, data):
    """Register server handlers for the Revisions Errors Correlation subtab."""

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
            variables=[input.variable()],
            metrics=[input.transform()],
            sources=all_sources,
        )

        if input.covid_filter() == "Yes":
            data_filtered.filter(custom_filter=fe.covid_filter)
        return fe.revisions_errors_correlation_analysis(data=data_filtered, k=int(input.k()))

    @render.data_frame
    def revisions_errors_table():
        rev_err_results = get_data()
        return render.DataGrid(rev_err_results.to_df(), filters=True)

    @render.download(filename="revisions_errors_correlation.csv")
    def download_revisions_errors():
        rev_err_results = get_data()
        csv_bytes = rev_err_results.to_csv()
        return io.BytesIO(csv_bytes.encode())

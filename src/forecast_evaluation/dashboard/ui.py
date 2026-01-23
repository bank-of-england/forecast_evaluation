"""UI components for the dashboard."""

from shiny import ui
from forecast_evaluation import DensityForecastData


def get_selector_info(col, data):
    """Generate selector info for a given column."""

    if isinstance(data, DensityForecastData):
        col_choices = sorted(data._density_forecasts[col].dropna().unique().tolist())
    else:
        col_choices = sorted(data.forecasts[col].dropna().unique().tolist())

    id_single = col.replace(" ", "_")
    id_multi = id_single + "_s"
    return col_choices, id_single, id_multi


def create_sidebar(data):
    """Create the sidebar with all conditional inputs"""

    # Define the range of options for the dynamic parameters
    # Map frequency codes to period names
    frequency_labels = {"Q": "quarters", "M": "months"}

    if hasattr(data, "_forecasts") and not data.forecasts.empty:
        vintages_set = set([str(v)[:10] for v in data.forecasts["vintage_date"].unique().tolist()])
        outturn_dates_set = set([str(v)[:10] for v in data.forecasts["date"].unique().tolist()])
        variable_set = set(data.forecasts["variable"].unique().tolist())
        sources_set = set(data.forecasts["source"].unique().tolist())
        unique_ids_set = set(data.forecasts["unique_id"].unique().tolist())
        transformations_set = set(data.forecasts["metric"].unique().tolist())
        # Get frequency label from data
        freq_code = data.forecasts["frequency"].iloc[0] if not data.forecasts["frequency"].empty else "Q"
        period_label = frequency_labels.get(freq_code, "periods")
    else:
        vintages_set = set()
        outturn_dates_set = set()
        variable_set = set()
        sources_set = set()
        unique_ids_set = set()
        transformations_set = set()
        period_label = "periods"

    if hasattr(data, "_density_forecasts") and not data._density_forecasts.empty:
        vintages_set.update([str(v)[:10] for v in data._density_forecasts["vintage_date"].unique().tolist()])
        outturn_dates_set.update([str(v)[:10] for v in data._density_forecasts["date"].unique().tolist()])
        variable_set.update(data._density_forecasts["variable"].unique().tolist())
        sources_set.update(data._density_forecasts["source"].unique().tolist())
        unique_ids_set.update(data._density_forecasts["unique_id"].unique().tolist())
        transformations_set.update(data._density_forecasts["metric"].unique().tolist())

    vintages = sorted(list(vintages_set))
    outturn_dates = sorted(list(outturn_dates_set))
    variable = sorted(list(variable_set))
    sources = sorted(list(sources_set))
    unique_ids = sorted(list(unique_ids_set))
    transformations = sorted(list(transformations_set))

    horizons = list(range(0, 13))
    loss_functions = ["rmse", "rmedse", "mean_abs_error"]
    loss_functions_tests = ["mse", "mae"]
    k_values = list(range(len(vintages) + 1))

    if hasattr(data, "_density_forecasts") and not data._density_forecasts.empty:
        quantiles = data._density_forecasts["quantile"].unique()
        closest_to_16 = min(quantiles, key=lambda x: abs(x - 0.16))
        closest_to_84 = min(quantiles, key=lambda x: abs(x - 0.84))

        # rounding
        closest_to_16 = round(float(closest_to_16), 2)
        closest_to_84 = round(float(closest_to_84), 2)
        quantiles = [round(float(q), 2) for q in quantiles]
        quantiles = sorted(list(set(quantiles)))
    else:
        quantiles = [0.0]
        closest_to_84 = 0.0
        closest_to_16 = 0.0

    _or = "||"
    _and = "&&"

    # efficiency tabs
    bl_tab = "input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Blanchard-Leigh'"
    revisions_tab = "input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Revisions predictability'"
    weak_efficiency_tab = "input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Optimal scaling'"
    revisions_errors_tab = (
        "input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Correlation of revisions and errors'"
    )

    # bias
    errors_tab = "input.tabs == 'Bias' && input.bias_subtabs == 'Errors'"
    rolling_errors_tab = "input.tabs == 'Bias' && input.bias_subtabs == 'Rolling errors'"
    bias_tab = "input.tabs == 'Bias' && input.bias_subtabs == 'Bias across horizons'"
    rolling_bias_tab = "input.tabs == 'Bias' && input.bias_subtabs == 'Rolling bias'"

    time_machine_tab = "input.tabs == 'Time Machine'"
    hedgehog_tab = "input.tabs == 'Hedgehog'"
    outturn_revisions_tab = "input.tabs == 'Outturn Revisions'"
    outturn_revisions_subtab = (
        "input.tabs == 'Outturn Revisions' && input.outturn_revisions_subtabs == 'Outturn Revisions'"
    )
    outturns_subtab = "input.tabs == 'Outturn Revisions' && input.outturn_revisions_subtabs == 'Outturns'"

    if hasattr(data, "_density_forecasts") and data._density_forecasts is not None:
        quantile_time_machine_tab = "input.tabs == 'Quantile Forecasts'"
    else:
        quantile_time_machine_tab = "false"

    # Accuracy tabs
    average_accuracy_tab = "input.tabs == 'Accuracy' && input.accuracy_subtabs == 'Average Accuracy'"
    relative_accuracy_tab = "input.tabs == 'Accuracy' && input.accuracy_subtabs == 'Relative Accuracy'"
    rolling_accuracy_tab = "input.tabs == 'Accuracy' && input.accuracy_subtabs == 'Rolling Accuracy'"
    rolling_relative_accuracy_tab = "input.tabs == 'Accuracy' && input.accuracy_subtabs == 'Rolling Relative Accuracy'"
    dm_tab = "input.tabs == 'Accuracy' && input.accuracy_subtabs == 'Diebold Mariano'"
    error_dist_tab = "input.tabs == 'Accuracy' && input.accuracy_subtabs == 'Error Distribution'"

    # Prepare potential additional selectors for labelling/id variables
    id_columns = [col for col in data.id_columns if col != "source"]  # source is always a selector
    additional_selectors = []

    if len(id_columns) > 0:
        additional_selectors = []
        for col in id_columns:
            col_choices, id_single, id_multi = get_selector_info(col, data)

            # Create the multiple input selector
            additional_selectors.append(
                ui.panel_conditional(
                    average_accuracy_tab
                    + _or
                    + relative_accuracy_tab
                    + _or
                    + rolling_accuracy_tab
                    + _or
                    + dm_tab
                    + _or
                    + time_machine_tab
                    + _or
                    + quantile_time_machine_tab
                    + _or
                    + revisions_tab
                    + _or
                    + weak_efficiency_tab
                    + _or
                    + revisions_errors_tab
                    + _or
                    + errors_tab
                    + _or
                    + rolling_errors_tab,
                    ui.input_selectize(id_multi, f"{col}s:", choices=col_choices, multiple=True, selected=col_choices),
                )
            )

            # Create the single input selector
            additional_selectors.append(
                ui.panel_conditional(
                    rolling_relative_accuracy_tab
                    + _or
                    + error_dist_tab
                    + _or
                    + bl_tab
                    + _or
                    + bias_tab
                    + _or
                    + rolling_bias_tab
                    + _or
                    + hedgehog_tab,
                    ui.input_selectize(id_single, f"{col}:", choices=col_choices, multiple=False),
                )
            )

    return ui.sidebar(
        # Update button
        ui.panel_conditional(
            "input.tabs != 'About'",
            ui.input_action_button("update", "Update", class_="btn-primary"),
        ),
        # Auto-click the button on load
        ui.tags.script("""
        $(document).on('shiny:connected', function() {
            $('#update').click();
        });
        """),
        # sources accordion
        ui.panel_conditional(
            "input.tabs != 'About' && input.tabs != 'Outturn Revisions'",
            ui.accordion(
                ui.accordion_panel(
                    "Source filters",
                    # Sources
                    ui.panel_conditional(
                        average_accuracy_tab
                        + _or
                        + relative_accuracy_tab
                        + _or
                        + rolling_accuracy_tab
                        + _or
                        + dm_tab
                        + _or
                        + time_machine_tab
                        + _or
                        + quantile_time_machine_tab
                        + _or
                        + revisions_tab
                        + _or
                        + weak_efficiency_tab
                        + _or
                        + revisions_errors_tab
                        + _or
                        + errors_tab
                        + _or
                        + rolling_errors_tab,
                        ui.input_selectize("sources", "Sources:", choices=sources, multiple=True, selected=sources),
                    ),
                    ui.panel_conditional(
                        rolling_relative_accuracy_tab
                        + _or
                        + error_dist_tab
                        + _or
                        + bl_tab
                        + _or
                        + bias_tab
                        + _or
                        + rolling_bias_tab
                        + _or
                        + hedgehog_tab,
                        ui.input_selectize("source", "Source:", choices=sources, multiple=False, selected="mpr"),
                    ),
                    # Dynamic additional selectors for labelling/id variables
                    *additional_selectors,
                ),
                open=False,  # Set to False if you want it collapsed by default
            ),
        ),
        # Grouped Filters accordion
        ui.panel_conditional(
            "input.tabs != 'About'",
            ui.accordion(
                ui.accordion_panel(
                    "Date filters",
                    # Date filters
                    ui.panel_conditional(
                        "input.tabs != 'Outturn Revisions'",
                        ui.input_select(
                            "start_date", "Data start date:", choices=outturn_dates, selected=outturn_dates[0]
                        ),
                    ),
                    ui.panel_conditional(
                        "input.tabs != 'Time Machine' && input.tabs != 'Outturn Revisions'",
                        ui.input_select(
                            "end_date", "Data end date:", choices=outturn_dates, selected=outturn_dates[-1]
                        ),
                    ),
                    # Vintage filters
                    ui.panel_conditional(
                        "input.tabs != 'Time Machine'",
                        ui.input_select(
                            "start_vintage", "First forecast vintage:", choices=vintages[:-1], selected=vintages[0]
                        ),
                    ),
                    ui.panel_conditional(
                        "input.tabs != 'Time Machine'",
                        ui.input_select(
                            "end_vintage", "Last forecast vintage:", choices=vintages, selected=vintages[-1]
                        ),
                    ),
                    # Covid filter
                    ui.panel_conditional(
                        "input.tabs != 'Outturn Revisions'",
                        ui.input_select("covid_filter", "Use COVID filter:", choices=["Yes", "No"], selected="No"),
                    ),
                ),
                open=False,  # Set to False if you want it collapsed by default
            ),
        ),
        # Variables
        ui.panel_conditional(
            "input.tabs != 'About' && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Blanchard-Leigh')",
            ui.input_selectize("variable", "Variable:", choices=variable, multiple=False, selected=["cpisa"]),
        ),
        # Error for rolling accuracy
        ui.panel_conditional(
            rolling_accuracy_tab,
            ui.input_select("error", "Error:", choices=["absolute", "squared"], selected="absolute"),
        ),
        # Statistic selector
        ui.panel_conditional(
            average_accuracy_tab + _or + relative_accuracy_tab,
            ui.input_select("stat", "Statistic:", choices=loss_functions, selected="RMSE"),
        ),
        # Test loss function selector
        ui.panel_conditional(
            rolling_relative_accuracy_tab + _or + dm_tab,
            ui.input_select("loss_function", "Loss function:", choices=loss_functions_tests, selected="MSE"),
        ),
        # Outturn taken at t + (single selection)
        ui.panel_conditional(
            "input.tabs != 'About' && input.tabs != 'Time Machine' && input.tabs != 'Outturn Revisions' && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Blanchard-Leigh') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Revisions predictability')",
            ui.input_select("k", f"Data vintage ({period_label} after first release)", choices=k_values, selected=12),
        ),
        # Outturn taken at t + (multiple selection for outturn revisions)
        ui.panel_conditional(
            outturn_revisions_subtab,
            ui.input_selectize(
                "k_multiple",
                f"Data vintage ({period_label} after first release)",
                choices=k_values[1:],
                multiple=True,
                selected=[12],
            ),
        ),
        # Outturn taken at t + (multiple selection for outturns)
        ui.panel_conditional(
            outturns_subtab,
            ui.input_selectize(
                "k_multiple_outturns",
                f"Data vintage ({period_label} after first release)",
                choices=k_values,
                multiple=True,
                selected=[12],
            ),
        ),
        # Vintage selector
        ui.panel_conditional(
            time_machine_tab + _or + quantile_time_machine_tab,
            ui.input_select("vintage", "Vintage:", choices=vintages, selected=vintages[-1]),
        ),
        # Transformation
        ui.panel_conditional(
            "!(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Blanchard-Leigh') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Revisions predictability') && input.tabs != 'About'",
            ui.input_selectize(
                "transform", "Transformation:", choices=transformations, multiple=False, selected=["yoy"]
            ),
        ),
        # Window size for rolling tabs
        ui.panel_conditional(
            rolling_bias_tab
            + _or
            + rolling_relative_accuracy_tab
            + _or
            + rolling_accuracy_tab
            + _or
            + rolling_errors_tab,
            ui.input_selectize(
                "window_size", "Window size:", choices=list(range(1, len(vintages))), multiple=False, selected=20
            ),
        ),
        # Moving average window for outturn revisions
        ui.panel_conditional(
            outturn_revisions_subtab,
            ui.input_selectize(
                "ma_window", "Moving average window:", choices=list(range(1, 21)), multiple=False, selected=4
            ),
        ),
        ui.panel_conditional(
            rolling_bias_tab
            + _or
            + dm_tab
            + _or
            + rolling_relative_accuracy_tab
            + _or
            + rolling_accuracy_tab
            + _or
            + rolling_errors_tab
            + _or
            + errors_tab,
            ui.input_selectize("horizons", "Horizons:", choices=horizons, multiple=True, selected=[0, 1, 4]),
        ),
        ui.panel_conditional(
            error_dist_tab,
            ui.input_select("horizon", "Horizon:", choices=horizons, selected=0),
        ),
        # Benchmark
        ui.panel_conditional(
            dm_tab + _or + relative_accuracy_tab + _or + rolling_relative_accuracy_tab,
            ui.input_select("benchmark", "Benchmark:", choices=unique_ids, selected=unique_ids[-1]),
        ),
        # Highlighted dates
        ui.panel_conditional(
            error_dist_tab,
            ui.input_selectize(
                "dates_to_highlight",
                "Errors to highlight:",
                choices=outturn_dates,
                multiple=True,
                selected=outturn_dates[-1],
            ),
        ),
        # BL specific inputs
        ui.panel_conditional(
            bl_tab,
            ui.input_selectize(
                "outcome_var", "Outcome variable:", choices=variable, multiple=False, selected=variable[0]
            ),
        ),
        ui.panel_conditional(
            bl_tab,
            ui.input_selectize(
                "instrument_var", "Instrument variable:", choices=variable, multiple=False, selected=variable[-1]
            ),
        ),
        ui.panel_conditional(
            bl_tab,
            ui.input_selectize(
                "outcome_metric", "Outcome Metric:", choices=transformations, multiple=False, selected=["yoy"]
            ),
        ),
        ui.panel_conditional(
            bl_tab,
            ui.input_selectize(
                "instrument_metric", "Instrument Metric:", choices=transformations, multiple=False, selected=["yoy"]
            ),
        ),
        ui.panel_conditional(
            bl_tab,
            ui.input_select("correct_bias", "Bias Correction:", choices=["Yes", "No"], selected="No"),
        ),
        ui.panel_conditional(
            rolling_relative_accuracy_tab + _or + rolling_bias_tab,
            ui.input_select("fluctuation_test", "Run fluctuation test:", choices=["Yes", "No"], selected="No"),
        ),
        ui.panel_conditional(
            quantile_time_machine_tab,
            ui.input_select("lower_quantile", "Lower Quantile:", choices=quantiles, selected=closest_to_16),
        ),
        ui.panel_conditional(
            quantile_time_machine_tab,
            ui.input_select("upper_quantile", "Upper Quantile:", choices=quantiles, selected=closest_to_84),
        ),
        # Legend
        ui.panel_conditional(
            "input.tabs != 'About' && !(input.tabs == 'Accuracy' && input.accuracy_subtabs == 'Diebold Mariano') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Revisions predictability') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Optimal scaling') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Correlation of revisions and errors')",
            ui.input_checkbox("show_legend", "Legend in plot", value=False),
        ),
        # Plot height control
        ui.panel_conditional(
            "input.tabs != 'About' && !(input.tabs == 'Accuracy' && input.accuracy_subtabs == 'Diebold Mariano') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Revisions predictability') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Optimal scaling') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Correlation of revisions and errors')",
            ui.input_slider("plot_height", "Plot height (px):", min=300, max=2000, value=500, step=50),
        ),
        # Legend height control
        ui.panel_conditional(
            "input.tabs != 'About' && !(input.tabs == 'Accuracy' && input.accuracy_subtabs == 'Diebold Mariano') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Revisions predictability') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Optimal scaling') && !(input.tabs == 'Efficiency' && input.efficiency_subtabs == 'Correlation of revisions and errors')",
            ui.input_slider("legend_height", "Legend height (px):", min=50, max=1000, value=200, step=50),
        ),
    )


def create_accuracy_tab():
    """Create the Accuracy tab UI"""
    return ui.nav_panel(
        "Accuracy",
        ui.navset_tab(
            ui.nav_panel(
                "Average Accuracy",
                ui.output_ui("accuracy_plot_ui"),
                ui.output_ui("accuracy_legend_ui"),
                ui.accordion(
                    ui.accordion_panel(
                        "Sample information",
                        ui.output_data_frame("info_accuracy"),
                    ),
                    open=False,
                ),
                ui.download_button("download_accuracy", "Download data behind chart"),
            ),
            ui.nav_panel(
                "Rolling Accuracy",
                ui.card(
                    ui.output_ui("rolling_accuracy_plot_ui"),
                    ui.output_ui("rolling_accuracy_legend_ui"),
                ),
                ui.download_button("download_rolling_accuracy", "Download data behind chart"),
            ),
            ui.nav_panel(
                "Relative Accuracy",
                ui.output_ui("relative_accuracy_plot_ui"),
                ui.output_ui("relative_accuracy_legend_ui"),
                ui.accordion(
                    ui.accordion_panel(
                        "Sample information",
                        ui.output_data_frame("info_relative_accuracy"),
                    ),
                    open=False,
                ),
                ui.download_button("download_relative_accuracy", "Download data behind chart"),
            ),
            ui.nav_panel(
                "Rolling Relative Accuracy",
                ui.output_ui("rolling_relative_accuracy_plot_ui"),
                ui.output_ui("rolling_relative_accuracy_legend_ui"),
                ui.accordion(
                    ui.accordion_panel(
                        "Sample information",
                        ui.output_data_frame("info_rolling_relative_accuracy"),
                    ),
                    open=False,
                ),
                ui.download_button("download_rolling_relative_accuracy", "Download data behind chart"),
            ),
            ui.nav_panel(
                "Diebold Mariano",
                ui.output_data_frame("DM_test"),
                ui.download_button("download_DM", "Download data behind table"),
            ),
            ui.nav_panel(
                "Error Distribution",
                ui.output_ui("error_density_plot_ui"),
                ui.output_ui("error_density_legend_ui"),
                ui.accordion(
                    ui.accordion_panel(
                        "Sample information",
                        ui.output_data_frame("info_error_density"),
                    ),
                    open=False,
                ),
                ui.download_button("download_error_density", "Download data behind chart"),
            ),
            id="accuracy_subtabs",
        ),
    )


def create_bias_tab():
    """Create the Bias tab UI"""
    return ui.nav_panel(
        "Bias",
        ui.navset_tab(
            ui.nav_panel(
                "Errors",
                ui.card(
                    ui.output_ui("errors_plot_ui"),
                    ui.output_ui("errors_legend_ui"),
                ),
                ui.download_button("download_errors", "Download data behind chart"),
            ),
            ui.nav_panel(
                "Rolling errors",
                ui.card(
                    ui.output_ui("rolling_errors_plot_ui"),
                    ui.output_ui("rolling_errors_legend_ui"),
                ),
                ui.download_button("download_rolling_errors", "Download data behind chart"),
            ),
            ui.nav_panel(
                "Bias across horizons",
                ui.output_ui("bias_plot_ui"),
                ui.output_ui("bias_legend_ui"),
                ui.download_button("download_bias", "Download data behind chart"),
            ),
            ui.nav_panel(
                "Rolling bias",
                ui.output_ui("rolling_bias_plot_ui"),
                ui.output_ui("rolling_bias_legend_ui"),
                ui.download_button("download_rolling_bias", "Download data behind chart"),
            ),
            id="bias_subtabs",
        ),
    )


def create_efficiency_tab():
    """Create the Efficiency tab UI"""
    return ui.nav_panel(
        "Efficiency",
        ui.navset_tab(
            ui.nav_panel(
                "Blanchard-Leigh",
                ui.output_ui("BL_plot_ui"),
                ui.output_ui("BL_legend_ui"),
                ui.download_button("download_bl_btn", "Download data behind chart"),
            ),
            ui.nav_panel(
                "Revisions predictability",
                ui.output_data_frame("revisions_reg"),
                ui.download_button("download_revision_pred", "Download data behind table"),
            ),
            ui.nav_panel(
                "Optimal scaling",
                ui.output_data_frame("weak_efficiency_table"),
                ui.download_button("download_weak_efficiency", "Download data behind table"),
            ),
            ui.nav_panel(
                "Correlation of revisions and errors",
                ui.output_data_frame("revisions_errors_table"),
                ui.download_button("download_revisions_errors", "Download data behind table"),
            ),
            id="efficiency_subtabs",
        ),
    )


def create_time_machine_tab():
    """Create the Time Machine tab UI"""
    return ui.nav_panel(
        "Time Machine",
        ui.output_ui("time_machine_plot_ui"),
        ui.output_ui("time_machine_legend_ui"),
        ui.download_button("download_time_machine", "Download data behind chart"),
    )


def create_hedgehog_tab():
    """Create the Hedgehog tab UI"""
    return ui.nav_panel(
        "Hedgehog",
        ui.output_ui("hedgehog_plot_ui"),
        ui.output_ui("hedgehog_legend_ui"),
        ui.download_button("download_hedgehog", "Download data behind chart"),
    )


def create_quantile_time_machine_tab():
    """Create the Time Machine tab UI"""
    return ui.nav_panel(
        "Quantile Forecasts",
        ui.output_ui("quantile_time_machine_plot_ui"),
        ui.output_ui("quantile_time_machine_legend_ui"),
        ui.download_button("quantile_time_machine_download", "Download data behind chart"),
    )


def create_outturn_revisions_tab():
    """Create the Outturn Revisions tab UI"""
    return ui.nav_panel(
        "Outturn Revisions",
        ui.navset_tab(
            ui.nav_panel(
                "Outturn Revisions",
                ui.output_ui("outturn_revisions_plot_ui"),
                ui.output_ui("outturn_revisions_legend_ui"),
                ui.download_button("download_outturn_revisions", "Download data behind chart"),
            ),
            ui.nav_panel(
                "Outturns",
                ui.output_ui("outturns_plot_ui"),
                ui.output_ui("outturns_legend_ui"),
                ui.download_button("download_outturns", "Download data behind chart"),
            ),
            id="outturn_revisions_subtabs",
        ),
    )

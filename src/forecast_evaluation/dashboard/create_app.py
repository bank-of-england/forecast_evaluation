"""Main dashboard application."""

import base64
from pathlib import Path
from importlib.resources import files

from shiny import App, ui

from .tabs.accuracy import (
    average_accuracy,
    relative_accuracy,
    rolling_accuracy,
    rolling_relative_accuracy,
    diebold_mariano,
    error_distribution,
)
from .tabs.about import about
from .tabs.bias import errors, rolling_errors, bias, rolling_bias
from .tabs.efficiency import blanchard_leigh, revisions_predictability, weak_efficiency, revisions_errors_correlation
from .tabs.hedgehog import hedgehog
from .tabs.outturn_revisions import outturn_revisions, outturns
from .tabs.time_machine import time_machine
from .tabs.quantile_time_machine import quantile_time_machine
from .ui import (
    create_accuracy_tab,
    create_bias_tab,
    create_efficiency_tab,
    create_hedgehog_tab,
    create_outturn_revisions_tab,
    create_sidebar,
    create_time_machine_tab,
    create_quantile_time_machine_tab,
)

from .theme.brand import brand as _brand
from .utils import patch_render_plot

# Apply global error handling
patch_render_plot()


def dashboard_app(data) -> App:
    # data is ForecastData instance

    def app_ui(request):
        """Main UI function"""

        tabs = [
            about(),
            create_accuracy_tab(),
            create_bias_tab(),
            create_efficiency_tab(),
            create_time_machine_tab(),
            create_hedgehog_tab(),
            create_outturn_revisions_tab(),
        ]

        if hasattr(data, "_density_forecasts") and not data._density_forecasts.empty:
            tabs.append(create_quantile_time_machine_tab())

        return ui.page_sidebar(
            create_sidebar(data),
            ui.navset_card_tab(
                *tabs,
                id="tabs",
            ),
            title=ui.div(
                ui.span("Forecast Evaluation Dashboard", style="font-size: 1.3rem;color: #12273f;"),
                style="display: flex; align-items: center;",
            ),
            theme=_brand,
        )

    def server(input, output, session):
        """Main server function"""

        # Register all handlers from different modules
        average_accuracy(input, output, session, data)
        relative_accuracy(input, output, session, data)
        rolling_accuracy(input, output, session, data)
        rolling_relative_accuracy(input, output, session, data)
        diebold_mariano(input, output, session, data)
        error_distribution(input, output, session, data)
        errors(input, output, session, data)
        rolling_errors(input, output, session, data)
        bias(input, output, session, data)
        rolling_bias(input, output, session, data)
        blanchard_leigh(input, output, session, data)
        revisions_predictability(input, output, session, data)
        weak_efficiency(input, output, session, data)
        revisions_errors_correlation(input, output, session, data)
        hedgehog(input, output, session, data)
        outturn_revisions(input, output, session, data)
        outturns(input, output, session, data)
        time_machine(input, output, session, data)

        if hasattr(data, "_density_forecasts") and not data._density_forecasts.empty:
            quantile_time_machine(input, output, session, data)

    # Returns the app
    return App(app_ui, server)

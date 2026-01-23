from shiny import ui


def about():
    return ui.nav_panel(
        "About",
        ui.markdown("""
        #### Welcome to the Forecast Evaluation Dashboard.

        This dashboard is an interactive interface to the forecast evaluation package.
        \\
        For any questions regarding the package or dashboard, please contact:\\
        Harry Li - Harry.Li@bankofengland.co.uk\\
        Paul Labonne - Paul.Labonne@bankofengland.co.uk\\
        James Hurley - James.Hurley@bankofengland.co.uk\\
        \\
        The source code for the package and dashboard is available [here](https://github.com/bank-of-england/forecast_evaluation).
        """),
        value="About",
    )

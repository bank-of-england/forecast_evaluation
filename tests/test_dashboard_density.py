"""Dashboard behaviour for density-only and merged density objects (review item A5)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from shiny import ui

from forecast_evaluation.dashboard.create_app import dashboard_app
from forecast_evaluation.dashboard.ui import create_sidebar, get_quantile_choices, get_selector_info
from forecast_evaluation.dashboard.utils import has_point_forecasts
from forecast_evaluation.data.DensityForecastData import DensityForecastData
from forecast_evaluation.data.ForecastData import ForecastData
from forecast_evaluation.data.sample_data import create_sample_forecasts

POINT_TAB_HANDLERS = ["average_accuracy", "bias", "hedgehog", "time_machine", "radar", "correlation_heatmap"]


def _sidebar_html(data) -> str:
    return str(ui.page_fluid(ui.layout_sidebar(create_sidebar(data), ui.div())))


def _point_forecasts_with_models(models=("a", "b")) -> pd.DataFrame:
    base = create_sample_forecasts().assign(metric="levels")
    return pd.concat([base.assign(model=m, value=base["value"] + i) for i, m in enumerate(models)], ignore_index=True)


def _merged(density_df, density_outturns_df, **density_kwargs) -> DensityForecastData:
    density = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=density_df, **density_kwargs)
    point = ForecastData(
        outturns_data=density_outturns_df,
        forecasts_data=_point_forecasts_with_models(),
        extra_ids=["model"],
        data_check=False,
    )
    density.merge(point)
    return density


def _density_with_levels(density_df, density_outturns_df, levels) -> DensityForecastData:
    base = density_df[density_df["quantile"] == density_df["quantile"].min()]
    frames = [base.assign(quantile=q, value=base["value"] + q) for q in levels]
    return DensityForecastData(outturns_data=density_outturns_df, forecasts_data=pd.concat(frames, ignore_index=True))


# -----------------------
# Selectors
# -----------------------
def test_sidebar_offers_density_sources_and_variables(density_two_sources):
    html = _sidebar_html(density_two_sources)

    assert 'value="mpr2"' in html
    assert 'value="toy"' in html
    assert 'value="gdpkp"' in html


def test_selector_choices_skip_empty_point_table(density_extra_ids):
    # Old code read only the density table; the new union must not KeyError on the point table lacking 'region'.
    assert density_extra_ids.forecasts.empty
    assert get_selector_info("region", density_extra_ids)[0] == ["UK"]


def test_selector_choices_list_point_models_after_merge(density_df, density_outturns_df):
    # Old code read only the density table, where model is NA after the merge, so returned [].
    merged = _merged(density_df, density_outturns_df)

    assert get_selector_info("model", merged) == (["a", "b"], "model", "model_s")


def test_selector_choices_union_when_density_has_model(density_df, density_outturns_df):
    # Old code returned ['d'] only.
    merged = _merged(density_df.assign(model="d"), density_outturns_df, extra_ids=["model"])

    assert get_selector_info("model", merged)[0] == ["a", "b", "d"]


# -----------------------
# Quantile choices
# -----------------------
def test_quantile_choices_equal_stored_levels_sample_grid(density_only):
    stored = np.sort(density_only._density_forecasts["quantile"].unique())

    choices, lower, upper = get_quantile_choices(density_only)

    assert len(choices) == len(stored)
    assert np.allclose([float(c) for c in choices], stored, atol=1e-9)
    assert lower in choices and upper in choices
    assert abs(float(lower) - 0.16) <= 0.0101
    assert abs(float(upper) - 0.84) <= 0.0101


def test_quantile_choices_keep_levels_finer_than_two_decimals(density_df, density_outturns_df):
    # Old code rounded to 2 dp, so 0.025 / 0.975 were offered as other levels that are not stored.
    levels = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
    data = _density_with_levels(density_df, density_outturns_df, levels)

    choices, _, _ = get_quantile_choices(data)
    html = _sidebar_html(data)

    assert [float(c) for c in choices] == levels
    assert 'value="0.025"' in html
    assert 'value="0.975"' in html
    assert 'value="0.03"' not in html
    assert 'value="0.97"' not in html


def test_quantile_choices_distinct_levels_not_collapsed(density_df, density_outturns_df):
    # Old code collapsed a 0.005-step grid (199 levels) to at most 100 rounded choices.
    levels = [round(0.005 * i, 3) for i in range(1, 200)]
    data = _density_with_levels(density_df, density_outturns_df, levels)

    choices, _, _ = get_quantile_choices(data)

    assert len(choices) == 199


# -----------------------
# Empty forecasts
# -----------------------
def test_create_sidebar_without_forecasts_raises_clear_error(density_outturns_df):
    # Old code raised IndexError at sources[0] (or an earlier [0] lookup).
    data = ForecastData(outturns_data=density_outturns_df)

    with pytest.raises(ValueError, match="at least one forecast"):
        create_sidebar(data)


def test_create_sidebar_density_only_selects_density_source(density_only):
    html = _sidebar_html(density_only)

    assert 'id="source"' in html
    assert 'value="mpr2"' in html


# -----------------------
# Point-only tabs on density-only objects
# -----------------------
def test_has_point_forecasts(request):
    # density_and_point mutates the density_only instance, so resolve them one after the other.
    assert not has_point_forecasts(request.getfixturevalue("density_only"))
    assert has_point_forecasts(request.getfixturevalue("density_and_point"))


def test_dashboard_density_only_shows_placeholders_and_skips_point_handlers(density_only):
    # Old code built the real Accuracy tab (and registered its handlers), which can only show errors here.
    app = dashboard_app(density_only)
    page_html = str(app.ui(None))

    assert "No point forecasts: call to_point_forecast()" in page_html
    assert "accuracy_plot_ui" not in page_html
    assert "bias_plot_ui" not in page_html
    for title in ["Accuracy", "Bias", "Efficiency", "Time Machine", "Hedgehog", "Correlation", "Radar"]:
        assert f'data-value="{title}"' in page_html
    assert 'data-value="Quantile Forecasts"' in page_html

    patches = [patch(f"forecast_evaluation.dashboard.create_app.{name}") for name in POINT_TAB_HANDLERS]
    with patch("forecast_evaluation.dashboard.create_app.quantile_time_machine") as mock_quantile:
        mocks = [p.start() for p in patches]
        try:
            app.server(MagicMock(), MagicMock(), MagicMock())
        finally:
            for p in patches:
                p.stop()

    for mock in mocks:
        mock.assert_not_called()
    mock_quantile.assert_called_once()


def test_dashboard_density_and_point_keeps_point_tabs(density_and_point):
    app = dashboard_app(density_and_point)
    page_html = str(app.ui(None))

    assert "accuracy_plot_ui" in page_html
    assert "No point forecasts" not in page_html

    with patch("forecast_evaluation.dashboard.create_app.average_accuracy") as mock_accuracy:
        app.server(MagicMock(), MagicMock(), MagicMock())

    mock_accuracy.assert_called_once()

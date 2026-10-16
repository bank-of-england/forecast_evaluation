"""Tests for DensityForecastData.sample_from_density (A2, A8, A12 sampling part, A22)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, spearmanr

from forecast_evaluation.data.DensityForecastData import DensityForecastData, _interpolate_quantile_function

DATES = pd.date_range("2025-12-31", periods=13, freq="QE")
ID_AND_META_COLUMNS = [
    "date",
    "variable",
    "vintage_date",
    "source",
    "frequency",
    "forecast_horizon",
    "metric",
    "target_minus_vintage",
    "unique_id",
]


def _normal_density_df(grid) -> pd.DataFrame:
    """Normal quantiles (mean 115 + i, sd 0.5) on ``grid`` for 13 dates."""
    rows = [
        {
            "date": date,
            "variable": "gdpkp",
            "vintage_date": pd.Timestamp("2025-12-31"),
            "source": "toy_grid",
            "frequency": "Q",
            "quantile": float(q),
            "value": float(norm.ppf(q, loc=115 + i, scale=0.5)),
            "forecast_horizon": i,
            "metric": "levels",
        }
        for i, date in enumerate(DATES)
        for q in grid
    ]
    return pd.DataFrame(rows)


def _sample_recording_tail_warnings(dfd, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        samples = dfd.sample_from_density(**kwargs)
    tail = [w for w in caught if issubclass(w.category, UserWarning) and "outer quantile levels" in str(w.message)]
    return samples, tail


def _values_by_date(samples: pd.DataFrame) -> pd.DataFrame:
    return samples.pivot(index="sample_id", columns="date", values="value")


# -----------------------
# A2: random generator
# -----------------------
@pytest.mark.parametrize("random_state", [5, None])
def test_sample_from_density_leaves_global_rng_untouched(density_only, random_state):
    np.random.seed(123)
    before = np.random.get_state()

    density_only.sample_from_density(n_samples=200, random_state=random_state)

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_sample_from_density_reproducible_and_seed_sensitive(density_only):
    first = density_only.sample_from_density(n_samples=300, random_state=42)
    second = density_only.sample_from_density(n_samples=300, random_state=42)
    other_seed = density_only.sample_from_density(n_samples=300, random_state=43)
    unseeded_a = density_only.sample_from_density(n_samples=300)
    unseeded_b = density_only.sample_from_density(n_samples=300)

    pd.testing.assert_frame_equal(first, second)
    assert not np.array_equal(first["value"].to_numpy(), other_seed["value"].to_numpy())
    assert not np.array_equal(unseeded_a["value"].to_numpy(), unseeded_b["value"].to_numpy())


def test_sample_from_density_seeded_draws_independent_across_densities(density_only):
    samples = density_only.sample_from_density(n_samples=5000, random_state=42)

    by_date = _values_by_date(samples)
    assert by_date.shape[1] == 13
    corr = spearmanr(by_date.to_numpy())[0]
    off_diagonal = np.abs(corr[np.triu_indices(13, k=1)])

    assert off_diagonal.max() < 0.1


def test_sample_from_density_accepts_generator(density_only):
    from_int = density_only.sample_from_density(n_samples=200, random_state=7)
    shared = np.random.default_rng(7)
    from_generator = density_only.sample_from_density(n_samples=200, random_state=shared)
    from_same_generator_again = density_only.sample_from_density(n_samples=200, random_state=shared)

    pd.testing.assert_frame_equal(from_int, from_generator)
    assert not np.array_equal(from_generator["value"].to_numpy(), from_same_generator_again["value"].to_numpy())


# -----------------------
# A12 (sampling part): NA ids, empty table, short densities
# -----------------------
def test_sample_from_density_keeps_densities_with_na_in_some_extra_ids(density_two_regions):
    table = density_two_regions._density_forecasts
    table.loc[table["region"] == "EW", "region"] = pd.NA
    n = 20

    samples = density_two_regions.sample_from_density(n_samples=n, random_state=1)

    assert len(samples) == 26 * n
    assert samples["region"].isna().sum() == 13 * n
    assert (samples["region"] == "UK").sum() == 13 * n


def test_sample_from_density_keeps_densities_when_all_extra_ids_are_na(density_extra_ids):
    density_extra_ids._density_forecasts["region"] = pd.NA
    n = 20

    samples = density_extra_ids.sample_from_density(n_samples=n, random_state=1)

    assert len(samples) == 13 * n
    assert samples["region"].isna().all()


def test_sample_from_density_keeps_densities_after_merge_with_different_extra_ids(density_outturns_df, density_df):
    first = DensityForecastData(
        outturns_data=density_outturns_df, forecasts_data=density_df.assign(label1="A"), extra_ids=["label1"]
    )
    second = DensityForecastData(
        outturns_data=density_outturns_df,
        forecasts_data=density_df.assign(label2="B", source="toy"),
        extra_ids=["label2"],
    )
    merged = first.copy()
    merged.merge(second)
    n = 10

    samples = merged.sample_from_density(n_samples=n, random_state=1)

    assert len(samples) == 26 * n
    assert set(samples["source"]) == {"mpr2", "toy"}


def test_sample_from_density_empty_table_raises_value_error(density_only):
    # Emptied directly: a filter that removes everything may also fail earlier for unrelated reasons.
    density_only._density_forecasts = density_only._density_forecasts.iloc[0:0]

    with pytest.raises(ValueError, match="No density forecasts to sample from"):
        density_only.sample_from_density(n_samples=10)


def test_sample_from_density_no_density_loaded_raises_value_error(density_outturns_df):
    dfd = DensityForecastData(outturns_data=density_outturns_df)

    with pytest.raises(ValueError, match="No density forecasts to sample from"):
        dfd.sample_from_density(n_samples=10)


def test_sample_from_density_rejects_densities_with_fewer_than_two_quantiles(density_outturns_df, density_df):
    short_date = DATES[2]
    keep = (density_df["date"] != short_date) | (density_df["quantile"] == density_df["quantile"].min())
    dfd = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=density_df[keep])

    with pytest.raises(ValueError, match="fewer than 2") as excinfo:
        dfd.sample_from_density(n_samples=10)

    assert "2026-06-30" in str(excinfo.value)


# -----------------------
# A22: output unchanged by the vectorised construction
# -----------------------
def test_sample_from_density_shape_columns_and_sample_ids(density_only):
    n = 50

    samples = density_only.sample_from_density(n_samples=n, random_state=3)

    assert list(samples.columns) == [*ID_AND_META_COLUMNS, "sample_id", "value"]
    assert samples.shape == (13 * n, len(ID_AND_META_COLUMNS) + 2)
    assert samples["sample_id"].dtype == np.int64
    for _, block in samples.groupby("date"):
        assert block["sample_id"].tolist() == list(range(n))
    assert samples["date"].is_monotonic_increasing
    assert (samples.index == np.arange(len(samples))).all()


def test_sample_from_density_matches_reference_loop(density_two_sources):
    n = 30
    density = density_two_sources.density_forecasts
    group_cols = [column for column in density.columns if column not in ("quantile", "value")]

    rng = np.random.default_rng(11)
    parts = []
    for _, group in density.groupby(group_cols, sort=True):
        group = group.sort_values("quantile")
        draws = np.interp(rng.random(n), group["quantile"], group["value"])
        part = pd.concat([group[group_cols].iloc[[0]]] * n, ignore_index=True)
        part["sample_id"] = range(n)
        part["value"] = draws
        parts.append(part)
    expected = pd.concat(parts, ignore_index=True)

    actual = density_two_sources.sample_from_density(n_samples=n, random_state=11)

    pd.testing.assert_frame_equal(actual, expected)


# -----------------------
# A8: tail clamping
# -----------------------
@pytest.mark.parametrize(
    "fixture_name, outer_mass_text, outer_levels_text",
    [("sparse_grid_density_df", "10%", "0.1 and 0.9"), ("bvar_grid_density_df", "16%", "0.16 and 0.84")],
)
def test_sample_from_density_warns_on_sparse_tails(
    request, density_outturns_df, fixture_name, outer_mass_text, outer_levels_text
):
    dfd = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=request.getfixturevalue(fixture_name))

    _, tail = _sample_recording_tail_warnings(dfd, n_samples=50, random_state=0)

    assert len(tail) == 1
    message = str(tail[0].message)
    assert outer_mass_text in message
    assert outer_levels_text in message
    assert "standard deviation" in message
    assert "13 of 13" in message


def test_sample_from_density_sparse_grid_puts_outer_mass_on_end_values(density_outturns_df, sparse_grid_density_df):
    dfd = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=sparse_grid_density_df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        samples = dfd.sample_from_density(n_samples=20000, random_state=0)

    for _, values in _values_by_date(samples).items():
        values = values.to_numpy()
        assert abs(np.mean(values == values.min()) - 0.10) < 0.01
        assert abs(np.mean(values == values.max()) - 0.10) < 0.01


@pytest.mark.parametrize(
    "grid", [np.linspace(0.001, 0.999, 50), np.linspace(0.02, 0.98, 25), np.linspace(0.01, 0.99, 50)]
)
def test_sample_from_density_no_tail_warning_within_threshold(density_outturns_df, grid):
    dfd = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=_normal_density_df(grid))

    _, tail = _sample_recording_tail_warnings(dfd, n_samples=50, random_state=0)

    assert tail == []


def test_sample_from_density_no_tail_warning_for_sample_density(density_only):
    _, tail = _sample_recording_tail_warnings(density_only, n_samples=50, random_state=0)

    assert tail == []


def test_sample_from_density_fine_grid_standard_deviation(density_outturns_df):
    grid = np.linspace(0.01, 0.99, 50)
    dfd = DensityForecastData(outturns_data=density_outturns_df, forecasts_data=_normal_density_df(grid))

    samples = dfd.sample_from_density(n_samples=20000, random_state=0)

    ratios = _values_by_date(samples).std() / 0.5
    assert (ratios >= 0.96).all()
    assert (ratios <= 1.03).all()


def test_interpolate_quantile_function_clamps_outside_range():
    func = _interpolate_quantile_function(np.array([0.1, 0.5, 0.9]), np.array([-1.0, 0.0, 2.0]))

    np.testing.assert_array_equal(func(np.array([0.0, 0.05, 0.95, 1.0])), [-1.0, -1.0, 2.0, 2.0])


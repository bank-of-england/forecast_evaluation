import pandas as pd
import pytest

from forecast_evaluation.utils import filter_k


def _maturities(maturities_by_series: dict[str, list[int]]) -> pd.DataFrame:
    rows = []
    for series, maturities in maturities_by_series.items():
        for maturity in maturities:
            rows.append(
                {
                    "date": pd.Timestamp("2020-01-31"),
                    "variable": "target",
                    "frequency": "M",
                    "metric": "levels",
                    "unique_id": series,
                    "k": maturity,
                    "vintage_date_outturn": pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(maturity + 1),
                    "latest_vintage": pd.Timestamp("2025-01-31"),
                }
            )
    return pd.DataFrame(rows)


def test_filter_k_prefers_exact_maturity():
    data = _maturities({"model": [5, 11, 12, 14]})

    result = filter_k(data, k=12)

    assert result["k"].tolist() == [12]


def test_filter_k_uses_nearest_lower_maturity_when_exact_is_missing():
    data = _maturities({"model": [5, 8, 11, 14]})

    result = filter_k(data, k=12)

    assert result["k"].tolist() == [11]


def test_filter_k_selects_maturity_independently_for_each_series():
    data = _maturities({"model_a": [5, 11, 14], "model_b": [8, 13]})

    result = filter_k(data, k=12).sort_values("unique_id")

    assert result[["unique_id", "k"]].to_dict("records") == [
        {"unique_id": "model_a", "k": 11},
        {"unique_id": "model_b", "k": 8},
    ]


def test_filter_k_without_fill_requires_exact_maturity():
    data = _maturities({"model": [5, 8, 11, 14]})

    result = filter_k(data, k=12, fill_k=False)

    assert result.empty


def test_filter_k_uses_earliest_later_maturity_when_no_lower_maturity_exists():
    data = _maturities({"model": [14]})

    result = filter_k(data, k=12)

    assert result["k"].tolist() == [14]


def test_filter_k_keeps_all_rows_when_outturn_vintages_are_unavailable():
    data = _maturities({"model": [5, 11, 14]})
    data["latest_vintage"] = pd.NaT

    result = filter_k(data, k=12)

    pd.testing.assert_frame_equal(result, data)


@pytest.mark.parametrize(
    ("available_maturities", "expected_maturities"),
    [
        ([5], [5]),
        ([5, 8, 11], [11]),
        ([5, 8, 11, 14], [11]),
        ([12, 14], [12]),
        ([14], [14]),
    ],
)
def test_filter_k_selection_table(available_maturities, expected_maturities):
    """For k=12, prefer the largest lower maturity and otherwise use the next one."""
    data = _maturities({"model": available_maturities})

    result = filter_k(data, k=12)

    assert result["k"].tolist() == expected_maturities

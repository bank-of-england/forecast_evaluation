import pandas as pd
import pandera.pandas as pa
import pytest

from forecast_evaluation.data.schema import OUTTURN_REQUIRED_COLUMNS, create_data_schema
from forecast_evaluation.data.utils import compute_target_minus_vintage


def test_target_minus_vintage_quarterly_and_monthly():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-31", "2024-09-30"]),
            "vintage_date": pd.to_datetime(["2024-01-15", "2024-07-15"]),
            "frequency": ["Q", "M"],
        }
    )

    result = compute_target_minus_vintage(data)

    assert result["target_minus_vintage"].tolist() == [0, 2]


def test_target_minus_vintage_matches_previous_values():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-03-31", "2022-06-30", "2022-09-30"]),
            "vintage_date": pd.to_datetime(["2021-12-31", "2022-03-31", "2022-09-30"]),
            "frequency": ["Q", "Q", "Q"],
        }
    )

    result = compute_target_minus_vintage(data)

    assert result["target_minus_vintage"].tolist() == [1, 1, 0]


def test_target_minus_vintage_null_vintage_yields_na():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
            "vintage_date": pd.to_datetime([None, "2024-03-31"]),
            "frequency": ["Q", "Q"],
        }
    )

    result = compute_target_minus_vintage(data)

    assert pd.isna(result.loc[0, "target_minus_vintage"])
    assert result.loc[0, "target_minus_vintage"] is pd.NA
    assert result.loc[0, "target_minus_vintage"] is not -9223372036854775808
    assert result.loc[1, "target_minus_vintage"] == 1
    assert str(result["target_minus_vintage"].dtype) == "Int64"


def test_forecast_schema_requires_forecast_horizon():
    forecast = pd.DataFrame(
        {
            "date": ["2024-03-31"],
            "vintage_date": ["2024-01-15"],
            "variable": ["gdp"],
            "frequency": ["Q"],
            "value": [1.0],
            "source": ["model"],
        }
    )

    with pytest.raises(pa.errors.SchemaError, match="forecast_horizon"):
        create_data_schema(forecast=True).validate(forecast)


def test_outturn_schema_omits_forecast_horizon():
    assert "forecast_horizon" not in OUTTURN_REQUIRED_COLUMNS

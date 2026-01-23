"""Data models and validation schemas for forecast evaluation."""

from typing import Optional

import pandas as pd
import pandera.pandas as pa

ALLOWED_FREQUENCIES = ["Q", "M"]


def create_data_schema(forecast: bool = False, optional_columns: Optional[list[str]] = None):
    """Create validation schema for forecast/outturn data.

    Parameters
    ----------
    forecast : bool, optional
        Whether the data is forecast data (True) or outturn data (False). Default is False.
    optional_columns : list of str, optional
        Additional optional columns to include in the schema.

    Returns
    -------
    pa.DataFrameSchema
        Pandera schema for validation.
    """
    # Required columns
    columns = {
        "date": pa.Column(pd.Timestamp, coerce=True),
        "vintage_date": pa.Column(pd.Timestamp, coerce=True),
        "variable": pa.Column(str, pa.Check(lambda s: s.str.len() >= 1)),
        "frequency": pa.Column(
            str, pa.Check(lambda s: s.isin(ALLOWED_FREQUENCIES), name=f"must be one of {ALLOWED_FREQUENCIES}")
        ),
        "forecast_horizon": pa.Column(int, coerce=True),
        "value": pa.Column(float, nullable=False, coerce=True),
    }

    # add source if data is forecast
    if forecast:
        columns["source"] = pa.Column(str, pa.Check(lambda s: s.str.len() >= 1))

    # Add optional columns
    if optional_columns:
        for col in optional_columns:
            # metric column has specific validation
            if col == "metric":
                columns[col] = pa.Column(
                    str,
                    pa.Check(
                        lambda s: s.isin(["levels", "pop", "yoy"]), name="must be one of ['levels', 'pop', 'yoy']"
                    ),
                )
            else:
                columns[col] = pa.Column(str, pa.Check(lambda s: s.str.len() >= 1))

            if col == "quantile":
                columns[col] = pa.Column(
                    float,
                    pa.Check(lambda s: (s >= 0) & (s <= 1), name="quantile must be between 0 and 1"),
                    coerce=True,
                )
            else:
                columns[col] = pa.Column(str, pa.Check(lambda s: s.str.len() >= 1), nullable=True)

    return pa.DataFrameSchema(
        columns=columns,
        strict="filter",  # keeps only columns defined in schema
        coerce=True,
    )


# Extract required columns from the base schema
FORECAST_REQUIRED_COLUMNS = list(create_data_schema(forecast=True).columns.keys())
OUTTURN_REQUIRED_COLUMNS = list(create_data_schema(forecast=False).columns.keys())

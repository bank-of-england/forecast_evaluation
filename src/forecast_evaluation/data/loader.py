import importlib.resources as resources

import pandas as pd


def load_fer_outturns() -> pd.DataFrame:
    """Load FER outturns data from embedded parquet file."""
    try:
        # Python 3.9+
        with resources.files("forecast_evaluation.data.files").joinpath("fer_outturns.parquet").open("rb") as f:
            return pd.read_parquet(f)
    except AttributeError:
        # Python 3.7-3.8 fallback
        with resources.open_binary("forecast_evaluation.data.files", "fer_outturns.parquet") as f:
            return pd.read_parquet(f)


def load_fer_forecasts() -> pd.DataFrame:
    """Load FER forecasts data from embedded parquet file."""
    try:
        with resources.files("forecast_evaluation.data.files").joinpath("fer_forecasts.parquet").open("rb") as f:
            return pd.read_parquet(f)
    except AttributeError:
        with resources.open_binary("forecast_evaluation.data.files", "fer_forecasts.parquet") as f:
            return pd.read_parquet(f)

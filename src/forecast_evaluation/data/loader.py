import importlib.resources as resources

import pandas as pd


def load_fer_outturns(minimal: bool = False) -> pd.DataFrame:
    """Load FER outturns data from embedded parquet file.
    Args:
        minimal: If True, load the minimal outturns dataset (only 2018).
        Defaults to False (full dataset).
    """
    outturns_file = "fer_outturns_minimal.parquet" if minimal else "fer_outturns.parquet"
    try:
        # Python 3.9+
        with resources.files("forecast_evaluation.data.files").joinpath(outturns_file).open("rb") as f:
            return pd.read_parquet(f)
    except AttributeError:
        # Python 3.7-3.8 fallback
        with resources.open_binary("forecast_evaluation.data.files", outturns_file) as f:
            return pd.read_parquet(f)


def load_fer_forecasts(minimal: bool = False) -> pd.DataFrame:
    """Load FER forecasts data from embedded parquet file.
    Args:
        minimal: If True, load the minimal forecasts dataset (only 2018, 2 sources and 2 variables).
        Defaults to False (full dataset).
    """
    forecast_file = "fer_forecasts_minimal.parquet" if minimal else "fer_forecasts.parquet"

    try:
        with resources.files("forecast_evaluation.data.files").joinpath(forecast_file).open("rb") as f:
            return pd.read_parquet(f)
    except AttributeError:
        with resources.open_binary("forecast_evaluation.data.files", forecast_file) as f:
            return pd.read_parquet(f)

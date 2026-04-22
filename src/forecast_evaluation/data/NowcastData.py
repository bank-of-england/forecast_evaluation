from typing import Literal, Optional

import pandas as pd

from forecast_evaluation.data.ForecastData import ForecastData


class NowcastData(ForecastData):
    """ForecastData subclass for nowcasting evaluation.

    Designed for environments where forecasts (and outturns) are released
    multiple times per period with intra-period vintage dates (e.g. weekly).
    Integer-period horizons are used (h=-1 backcast, h=0 nowcast, h=1 one-
    period-ahead) so that multiple weekly vintages per horizon provide many
    observations for accuracy statistics.

    Differences from ForecastData:

    - ``first_forecast_horizon`` defaults to -1 (include backcasts).
    - The main table includes a ``days_to_publication`` column computed as
      ``(vintage_date_outturn - vintage_date_forecast).days``, which can be
      used for intra-period accuracy analysis.
    - Efficiency analyses (weak/strong efficiency, Blanchard-Leigh, revision
      predictability, revisions-errors correlation) are not available.
    """

    def __init__(
        self,
        outturns_data: Optional[pd.DataFrame] = None,
        forecasts_data: Optional[pd.DataFrame] = None,
        *,
        extra_ids: Optional[list[str]] = None,
        metric: Literal["levels", "pop", "yoy"] = "levels",
        compute_levels: bool = True,
        data_check: bool = True,
        first_forecast_horizon: int = -1,
    ):
        """Initialise NowcastData.

        Parameters
        ----------
        outturns_data : pd.DataFrame, optional
            DataFrame containing outturn records.
        forecasts_data : pd.DataFrame, optional
            DataFrame containing forecast records.
        extra_ids : list of str, optional
            Extra label columns in addition to 'source'.
        metric : str, optional
            Default metric if not present in the data. Default is 'levels'.
        compute_levels : bool, optional
            Whether to auto-transform non-levels forecasts to levels.
        data_check : bool, optional
            Whether to run data checks when adding forecasts.
        first_forecast_horizon : int, optional
            Minimum forecast horizon to retain. Default is -1 (include backcasts).
        """
        super().__init__(
            outturns_data=outturns_data,
            forecasts_data=forecasts_data,
            extra_ids=extra_ids,
            metric=metric,
            compute_levels=compute_levels,
            data_check=data_check,
            first_forecast_horizon=first_forecast_horizon,
        )

    def add_forecasts(self, df, **kwargs):
        """Add forecasts, aligning outturn vintages to forecast vintages first."""
        self._align_outturn_vintages(df)
        super().add_forecasts(df, **kwargs)
        self._add_days_to_publication()

    def _align_outturn_vintages(self, forecasts_df: pd.DataFrame):
        """Expand outturns so every forecast vintage has matching outturn rows.

        For each variable and forecast vintage V that doesn't already exist in
        the outturns, we copy the outturn rows from the latest outturn vintage
        <= V for that variable and stamp them with vintage_date = V.  This
        ensures that downstream transforms (pop, yoy) which group by
        vintage_date can pair forecast and outturn rows naturally, without
        modifying the transformation logic.

        The lookup is per-variable because different variables may have
        different publication lags (e.g. GDP 42 days, CPI 14 days) and
        therefore different outturn vintage calendars.
        """
        if self._raw_outturns.empty:
            return

        forecast_vintages = pd.to_datetime(forecasts_df["vintage_date"]).unique()

        new_rows = []
        for variable, var_outturns in self._raw_outturns.groupby("variable"):
            var_outturn_vintages = sorted(pd.to_datetime(var_outturns["vintage_date"]).unique())
            missing = set(forecast_vintages) - set(var_outturn_vintages)
            if not missing:
                continue

            for vintage in sorted(missing):
                valid = [v for v in var_outturn_vintages if v <= vintage]
                if not valid:
                    continue
                closest = valid[-1]
                rows = var_outturns[var_outturns["vintage_date"] == closest].copy()
                rows["vintage_date"] = vintage
                new_rows.append(rows)

        if new_rows:
            from forecast_evaluation.core.transformations import prepare_outturns
            from forecast_evaluation.data.utils import compute_forecast_horizon

            expanded = pd.concat(new_rows, ignore_index=True)
            expanded = compute_forecast_horizon(expanded)
            expanded["_aligned"] = True
            if "_aligned" not in self._raw_outturns.columns:
                self._raw_outturns["_aligned"] = False
            self._raw_outturns = pd.concat([self._raw_outturns, expanded], ignore_index=True)
            self._outturns = prepare_outturns(self._raw_outturns)

    def _add_days_to_publication(self):
        """Add days_to_publication column to the main table.

        Computed as the number of days between the outturn vintage and the
        forecast vintage. A larger value means the forecast was made further
        from publication; the value decreases (and can become negative for
        backcasts) as the vintage approaches the outturn release.
        """
        if self._main_table.empty:
            return
        if "vintage_date_forecast" in self._main_table.columns and "vintage_date_outturn" in self._main_table.columns:
            self._main_table["days_to_publication"] = (
                pd.to_datetime(self._main_table["vintage_date_outturn"])
                - pd.to_datetime(self._main_table["vintage_date_forecast"])
            ).dt.days

    # --- Methods not supported for nowcasting data ---

    def add_fer_data(self):
        raise NotImplementedError("FER data loading is not available for NowcastData.")

    def add_fer_outturns(self):
        raise NotImplementedError("FER data loading is not available for NowcastData.")

    def add_fer_forecasts(self):
        raise NotImplementedError("FER data loading is not available for NowcastData.")

    def filter_fer(self):
        raise NotImplementedError("FER filtering is not available for NowcastData.")

    def create_pseudo_vintages(self, *args, **kwargs):
        raise NotImplementedError("Pseudo vintage creation is not available for NowcastData.")

    def add_benchmarks(self, *args, **kwargs):
        raise NotImplementedError("Benchmark models are not yet available for NowcastData.")

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
        self._set_revision_index_k()
        self._add_days_to_publication()

    def _set_revision_index_k(self):
        """Set ``k`` as a dense revision index over outturn vintages.

        Generalises the parent's calendar-based ``k`` to the nowcasting case
        where multiple outturn vintages can fall within a single period.
        Within each ``(variable, metric, frequency, date)`` group:

        - Post-release vintages (``vintage_date_outturn >= date``) are dense-
          ranked in ascending order and indexed ``0, 1, 2, ...`` so ``k=0`` is
          the first release, ``k=1`` the first revision, etc.
        - Pre-release vintages (``vintage_date_outturn < date``, i.e. backcast
          snapshots taken before the period ended) are dense-ranked in
          descending order and indexed ``-1, -2, -3, ...`` so ``k=-1`` is the
          latest pre-release snapshot, ``k=-2`` the one before, etc.

        Guarantees one row per ``(date, k)`` for each forecast vintage / source.
        """
        if self._main_table.empty:
            return

        mt = self._main_table.copy()
        group_cols = ["variable", "metric", "frequency", "date"]
        key_cols = group_cols + ["vintage_date_outturn"]

        def _ranked_k(subset: pd.DataFrame, ascending: bool, offset: int) -> pd.DataFrame:
            unique_vintages = subset[key_cols].drop_duplicates().sort_values(key_cols)
            ranks = (
                unique_vintages.groupby(group_cols)["vintage_date_outturn"]
                .rank(method="dense", ascending=ascending)
                .astype(int)
            )
            unique_vintages = unique_vintages.assign(k=(ranks + offset) if ascending else -(ranks + offset))
            return subset.drop(columns=["k"]).merge(unique_vintages, on=key_cols, how="left")

        pre_release_mask = mt["vintage_date_outturn"] < mt["date"]
        pieces = []
        if pre_release_mask.any():
            pieces.append(_ranked_k(mt.loc[pre_release_mask], ascending=False, offset=0))
        if (~pre_release_mask).any():
            pieces.append(_ranked_k(mt.loc[~pre_release_mask], ascending=True, offset=-1))

        mt = pd.concat(pieces, ignore_index=True)
        mt["k"] = mt["k"].astype(int)
        self._main_table = mt

    def _align_outturn_vintages(self, forecasts_df: pd.DataFrame):
        """Build outturn snapshots so every forecast vintage has a full history.

        For each variable and forecast vintage *V* that doesn't already exist
        in the outturns, this method builds a **point-in-time snapshot** of the
        outturn data that was available at *V*.  For every target date *D*
        whose outturn had been released by *V*, the row with the **latest**
        outturn ``vintage_date <= V`` is selected.  The resulting snapshot is
        stamped with ``vintage_date = V``.
        # TODO: Maybe this could be generated on the fly to save memory


        ===
        The following is important when backcasting outturns that have already
        been released (possibly because the forecaster think they can do better):

        Dates that the forecasts themselves target at vintage *V* are excluded
        from the snapshot.  Without this, backcast rows (h=-1) would share a
        date with the prepended outturn rows inside ``prepare_forecasts``,
        creating duplicate rows that corrupt ``pct_change``.

        This is crucial for the transformation pipeline: ``pct_change(n)``
        requires a gap-free quarterly series.  The previous approach of copying
        rows from one closest outturn vintage produced incomplete series
        (first-release vintages typically contain only the newly released
        quarter).  Building a proper snapshot ensures the full historical time
        series is available.

        The lookup is per-variable because different variables may have
        different publication lags (e.g. GDP 42 days, CPI 14 days) and
        therefore different outturn vintage calendars.
        """
        if self._raw_outturns.empty:
            return

        forecast_vintages = pd.to_datetime(forecasts_df["vintage_date"]).unique()
        forecast_variables = set(forecasts_df["variable"].unique())

        # Pre-compute which (variable, vintage_date) -> set of target dates
        # the forecasts cover, so we can exclude them from each snapshot.
        fc_targets = forecasts_df.groupby(["variable", "vintage_date"])["date"].apply(set).to_dict()

        new_rows = []
        for variable, var_outturns in self._raw_outturns.groupby("variable"):
            # Only align outturn variables that have corresponding forecasts.
            if variable not in forecast_variables:
                continue
            existing_vintages = set(pd.to_datetime(var_outturns["vintage_date"]).unique())
            missing = set(forecast_vintages) - existing_vintages
            if not missing:
                continue

            for vintage in sorted(missing):
                # All outturn rows for this variable released on or before *vintage*
                available = var_outturns[var_outturns["vintage_date"] <= vintage]
                if available.empty:
                    continue

                # For each target date, keep the row from the latest outturn
                # vintage — this is the best estimate available at time *vintage*.
                idx = available.groupby("date")["vintage_date"].idxmax()
                snapshot = available.loc[idx].copy()

                # Exclude dates that the forecasts already cover at this
                # vintage.  This avoids duplicate (variable, vintage_date,
                # date) rows when the outturn + forecast DataFrames are
                # concatenated inside prepare_forecasts.
                forecast_dates = fc_targets.get((variable, vintage), set())
                if forecast_dates:
                    snapshot = snapshot[~snapshot["date"].isin(forecast_dates)]
                    if snapshot.empty:
                        continue

                snapshot["vintage_date"] = vintage
                new_rows.append(snapshot)

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

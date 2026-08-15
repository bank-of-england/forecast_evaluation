from typing import Literal, Optional

import pandas as pd

from forecast_evaluation.data.ForecastData import _FORECAST_STATE, ForecastData, _atomic_state
from forecast_evaluation.data.utils import compute_target_minus_vintage


class NowcastData(ForecastData):
    """ForecastData subclass for nowcasting evaluation.

    Designed for environments where forecasts (and outturns) are released
    multiple times per period with intra-period vintage dates (e.g. weekly).
    Non-negative information horizons are used (h=0 nowcast, h=1 one-period-
    ahead) so that multiple weekly vintages per horizon provide many
    observations for accuracy statistics. Vintage-relative backcasts are
    identified by ``target_minus_vintage`` instead.

    Differences from ForecastData:

    - The main table includes a ``days_to_publication`` column computed as
      ``(vintage_date_outturn - vintage_date_forecast).days``, which can be
      used for intra-period accuracy analysis.
    - Efficiency analyses (weak/strong efficiency, Blanchard-Leigh, revision
      predictability, revisions-errors correlation) are not available.

    Notes
    -----
    Two different time axes appear in nowcast analysis and should not be
    confused:

    - ``days_to_publication`` (this class) is the distance from the forecast
      vintage to the *release date* of the selected outturn vintage. It
      therefore depends on ``k`` and on the publication lag of the series.
    - ``days_to_period_end`` is the distance from the forecast vintage to the
      *end of the target period*, independent of when the outturn is
      published.

    The two differ by the publication lag. Intra-period accuracy and bias
    functions take an ``axis`` argument (``'period_end'``, the default, or
    ``'publication'``) to choose between them, and bin the result to the
    nearest 7 days.
    """

    default_k = 0

    def __init__(
        self,
        outturns_data: Optional[pd.DataFrame] = None,
        forecasts_data: Optional[pd.DataFrame] = None,
        *,
        extra_ids: Optional[list[str]] = None,
        metric: Literal["levels", "pop", "yoy"] = "levels",
        compute_levels: bool = True,
        data_check: bool = True,
        default_k: Optional[int] = None,
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
        """
        super().__init__(
            outturns_data=outturns_data,
            forecasts_data=forecasts_data,
            extra_ids=extra_ids,
            metric=metric,
            compute_levels=compute_levels,
            data_check=data_check,
            default_k=default_k,
        )

    def add_outturns(self, df, **kwargs):
        """Add outturns, then normalise the internal ``_aligned`` marker column.

        Genuine new outturn releases never carry the ``_aligned`` marker (it
        is only set by ``_align_outturn_vintages`` for forward-filled
        snapshots). Concatenating them with existing aligned rows can
        introduce NaNs in this Boolean column, which would break
        ``~outturns["_aligned"]`` expressions elsewhere. Treat missing values
        as False (i.e. not aligned).
        """
        super().add_outturns(df, **kwargs)
        if "_aligned" in self._raw_outturns.columns:
            self._raw_outturns["_aligned"] = self._raw_outturns["_aligned"].fillna(False).astype(bool)
        if "_aligned" in self._outturns.columns:
            self._outturns["_aligned"] = self._outturns["_aligned"].fillna(False).astype(bool)

    def add_forecasts(self, df, **kwargs):
        """Add forecasts, aligning outturn vintages to forecast vintages first.

        Alignment mutates the outturns before the parent validates the input, so the
        rollback discards those snapshots if the forecasts are rejected.
        """
        with _atomic_state(self, *_FORECAST_STATE, "_raw_outturns", "_outturns"):
            self._align_outturn_vintages(df)
            super().add_forecasts(df, **kwargs)
            self._set_revision_index_k()
            self._add_days_to_publication()

    def merge(self, other: "ForecastData", compute_levels: bool = True) -> None:
        """Merge another instance into this one, excluding synthetic outturn snapshots.

        The parent implementation passes ``other._raw_outturns`` straight
        through ``add_outturns``, which strips the ``_aligned`` marker
        (it isn't part of the outturn schema). Forward-filled snapshots
        built by ``_align_outturn_vintages`` would then be indistinguishable
        from genuine releases and could corrupt revision indices and
        outturn-revision analyses.

        This override merges only ``other``'s genuine outturn releases
        (``_aligned`` is False or absent). Forecasts are merged as usual;
        ``add_forecasts`` (overridden above) regenerates aligned snapshots
        and recomputes ``k`` / ``days_to_publication`` for the merged data.
        """
        if self._outturn_vintages != other._outturn_vintages:
            raise ValueError("Cannot merge ForecastData instances with different outturn_vintages settings.")

        if not other._raw_outturns.empty:
            genuine_outturns = other._raw_outturns
            if "_aligned" in genuine_outturns.columns:
                genuine_outturns = genuine_outturns[~genuine_outturns["_aligned"]]
            if not genuine_outturns.empty:
                self.add_outturns(genuine_outturns.drop(columns=["_aligned"], errors="ignore"))

        if not other._raw_forecasts.empty:
            # Filter out 'source' from id_columns to get only the extra_ids
            extra_ids = [col for col in other._id_columns if col != "source"] if other._id_columns else None
            extra_ids = extra_ids if extra_ids else None  # Convert empty list to None
            self.add_forecasts(
                other._raw_forecasts, extra_ids=extra_ids, compute_levels=compute_levels, data_check=False
            )

    def clear_filter(self) -> None:
        """Reset the forecasts, main and revisions tables to include all original data.

        Overrides the parent implementation to reapply the nowcast-specific
        ``k`` revision index and ``days_to_publication`` column, which the
        parent's calendar-based rebuild does not preserve.
        """
        super().clear_filter()
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

        For each ``(variable, metric)`` series and forecast vintage *V* that
        has no genuine outturn release, this method builds a
        **point-in-time snapshot** of the outturn data that was available at
        *V*.  For every target date *D* whose outturn had been released by
        *V*, the row with the **latest** outturn ``vintage_date <= V`` is
        selected.  The resulting snapshot is stamped with ``vintage_date = V``.

        These synthetic snapshot rows exist solely to feed the pop/yoy
        transformation pipeline (``transform_forecast_to_levels`` /
        ``transform_series``, which need a vintage-matched, gap-free outturn
        history); they are **not** used for evaluation, since
        ``build_main_table`` strips ``_aligned`` rows before comparing
        forecasts against outturns. History is therefore bounded to the
        window ``prepare_forecasts`` actually consumes rather than kept in
        full.

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

        Snapshots are *built* per ``(variable, metric)`` because different
        variables (and different metrics of the same variable) may have
        different publication lags (e.g. GDP 42 days, CPI 14 days) and
        therefore different outturn vintage calendars. Grouping by ``variable``
        alone would let a vintage that only exists for one metric look as
        though it exists for every metric, and could cause the latest-release
        lookup (one row per ``date``) to mix or drop rows across metrics.

        Which series are *eligible* for alignment is decided by ``variable``
        alone, however. With ``compute_levels=True`` a 'pop' or 'yoy' forecast
        is converted to levels using the levels outturn history, so the levels
        series needs aligning even when no forecast carries ``metric='levels'``.

        Existing snapshots are discarded and rebuilt from every known forecast
        on each call. A snapshot depends on the forecasts at its vintage and on
        which outturns had been released by then, so both a later
        ``add_forecasts`` reaching a deeper horizon and a later ``add_outturns``
        revising history would otherwise leave it stale.

        How much history each snapshot needs is derived from the forecasts
        themselves: the smallest vintage distance for that variable, less the
        ``n_periods + 1`` base rows that ``pct_change`` needs for the
        year-on-year transform.
        """
        if self._raw_outturns.empty:
            return

        # Snapshots depend on every known forecast and on which outturns had been
        # released by each vintage, so rebuild them all rather than patching.
        if "_aligned" in self._raw_outturns.columns:
            self._raw_outturns = self._raw_outturns[~self._raw_outturns["_aligned"]].copy()
        if not self._raw_forecasts.empty:
            forecasts_df = pd.concat([self._raw_forecasts, forecasts_df], ignore_index=True)
        forecasts_df = forecasts_df.assign(vintage_date=pd.to_datetime(forecasts_df["vintage_date"]))

        forecast_vintages = forecasts_df["vintage_date"].unique()
        forecast_variables = set(forecasts_df["variable"])

        # (variable, vintage_date) -> set of target dates the forecasts cover,
        # so we can exclude them from each snapshot. Keyed by variable rather
        # than (variable, metric) because a forecast in any metric can become a
        # levels forecast via ``compute_levels`` and collide with a levels
        # snapshot row for the same date.
        fc_targets = forecasts_df.groupby(["variable", "vintage_date"])["date"].apply(set).to_dict()

        # Deepest vintage distance being added per variable. The outturn
        # history for each snapshot is bounded relative to this distance.
        fc_distances = compute_target_minus_vintage(
            forecasts_df[["variable", "date", "vintage_date", "frequency"]].copy()
        )
        min_fc_distance = fc_distances.groupby("variable")["target_minus_vintage"].min().to_dict()

        n_periods_by_frequency = {"Q": 4, "M": 12}

        new_rows = []
        for (variable, _metric), var_outturns in self._raw_outturns.groupby(["variable", "metric"]):
            # Only align outturn series for variables that have forecasts.
            if variable not in forecast_variables:
                continue
            missing = set(forecast_vintages) - set(pd.to_datetime(var_outturns["vintage_date"]).unique())

            # ``pct_change`` needs n_periods+1 rows of base history below the
            # deepest forecast distance, so don't build older snapshots.
            frequencies = var_outturns["frequency"].unique()
            if len(frequencies) > 1:
                raise ValueError(f"Expected a single frequency for variable {variable!r}, got {sorted(frequencies)}.")
            freq = frequencies[0]
            n_periods = n_periods_by_frequency.get(freq)
            min_distance = None if n_periods is None else min_fc_distance[variable] - (n_periods + 1)

            for vintage in sorted(missing):
                # All outturn rows for this series released on or before *vintage*
                available = var_outturns[var_outturns["vintage_date"] <= vintage]
                if available.empty:
                    continue

                if min_distance is not None:
                    release_vintages = available["vintage_date"].copy()
                    available = compute_target_minus_vintage(available.assign(vintage_date=pd.Timestamp(vintage)))
                    available["vintage_date"] = release_vintages
                    available = available[available["target_minus_vintage"] >= min_distance]
                    if available.empty:
                        continue

                # For each target date, keep the row from the latest outturn
                # vintage — this is the best estimate available at time *vintage*.
                snapshot = available.loc[available.groupby("date")["vintage_date"].idxmax()].copy()

                # Exclude dates that the forecasts already cover at this
                # vintage.  This avoids duplicate (variable, metric,
                # vintage_date, date) rows when the outturn + forecast
                # DataFrames are concatenated inside prepare_forecasts.
                snapshot = snapshot[~snapshot["date"].isin(fc_targets.get((variable, vintage), set()))]
                if snapshot.empty:
                    continue

                snapshot["vintage_date"] = vintage
                new_rows.append(snapshot)

        from forecast_evaluation.core.transformations import prepare_outturns

        if "_aligned" not in self._raw_outturns.columns:
            self._raw_outturns = self._raw_outturns.assign(_aligned=False)
        if new_rows:
            expanded = pd.concat(new_rows, ignore_index=True)
            expanded = compute_target_minus_vintage(expanded)
            expanded["_aligned"] = True
            self._raw_outturns = pd.concat([self._raw_outturns, expanded], ignore_index=True)
        self._outturns = prepare_outturns(self._raw_outturns)

    def _add_days_to_publication(self):
        """Add days_to_publication column to the main table.

        Computed as the number of days between the outturn vintage and the
        forecast vintage. A larger value means the forecast was made further
        from publication; the value decreases (and can become negative for
        backcasts) as the vintage approaches the outturn release.

        Not to be confused with ``days_to_period_end`` used by intra-period
        analysis, which measures the distance to the end of the target period
        rather than to the outturn release.
        """
        if self._main_table.empty:
            return
        if "vintage_date_forecast" in self._main_table.columns and "vintage_date_outturn" in self._main_table.columns:
            self._main_table["days_to_publication"] = (
                pd.to_datetime(self._main_table["vintage_date_outturn"])
                - pd.to_datetime(self._main_table["vintage_date_forecast"])
            ).dt.days

    @property
    def uses_intra_period_vintages(self) -> bool:
        """Whether forecast vintages are intra-period releases for the same target period."""
        return True

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

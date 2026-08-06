"""Helpers for fixed-lead-time forecast paths.

These functions select and optionally plot a comparable out-of-sample forecast
path: for each target period, use the forecast made a fixed number of days
before that period's first release.
"""

from __future__ import annotations

import pandas as pd


def get_lead_time_forecast_path(
    data,
    variable: str,
    days_before_release: int,
    source: str | None = None,
    metric: str | None = None,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Select a fixed-lead-time forecast path.

    For each target date, the first release date is inferred from
    ``data.outturns`` as the earliest ``vintage_date`` for ``variable`` and
    ``date``. Forecasts are then selected from ``data.forecasts`` at a fixed
    distance from that release date.

    Parameters
    ----------
    data : object
        Object with ``outturns`` and ``forecasts`` DataFrame attributes, such as
        ``forecast_evaluation.ForecastData``.
    variable : str
        Target variable to select, for example ``"GDP"``.
    days_before_release : int
        Desired number of calendar days before first release.
    source : str or None, optional
        Forecast source/model to keep. If ``None`` all sources are retained.
    metric : str or None, optional
        Forecast metric to keep. If ``None`` all metrics are retained.
    frequency : str or None, optional
        Frequency to keep, for example ``"Q"``. If ``None`` all frequencies are
        retained.
    Returns
    -------
    pd.DataFrame
        Selected forecasts, sorted by target date, with additional columns:
        ``release_date``, ``target_vintage_date``,
        ``actual_days_before_release`` and ``vintage_distance_days``.
    """
    if not isinstance(days_before_release, int) or days_before_release < 0:
        raise ValueError("days_before_release must be a non-negative integer")

    outturns = _copy_with_datetime_columns(
        data.outturns, "outturns"
    )  # just makes dates datetimes
    forecasts = _copy_with_datetime_columns(data.forecasts, "forecasts")

    _require_columns(
        outturns, {"date", "variable", "vintage_date"}, "outturns"
    )  # checks date variable and vintage all avail
    _require_columns(forecasts, {"date", "variable", "vintage_date"}, "forecasts")

    outturns = outturns[
        outturns["variable"] == variable
    ].copy()  # creates outturns of mgsx.q
    forecasts = forecasts[
        forecasts["variable"] == variable
    ].copy()  # creates forecast of mgsx.q

    if source is not None:
        _require_columns(forecasts, {"source"}, "forecasts")
        forecasts = forecasts[forecasts["source"] == source].copy()  # filters by source
    if metric is not None:
        _require_columns(forecasts, {"metric"}, "forecasts")
        forecasts = forecasts[forecasts["metric"] == metric].copy()  # filters by metric
        # Keep release-date inference on the same metric slice when available.
        if "metric" in outturns.columns:
            outturns = outturns[
                outturns["metric"] == metric
            ].copy()  # filters by metric
    if frequency is not None:
        if "frequency" in outturns.columns:
            outturns = outturns[
                outturns["frequency"] == frequency
            ].copy()  # filters by frequency
        if "frequency" in forecasts.columns:
            forecasts = forecasts[
                forecasts["frequency"] == frequency
            ].copy()  # filters by frequency

    if outturns.empty:
        return _empty_fixed_offset_result(forecasts)
    if forecasts.empty:
        return _empty_fixed_offset_result(forecasts)

    release_dates = (
        outturns.groupby(["variable", "date"], as_index=False)["vintage_date"]
        .min()
        .rename(columns={"vintage_date": "release_date"})
    )  # find earliest vintage date for each variable and date, rename to release_date - i.e. the earliest vintage that March data was avail would be May

    selected = forecasts.merge(release_dates, on=["variable", "date"], how="inner")
    if selected.empty:
        return _empty_fixed_offset_result(forecasts)

    selected["target_vintage_date"] = selected["release_date"] - pd.to_timedelta(
        days_before_release, unit="D"
    )
    selected["actual_days_before_release"] = (
        selected["release_date"] - selected["vintage_date"]
    ).dt.days
    selected["vintage_distance_days"] = (
        selected["vintage_date"] - selected["target_vintage_date"]
    ).dt.days
    selected = selected[selected["vintage_distance_days"] <= 0].copy()
    sort_cols = _group_columns(selected) + ["vintage_date"]
    selected = selected.sort_values(sort_cols)
    selected = selected.groupby(_group_columns(selected), as_index=False).tail(1)

    return selected.sort_values(_sort_columns(selected)).reset_index(drop=True)


def plot_lead_time_forecast_path(
    data,
    variable: str,
    days_before_release: int,
    source: str | None = None,
    metric: str | None = None,
    frequency: str | None = None,
    actuals: bool = True,
    ax=None,
    **plot_kwargs,
):
    """Plot a fixed-lead-time forecast path.

    Returns the matplotlib ``Axes`` object. ``matplotlib`` is imported lazily so
    the package can still be used without plotting dependencies installed.
    """
    path = get_lead_time_forecast_path(
        data=data,
        variable=variable,
        days_before_release=days_before_release,
        source=source,
        metric=metric,
        frequency=frequency,
    )

    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "plot_lead_time_forecast_path requires matplotlib. "
                "Install matplotlib to use plotting helpers."
            ) from exc
        _, ax = plt.subplots()

    label = plot_kwargs.pop("label", None)
    if source is not None:
        label = label or source
        ax.plot(path["date"], path["value"], label=label, **plot_kwargs)
    else:
        for group_label, group in path.groupby("source", dropna=False):
            ax.plot(
                group["date"], group["value"], label=str(group_label), **plot_kwargs
            )

    if actuals:
        actual_path = _latest_outturn_path(
            data.outturns,
            variable=variable,
            frequency=frequency,
        )
        if not actual_path.empty:
            ax.plot(
                actual_path["date"],
                actual_path["value"],
                color="black",
                linestyle="--",
                label=f"{variable} actual",
            )

    ax.set_title(f"{variable}: {days_before_release} days before release")
    ax.set_xlabel("Date")
    ax.set_ylabel(variable if metric is None else f"{variable} ({metric})")
    ax.legend()
    return ax


def _copy_with_datetime_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"data.{name} must be a pandas DataFrame")
    out = df.copy()
    for column in ("date", "vintage_date"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column]).dt.normalize()
    return out


def _require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"data.{name} is missing required columns: {sorted(missing)}")


def _group_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "date",
        "variable",
        "source",
        "metric",
        "frequency",
    ]
    return [column for column in candidates if column in df.columns]


def _sort_columns(df: pd.DataFrame) -> list[str]:
    candidates = ["date", "source", "metric", "frequency", "forecast_horizon"]
    return [column for column in candidates if column in df.columns]


def _empty_fixed_offset_result(forecasts: pd.DataFrame) -> pd.DataFrame:
    extra_columns = [
        "release_date",
        "target_vintage_date",
        "actual_days_before_release",
        "vintage_distance_days",
    ]
    columns = list(forecasts.columns)
    columns.extend(column for column in extra_columns if column not in columns)
    return pd.DataFrame(columns=columns)


def _latest_outturn_path(
    outturns: pd.DataFrame,
    variable: str,
    frequency: str | None = None,
) -> pd.DataFrame:
    actuals = _copy_with_datetime_columns(outturns, "outturns")
    actuals = actuals[actuals["variable"] == variable].copy()
    if frequency is not None and "frequency" in actuals.columns:
        actuals = actuals[actuals["frequency"] == frequency].copy()
    if actuals.empty:
        return actuals
    return (
        actuals.sort_values("vintage_date")
        .drop_duplicates(subset=["date", "variable"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

import numpy as np
import pandas as pd


def create_sample_outturns() -> pd.DataFrame:
    """Create deterministic quarterly index-level outturns for examples.

    Return 26 rows: 13 quarter-end targets from 2022-03-31 to 2025-03-31
    for each of ``cpisa`` and ``gdpkp``. Both variables have index levels
    100 through 112, increasing by one per quarter, frequency ``Q``, and
    a single outturn vintage of 2025-09-30. No random numbers are used.
    """
    df_outturn_1 = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=13, freq="QE"),
            "variable": ["cpisa"] * 13,
            "vintage_date": pd.to_datetime("2025-09-30"),
            "frequency": ["Q"] * 13,
            "value": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        }
    )

    df_outturn_2 = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=13, freq="QE"),
            "variable": ["gdpkp"] * 13,
            "vintage_date": pd.to_datetime("2025-09-30"),
            "frequency": ["Q"] * 13,
            "value": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        }
    )

    df_outturn = pd.concat([df_outturn_1, df_outturn_2], ignore_index=True)

    return df_outturn


def create_sample_forecasts() -> pd.DataFrame:
    """Create deterministic quarterly index-level point forecasts.

    Return 13 quarter-end targets from 2022-03-31 to 2025-03-31 for
    ``gdpkp`` from source ``mpr2``. Index levels run from 100 through 112
    and explicit forecast horizons run from 0 through 12. All rows have
    frequency ``Q`` and vintage 2025-03-31, independent of target dates.
    No random numbers are used.
    """
    df_forecasts = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=13, freq="QE"),
            "variable": ["gdpkp"] * 13,
            "vintage_date": pd.to_datetime("2025-03-31"),
            "source": ["mpr2"] * 13,
            "frequency": ["Q"] * 13,
            "value": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        }
    )

    df_forecasts["forecast_horizon"] = list(range(len(df_forecasts)))

    return df_forecasts


def create_sample_nowcast_outturns() -> pd.DataFrame:
    """Create sample outturns DataFrame for nowcasting tests and examples.

    Creates quarterly outturns for two variables (``gdp`` and ``cpi``) with
    multiple outturn vintages, covering 2020Q1 to 2025Q4 (24 quarters).

    Publication lags and revision frequencies are realistic:

    - **gdp**: first release 6 weeks (42 days) after the end of the target
      quarter, then revised once per quarter at each subsequent quarter-end.
    - **cpi**: first release 2 weeks (14 days) after the end of the target
      quarter, then revised monthly (14 days after each subsequent month-end).

    For each target quarter, the outturn data has:

    - A **first-release vintage** at the publication date.
    - Subsequent revision vintages (quarterly for GDP, monthly for CPI) up to
      2026-03-31, with white noise shrinking as the vintage matures.

    For zero-based target index ``i``, the underlying index-level paths are
    ``99 + 0.3*i + 0.5*sin(i/4)`` for GDP and
    ``98 + 0.45*i + 0.8*cos(i/4)`` for CPI. Each release adds independent
    zero-mean normal noise with standard deviation ``0.5 / (1 + abs(k))``,
    where ``k`` is the vintage quarter minus the target quarter, and values
    are rounded to four decimal places. Revision month-ends stop at
    2026-03-31; CPI's 14-day release lag extends vintages to 2026-04-14.
    A fresh local ``default_rng(42)`` makes calls repeatable without changing
    NumPy's global random state.

    The realistic first-release dates mean that intra-quarter nowcast vintages
    before the publication date have no backcast (h=-1) outturn available for
    pairing, while later vintages within that quarter do.

    Returns
    -------
    pd.DataFrame
        DataFrame with outturn data suitable for nowcasting evaluation.
    """
    rng = np.random.default_rng(42)
    target_dates = pd.date_range("2020-03-31", periods=24, freq="QE")

    # True outturn values
    truth = {
        "gdp": {d: 99.0 + (i * 0.3 + np.sin(i / 4) * 0.5) for i, d in enumerate(target_dates)},
        "cpi": {d: 98.0 + (i * 0.45 + np.cos(i / 4) * 0.8) for i, d in enumerate(target_dates)},
    }

    # Publication lag in days after quarter-end:
    #   gdp: 6 weeks = 42 days; cpi: 2 weeks = 14 days
    pub_lag = {"gdp": 42, "cpi": 14}

    rows = []
    for variable, values in truth.items():
        lag = pub_lag[variable]

        for target_date, true_value in values.items():
            first_release = target_date + pd.Timedelta(days=lag)

            if variable == "cpi":
                # CPI is revised monthly, 14 days after each month-end.
                first_release_me = first_release + pd.offsets.MonthEnd(0)
                if first_release_me <= first_release:
                    first_release_me = first_release + pd.offsets.MonthEnd(1)
                revision_month_ends = pd.date_range(first_release_me + pd.offsets.MonthEnd(1), "2026-03-31", freq="ME")
                revision_vintages = revision_month_ends + pd.Timedelta(days=14)
            else:
                # GDP is revised once per quarter at each subsequent quarter-end.
                first_release_qe = first_release + pd.offsets.QuarterEnd(0)
                if first_release_qe <= first_release:
                    first_release_qe = first_release + pd.offsets.QuarterEnd(1)
                revision_vintages = pd.date_range(first_release_qe + pd.offsets.QuarterEnd(1), "2026-03-31", freq="QE")

            target_vintages = [first_release] + list(revision_vintages)

            for vintage_date in target_vintages:
                k = (vintage_date.to_period("Q") - target_date.to_period("Q")).n
                noise = rng.normal(0, 0.5 / (1 + abs(k)))
                rows.append(
                    {
                        "date": target_date,
                        "variable": variable,
                        "vintage_date": vintage_date,
                        "frequency": "Q",
                        "value": round(true_value + noise, 4),
                    }
                )

    df_outturn = pd.DataFrame(rows)
    return df_outturn


def create_sample_nowcast_forecasts() -> pd.DataFrame:
    """Create sample nowcasting forecasts with weekly vintage dates.

    Generates index-level forecasts for 24 quarterly targets from 2020Q1 to
    2025Q4, for ``gdp`` and ``cpi`` from ``nowcast_dfm`` and ``nowcast_bridge``.
    Weekly vintages run every seven days from 2020-01-01 through 2026-03-31.
    Target quarter offsets are ``(-1, 0, 1)`` relative to the vintage quarter:
    previous, current, and next quarter. The previous-quarter path stops at
    publication (GDP: target end + 42 days; CPI: target end + 14 days).
    Explicit ``forecast_horizon`` is ``max(offset, 0)``, so both the previous
    and current quarter use horizon zero.

    Truth is ``99 + 0.3*i + 0.5*sin(i/4)`` for GDP and
    ``98 + 0.45*i + 0.8*cos(i/4)`` for CPI, with zero-based target index ``i``.
    Initial biases (GDP, CPI) are (2.0, -2.5) for the DFM and (-1.5, 3.0)
    for the bridge. Set ``r = min(1, max(0, days_before_publication) / 182)``.
    Each forecast is truth plus ``bias*r`` plus independent zero-mean normal
    noise of standard deviation ``0.25*abs(bias)*r``, rounded to two decimal
    places. Errors converge to zero at publication, not at target quarter-end.
    A fresh local ``default_rng(42)`` makes calls repeatable without changing
    NumPy's global random state.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, variable, vintage_date, source,
        frequency, value, and the explicitly supplied ``forecast_horizon``.

    Examples
    --------
    >>> df = create_sample_nowcast_forecasts()
    >>> df["source"].unique()
    array(['nowcast_dfm', 'nowcast_bridge'], dtype=object)

    """
    rng = np.random.default_rng(42)

    # 24 target quarters: 2020Q1 – 2025Q4
    target_dates = pd.date_range("2020-03-31", periods=24, freq="QE")

    # Map quarter period → quarter-end date for fast lookup
    target_by_period = {d.to_period("Q"): d for d in target_dates}

    # True outturn levels (same as create_sample_nowcast_outturns)
    truth = {
        "gdp": {d: 99.0 + (i * 0.3 + np.sin(i / 4) * 0.5) for i, d in enumerate(target_dates)},
        "cpi": {d: 98.0 + (i * 0.45 + np.cos(i / 4) * 0.8) for i, d in enumerate(target_dates)},
    }

    initial_bias = {
        "nowcast_dfm": {"gdp": 2.0, "cpi": -2.5},
        "nowcast_bridge": {"gdp": -1.5, "cpi": 3.0},
    }

    # Publication lag in days after quarter-end:
    #   gdp: 6 weeks = 42 days; cpi: 2 weeks = 14 days
    pub_lag = {"gdp": 42, "cpi": 14}

    max_days = 182.0  # reference scale (~2 quarters) for remaining-days fraction
    noise_scale = 0.25  # noise std as a fraction of |bias * remaining_fraction|

    # Weekly vintage dates: from the first week of 2020 through to after the
    # last publication date so all backcasts are fully covered.
    all_vintages = pd.date_range("2020-01-01", "2026-03-31", freq="7D")

    rows = []
    for source, biases in initial_bias.items():
        for variable, bias in biases.items():
            lag = pub_lag[variable]

            for vintage in all_vintages:
                v_period = vintage.to_period("Q")

                for target_offset in (-1, 0, 1):
                    # Resolve the target quarter for this horizon
                    target_period = v_period + target_offset
                    if target_period not in target_by_period:
                        continue
                    target = target_by_period[target_period]

                    pub_date = target + pd.Timedelta(days=lag)

                    # Stop the previous-period path once its official outturn is published.
                    if target_offset == -1 and vintage >= pub_date:
                        continue

                    # Unified error model: error decreases monotonically as the
                    # vintage approaches the publication date, across all horizons.
                    # At pub_date, days_before_pub=0 and error→0.
                    days_before_pub = max(0, (pub_date - vintage).days)
                    remaining_fraction = min(1.0, days_before_pub / max_days)
                    error = bias * remaining_fraction
                    noise = rng.normal(0, noise_scale * abs(bias) * remaining_fraction)

                    rows.append(
                        {
                            "date": target,
                            "variable": variable,
                            "vintage_date": vintage,
                            "source": source,
                            "frequency": "Q",
                            "forecast_horizon": max(target_offset, 0),
                            "value": round(truth[variable][target] + error + noise, 2),
                        }
                    )

    df = pd.DataFrame(rows)
    return df


def create_sample_density_forecasts() -> pd.DataFrame:
    """Create sample density forecasts DataFrame with multiple quantiles.

    Return 13 quarterly targets from 2025-12-31 through 2028-12-31 with
    explicit horizons 0 through 12, variable ``gdpkp``, source ``mpr2``,
    frequency ``Q``, and vintage 2025-12-31. Values are index levels aligned
    with the point/outturn examples, with a normal mean of ``115 + horizon``
    and constant standard deviation 0.5 at every horizon.

    Draw 10,000 independent normal samples per horizon, then extract 50
    noncrossing empirical quantiles equally spaced from 0.01 to 0.99.
    This even-sized grid does not contain 0.5; nearest-quantile conversion
    is needed to obtain a median path. Return 650 rows sorted by date and
    quantile. A fresh local ``default_rng(42)`` makes calls repeatable without
    changing NumPy's global random state.

    Returns
    -------
    pd.DataFrame
        DataFrame with density forecast data including a 'quantile' column.
        Contains forecasts for 50 quantiles from 0.01 to 0.99.

    Examples
    --------
    >>> df = create_sample_density_forecasts()
    >>> len(df['quantile'].unique())
    50
    >>> # Verify quantiles don't cross for a given date
    >>> sample = df[df['date'] == df['date'].iloc[0]].sort_values('quantile')
    >>> assert all(sample['value'].diff().dropna() > 0)  # Values strictly increasing
    """
    rng = np.random.default_rng(42)

    quantiles = np.linspace(0.01, 0.99, 50)

    # Create base structure
    dates = pd.date_range(start="2025-12-31", periods=13, freq="QE")
    vintage_date = pd.to_datetime("2025-12-31")

    density_forecasts = []

    # For each forecast horizon
    for i, date in enumerate(dates):
        # Mean value increases with time (trend)
        mean = 115 + i

        std = 0.5

        # Generate 10000 samples from normal distribution for this horizon
        samples = rng.normal(mean, std, 10000)

        # Extract quantiles from the samples
        for quantile in quantiles:
            quantile_value = np.quantile(samples, quantile)

            density_forecasts.append(
                {
                    "date": date,
                    "variable": "gdpkp",
                    "vintage_date": vintage_date,
                    "source": "mpr2",
                    "frequency": "Q",
                    "quantile": quantile,
                    "value": quantile_value,
                    "forecast_horizon": i,
                }
            )

    result = pd.DataFrame(density_forecasts)

    # Sort by date and quantile for clarity
    result = result.sort_values(["date", "quantile"]).reset_index(drop=True)

    return result

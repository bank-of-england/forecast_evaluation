import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_sample_outturns() -> pd.DataFrame:
    """Create sample outturns DataFrame for testing and examples."""
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

    df_outturn["forecast_horizon"] = (
        df_outturn["date"].dt.to_period("Q") - df_outturn["vintage_date"].dt.to_period("Q")
    ).apply(lambda x: x.n)

    return df_outturn


def create_sample_forecasts() -> pd.DataFrame:
    """Create sample forecasts DataFrame for testing and examples."""
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

    df_forecasts["forecast_horizon"] = (
        df_forecasts["date"].dt.to_period("Q") - df_forecasts["vintage_date"].dt.to_period("Q")
    ).apply(lambda x: x.n)

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

    The realistic first-release dates mean that intra-quarter nowcast vintages
    before the publication date have no backcast (h=-1) outturn available for
    pairing, while later vintages within that quarter do.

    Returns
    -------
    pd.DataFrame
        DataFrame with outturn data suitable for nowcasting evaluation.
    """
    np.random.seed(42)
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
                noise = np.random.normal(0, 0.5 / (1 + abs(k)))
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
    df_outturn["forecast_horizon"] = (
        df_outturn["date"].dt.to_period("Q") - df_outturn["vintage_date"].dt.to_period("Q")
    ).apply(lambda x: x.n)

    return df_outturn


def create_sample_nowcast_forecasts() -> pd.DataFrame:
    """Create sample nowcasting forecasts with weekly vintage dates.

    Generates nowcasts (2020-2025) from two models for two variables (``gdp``
    and ``cpi``). Vintages are produced once per week across the full sample.
    Each vintage targets two consecutive releases:

    - **h = 1** (forecast): the next quarter (not yet started or in progress).
    - **h = 0** (nowcast): the current quarter (in progress).
    - **h = -1** (backcast): the previous quarter, but **only** while its
      official outturn has not yet been published (gdp: 6 weeks after
      quarter-end; cpi: 2 weeks after quarter-end). Once the data is
      released, backcasting for that quarter stops.

    Forecast errors converge toward zero as the vintage date approaches the
    end of the target quarter, with additive noise so that convergence is
    realistic rather than perfectly monotonic.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, variable, vintage_date, source,
        frequency, value.  ``forecast_horizon`` is intentionally omitted so
        that :meth:`NowcastData.add_forecasts` computes the integer-period
        horizon automatically.

    Examples
    --------
    >>> df = create_sample_nowcast_forecasts()
    >>> df["source"].unique()
    array(['nowcast_dfm', 'nowcast_bridge'], dtype=object)

    Visualize how forecasts evolve over weekly vintages:

    >>> # Import the plotting function from the end of this module
    >>> import matplotlib.pyplot as plt
    >>> from forecast_evaluation.data.sample_data import plot_sample_nowcasts
    >>> plot_sample_nowcasts()  # Plots forecasts by horizon, colored by vintage date
    """
    np.random.seed(42)

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

                for horizon in (-1, 0, 1):
                    # Resolve the target quarter for this horizon
                    target_period = v_period + horizon
                    if target_period not in target_by_period:
                        continue
                    target = target_by_period[target_period]

                    pub_date = target + pd.Timedelta(days=lag)

                    # h=-1: stop once the official outturn has been published
                    if horizon == -1 and vintage >= pub_date:
                        continue

                    # Unified error model: error decreases monotonically as the
                    # vintage approaches the publication date, across all horizons.
                    # At pub_date, days_before_pub=0 and error→0.
                    days_before_pub = max(0, (pub_date - vintage).days)
                    remaining_fraction = min(1.0, days_before_pub / max_days)
                    error = bias * remaining_fraction
                    noise = np.random.normal(0, noise_scale * abs(bias) * remaining_fraction)

                    rows.append(
                        {
                            "date": target,
                            "variable": variable,
                            "vintage_date": vintage,
                            "source": source,
                            "frequency": "Q",
                            "value": round(truth[variable][target] + error + noise, 2),
                        }
                    )

    df = pd.DataFrame(rows)
    return df


def create_sample_density_forecasts() -> pd.DataFrame:
    """Create sample density forecasts DataFrame with multiple quantiles.

    Generates forecasts by sampling from a normal distribution for each horizon,
    then extracts 100 quantiles from the samples. This ensures quantiles don't cross.

    Returns
    -------
    pd.DataFrame
        DataFrame with density forecast data including a 'quantile' column.
        Contains forecasts for 100 quantiles from 0.01 to 0.99.

    Examples
    --------
    >>> df = create_sample_density_forecasts()
    >>> len(df['quantile'].unique())
    100
    >>> # Verify quantiles don't cross for a given date
    >>> sample = df[df['date'] == df['date'].iloc[0]].sort_values('quantile')
    >>> assert all(sample['value'].diff().dropna() > 0)  # Values strictly increasing
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define 100 quantiles from 0.01 to 0.99
    quantiles = np.linspace(0.01, 0.99, 50)

    # Create base structure
    dates = pd.date_range(start="2025-12-31", periods=13, freq="QE")
    vintage_date = pd.to_datetime("2025-12-31")

    density_forecasts = []

    # For each forecast horizon
    for i, date in enumerate(dates):
        # Mean value increases with time (trend)
        mean = i + 1

        # Standard deviation proportional to mean (5% uncertainty)
        std = 0.5

        # Generate 10000 samples from normal distribution for this horizon
        samples = np.random.normal(mean, std, 10000)

        # Extract quantiles from the samples
        for quantile in quantiles:
            quantile_value = np.quantile(samples, quantile)

            forecast_horizon = (pd.Period(date, freq="Q") - pd.Period(vintage_date, freq="Q")).n

            density_forecasts.append(
                {
                    "date": date,
                    "variable": "gdpkp",
                    "vintage_date": vintage_date,
                    "source": "mpr2",
                    "frequency": "Q",
                    "quantile": quantile,
                    "value": quantile_value,
                    "forecast_horizon": forecast_horizon,
                }
            )

    result = pd.DataFrame(density_forecasts)

    # Sort by date and quantile for clarity
    result = result.sort_values(["date", "quantile"]).reset_index(drop=True)

    return result


def plot_sample_nowcasts() -> None:
    """Plot sample nowcast forecasts and outturns on the same axes.

    Generates a single plot showing how forecast and outturn values evolve over
    target dates, with each forecast/outturn vintage colored differently:

    - **Nowcasts** (forecast vintages): Lines with circle markers, colored by
      viridis colormap. Shows how weekly forecasts are produced and how they
      converge as time passes.
    - **Outturns** (publication vintages): Lines with square markers, colored by
      plasma colormap. Illustrates publication timing and how values are revised
      over time as data matures.

    Examples
    --------
    >>> plot_sample_nowcasts()  # Displays a single matplotlib figure
    """
    fc = create_sample_nowcast_forecasts()
    out = create_sample_nowcast_outturns()

    variable = "gdp"
    model = "nowcast_dfm"

    # ========== Prepare forecast data ==========
    fc_var = fc[fc["variable"] == variable].copy()
    fc_var_model = fc_var[fc_var["source"] == model]

    unique_vintages_all = sorted(fc_var_model["vintage_date"].unique())
    # Show all weekly forecast vintages with transparency
    selected_vintages = unique_vintages_all
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_vintages)))
    vintage_colors = dict(zip(selected_vintages, colors))

    # ========== Prepare outturn data ==========
    out_var = out[out["variable"] == variable].copy()
    unique_vintages_out = sorted(out_var["vintage_date"].unique())
    step_out = max(1, len(unique_vintages_out) // 10)
    selected_out_vintages = unique_vintages_out[::step_out]
    colors_out = plt.cm.plasma(np.linspace(0, 1, len(selected_out_vintages)))
    out_vintage_colors = dict(zip(selected_out_vintages, colors_out))

    # ========== Single combined figure ==========
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        f"{variable.upper()}: Nowcasts and Outturns over time (colored by vintage)",
        fontsize=13,
        fontweight="bold",
    )

    # Plot forecasts (all weekly vintages with transparency)
    for i, vintage in enumerate(selected_vintages):
        fc_v = fc_var_model[fc_var_model["vintage_date"] == vintage].sort_values("date")
        # Only add label for first forecast to avoid cluttered legend
        label = "Forecast vintages (weekly)" if i == 0 else None
        ax.plot(
            fc_v["date"],
            fc_v["value"],
            marker="o",
            markersize=3,
            alpha=0.15,
            label=label,
            color=vintage_colors[vintage],
            linewidth=1,
        )

    # Plot outturns
    for vintage in selected_out_vintages:
        out_v = out_var[out_var["vintage_date"] == vintage].sort_values("date")
        ax.plot(
            out_v["date"],
            out_v["value"],
            marker="s",
            markersize=5,
            alpha=0.7,
            label=f"OT {vintage.date()}",
            color=out_vintage_colors[vintage],
            linewidth=1.5,
        )

    ax.set_xlabel("Target date (quarter-end)", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Create legend with handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Already have forecast (from single label) and outturn entries in handles/labels
    # Just add a separator line for clarity
    from matplotlib.lines import Line2D

    sep = Line2D([0], [0], color="none")
    sep_label = ""

    legend_handles = handles + [sep]
    legend_labels = labels + [sep_label]

    ax.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8,
        ncol=1,
    )

    fig.autofmt_xdate(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_sample_nowcasts()

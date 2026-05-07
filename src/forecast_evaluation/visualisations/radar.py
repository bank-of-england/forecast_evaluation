"""Radar (spider) chart visualisation for comparing forecast performance."""

from typing import TYPE_CHECKING, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_evaluation.utils import clean_unique_id
from forecast_evaluation.visualisations.theme import THEME

if TYPE_CHECKING:
    from forecast_evaluation.data import ForecastData
    from forecast_evaluation.tests.results import TestResult


def _normalise_series(series: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 1] range.

    If all values are identical the series is set to 0.5.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def plot_radar(
    df: Union[pd.DataFrame, "TestResult", "ForecastData"],
    mode: Literal["metrics", "variables", "tests"],
    *,
    # Fixed parameters (which ones are needed depends on ``mode``)
    variable: Optional[str] = None,
    variables: Optional[list[str]] = None,
    metric: Optional[str] = None,
    horizon: Optional[int] = None,
    frequency: Union[Literal["Q", "M"], None] = None,
    statistic: Literal["rmse", "rmedse", "mean_abs_error"] = "rmse",
    k: int = 12,
    test_type: Literal["accuracy", "bias", "efficiency"] = "accuracy",
    bias_type: Literal["mean", "mz"] = "mean",
    efficiency_type: Literal["revision_predictability", "revisions_errors"] = "revision_predictability",
    normalise: bool = True,
    return_plot: bool = False,
):
    """Create a radar / spider chart of forecast accuracy statistics.

    The chart can be configured in three modes that determine what appears on
    each edge of the radar:

    ``"metrics"``
        Edges are accuracy statistics (RMSE, RMedSE, MAE) for a
        **fixed variable and horizon**.  Each trace is a different *source*.
        ``df`` must be an accuracy ``TestResult`` or DataFrame.

    ``"variables"``
        Edges are variables for a **fixed metric and horizon**.
        Each trace is a different *source*.  The value plotted per variable
        depends on ``test_type``.
        ``df`` must be a ``ForecastData`` object (or a ``TestResult`` when
        ``test_type='accuracy'``).

    ``"tests"``
        Edges are test categories – Accuracy, Bias, and Efficiency –
        for a **fixed variable, metric and horizon**.  Each trace is a
        different *source*.  ``df`` must be a ``ForecastData`` object.

    Parameters
    ----------
    df : DataFrame, TestResult, or ForecastData
        For ``"metrics"`` mode: accuracy results (``TestResult`` or DataFrame).
        For ``"variables"`` / ``"tests"`` modes: a ``ForecastData`` object
        (or ``TestResult`` when ``test_type='accuracy'``).
    mode : {"metrics", "variables", "tests"}
        Determines what is plotted on the radar edges.
    variable : str, optional
        Variable to fix (required for ``"metrics"`` and ``"tests"`` modes).
    variables : list of str, optional
        Subset of variables to show on the edges in ``"variables"`` mode.
        If None, all available variables are shown.
    metric : str, optional
        Metric to fix (required for ``"variables"`` and ``"tests"`` modes).
    horizon : int, optional
        Forecast horizon to fix (required for all modes).
    frequency : {"Q", "M"} or None, default None
        Data frequency filter. If None, inferred from the data.
    statistic : str, default "rmse"
        Accuracy statistic to use when ``test_type='accuracy'``.
    k : int, default 12
        Number of revisions used to define outturns.
    test_type : {"accuracy", "bias", "efficiency"}, default "accuracy"
        Which test to evaluate in ``"variables"`` mode:

        - ``"accuracy"``: the selected ``statistic`` from accuracy results.
        - ``"bias"``: bias measure (see ``bias_type``).
        - ``"efficiency"``: efficiency measure (see ``efficiency_type``).
    bias_type : {"mean", "mz"}, default "mean"
        Bias definition (used in ``"tests"`` mode and
        ``"variables"`` mode with ``test_type='bias'``):

        - ``"mean"``: absolute mean forecast error from ``bias_analysis``.
        - ``"mz"``: Mincer-Zarnowitz joint test p-value from
          ``weak_efficiency_analysis``.
    efficiency_type : {"revision_predictability", "revisions_errors"}, default "revision_predictability"
        Efficiency definition (used in ``"tests"`` mode and
        ``"variables"`` mode with ``test_type='efficiency'``):

        - ``"revision_predictability"``: Mincer-Zarnowitz weak-efficiency
          joint test p-value from ``weak_efficiency_analysis``.
        - ``"revisions_errors"``: p-value of the slope coefficient from
          ``revisions_errors_correlation_analysis``.
    normalise : bool, default True
        If True, values are min-max normalised **per edge** so that
        different scales become comparable.
    return_plot : bool, default False
        If True return ``(fig, ax)`` instead of displaying the plot.

    Returns
    -------
    tuple of (Figure, PolarAxes) or None
    """
    if frequency is None:
        if hasattr(df, "_main_table") and df._main_table is not None:
            _freq_col = df._main_table["frequency"]
        elif hasattr(df, "to_df"):
            _freq_col = df.to_df()["frequency"]
        else:
            _freq_col = df["frequency"]
        inferred = _freq_col.unique()
        if len(inferred) != 1:
            raise ValueError(
                f"Could not infer a unique frequency from data; found: {list(inferred)}. "
                "Please specify the 'frequency' argument explicitly."
            )
        frequency = inferred[0]

    # ------------------------------------------------------------------
    # Build pivot table depending on mode
    # ------------------------------------------------------------------
    if mode == "metrics":
        # Spokes = accuracy statistics, traces = sources
        if hasattr(df, "to_df"):
            df = df.to_df()
        df = df.loc[df["frequency"] == frequency].copy()

        if variable is None or horizon is None:
            raise ValueError("`variable` and `horizon` are required for mode='metrics'")
        mask = (df["variable"] == variable) & (df["forecast_horizon"] == horizon)
        sub = df.loc[mask]
        if sub.empty:
            raise ValueError(f"No data for variable={variable!r}, horizon={horizon}")

        stat_cols = ["rmse", "rmedse", "mean_abs_error"]
        stat_labels = ["RMSE", "RMedSE", "MAE"]
        pivot = sub.set_index("unique_id")[stat_cols].copy()
        pivot.columns = stat_labels
        title = f"Accuracy metrics – {variable} (h={horizon})"

    elif mode == "variables":
        # Edges = variables, traces = sources
        if metric is None or horizon is None:
            raise ValueError("`metric` and `horizon` are required for mode='variables'")

        from forecast_evaluation.tests.accuracy import compute_accuracy_statistics
        from forecast_evaluation.tests.bias import bias_analysis
        from forecast_evaluation.tests.revisions_errors_correlation import revisions_errors_correlation_analysis
        from forecast_evaluation.tests.weak_efficiency import weak_efficiency_analysis

        stat_label_map = {"rmse": "RMSE", "rmedse": "RMedSE", "mean_abs_error": "MAE"}

        if test_type == "accuracy":
            # Can work with TestResult or ForecastData
            if hasattr(df, "_main_table"):
                acc = compute_accuracy_statistics(data=df, k=k).to_df()
            elif hasattr(df, "to_df"):
                acc = df.to_df()
            else:
                acc = df
            acc = acc.loc[
                (acc["frequency"] == frequency) & (acc["metric"] == metric) & (acc["forecast_horizon"] == horizon)
            ]
            if acc.empty:
                raise ValueError(f"No data for metric={metric!r}, horizon={horizon}")
            pivot = acc.pivot_table(index="unique_id", columns="variable", values=statistic, aggfunc="first")
            title = f"{stat_label_map.get(statistic, statistic.upper())} by variable – {metric} (h={horizon})"

        elif test_type == "bias":
            if not hasattr(df, "_main_table"):
                raise ValueError("test_type='bias' requires a ForecastData object.")
            if bias_type == "mean":
                bias_df = bias_analysis(data=df, k=k).to_df()
                bias_df = bias_df.loc[
                    (bias_df["frequency"] == frequency)
                    & (bias_df["metric"] == metric)
                    & (bias_df["forecast_horizon"] == horizon)
                ]
                bias_df["value"] = bias_df["bias_estimate"].abs()
                pivot = bias_df.pivot_table(index="unique_id", columns="variable", values="value", aggfunc="first")
                title = f"Abs. mean forecast error by variable – {metric} (h={horizon})"
            else:  # mz
                we_df = weak_efficiency_analysis(data=df, k=k).to_df()
                we_df = we_df.loc[
                    (we_df["frequency"] == frequency)
                    & (we_df["metric"] == metric)
                    & (we_df["forecast_horizon"] == horizon)
                ]
                we_df["value"] = we_df["joint_test_pvalue"].astype(float)
                pivot = we_df.pivot_table(index="unique_id", columns="variable", values="value", aggfunc="first")
                title = f"Bias MZ p-value by variable – {metric} (h={horizon})"

        elif test_type == "efficiency":
            if not hasattr(df, "_main_table"):
                raise ValueError("test_type='efficiency' requires a ForecastData object.")
            if efficiency_type == "revision_predictability":
                eff_df = weak_efficiency_analysis(data=df, k=k).to_df()
                eff_df = eff_df.loc[
                    (eff_df["frequency"] == frequency)
                    & (eff_df["metric"] == metric)
                    & (eff_df["forecast_horizon"] == horizon)
                ]
                eff_df["value"] = eff_df["joint_test_pvalue"].astype(float)
                pivot = eff_df.pivot_table(index="unique_id", columns="variable", values="value", aggfunc="first")
                title = f"Optimal scaling p-value by variable – {metric} (h={horizon})"
            else:  # revisions_errors
                re_df = revisions_errors_correlation_analysis(data=df, k=k).to_df()
                re_df = re_df.loc[
                    (re_df["frequency"] == frequency)
                    & (re_df["metric"] == metric)
                    & (re_df["forecast_horizon"] == horizon)
                ]
                re_df["value"] = re_df["beta_pvalue"].astype(float)
                pivot = re_df.pivot_table(index="unique_id", columns="variable", values="value", aggfunc="first")
                title = f"Revisions-errors corr. p-value by variable – {metric} (h={horizon})"
        else:
            raise ValueError(f"Invalid test_type {test_type!r}.")

        # Filter to selected variables if specified
        if variables is not None:
            keep = [col for col in pivot.columns if col in variables]
            pivot = pivot[keep]

    elif mode == "tests":
        # Spokes = [Accuracy, Bias, Efficiency] – definitions depend on bias_type / efficiency_type
        # df must be a ForecastData object
        if not hasattr(df, "_main_table"):
            raise ValueError("mode='tests' requires a ForecastData object as `df`, not an accuracy results DataFrame.")
        if variable is None or metric is None or horizon is None:
            raise ValueError("`variable`, `metric` and `horizon` are required for mode='tests'")

        from forecast_evaluation.tests.accuracy import compute_accuracy_statistics
        from forecast_evaluation.tests.bias import bias_analysis
        from forecast_evaluation.tests.revisions_errors_correlation import revisions_errors_correlation_analysis
        from forecast_evaluation.tests.weak_efficiency import weak_efficiency_analysis

        # -- Accuracy spoke --
        stat_label_map = {"rmse": "RMSE", "rmedse": "RMedSE", "mean_abs_error": "MAE"}
        acc_label = f"Accuracy ({stat_label_map.get(statistic, statistic.upper())})"

        accuracy = compute_accuracy_statistics(data=df, variable=variable, k=k).to_df()
        accuracy = accuracy.loc[
            (accuracy["variable"] == variable)
            & (accuracy["metric"] == metric)
            & (accuracy["forecast_horizon"] == horizon)
            & (accuracy["frequency"] == frequency)
        ]

        # -- Bias edge --
        if bias_type == "mean":
            bias_label = "Bias (abs. mean forecast error)"
            bias_results = bias_analysis(data=df, variable=variable, k=k).to_df()
            bias_results = bias_results.loc[
                (bias_results["variable"] == variable)
                & (bias_results["metric"] == metric)
                & (bias_results["forecast_horizon"] == horizon)
                & (bias_results["frequency"] == frequency)
            ]
        else:  # mz
            bias_label = "Bias (MZ p-value)"
            bias_results = weak_efficiency_analysis(data=df, variable=variable, k=k).to_df()
            bias_results = bias_results.loc[
                (bias_results["variable"] == variable)
                & (bias_results["metric"] == metric)
                & (bias_results["forecast_horizon"] == horizon)
                & (bias_results["frequency"] == frequency)
            ]

        # -- Efficiency edge --
        if efficiency_type == "revision_predictability":
            eff_label = "Efficiency\n(optimal scaling p-value)"
            eff_results = weak_efficiency_analysis(data=df, variable=variable, k=k).to_df()
            eff_results = eff_results.loc[
                (eff_results["variable"] == variable)
                & (eff_results["metric"] == metric)
                & (eff_results["forecast_horizon"] == horizon)
                & (eff_results["frequency"] == frequency)
            ]
        else:  # revisions_errors
            eff_label = "Efficiency\n(revisions-errors corr. p-value)"
            eff_results = revisions_errors_correlation_analysis(data=df, variable=variable, k=k).to_df()
            eff_results = eff_results.loc[
                (eff_results["variable"] == variable)
                & (eff_results["metric"] == metric)
                & (eff_results["forecast_horizon"] == horizon)
                & (eff_results["frequency"] == frequency)
            ]

        # -- Merge into a single pivot --
        rows = []
        for uid in accuracy["unique_id"].unique():
            acc_row = accuracy.loc[accuracy["unique_id"] == uid]
            bias_row = bias_results.loc[bias_results["unique_id"] == uid]
            eff_row = eff_results.loc[eff_results["unique_id"] == uid]

            acc_val = acc_row[statistic].values[0] if not acc_row.empty else np.nan

            if bias_type == "mean":
                bias_val = abs(bias_row["bias_estimate"].values[0]) if not bias_row.empty else np.nan
            else:
                bias_val = float(bias_row["joint_test_pvalue"].values[0]) if not bias_row.empty else np.nan

            if efficiency_type == "revision_predictability":
                eff_val = float(eff_row["joint_test_pvalue"].values[0]) if not eff_row.empty else np.nan
            else:
                eff_val = float(eff_row["beta_pvalue"].values[0]) if not eff_row.empty else np.nan

            rows.append(
                {
                    "unique_id": uid,
                    acc_label: acc_val,
                    bias_label: bias_val,
                    eff_label: eff_val,
                }
            )

        pivot = pd.DataFrame(rows).set_index("unique_id")
        title = f"Test summary – {variable} ({metric}, h={horizon})"

    else:
        raise ValueError(f"Invalid mode {mode!r}. Choose from 'metrics', 'variables', 'tests'.")

    # Clean unique_id labels
    pivot = clean_unique_id(pivot.reset_index()).set_index("unique_id")

    # Drop any spoke that is all NaN, then drop sources with missing spokes
    pivot = pivot.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError("No complete data available for the selected combination.")

    # ------------------------------------------------------------------
    # Normalise (per spoke) so different scales are comparable
    # ------------------------------------------------------------------
    if normalise:
        pivot = pivot.apply(_normalise_series, axis=0)

    # ------------------------------------------------------------------
    # Draw the radar chart
    # ------------------------------------------------------------------
    categories = list(pivot.columns)
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    # Close the polygon
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=THEME["figure"]["figsize"],
        subplot_kw={"polar": True},
        constrained_layout=True,
    )

    for idx, (source_name, row) in enumerate(pivot.iterrows()):
        values = row.tolist()
        values += values[:1]  # close
        ax.plot(angles, values, linewidth=2, label=source_name)
        ax.fill(angles, values, alpha=0.10)

    # Spoke labels
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)

    # Styling
    ax.set_title(title, size=THEME["axes"]["titlesize"], pad=20)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        fontsize=THEME["legend"]["fontsize"],
    )
    ax.grid(True)

    if return_plot:
        return fig, ax

    plt.show()
    return None

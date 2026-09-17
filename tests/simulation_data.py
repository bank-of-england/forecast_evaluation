from collections.abc import Sequence

import numpy as np
import pandas as pd


def create_monte_carlo_data(
    draws: int = 2,
    scenarios: Sequence[str] = ("baseline", "shock"),
    periods: int = 20,
    vintages: int = 2,
    seed: int = 42,
    include_vintages: bool = True,
) -> pd.DataFrame:
    """Create seeded multi-path outturns for simulation tests.

    Each draw shares its innovations across scenarios, which makes the output
    suitable for paired scenario tests. Scenarios receive distinct deterministic
    drifts, and every path contains multiple outturn vintages.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    if type(periods) is not int or periods < 2:
        raise ValueError("periods must be an integer greater than one")
    if type(vintages) is not int or not 1 <= vintages <= periods:
        raise ValueError("vintages must be an integer between one and periods")
    if type(include_vintages) is not bool:
        raise ValueError("include_vintages must be a boolean")
    scenarios = tuple(scenarios)
    if not scenarios or any(not isinstance(scenario, str) or not scenario.strip() for scenario in scenarios):
        raise ValueError("scenarios must contain non-empty labels")
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("scenarios must contain unique labels")

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-03-31", periods=periods, freq="QE")
    vintage_dates = dates[-vintages:]
    rows = []
    for draw in range(draws):
        innovations = rng.normal(0, 1, periods)
        base_path = 100 + np.cumsum(0.5 + innovations)
        for scenario_index, scenario in enumerate(scenarios):
            path = base_path + scenario_index * 10
            vintage_rows = enumerate(vintage_dates) if include_vintages else [(0, None)]
            for vintage_index, vintage_date in vintage_rows:
                available = dates <= vintage_date if include_vintages else np.ones(periods, dtype=bool)
                revision_scale = 0.2 / (vintage_index + 1)
                for date, value in zip(dates[available], path[available], strict=True):
                    row = {
                        "draw": draw,
                        "scenario": scenario,
                        "date": date,
                        "variable": "y",
                        "frequency": "Q",
                        "metric": "levels",
                        "value": value + rng.normal(0, revision_scale),
                    }
                    if include_vintages:
                        row["vintage_date"] = vintage_date
                    rows.append(row)

    return pd.DataFrame(rows)

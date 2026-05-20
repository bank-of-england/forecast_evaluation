"""Tests for ensure_consistent_date_range."""

from pathlib import Path

import pandas as pd

from forecast_evaluation.utils import ensure_consistent_date_range

DATA_DIR = Path(__file__).parent / "data"


def _make_rows(unique_id: str, variable: str, vintage_dates: list[str], dates: list[str]) -> list[dict]:
    """Helper to create rows for each combination of vintage_date_forecast and date."""
    rows = []
    for v in vintage_dates:
        for d in dates:
            rows.append(
                {
                    "unique_id": unique_id,
                    "variable": variable,
                    "vintage_date_forecast": pd.Timestamp(v),
                    "date": pd.Timestamp(d),
                }
            )
    return rows


class TestEnsureConsistentDateRange:
    """Tests for ensure_consistent_date_range with overlapping vintages."""

    def test_overlapping_vintages_keeps_common_only(self):
        """Sources with partially overlapping vintages should be reduced to the common set."""
        # Source A has vintages Q1-Q4, Source B has vintages Q2-Q5
        # Common vintages: Q2, Q3, Q4
        rows = _make_rows("A", "gdp", ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"], ["2019-12-31"])
        rows += _make_rows("B", "gdp", ["2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31"], ["2019-12-31"])
        df = pd.DataFrame(rows)

        result = ensure_consistent_date_range(df)

        for uid in ["A", "B"]:
            vintages = set(result.loc[result["unique_id"] == uid, "vintage_date_forecast"])
            assert vintages == {
                pd.Timestamp("2020-06-30"),
                pd.Timestamp("2020-09-30"),
                pd.Timestamp("2020-12-31"),
            }

    def test_overlapping_dates_keeps_common_only(self):
        """Sources with partially overlapping date sets should be reduced to the common set."""
        # Both sources have the same vintages, but different date coverage
        # Source A covers dates Q1-Q3, Source B covers dates Q2-Q4
        # Common dates: Q2, Q3
        common_vintages = ["2021-03-31", "2021-06-30"]
        rows = _make_rows("A", "gdp", common_vintages, ["2019-03-31", "2019-06-30", "2019-09-30"])
        rows += _make_rows("B", "gdp", common_vintages, ["2019-06-30", "2019-09-30", "2019-12-31"])
        df = pd.DataFrame(rows)

        result = ensure_consistent_date_range(df)

        for uid in ["A", "B"]:
            dates = set(result.loc[result["unique_id"] == uid, "date"])
            assert dates == {pd.Timestamp("2019-06-30"), pd.Timestamp("2019-09-30")}

    def test_both_dimensions_filtered(self):
        """Both vintage dates and dates should be filtered to common sets simultaneously."""
        # Source A: vintages V1, V2; dates D1, D2
        # Source B: vintages V2, V3; dates D2, D3
        # Common: vintage V2, date D2
        rows = _make_rows("A", "gdp", ["2020-03-31", "2020-06-30"], ["2019-03-31", "2019-06-30"])
        rows += _make_rows("B", "gdp", ["2020-06-30", "2020-09-30"], ["2019-06-30", "2019-09-30"])
        df = pd.DataFrame(rows)

        result = ensure_consistent_date_range(df)

        assert len(result[result["unique_id"] == "A"]) == len(result[result["unique_id"] == "B"])
        for uid in ["A", "B"]:
            sub = result[result["unique_id"] == uid]
            assert set(sub["vintage_date_forecast"]) == {pd.Timestamp("2020-06-30")}
            assert set(sub["date"]) == {pd.Timestamp("2019-06-30")}

    def test_equal_obs_per_source(self):
        """After filtering, all sources must have the same number of observations."""
        # 3 sources with staggered vintage ranges
        rows = _make_rows("A", "gdp", ["2020-03-31", "2020-06-30", "2020-09-30"], ["2019-06-30", "2019-12-31"])
        rows += _make_rows("B", "gdp", ["2020-06-30", "2020-09-30", "2020-12-31"], ["2019-06-30", "2019-12-31"])
        rows += _make_rows("C", "gdp", ["2020-06-30", "2020-09-30"], ["2019-03-31", "2019-06-30", "2019-12-31"])
        df = pd.DataFrame(rows)

        result = ensure_consistent_date_range(df)

        counts = result.groupby("unique_id").size()
        assert counts.nunique() == 1, f"Observation counts differ across sources: {counts.to_dict()}"

    def test_non_rectangular_data(self):
        """Sources with different (vintage, date) pairs but overlapping sets."""
        # Source A: V1→D1, V2→D1+D2, V3→D2
        # Source B: V2→D1+D2, V3→D2+D3
        # Common vintages: V2, V3. After vintage filter:
        #   A: (V2,D1),(V2,D2),(V3,D2)   B: (V2,D1),(V2,D2),(V3,D2),(V3,D3)
        # Common dates from filtered: D1, D2 (D3 only in B)
        # Final: A: 3 rows, B: 3 rows
        rows = [
            {
                "unique_id": "A",
                "variable": "gdp",
                "vintage_date_forecast": pd.Timestamp("2020-03-31"),
                "date": pd.Timestamp("2019-03-31"),
            },
            {
                "unique_id": "A",
                "variable": "gdp",
                "vintage_date_forecast": pd.Timestamp("2020-06-30"),
                "date": pd.Timestamp("2019-03-31"),
            },
            {
                "unique_id": "A",
                "variable": "gdp",
                "vintage_date_forecast": pd.Timestamp("2020-06-30"),
                "date": pd.Timestamp("2019-06-30"),
            },
            {
                "unique_id": "A",
                "variable": "gdp",
                "vintage_date_forecast": pd.Timestamp("2020-09-30"),
                "date": pd.Timestamp("2019-06-30"),
            },
            {
                "unique_id": "B",
                "variable": "gdp",
                "vintage_date_forecast": pd.Timestamp("2020-06-30"),
                "date": pd.Timestamp("2019-03-31"),
            },
            {
                "unique_id": "B",
                "variable": "gdp",
                "vintage_date_forecast": pd.Timestamp("2020-06-30"),
                "date": pd.Timestamp("2019-06-30"),
            },
            {
                "unique_id": "B",
                "variable": "gdp",
                "vintage_date_forecast": pd.Timestamp("2020-09-30"),
                "date": pd.Timestamp("2019-06-30"),
            },
            {
                "unique_id": "B",
                "variable": "gdp",
                "vintage_date_forecast": pd.Timestamp("2020-09-30"),
                "date": pd.Timestamp("2019-09-30"),
            },
        ]
        df = pd.DataFrame(rows)
        result = ensure_consistent_date_range(df)

        counts = result.groupby("unique_id").size()
        assert counts.nunique() == 1, f"Obs counts differ: {counts.to_dict()}"
        for uid in ["A", "B"]:
            sub = result[result["unique_id"] == uid]
            assert set(sub["vintage_date_forecast"]) == {pd.Timestamp("2020-06-30"), pd.Timestamp("2020-09-30")}
            assert set(sub["date"]) == {pd.Timestamp("2019-03-31"), pd.Timestamp("2019-06-30")}

    def test_multiple_variables_different_coverage(self):
        """Multiple variables where sources have different vintage coverage per variable."""
        # Variable GDP: A has V1,V2; B has V2,V3 → common V2
        # Variable CPI: A has V1,V2,V3; B has V2,V3 → common V2,V3
        # Each source should have different obs counts across variables but equal within each variable
        rows = _make_rows("A", "gdp", ["2020-03-31", "2020-06-30"], ["2019-06-30"])
        rows += _make_rows("B", "gdp", ["2020-06-30", "2020-09-30"], ["2019-06-30"])
        rows += _make_rows("A", "cpi", ["2020-03-31", "2020-06-30", "2020-09-30"], ["2019-06-30"])
        rows += _make_rows("B", "cpi", ["2020-06-30", "2020-09-30"], ["2019-06-30"])
        df = pd.DataFrame(rows)

        result = ensure_consistent_date_range(df)

        for var in ["gdp", "cpi"]:
            sub = result[result["variable"] == var]
            counts = sub.groupby("unique_id").size()
            assert counts.nunique() == 1, f"Obs counts differ for {var}: {counts.to_dict()}"

    def test_identical_sources_unchanged(self):
        """Sources with identical vintages and dates should pass through unchanged."""
        vintages = ["2020-03-31", "2020-06-30"]
        dates = ["2019-06-30", "2019-12-31"]
        rows = _make_rows("A", "gdp", vintages, dates)
        rows += _make_rows("B", "gdp", vintages, dates)
        df = pd.DataFrame(rows)

        result = ensure_consistent_date_range(df)

        assert len(result) == len(df)

    def test_real_data_mixed_ranges(self):
        """All sources should have equal obs after filtering real data with mixed (vintage, date) coverage."""
        df = pd.read_parquet(DATA_DIR / "df_mixed_ranges.parquet")
        df = ensure_consistent_date_range(df)

        # check that all unique_id have the same number of observations
        obs_counts = df.groupby("unique_id").size()
        assert obs_counts.nunique() == 1, f"Observation counts differ across sources: {obs_counts.to_dict()}"

        # check that all unique_id have the same set of (vintage_date_forecast, date) pairs
        pair_sets = {
            uid: set(zip(group["vintage_date_forecast"], group["date"])) for uid, group in df.groupby("unique_id")
        }
        first_uid = next(iter(pair_sets))
        first_pairs = pair_sets[first_uid]
        for uid, pairs in pair_sets.items():
            assert pairs == first_pairs, f"Pairs differ for {uid}: {len(pairs)} vs {len(first_pairs)}"

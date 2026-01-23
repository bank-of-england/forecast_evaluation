"""
Unit tests for the unified TestResult class.

Tests the TestResult class to ensure it properly stores data, provides
filtering, visualization, and export capabilities for all test types.
"""

import pandas as pd
import pytest

from forecast_evaluation.tests.accuracy import compare_to_benchmark
from forecast_evaluation.tests.results import TestResult


class TestTestResultWithBias:
    """Tests for TestResult class with bias test data."""

    def test_initialization(self):
        """Test that BiasResults can be initialized with DataFrame and metadata."""
        # Create sample data
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "gdpkp"],
                "unique_id": ["mpr", "mpr"],
                "metric": ["yoy", "yoy"],
                "frequency": ["Q", "Q"],
                "forecast_horizon": [0, 1],
                "bias_estimate": [0.1, 0.2],
                "std_error": [0.05, 0.06],
                "t_statistic": [2.0, 3.33],
                "p_value": [0.045, 0.001],
                "bias_conclusion": ["Biased", "Biased"],
                "n_observations": [50, 48],
                "ci_lower": [0.0, 0.08],
                "ci_upper": [0.2, 0.32],
            }
        )

        metadata = {
            "test_name": "bias_analysis",
            "parameters": {"k": 12, "same_date_range": True},
            "filters": {"unique_id": "mpr", "variable": "gdpkp"},
        }

        # Initialize result object
        result = TestResult(df, metadata=metadata)

        # Test that data is stored correctly
        assert len(result) == 2
        assert result._metadata["test_name"] == "bias_analysis"
        assert result._metadata["parameters"]["k"] == 12

    def test_to_df(self):
        """Test DataFrame accessor returns a copy."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp"],
                "unique_id": ["mpr"],
                "metric": ["yoy"],
                "frequency": ["Q"],
                "forecast_horizon": [0],
                "bias_estimate": [0.1],
                "p_value": [0.045],
            }
        )

        result = TestResult(df)
        df_copy = result.to_df()

        # Should be a copy, not the same object
        assert df_copy is not result._df
        # But should have same data
        assert df_copy.equals(result._df)

    def test_filter_by_variable(self):
        """Test filtering results by variable."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "cpisa", "gdpkp"],
                "unique_id": ["mpr", "mpr", "mpr"],
                "metric": ["yoy", "yoy", "yoy"],
                "frequency": ["Q", "Q", "Q"],
                "forecast_horizon": [0, 0, 1],
                "bias_estimate": [0.1, 0.2, 0.15],
            }
        )

        result = TestResult(df)
        filtered = result.filter(variable="gdpkp")

        assert len(filtered) == 2
        assert all(filtered._df["variable"] == "gdpkp")

    def test_filter_by_horizon(self):
        """Test filtering results by forecast horizon."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "gdpkp", "gdpkp"],
                "unique_id": ["mpr", "mpr", "mpr"],
                "forecast_horizon": [0, 1, 2],
                "bias_estimate": [0.1, 0.2, 0.15],
            }
        )

        result = TestResult(df)
        filtered = result.filter(horizon=[0, 1])

        assert len(filtered) == 2
        assert all(filtered._df["forecast_horizon"].isin([0, 1]))

    def test_dataframe_like_indexing(self):
        """Test that result objects support DataFrame-like indexing."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "cpisa"],
                "unique_id": ["mpr", "mpr"],
                "forecast_horizon": [0, 0],
                "bias_estimate": [0.1, 0.2],
            }
        )

        result = TestResult(df)

        # Test column access
        assert len(result["variable"]) == 2
        assert result["bias_estimate"].iloc[0] == 0.1

        # Test slicing
        subset = result[0:1]
        assert len(subset) == 1

    def test_len(self):
        """Test that len() returns number of rows."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "cpisa", "unemp"],
                "bias_estimate": [0.1, 0.2, 0.3],
            }
        )

        result = TestResult(df)
        assert len(result) == 3

    def test_summary(self):
        """Test summary method generates formatted output."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "cpisa"],
                "bias_estimate": [0.1, 0.2],
                "p_value": [0.045, 0.001],
                "unique_id": ["mpr", "mpr"],
            }
        )

        metadata = {"test_name": "bias_analysis", "parameters": {"k": 12}}

        result = TestResult(df, metadata=metadata)
        summary = result.summary()

        # Check that summary contains expected information
        assert "BIAS_ANALYSIS" in summary
        assert "k: 12" in summary
        assert "Number of Results: 2" in summary

    def test_describe(self):
        """Test describe method returns descriptive statistics."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "cpisa", "unemp"],
                "bias_estimate": [0.1, 0.2, 0.15],
                "p_value": [0.045, 0.001, 0.5],
            }
        )

        result = TestResult(df)
        desc = result.describe()

        # Should return DataFrame with descriptive stats
        assert isinstance(desc, pd.DataFrame)
        assert "bias_estimate" in desc.columns
        assert "mean" in desc.index

    def test_to_csv(self, tmp_path):
        """Test CSV export functionality."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "cpisa"],
                "bias_estimate": [0.1, 0.2],
            }
        )

        result = TestResult(df)

        # Export to CSV
        csv_path = tmp_path / "bias_results.csv"
        result.to_csv(str(csv_path), index=False)

        # Read back and verify
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == 2
        assert "variable" in df_read.columns


class TestTestResultWithAccuracy:
    """Tests for TestResult class with accuracy test data."""

    def test_initialization(self):
        """Test TestResult initialization with accuracy data."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "gdpkp"],
                "unique_id": ["mpr", "compass conditional"],
                "metric": ["yoy", "yoy"],
                "frequency": ["Q", "Q"],
                "forecast_horizon": [0, 0],
                "rmse": [0.5, 0.6],
                "rmedse": [0.45, 0.55],
                "mean_abs_error": [0.4, 0.5],
                "n_observations": [50, 48],
            }
        )

        metadata = {"test_name": "compute_accuracy_statistics"}
        result = TestResult(df, metadata=metadata)
        assert len(result) == 2
        assert result._metadata["test_name"] == "compute_accuracy_statistics"

    def test_compare_to_benchmark(self):
        """Test compare_to_benchmark standalone function."""
        df = pd.DataFrame(
            {
                "variable": ["gdpkp", "gdpkp"],
                "unique_id": ["mpr", "compass conditional"],
                "metric": ["yoy", "yoy"],
                "frequency": ["Q", "Q"],
                "forecast_horizon": [0, 0],
                "rmse": [0.5, 0.6],
            }
        )

        metadata = {"test_name": "compute_accuracy_statistics"}
        result = TestResult(df, metadata=metadata)

        # Compare to benchmark using standalone function
        compared = compare_to_benchmark(result, "mpr", "rmse")

        # Should have ratio column
        assert "rmse_to_benchmark" in compared.columns
        # Benchmark should have ratio of 1.0
        assert compared[compared["unique_id"] == "mpr"]["rmse_to_benchmark"].iloc[0] == 1.0
        # Other model should have ratio > 1 (worse than benchmark)
        assert compared[compared["unique_id"] == "compass conditional"]["rmse_to_benchmark"].iloc[0] == 1.2


class TestTestResultUniversalMethods:
    """Tests for universal methods of the TestResult class."""

    def test_testresult_has_all_required_methods(self):
        """Verify TestResult class has all required methods."""
        assert hasattr(TestResult, "plot"), "TestResult missing plot method"
        assert hasattr(TestResult, "to_df"), "TestResult missing to_df method"
        assert hasattr(TestResult, "filter"), "TestResult missing filter method"
        assert hasattr(TestResult, "summary"), "TestResult missing summary method"
        assert hasattr(TestResult, "describe"), "TestResult missing describe method"
        assert hasattr(TestResult, "to_csv"), "TestResult missing to_csv method"

    def test_testresult_works_with_different_test_types(self):
        """Verify TestResult works with different test metadata."""
        df = pd.DataFrame({"unique_id": ["mpr"], "variable": ["gdpkp"], "value": [0.1]})

        test_types = [
            "bias_analysis",
            "compute_accuracy_statistics",
            "weak_efficiency_analysis",
            "strong_efficiency_analysis",
            "blanchard_leigh_horizon_analysis",
            "diebold_mariano_table",
            "revisions_errors_correlation_analysis",
            "revision_predictability_analysis",
            "rolling_analysis",
        ]

        for test_type in test_types:
            metadata = {"test_name": test_type}
            result = TestResult(df, metadata=metadata)

            # Check common methods exist
            assert hasattr(result, "to_df")
            assert hasattr(result, "filter")
            assert hasattr(result, "summary")
            assert hasattr(result, "describe")
            assert hasattr(result, "to_csv")
            assert hasattr(result, "__len__")
            assert hasattr(result, "__getitem__")
            assert result._metadata["test_name"] == test_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

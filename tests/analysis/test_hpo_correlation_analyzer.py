import pytest
import pandas as pd
import numpy as np
from ppkt2synergy.analysis.hpo_correlation_analyzer import HPOStatisticsAnalyzer

@pytest.fixture
def mock_matrices():
    n_samples = 50
    rng = np.random.default_rng(seed=42)

    hpo_matrix = pd.DataFrame({
        "HP:0000001": rng.integers(0, 2, size=n_samples),
        "HP:0000002": rng.integers(0, 2, size=n_samples),
    })

    target_matrix = pd.DataFrame({
        "Disease_A": rng.integers(0, 2, size=n_samples),
        "Disease_B": rng.integers(0, 2, size=n_samples),
    })

    return hpo_matrix, target_matrix

@pytest.fixture
def analyzer(mock_matrices):
    hpo_matrix, target_matrix = mock_matrices
    return HPOStatisticsAnalyzer(
        hpo_data=(hpo_matrix, None),
        target_matrix=target_matrix,
        min_individuals_for_correlation_test=30
    )

def test_calculate_pairwise_stats_spearman(analyzer):
    result = analyzer._calculate_pairwise_stats(
        analyzer.combined_matrix["HP:0000001"],
        analyzer.combined_matrix["Disease_A"],
        stats_name="spearman"
    )
    assert "spearman" in result
    assert "spearman_p_value" in result
    assert isinstance(result["spearman"], float)
    assert 0 <= result["spearman_p_value"] <= 1

def test_calculate_pairwise_stats_kendall(analyzer):
    result = analyzer._calculate_pairwise_stats(
        analyzer.combined_matrix["HP:0000001"],
        analyzer.combined_matrix["Disease_B"],
        stats_name="kendall"
    )
    assert "kendall" in result
    assert "kendall_p_value" in result
    assert isinstance(result["kendall"], float)
    assert 0 <= result["kendall_p_value"] <= 1

def test_calculate_pairwise_stats_phi(analyzer):
    result = analyzer._calculate_pairwise_stats(
        analyzer.combined_matrix["HP:0000001"],
        analyzer.combined_matrix["Disease_A"],
        stats_name="phi"
    )
    assert "phi" in result
    assert "phi_p" in result
    assert isinstance(result["phi"], float)
    assert 0 <= result["phi_p"] <= 1

def test_invalid_stats_name(analyzer):
    with pytest.raises(ValueError, match="Unsupported stats_name"):
        analyzer._calculate_pairwise_stats(
            analyzer.combined_matrix["HP:0000001"],
            analyzer.combined_matrix["Disease_A"],
            stats_name="unsupported_method"
        )

def test_invalid_columns_all_zeros_or_ones(analyzer):
    n_samples = 40
    rng = np.random.default_rng(seed=42)
    hpo_matrix = pd.DataFrame({
        "HP:0000001": rng.integers(0, 3, size=n_samples),
        "HP:0000002": rng.integers(0, 2, size=n_samples),
    })

    target_matrix = pd.DataFrame({
        "Disease_A": rng.integers(0, 2, size=n_samples),
        "Disease_B": rng.integers(0, 2, size=n_samples),
    })

    with pytest.raises(ValueError, match="Non-NaN values in HPO Matrix must be either 0 or 1"):
        HPOStatisticsAnalyzer(
        hpo_data=(hpo_matrix, None),
        target_matrix=target_matrix,
        min_individuals_for_correlation_test=30
    )

def test_insufficient_samples_for_correlation(analyzer):
    analyzer.combined_matrix = pd.DataFrame({
        "HP:0000001": [0, np.nan, np.nan],
        "Disease_A": [1, np.nan, np.nan],
    })
    assert analyzer._calculate_pairwise_correlation(0, 1) is None

def test_compute_correlation_matrix(analyzer):
    corr_matrix, pval_matrix = analyzer.compute_correlation_matrix(stats_name="spearman")
    n = analyzer.combined_matrix.shape[1]
    assert corr_matrix.shape == (n, n)
    assert pval_matrix.shape == (n, n)
    assert not corr_matrix.isna().all().all()
    assert not pval_matrix.isna().all().all()

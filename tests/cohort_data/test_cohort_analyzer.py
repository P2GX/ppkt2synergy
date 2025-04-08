import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Assuming the module name is hpo_analysis
from ppkt2synergy import HPOCorrelationAnalyzer

MIN_INDIVIDUALS_FOR_CORRELATION_TEST=40
@pytest.fixture
def mock_cohort_matrix_generator():
    """
    Create a mock cohort_matrix_generator that simulates an HPO observation matrix
    and a disease matrix for testing purposes with specific data values.
    """
    mock = MagicMock()
    
    # Define a dataset with exactly 10 individuals and specific values
    hpo_matrix = pd.DataFrame(
        {
            "HP:0000123": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
            "HP:0000456": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        }
    )
    target_matrix = pd.DataFrame(
        {
            "Disease_A": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            "Disease_B": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        }
    )
    
    # Assign matrices to the mock object
    mock.hpo_term_observation_matrix = hpo_matrix
    mock.target_matrix = target_matrix
    mock.hpo_labels = {"HP:0000123": "Phenotype 1", "HP:0000456": "Phenotype 2"}
    mock.target_labels = {"Disease_A": "Disease A", "Disease_B": "Disease B"}
    
    return mock

@pytest.fixture
def hpo_analyzer(mock_cohort_matrix_generator):
    """
    Create an instance of HPOCorrelationAnalyzer with the mock cohort_matrix_generator.
    """
    return HPOCorrelationAnalyzer(mock_cohort_matrix_generator)


def test_combine_matrices(hpo_analyzer):
    """
    Test if the HPO and disease matrices are correctly combined into a single matrix.
    The combined matrix should have all terms from both matrices.
    """
    combined_matrix = hpo_analyzer._combine_matrices()
    assert combined_matrix.shape == (10, 4)  # 10 individuals and 4 features
    assert "HP:0000123" in combined_matrix.columns  # Ensure HPO terms exist
    assert "Disease_A" in combined_matrix.columns  # Ensure disease terms exist


def test_validate_hpo_terms(hpo_analyzer):
    """
    Test the _validate_hpo_terms method to ensure that it correctly identifies
    identical terms and ancestor-descendant relationships.
    """
    hpo_analyzer.hpo = MagicMock()
    hpo_analyzer.hpo.graph.is_ancestor_of.return_value = True  # Simulating ancestor relationship
    
    with pytest.raises(ValueError, match="HP:0000123 is an ancestor of HP:0000456"):
        hpo_analyzer._validate_hpo_terms("HP:0000123", "HP:0000456")


def test_calculate_pairwise_correlation_insufficient_data(hpo_analyzer):
    """
    Test if _calculate_pairwise_correlation raises an error when there are insufficient
    valid data points (less than MIN_INDIVIDUALS_FOR_CORRELATION_TEST) to compute correlation.
    """
    hpo_analyzer.combined_matrix = pd.DataFrame({
        "HP:0000123": [1, np.nan, np.nan],
        "HP:0000456": [0, np.nan, np.nan],
    })
    
    with pytest.raises(ValueError, match=f"Insufficient data.*{MIN_INDIVIDUALS_FOR_CORRELATION_TEST}"):
        hpo_analyzer._calculate_pairwise_correlation("HP:0000123", "HP:0000456")


def test_generate_correlation_matrix(hpo_analyzer):
    """
    Test the generate_correlation_matrix method to verify that the correlation
    and p-value matrices are correctly calculated.
    The test checks that correlation coefficients and p-values are properly assigned
    and that the correct statistical calculations are performed.
    """
    # Mock the correlation calculation function to return predefined values
    hpo_analyzer._calculate_pairwise_correlation = MagicMock(return_value={"Spearman": 0.8, "Spearman_p_value": 0.05})
    
    correlation_matrix, pvalue_matrix = hpo_analyzer.generate_correlation_matrix("Spearman")
    
    assert correlation_matrix.shape == (4, 4)  # Ensure matrix has correct dimensions
    assert pvalue_matrix.shape == (4, 4)  # Ensure p-value matrix has the same shape
    assert correlation_matrix.loc["Phenotype 1", "Phenotype 2"] == 0.8  # Check correlation value
    assert pvalue_matrix.loc["Phenotype 1", "Phenotype 2"] == 0.05  # Check p-value
import pytest
import numpy as np
from ppkt2synergy import MutualInformationCalculator

@pytest.fixture
def valid_data_I():
    feature_matrix = np.array([[0, 1, 0, 1], 
                               [1, 0, 1, 0], 
                               [1, 1, 1, 0], 
                               [0, 0, 1, 1],
                               [1, 0, 0, 0], 
                               [0, 1, 1, 1], 
                               [1, 1, 0, 1], 
                               [0, 0, 0, 1]])
    target = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    return feature_matrix, target

@pytest.fixture
def valid_data_II():
    feature_matrix = np.array([[0, 1],
                               [1, 1], 
                               [1, 0], 
                               [1, 1], 
                               [0, 0]])
    target = np.array([0, 1, 1, 1, 0])
    return feature_matrix, target

def test_valid_input_I(valid_data_I):
    feature_matrix, target = valid_data_I
    calculator = MutualInformationCalculator(feature_matrix, target)
    assert calculator.feature_matrix.shape == (8, 4)
    assert calculator.target.shape == (8,)

def test_valid_input_II(valid_data_II):
    feature_matrix, target = valid_data_II
    calculator = MutualInformationCalculator(feature_matrix, target)
    assert calculator.feature_matrix.shape == (5, 2)
    assert calculator.target.shape == (5,)

def test_target_no_variation():
    """
    Test case where the target variable has no variation.
    Expected: MutualInformationCalculator should raise a ValueError.
    """
    feature_matrix = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    target = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError):
        MutualInformationCalculator(feature_matrix, target)

def test_mismatched_dimensions():
    """
    Test case where feature matrix and target length do not match.
    Expected: MutualInformationCalculator should raise a ValueError.
    """
    feature_matrix = np.array([[0, 1], [1, 0], [1, 1]])  # 3 samples
    target = np.array([0, 1])  # 2 samples
    with pytest.raises(ValueError):
        MutualInformationCalculator(feature_matrix, target)

def test_non_binary_values():
    """
    Test case where feature matrix contains non-binary values.
    Expected: MutualInformationCalculator should raise a ValueError.
    """
    feature_matrix = np.array([[0, 2], [1, 0], [1, 1]])
    target = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        MutualInformationCalculator(feature_matrix, target)

def test_compute_mi_I(valid_data_I):
    """
    Test the compute_mi function on a valid dataset.
    Example calculation:
    Mutual Information (MI) is calculated as:
    MI(X, Y) = H(X) + H(Y) - H(X, Y)
    where H represents entropy.
    
    Example:
    Suppose feature (0,) has the following probability distribution:
    X = [0, 1, 1, 0, 1, 0, 1, 0]
    Y = [0, 1, 1, 0, 0, 1, 1, 0]
    P(X=0) = 0.5, P(X=1) = 0.5  H(X)= -0.5*log2(0.5)-0.5*log2(0.5)=1
    P(Y=0) = 0.5, P(Y=1) = 0.5   H(Y)= -0.5*log2(0.5)-0.5*log2(0.5)=1
    P(X=0,Y=0)=0.375  P(X=0,Y=1)=0.125
    P(X=1,Y=0)=0.125  P(X=1,Y=1)=0.375  H(X,Y)= −0.125*log2(0.125)*2 -0.375*log2(0.375)*2 = 1.811
    MI(X,Y) = 0.189.
    """
    feature_matrix, target = valid_data_I
    calculator = MutualInformationCalculator(feature_matrix, target)
    mi_score = calculator.compute_mi((0,))
    assert mi_score == pytest.approx(0.189, rel=1e-2)  

def test_compute_mi_II(valid_data_II):
    """
    Test the compute_mi function on a dataset with multiple features.
    Example calculation:
    If feature set (0,1) provides no additional information, MI >= 0.
    
    Example:
    Suppose feature set (0,1) has a joint probability distribution:
    X = [01, 11, 10, 11, 00]
    Y = [0, 1, 1, 1, 0]
    P(X=00)=0.2, P(X=01)=0.2 P(X=10)=0.2 P(X=11)=0.4 H(X)= -0.2*log(0.2)*3-0.4*log(0.4)=1.922
    P(Y=0)=0.4, P(Y=1)=0.6   H(Y)= -0.4*log2(0.4)-0.6*log2(0.6)=0.971
    P(X=00, Y=0) = 0.2, P(X=01, Y=0) = 0.2
    P(X=10, Y=1) = 0.2, P(X=11, Y=1) = 0.4 H(X, Y)= -0.2*log2(0.2)*3-0.4*log2(0.4)=1.922
    MI(X,Y) = 0.971.
    """
    feature_matrix, target = valid_data_II
    calculator = MutualInformationCalculator(feature_matrix, target)
    mi_score = calculator.compute_mi((0, 1))
    assert mi_score == pytest.approx(0.971, rel=1e-2)  


def test_lru_cache(valid_data_II):
    """
    Test whether compute_mi function correctly caches results.
    Example:
    Calling compute_mi((0,)) twice should return the same result and hit cache.
    """
    feature_matrix, target = valid_data_II
    calculator = MutualInformationCalculator(feature_matrix, target)
    
    mi_score_1 = calculator.compute_mi((0,))
    mi_score_2 = calculator.compute_mi((0,))
    
    assert mi_score_1 == mi_score_2
    assert calculator.compute_mi.cache_info().hits > 0  # Ensure cache hit occurred

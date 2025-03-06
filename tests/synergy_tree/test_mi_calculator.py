import pytest
import numpy as np
from ppkt2synergy import MutualInformationCalculator

@pytest.fixture
def valid_data_small():
    feature_matrix = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    target = np.array([0, 1, 1, 0])
    return feature_matrix, target

@pytest.fixture
def valid_data_large():
    feature_matrix = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 1],
                               [1, 0, 0, 0], [0, 1, 1, 1], [1, 1, 0, 1], [0, 0, 0, 1]])
    target = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    return feature_matrix, target

@pytest.fixture
def valid_data_many_features():
    feature_matrix = np.random.randint(0, 2, size=(100, 20))  # 100 samples, 20 features
    target = np.random.randint(0, 2, size=100)
    return feature_matrix, target

def test_valid_input(valid_data_small):
    feature_matrix, target = valid_data_small
    calculator = MutualInformationCalculator(feature_matrix, target)
    assert calculator.feature_matrix.shape == (4, 2)
    assert calculator.target.shape == (4,)

def test_valid_input_large(valid_data_large):
    feature_matrix, target = valid_data_large
    calculator = MutualInformationCalculator(feature_matrix, target)
    assert calculator.feature_matrix.shape == (8, 4)
    assert calculator.target.shape == (8,)

def test_valid_input_many_features(valid_data_many_features):
    feature_matrix, target = valid_data_many_features
    calculator = MutualInformationCalculator(feature_matrix, target)
    assert calculator.feature_matrix.shape == (100, 20)
    assert calculator.target.shape == (100,)

def test_target_no_variation():
    feature_matrix = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    target = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError):
        MutualInformationCalculator(feature_matrix, target)

def test_mismatched_dimensions():
    feature_matrix = np.array([[0, 1], [1, 0], [1, 1]])  # 3 samples
    target = np.array([0, 1])  # 2 samples
    with pytest.raises(ValueError):
        MutualInformationCalculator(feature_matrix, target)

def test_non_binary_values():
    feature_matrix = np.array([[0, 2], [1, 0], [1, 1]])
    target = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        MutualInformationCalculator(feature_matrix, target)

def test_compute_mi(valid_data_small):
    feature_matrix, target = valid_data_small
    calculator = MutualInformationCalculator(feature_matrix, target)
    mi_score = calculator.compute_mi((0,))
    assert isinstance(mi_score, float)
    assert mi_score >= 0  # Mutual information should be non-negative

def test_compute_mi_large(valid_data_large):
    feature_matrix, target = valid_data_large
    calculator = MutualInformationCalculator(feature_matrix, target)
    mi_score = calculator.compute_mi((0, 1))
    assert isinstance(mi_score, float)
    assert mi_score >= 0  # Mutual information should be non-negative

def test_compute_mi_many_features(valid_data_many_features):
    feature_matrix, target = valid_data_many_features
    calculator = MutualInformationCalculator(feature_matrix, target)
    mi_score = calculator.compute_mi((0, 5, 10))
    assert isinstance(mi_score, float)
    assert mi_score >= 0  # Mutual information should be non-negative

def test_lru_cache(valid_data_small):
    feature_matrix, target = valid_data_small
    calculator = MutualInformationCalculator(feature_matrix, target)
    
    mi_score_1 = calculator.compute_mi((0,))
    mi_score_2 = calculator.compute_mi((0,))
    
    assert mi_score_1 == mi_score_2
    assert calculator.compute_mi.cache_info().hits > 0  # Ensure cache hit occurred

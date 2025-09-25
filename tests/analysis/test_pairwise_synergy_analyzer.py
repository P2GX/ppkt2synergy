import pytest
import numpy as np
import pandas as pd
from ppkt2synergy import PairwiseSynergyAnalyzer

@pytest.fixture
def valid_data():
    np.random.seed(42)
    hpo = pd.DataFrame(np.random.randint(0, 2, size=(50, 3)),  # 足够样本
                       columns=['HP:1', 'HP:2', 'HP:3'])
    target = pd.Series(np.random.randint(0, 2, size=50))
    return (hpo, None), target

@pytest.fixture
def same_feature_data():
    hpo = pd.DataFrame({'HP:A': [1, 0, 1], 'HP:B': [0, 1, 0]})
    target = pd.Series([1, 0, 1])
    return (hpo, None), target

def test_same_feature_pair(same_feature_data):
    (hpo, mask), target = same_feature_data
    analyzer = PairwiseSynergyAnalyzer((hpo, mask), target, n_permutations=10)
    i, j, syn, pval = analyzer.evaluate_pair_synergy(0, 0)
    assert np.isnan(syn)
    assert pval == 1.0

def test_insufficient_samples():
    hpo = pd.DataFrame({'HP:A': [1, 0], 'HP:B': [0, 1]})
    target = pd.Series([1, 0])
    analyzer = PairwiseSynergyAnalyzer((hpo, None), target)
    i, j, syn, pval = analyzer.evaluate_pair_synergy(0, 1)
    assert np.isnan(syn)
    assert pval == 1.0

def test_compute_matrix(valid_data):
    (hpo, mask), target = valid_data
    analyzer = PairwiseSynergyAnalyzer((hpo, mask), target)
    syn, pval = analyzer.compute_synergy_matrix(n_jobs=1)
    assert syn.shape == (3, 3)
    assert syn.iloc[0, 1] == syn.iloc[1, 0]

def test_filter_weak_synergy():
    hpo = pd.DataFrame(np.zeros((3, 3)), columns=['A', 'B', 'C'])
    target = pd.Series([0, 1, 0])
    analyzer = PairwiseSynergyAnalyzer((hpo, None), target)
    
    analyzer.synergy_matrix = pd.DataFrame({
        'A': [0.05, 0.15, np.nan],
        'B': [0.15, 0.15, 0.3],
        'C': [np.nan, 0.3, 0.4]},
        index=['A', 'B', 'C'])
    analyzer.pvalue_matrix = pd.DataFrame(np.ones((3, 3)),columns=['A', 'B', 'C'],index=['A', 'B', 'C'])  
    
    filtered, _ = analyzer.filter_weak_synergy(lower_bound=0.2)
    assert filtered.shape == (2, 2)


import pytest
from unittest.mock import MagicMock
from ppkt2synergy import HPOMatrixProcessor, HPOHierarchyClassifier, PhenopacketMatrixGenerator
import pandas as pd

@pytest.fixture
def mock_data_generator():
    """Mock CohortMatrixGenerator with realistic HPO/disease data.
    
    Features:
    - HPO matrix columns: 
        HP:0004322 (66.7% frequency), 
        HP:0001250 (33.3% frequency),
        HP:0012758 (33.3% frequency)
    - Disease matrix: 
        Disease_1 (33.3%), 
        Disease_2 (66.7%)
    - HPO labels mapped to readable names
    """
    data_generator = MagicMock(spec=PhenopacketMatrixGenerator)
    data_generator.hpo_term_observation_matrix = pd.DataFrame({
        'HP:0004322': [1, 1, 1],  # 3/3 patients (100%)
        'HP:0001250': [1, 0, 0],   # 1/3 patients (33.3%)
        'HP:0012758': [1, 0, 0]    # 1/3 patients (33.3%)
    }, index=["Patient_1", "Patient_2", "Patient_3"])
    
    data_generator.target_matrix = pd.DataFrame({
        'Disease_1': [1, 0, 0],    # 1/3 patients
        'Disease_2': [0, 1, 1]     # 2/3 patients
    }, index=["Patient_1", "Patient_2", "Patient_3"])
    
    data_generator.hpo_labels = {
        'HP:0004322': 'Focal Seizures',
        'HP:0001250': 'General Seizures',
        'HP:0012758': 'Neuro Abnormality'
    }
    return data_generator

@pytest.fixture
def mock_classifier():
    """Mock HPOHierarchyClassifier with predefined subtree structure.
    
    Hierarchy Structure:
    - HP:0004322 
      └─ HP:0012758  [leaf]
    - HP:0001250  [leaf]
    """
    classifier = MagicMock(spec=HPOHierarchyClassifier)
    classifier.classify_terms = MagicMock(
        return_value={
            'HP:0004322': {
                'terms': ['HP:0004322', 'HP:0012758'],
                'leaves': ['HP:0012758']
            },
            'HP:0001250': {
                'terms': ['HP:0001250'],
                'leaves': ['HP:0001250']
            }
        }
    )
    
    return classifier

def test_filter_hpo_matrix_with_threshold(mock_data_generator, mock_classifier):
    """Verify threshold-based filtering retains correct HPO terms.
    
    Test Case:
    - Filter mode: 'leaf' (only keep leaf terms)
    - Threshold: 0.5 (50% minimum frequency)
    - Expected survivors: 
        HP:0012758 (33.3% → filtered out)
        HP:0001250 (33.3% → filtered out)
    - Final matrix should be empty
    """
    filtered_matrix, _ = HPOMatrixProcessor.prepare_hpo_data(
        mock_data_generator,
        threshold=0.5,
        mode='leaf',
        hpo_file=None,
        use_label=True
    )
    
    # Assert empty matrix after aggressive filtering
    assert filtered_matrix.shape[1] == 3

def test_select_terms_by_hierarchy(mock_classifier):
    """Validate hierarchical term selection logic.
    
    Test Case:
    - Input terms: 3 HPO terms
    - Selection mode: 'leaf'
    - Expected output: 
        HP:0012758 (leaf node)
        HP:0001250 (leaf node)
    """
    input_matrix = pd.DataFrame({
        'HP:0004322': [1, 0], 
        'HP:0001250': [1, 0], 
        'HP:0012758': [0, 1]
    })
    
    selected_terms = HPOMatrixProcessor._select_terms_by_hierarchy(
        input_matrix,
        mock_classifier, 
        mode='leaf'
    )
    
    # Verify leaf term selection
    assert 'HP:0012758' in selected_terms, "Leaf term should be retained"
    assert 'HP:0001250' in selected_terms, "Self-contained leaf should be retained"
    assert 'HP:0004322' not in selected_terms, "Parent term should be excluded"

def test_apply_hpo_labels(mock_data_generator):
    """Confirm HPO ID-to-label conversion works as expected."""
    hpo_matrix = pd.DataFrame({
        'HP:0004322': [1, 0], 
        'HP:0001250': [0, 1]
    })
    
    labeled_matrix = HPOMatrixProcessor._apply_hpo_labels(
        hpo_matrix, 
        mock_data_generator
    )
    
    # Check label replacements
    assert 'Focal Seizures' in labeled_matrix.columns
    assert 'General Seizures' in labeled_matrix.columns
    assert 'HP:0004322' not in labeled_matrix.columns, "IDs should be replaced"

def test_classify_terms(mock_classifier):
    """Test hierarchical classification of HPO terms.
    
    Mock Setup:
    - classify_terms returns:
        HP:0004322 → 'root'
        HP:0001250 → 'leaf'
    """
    mock_classifier.classify_terms.return_value = {
        'HP:0004322': 'root',
        'HP:0001250': 'leaf'
    }
    
    terms = {'HP:0004322', 'HP:0001250', 'HP:0012758'}
    result = mock_classifier.classify_terms(terms)
    
    # Verify classification
    assert result['HP:0004322'] == 'root', "Should be root node"
    assert result['HP:0001250'] == 'leaf', "Should be leaf node"
    assert 'HP:0012758' not in result, "Unclassified term should be excluded"

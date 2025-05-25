import pytest
from unittest.mock import MagicMock
import pandas as pd
from ppkt2synergy import PhenopacketMatrixProcessor
from ppkt2synergy import PhenopacketMatrixGenerator
from ppkt2synergy import HPOHierarchyUtils


@pytest.fixture
def full_mock_data_generator():
    """Mock a complete PhenopacketMatrixGenerator."""
    generator = MagicMock(spec=PhenopacketMatrixGenerator)

    generator.phenopackets = [MagicMock(), MagicMock(), MagicMock()]

    generator.hpo_term_observation_matrix = pd.DataFrame({
        'HP:0020219': [1, 1, 1],
        'HP:0001250': [1, 0, None],
        'HP:0012759': [1, 0, 0]
    }, index=["Patient_1", "Patient_2", "Patient_3"])

    generator.target_matrix = pd.DataFrame({
        'Disease_1': [1, 0, 0],
        'Disease_2': [0, 1, 1]
    }, index=["Patient_1", "Patient_2", "Patient_3"])

    generator.hpo_labels = {
        'HP:0020219': 'Motor seizure',
        'HP:0001250': 'General Seizures',
        'HP:0012759': 'Neuro Abnormality'
    }

    generator.target_labels = {
        'Disease_1': 'Marfan Syndrome',
        'Disease_2': 'Ehlers-Danlos syndrome'
    }

    return generator


@pytest.fixture
def mock_hpo_classifier():
    """Mock HPOHierarchyUtils with fixed tree logic."""
    classifier = MagicMock()
    classifier.classify_terms.return_value = {
        'HP:0001250': {
            'terms': ['HP:0001250', 'HP:0020219'],
            'leaves': ['HP:0020219']
        },
        'HP:0012759': {
            'terms': ['HP:0012759'],
            'leaves': ['HP:0012759']
        }
    }
    classifier.build_relationship_mask.return_value = pd.DataFrame(
        [[1, 1, None],
         [1, 1, None],
         [None, None, 1]],
        index=['HP:0020219', 'HP:0001250', 'HP:0012759'],
        columns=['HP:0020219', 'HP:0001250', 'HP:0012759']
    )
    return classifier


def test_prepare_hpo_data_leaf_mode(monkeypatch, full_mock_data_generator, mock_hpo_classifier):
    """Test prepare_hpo_data with leaf filtering and label application."""
    monkeypatch.setattr('ppkt2synergy.preprocessing.phenopacket_matrix_processor.PhenopacketMatrixGenerator', lambda *args, **kwargs: full_mock_data_generator)
    monkeypatch.setattr('ppkt2synergy.preprocessing.phenopacket_matrix_processor.HPOHierarchyUtils', lambda *args, **kwargs: mock_hpo_classifier)

    (hpo_matrix, relationship_mask), target_matrix = PhenopacketMatrixProcessor.prepare_hpo_data(
        phenopackets=full_mock_data_generator.phenopackets,
        hpo_file="dummy/path.owl",
        threshold=0.3,
        mode='leaf',
        use_label=True,
        nan_strategy="impute_zero"
    )

    assert isinstance(hpo_matrix, pd.DataFrame)
    assert isinstance(target_matrix, pd.DataFrame)
    assert relationship_mask is None
    assert 'Motor seizure' in hpo_matrix.columns  # only leaves
    assert 'General Seizures' not in hpo_matrix.columns  # root excluded
    assert 'Neuro Abnormality' in hpo_matrix.columns
    assert hpo_matrix.isnull().sum().sum() == 0  # NaNs should be imputed


def test_prepare_hpo_data_root_mode(monkeypatch, full_mock_data_generator, mock_hpo_classifier):
    """Test prepare_hpo_data with root selection and no label application."""
    monkeypatch.setattr('ppkt2synergy.preprocessing.phenopacket_matrix_processor.PhenopacketMatrixGenerator', lambda *args, **kwargs: full_mock_data_generator)
    monkeypatch.setattr('ppkt2synergy.preprocessing.phenopacket_matrix_processor.HPOHierarchyUtils', lambda *args, **kwargs: mock_hpo_classifier)

    (hpo_matrix, relationship_mask), _ = PhenopacketMatrixProcessor.prepare_hpo_data(
        phenopackets=full_mock_data_generator.phenopackets,
        hpo_file=None,
        threshold=0.0,
        mode='root',
        use_label=False,
        nan_strategy=None
    )

    assert 'HP:0001250' in hpo_matrix.columns  # root retained
    assert 'HP:0020219' not in hpo_matrix.columns  # leaf excluded
    assert 'HP:0012759' in hpo_matrix.columns  # also a root
    assert pd.isna(hpo_matrix.values).sum() > 0  # NaNs preserved


def test_prepare_hpo_data_relationship_mode(monkeypatch, full_mock_data_generator, mock_hpo_classifier):
    """Test prepare_hpo_data in mode=None with relationship_mask returned."""
    monkeypatch.setattr('ppkt2synergy.preprocessing.phenopacket_matrix_processor.PhenopacketMatrixGenerator', lambda *args, **kwargs: full_mock_data_generator)
    monkeypatch.setattr('ppkt2synergy.preprocessing.phenopacket_matrix_processor.HPOHierarchyUtils', lambda *args, **kwargs: mock_hpo_classifier)

    (hpo_matrix, relationship_mask), _ = PhenopacketMatrixProcessor.prepare_hpo_data(
        phenopackets=full_mock_data_generator.phenopackets,
        hpo_file=None,
        threshold=0.3,
        mode=None,
        use_label=False,
        nan_strategy="impute_zero"
    )

    assert isinstance(relationship_mask, pd.DataFrame)
    assert relationship_mask.shape[0] == relationship_mask.shape[1]
    assert 'HP:0020219' in relationship_mask.columns
    assert pd.isna(relationship_mask.loc['HP:0020219', 'HP:0012759']) or relationship_mask.loc['HP:0020219', 'HP:0012759'] in {1, None}


def test_prepare_hpo_data_invalid_threshold_raises(monkeypatch, full_mock_data_generator):
    with pytest.raises(ValueError):
        PhenopacketMatrixProcessor.prepare_hpo_data(
            phenopackets=full_mock_data_generator.phenopackets,
            threshold=1.5
        )


def test_prepare_hpo_data_invalid_nan_strategy(monkeypatch, full_mock_data_generator, mock_hpo_classifier):
    monkeypatch.setattr('ppkt2synergy.preprocessing.phenopacket_matrix_processor.PhenopacketMatrixGenerator', lambda *args, **kwargs: full_mock_data_generator)
    monkeypatch.setattr('ppkt2synergy.preprocessing.phenopacket_matrix_processor.HPOHierarchyUtils', lambda *args, **kwargs: mock_hpo_classifier)

    with pytest.raises(ValueError):
        PhenopacketMatrixProcessor.prepare_hpo_data(
            phenopackets=full_mock_data_generator.phenopackets,
            nan_strategy="unknown_strategy"
        )

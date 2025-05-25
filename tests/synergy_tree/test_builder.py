import pytest
import pandas as pd
import numpy as np
from ppkt2synergy import SynergyTreeBuilder, LeafNode, InternalNode

@pytest.fixture
def valid_data():
    """Valid binary input data with 20 samples"""
    np.random.seed(42)
    X = pd.DataFrame({
        'A': np.random.randint(0, 2, 20),
        'B': np.random.randint(0, 2, 20),
        'C': np.random.randint(0, 2, 20),
    })
    y = pd.Series(np.random.randint(0, 2, 20))
    return X, y

@pytest.fixture
def invalid_data():
    """Invalid input data with non-binary values"""
    X = pd.DataFrame({
        'A': [0, 1, 2]*7,  # Invalid: contains non-binary values
        'B': np.random.randint(0, 2, 21)
    })
    y = pd.Series(np.random.randint(0, 2, 21))
    return X, y

@pytest.fixture
def empty_data():
    """Empty input data"""
    X = pd.DataFrame()
    y = pd.Series()
    return X, y

@pytest.fixture
def single_feature_data():
    """Input data with only one feature"""
    X = pd.DataFrame({'A': np.random.randint(0, 2, 20)})
    y = pd.Series(np.random.randint(0, 2, 20))
    return X, y

@pytest.fixture
def perfect_correlation_data():
    """Data where two features perfectly predict the target"""
    X = pd.DataFrame({
        'A': [1, 0]*10,
        'B': [1, 0]*10,  # Perfect correlation with A
        'C': [0, 1]*10   # Perfect anti-correlation with target
    })
    y = pd.Series([1, 0]*10)
    return X, y

def test_validate_input(valid_data, invalid_data, empty_data, single_feature_data):
    """Test input validation"""
    X_valid, y_valid = valid_data
    X_invalid, y_invalid = invalid_data
    X_empty, y_empty = empty_data
    X_single, y_single = single_feature_data

    # Test valid input
    builder = SynergyTreeBuilder(X_valid, y_valid, max_k=10)
    assert builder.max_k == 3  # Should be min(10, num_features=3)

    # Test max_k validation
    with pytest.raises(ValueError, match="max_k must be between 2 and"):
        SynergyTreeBuilder(X_valid, y_valid, max_k=0)

    # Test invalid input (non-binary values)
    with pytest.raises(ValueError, match="must contain only 0 and 1"):
        SynergyTreeBuilder(X_invalid, y_invalid, max_k=2)

    # Test empty input
    with pytest.raises(ValueError, match="At least 2 features are required"):
        SynergyTreeBuilder(X_empty, y_empty, max_k=2)

    # Test insufficient features
    with pytest.raises(ValueError, match="At least 2 features are required"):
        SynergyTreeBuilder(X_single, y_single, max_k=2)

def test_init_leaf_nodes(valid_data):
    """Test leaf node initialization"""
    X, y = valid_data
    builder = SynergyTreeBuilder(X, y, max_k=2)
    builder._init_leaf_nodes()

    # Check if leaf nodes are initialized correctly
    assert len(builder.nodes) == 3  # 3 features
    for i, name in enumerate(X.columns):
        assert (i,) in builder.nodes
        node = builder.nodes[(i,)]
        assert isinstance(node, LeafNode)
        assert node.get_feature_indices == (i,)
        assert 0 <= node.get_mi <= 1  # MI should be between 0 and 1


@pytest.mark.parametrize(
    'X, y, max_k, expected_roots_len, expected_leaf_nodes_count, expected_internal_nodes_count',
    [
        (
            pd.DataFrame({
                'feature_1': [0, 1]*10,
                'feature_2': [0, 0, 1, 1]*5,
                'feature_3': [1, 1, 0, 0]*5,
                'feature_4': [1, 0]*10,
            }),
            pd.Series([0, 1]*10),
            3,  # max_k
            4,  # expected number of roots
            4,  # expected number of leaf nodes
            4   # expected number of internal nodes
        ),
    ]
)
def test_synergy_tree(X, y, max_k, expected_roots_len, expected_leaf_nodes_count, expected_internal_nodes_count):
    """Test the synergy tree builder with different datasets and assert the expected values."""
    synergy_tree_builder = SynergyTreeBuilder(X, y, max_k=max_k)
    synergy_tree_builder.build()
    
    assert len(synergy_tree_builder.roots) == expected_roots_len
    
    leaf_nodes_count = sum(1 for node in synergy_tree_builder.nodes.values() if node.is_leaf())
    assert leaf_nodes_count == expected_leaf_nodes_count
  
    internal_nodes_count = sum(1 for node in synergy_tree_builder.nodes.values() if not node.is_leaf())
    assert internal_nodes_count == expected_internal_nodes_count
import pytest
import pandas as pd
from ppkt2synergy import SynergyTreeBuilder


@pytest.mark.parametrize(
    'X, y, max_k, expected_roots_len, expected_leaf_nodes_count, expected_internal_nodes_count',
    [
        (
            pd.DataFrame({
                'feature_1': [0, 1, 0, 1, 0, 1, 0, 1],
                'feature_2': [0, 0, 1, 1, 0, 0, 1, 1],
                'feature_3': [1, 1, 0, 0, 1, 1, 0, 0],
                'feature_4': [1, 0, 1, 0, 1, 0, 1, 0],
            }),
            pd.Series([0, 1, 0, 1, 0, 1, 0, 1]),
            3,  # max_k
            4,  # expected number of roots
            4,  # expected number of leaf nodes
            4   # expected number of internal nodes
        ),
        # Second dataset (3 features)
        (
            pd.DataFrame({
                'feature_1': [1, 0, 1, 0],
                'feature_2': [1, 1, 0, 0],
                'feature_3': [0, 1, 1, 0],
            }),
            pd.Series([1, 0, 1, 0]),
            2,  # max_k
            1,  # expected number of roots
            3,  # expected number of leaf nodes
            1  # expected number of internal nodes
        ),
        # Third dataset (2 features)
        (
            pd.DataFrame({
                'feature_1': [1, 0, 1, 0],
                'feature_2': [0, 1, 0, 1],
            }),
            pd.Series([1, 0, 1, 0]),
            3,  # max_k
            0,  # expected number of roots
            2,  # expected number of leaf nodes
            0   # expected number of internal nodes
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

import pytest
import pandas as pd
from ppkt2synergy import SynergyTreeBuilder
from ppkt2synergy import LeafNode,InternalNode

# Fixture for valid input data
@pytest.fixture
def valid_data():
    X = pd.DataFrame({
        'A': [1, 0, 1, 0],
        'B': [1, 1, 0, 0],
        'C': [0, 1, 1, 0],
    })
    y = pd.Series([1, 0, 1, 0])
    return X, y

# Fixture for invalid input data
@pytest.fixture
def invalid_data():
    X = pd.DataFrame({
        'A': [0, 1, 2],  # Invalid: contains non-binary values
        'B': [1, 0, 1]
    })
    y = pd.Series([0, 1, 1])
    return X, y

# Fixture for empty input data
@pytest.fixture
def empty_data():
    X = pd.DataFrame()
    y = pd.Series()
    return X, y

# Fixture for single feature input data
@pytest.fixture
def single_feature_data():
    X = pd.DataFrame({'A': [0, 1, 0, 1]})
    y = pd.Series([0, 1, 1, 1])
    return X, y

# Test valid input
# H(y) = −0.5*log2(0.5)*2=1
# H(A,y)=−0.5*log2(0.5)*2=1  H(A) =−0.5*log2(0.5)*2=1  MI(A,y)=1
# H(B,y)=−0.25*log2(0.25)*4=2  H(B) =−0.5*log2(0.5)*2=1  MI(B,y)=0
# H(C,y)=−0.25*log2(0.25)*4=2  H(C) =−0.5*log2(0.5)*2=1  MI(B,y)=0
# H(AB,y)=2  H(AB)=−0.25*log2(0.25)*4=2  MI(AB,y)=1
# H(AC,y)=2  H(AC)=−0.25*log2(0.25)*4=2  MI(AC,y)=1
# H(BC,y)=2  H(BC)=−0.25*log2(0.25)*4=2  MI(BC,y)=1
# H(ABC,y)=2  H(ABC)=−0.25*log2(0.25)*4=2  MI(ABC,y)=1      
# Synergy(BC,y) = MI(BC,y)-MI(B,y)-MI(C,y) >0

# Test validate_input
def test_validate_input(valid_data, invalid_data, empty_data, single_feature_data):
    X_valid, y_valid = valid_data
    X_invalid, y_invalid = invalid_data
    X_empty, y_empty = empty_data
    X_single_feature, y_single_feature =single_feature_data

    # Test valid input
    builder = SynergyTreeBuilder(X_valid, y_valid, max_k=10)
    assert builder.max_k == 3  # Should be min(10,3) 
    # Test max_k validation
    with pytest.raises(ValueError):
        SynergyTreeBuilder(X_valid, y_valid, max_k=0)  # max_k <= 0

    # Test invalid input (non-binary values)
    with pytest.raises(ValueError):
        SynergyTreeBuilder(X_invalid, y_invalid, max_k=2)

    # Test empty input
    with pytest.raises(ValueError):
        SynergyTreeBuilder(X_empty, y_empty, max_k=2)


    # Test insufficient features
    with pytest.raises(ValueError, match="At least 2 features are required to build a synergy tree"):
        SynergyTreeBuilder(X_single_feature, y_single_feature, max_k=2)

# Test SynergyTreeBuilder._init_leaf_nodes  3 Leafnodes will be updated in builder.nodes
def test_init_leaf_nodes(valid_data):
    X, y = valid_data
    builder = SynergyTreeBuilder(X, y, max_k=2)
    builder._init_leaf_nodes()

    # Check if leaf nodes are initialized correctly
    assert len(builder.nodes) == 3  # 3 features
    for i, name in enumerate(X.columns):
        assert (i,) in builder.nodes
        assert isinstance(builder.nodes[(i,)], LeafNode)
        assert builder.nodes[(i,)].get_feature_indices == (i,)
        assert builder.nodes[(i,)].get_mi >= 0

# Test SynergyTreeBuilder._build_layer
def test_build_layer(valid_data):
    X, y = valid_data
    builder = SynergyTreeBuilder(X, y, max_k=2)
    builder._init_leaf_nodes()
    builder._build_layer(2)

    # Check if layer 2 nodes are built correctly  
    # Synergy(BC,y) >0  InternalNode(BC) will be updated in builder.nodes
    assert len(builder.nodes) == 4 
    assert isinstance(builder.nodes[(1,2)], InternalNode)
    assert builder.nodes[(1,2)].get_synergy >= 0

# Test SynergyTreeBuilder._update_roots
def test_update_roots(valid_data):
    X, y = valid_data
    builder = SynergyTreeBuilder(X, y, max_k=2)
    builder._init_leaf_nodes()
    builder._build_layer(2)

    # Check if roots are updated correctly
    assert len(builder.roots) == 1
    for feature_comb in builder.roots:
        assert feature_comb in builder.nodes
        assert isinstance(builder.roots[feature_comb], InternalNode)

def test_build(valid_data):
    X, y = valid_data
    builder = SynergyTreeBuilder(X, y, max_k=2)
    trees = builder.build()

    # Check if the correct number of trees is returned
    assert len(trees) == 1

    # Check if the tree structure is correct
    root = trees[0]
    assert isinstance(root, InternalNode)
    assert root.get_feature_indices == (1, 2)  # Expected root node
    assert len(root.get_children) == 2  # Expected two children

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
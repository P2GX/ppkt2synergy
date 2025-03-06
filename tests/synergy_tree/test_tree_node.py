import pytest
from ppkt2synergy import  LeafNode, InternalNode


class TestTreeNode:

    @pytest.mark.parametrize(
        'features, mi, label, expected_features, expected_mi, expected_is_leaf',
        [
            ((0,), 0.5, "Feature 0", (0,), 0.5, True),
            ((1,), 0.8, "Feature 1", (1,), 0.8, True),
            ((2,), 0.3, "Feature 2", (2,), 0.3, True)
        ]
    )
    def test_leaf_node(
        self, 
        features, 
        mi, 
        label, 
        expected_features, 
        expected_mi, 
        expected_is_leaf):
        """Test properties of LeafNode."""

        node = LeafNode(features, mi, label)
        assert node.features == expected_features
        assert node.mi == expected_mi
        assert node.is_leaf() is expected_is_leaf


    @pytest.mark.parametrize(
        'features, mi, synergy, children, expected_features, expected_mi, expected_synergy, expected_is_leaf, expected_children_features',
        [
            ((0, 1), 0.9, 0.1, ["Feature 0", "Feature 1"], (0, 1), 0.9, 0.1, False, [(0,), (1,)]),
            ((1, 2), 0.7, 0.2, ["Feature 1", "Feature 2"], (1, 2), 0.7, 0.2, False, [(1,), (2,)]),
            ((0, 2), 0.6, 0.15, ["Feature 0", "Feature 2"], (0, 2), 0.6, 0.15, False, [(0,), (2,)]),
        ]
    )
    def test_internal_node(self, features, mi, synergy, children, expected_features, expected_mi, expected_synergy, expected_is_leaf, expected_children_features):
        """Test properties of InternalNode including children."""

        leaf_nodes = [LeafNode((i,), 0.5 + i * 0.1, f"Feature {i}") for i in range(3)]
        feature_to_leaf = {f"Feature {i}": leaf_nodes[i] for i in range(3)}
        child_nodes = [feature_to_leaf[child] for child in children]
        node = InternalNode(features, mi, synergy, child_nodes)

        assert node.features == expected_features
        assert node.mi == expected_mi
        assert node.synergy == expected_synergy
        assert node.is_leaf() is expected_is_leaf
        children_features = [child.features for child in node.children]
        assert children_features == expected_children_features


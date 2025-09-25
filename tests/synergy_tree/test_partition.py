import pytest
from ppkt2synergy import  LeafNode, InternalNode
from ppkt2synergy import PartitionGenerator

@pytest.fixture
def nodes():
   
    leaf1 = LeafNode(feature_indices=(0,), mi=0.5, label="A")
    leaf2 = LeafNode(feature_indices=(1,), mi=0.7, label="B")
    leaf3 = LeafNode(feature_indices=(2,), mi=0.4, label="C")
    internal_node = InternalNode(feature_indices=(0, 1), mi=1.3, synergy=0.3, children=[leaf1, leaf2])
    
    nodes_dict = {
        (0,): leaf1,
        (1,): leaf2,
        (2,): leaf3,
        (0, 1): internal_node
    }
    return nodes_dict



@pytest.mark.parametrize("feature_indices, expected_partitions", [
    ([0, 1, 2], [
        [(0,), (1,), (2,)],
        [(2,), (0, 1)],
        [(1,), (0, 2)],
        [(0,), (1, 2)],
        [(0, 1, 2)]
    ]),
    ([0, 1], [
        [(0,), (1,)],
        [(0, 1)]
    ]),
    ([0], [
        [(0,)]
    ]),
    ([], [
        []
    ])
])
def test_generate_partitions(nodes, feature_indices, expected_partitions):
    partition_gen = PartitionGenerator(nodes)
    partitions = partition_gen.generate_partitions(feature_indices)
    assert sorted(partitions) == sorted(expected_partitions) 



@pytest.mark.parametrize(
    "features, expected_partition, should_return_none",
    [
        ([0, 1, 2], [(0, 1), (2,)], False),  # Standard case with three features
        ([1, 2], [(1,), (2,)], False),  # Standard case with two features
        ([0], [(0,)], False),  # Single feature should return itself as a partition
        ([], None, True),  # Empty feature list should return None
        ([0, 1, 3], None, True),  # Unknown feature (3) should return None
    ],
)
def test_find_max_partition(nodes, features, expected_partition, should_return_none):
    partition_gen = PartitionGenerator(nodes)

    # Compute the best partition
    best_partition = partition_gen.find_max_partition(features)

    if should_return_none:
        # If None is expected, verify that the result is None
        assert best_partition is None
    else:
        # Convert TreeNode objects into feature index tuples for comparison
        actual_partition = [tuple(sorted(node.get_feature_indices)) for node in best_partition]

        # Ensure the actual partition matches the expected one, ignoring order
        assert sorted(actual_partition) == sorted(expected_partition)
 


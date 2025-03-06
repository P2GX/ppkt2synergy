import pytest
from ppkt2synergy import  LeafNode, InternalNode
from ppkt2synergy import PartitionGenerator

@pytest.fixture
def nodes():
   
    leaf1 = LeafNode(features=(0,), mi=0.5, label="A")
    leaf2 = LeafNode(features=(1,), mi=0.7, label="B")
    leaf3 = LeafNode(features=(2,), mi=0.4, label="C")
    internal_node = InternalNode(features=(0, 1), mi=1.2, synergy=0.3, children=[leaf1, leaf2])
    
    nodes_dict = {
        (0,): leaf1,
        (1,): leaf2,
        (2,): leaf3,
        (0, 1): internal_node
    }
    return nodes_dict


def test_partition_generator_init(nodes):
    partition_gen = PartitionGenerator(nodes)
    assert partition_gen.nodes == nodes
    assert len(partition_gen.nodes) == 4 


def test_generate_partitions(nodes):
    partition_gen = PartitionGenerator(nodes)
    features = [0, 1, 2]
    
    partitions = partition_gen.generate_partitions(features)
    
    for partition in partitions:
        print(f"Generated Partition: {partition}")
    
    seen_partitions = set()
    for partition in partitions:
        seen_partitions.add(tuple(map(tuple, partition)))  
    
    assert len(partitions) == len(seen_partitions)  

def test_find_max_partition(nodes):
    partition_gen = PartitionGenerator(nodes)
    features = [0, 1]
    
    best_partition = partition_gen.find_max_partition(features)
    
    assert best_partition is not None
    assert len(best_partition) == 2  
    assert best_partition[0].mi == 0.5  
    print(f"Best Partition (MI = 1.2): {best_partition[0].features}")

def test_find_max_partition_no_valid_partition(nodes):
    partition_gen = PartitionGenerator(nodes)
    features = [3]  
    
    best_partition = partition_gen.find_max_partition(features)
    assert best_partition is None  

def test_generate_partitions_no_leaf(nodes):
    internal_node = InternalNode(features=(0, 1), mi=0.8, synergy=0.5, children=[])
    nodes_no_leaf = {
        (0, 1): internal_node
    }
    partition_gen = PartitionGenerator(nodes_no_leaf)
    partitions = partition_gen.generate_partitions([0, 1])

    for partition in partitions:
        print(f"Generated Partition: {partition}")
    
    assert len(partitions) > 0  

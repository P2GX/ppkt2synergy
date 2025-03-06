import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from .tree_node import TreeNode

class PartitionGenerator:
    """Generates and evaluates feature partitions for mutual information optimization."""

    def __init__(self, nodes: Dict[Tuple[int], TreeNode]):
        """
        Initializes the PartitionGenerator with a dictionary of TreeNode objects.

        Args:
            nodes (Dict[Tuple[int], TreeNode]): A dictionary where keys are tuples representing feature indices,
                                                 and values are TreeNode objects containing information about these feature subsets.
        """
        self.nodes = nodes

    def generate_partitions(self, features: List[int]) -> List[List[Tuple[int]]]:
        """
        Generates all unique partitions of feature_indices.
        Ensures that order of subsets does not affect uniqueness.
        
        features: List of feature indices
        return: List of unique partitions
        """
        n = len(features)
        all_partitions = []
        seen_partitions = set()
        
        if n == 1:
            return [[(features[0],)]]  
        
        for k in range(1, n):  
            for subset in combinations(features, k):
                remaining = tuple(sorted(set(features) - set(subset)))  # Remaining elements
                sub_partitions = self.generate_partitions(remaining)  # Recursively generate partitions
                
                for part in sub_partitions:
                    partition = tuple(sorted([(x,) if isinstance(x, int) else x for x in [subset] + part]))  
                    if partition not in seen_partitions:
                        seen_partitions.add(partition)
                        all_partitions.append([subset] + part)  #

        return all_partitions

    def find_max_partition(self, features: List[int]) -> Optional[List[TreeNode]]:
        """
        Find the optimal valid partition with the maximum mutual information (MI) sum.

        Args:
            features (List[int]): A list of feature indices to be partitioned and evaluated.

        Returns:
            Optional[List[TreeNode]]: A list of TreeNode objects corresponding to the best partition with the highest MI sum.
                                      If no valid partition is found, returns None.

        """
        all_partitions = self.generate_partitions(features)
        best_partition = None
        max_sum = -np.inf

        for partition in all_partitions:
            # Validate partition components
            if not all(tuple(sorted(subset)) in self.nodes for subset in partition):
                continue

            # Calculate partition score
            current_sum = sum(self.nodes[tuple(sorted(subset))].mi for subset in partition)
            if current_sum > max_sum:
                max_sum = current_sum
                best_partition = [self.nodes[tuple(sorted(subset))] for subset in partition]

        return best_partition
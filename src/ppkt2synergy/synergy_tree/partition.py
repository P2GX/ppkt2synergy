import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from .tree_node import TreeNode

class PartitionGenerator:
    """Generates and evaluates feature partitions for mutual information optimization."""

    def __init__(
            self, 
            nodes: Dict[Tuple[int], TreeNode]
            ):
        """
        Initializes the PartitionGenerator with a dictionary of TreeNode objects.

        Args:
            nodes (Dict[Tuple[int], TreeNode]): A dictionary where keys are tuples representing feature indices,
                                                 and values are TreeNode objects containing information about these feature subsets.
        """
        self.nodes = nodes

    def generate_partitions(
            self, 
            feature_indices: Tuple[int]
            ) -> List[List[Tuple[int]]]:
        """
        Generates all unique partitions of feature_indices.
        Ensures that the order of subsets does not affect uniqueness.

        The goal is to split the given Tuple of feature indices into all possible 
        partitions where subsets are unordered.

        Args:
        - feature_indices (Tuple[int]): A Tuple of feature indices to partition.

        Returns:
        - List[List[Tuple[int]]]: A list of unique partitions, where each partition 
        is a list of tuples representing feature subsets.
        
        Example:
        - Input: (1, 2, 3)
        - Possible partitions:
        [
            [(1,), (2, 3)],
            [(2,), (1, 3)],
            [(3,), (1, 2)],
            [(1,), (2,), (3,)],
            [(1,2,3)]
        ]
        """
        if not isinstance(feature_indices, tuple):
            feature_indices = tuple(sorted(feature_indices))
        n = len(feature_indices)
        all_partitions = []
        seen_partitions = set()
        
        if n == 0:
            return [[]]
        if n == 1:
            return [[feature_indices]]  
        
        # Iterate over all possible subset sizes (except the full set)
        for k in range(1, n):  

            # Example:
            # feature_indices = [1, 2, 3]
            # k = 2
            # combinations([1, 2, 3], 2) will return: [(1, 2), (1, 3), (2, 3)]
            for subset in combinations(feature_indices, k):
                
                # Example:
                # feature_indices = [1, 2, 3, 4]
                # subset = (1, 3)
                # remaining = [2, 4]
                remaining = tuple(sorted(set(feature_indices) - set(subset)))  

                # Recursively generate partitions for the remaining elements
                # Example:
                # Possible sub partitions:
                # [
                #    [(2,), (4,)],
                #    [(2,4)]
                # ]
                sub_partitions = self.generate_partitions(remaining) 
                
                # Combine the current subset with each of the sub_partitions
                # Example: if subset = (1, 3) and sub_partition = 
                # [
                #    [(2,), (4,)],
                #    [(2,4)]
                # ],
                # the resulting partition will be [(1, 3), (2,), (4,)],[(1, 3), (2, 4)]
                for part in sub_partitions:
                    partition = tuple(sorted([subset] + part))  
                    if partition not in seen_partitions:
                        seen_partitions.add(partition)
                        all_partitions.append([subset] + part)

        # Finally, add the partition where all features are in a single subset
        # Example: if feature_indices = [1, 2, 3, 4], the final partition will be [(1, 2, 3, 4)]
        all_partitions.append([tuple(sorted(feature_indices))])
        return all_partitions

    
    def find_max_partition(
            self, 
            feature_indices: Tuple[int]
            ) -> Optional[List[TreeNode]]:
        """
        Find the optimal valid partition with the maximum mutual information (MI) sum.

        This function generates all possible partitions of the provided list of feature indices and 
        evaluates them by summing the mutual information (MI) for each subset in the partition.
        The partition with the highest MI sum is returned.

        Args:
            features (Tuple[int]): A Tuple of feature indices to be partitioned and evaluated.
                                Example: (1, 2, 3)

        Returns:
            Optional[List[TreeNode]]: A list of TreeNode objects corresponding to the best partition
                                    with the highest MI sum.
                                    If no valid partition is found, returns None.
                                    Example: [TreeNode1, TreeNode2]

        Example:
            If the input features are (1, 2, 3), the function will first generate all possible 
            partitions like:
            [
                [(1,), (2,), (3,)]
                [(1,), (2, 3)],
                [(2,), (1, 3)],
                [(3,), (1, 2)],
                [(1, 2, 3)]  # This partition will be excluded as it contains all features.
            ]
            Then, it will calculate the mutual information for each valid partition, and the one 
            with the highest sum will be returned.
        """
        
        n = len(feature_indices)
        if n == 0:
            return None  # Return None if the input is empty
        if n== 1:
            return [self.nodes[tuple(feature_indices)]]
        
        # Generate all possible partitions of the feature indices
        all_partitions = self.generate_partitions(feature_indices)

        # Filter out partitions that contain all features as one subset
        # Example: If the features = [1, 2, 3], partitions like [(1, 2, 3)] will be excluded
        all_partitions = [part for part in all_partitions if len(part) != 1]

        best_partition = None
        max_sum = -np.inf  # Initialize the maximum sum to a very low value

        # Iterate through all valid partitions
        for partition in all_partitions:
            # Validate partition components to ensure they are present in self.nodes
            # Example: For partition [(1,), (2, 3)], it checks if (1,) and (2, 3) exist in self.nodes
            if not all(tuple(sorted(subset)) in self.nodes for subset in partition):
                continue  # Skip invalid partitions

            # Get the mutual information (MI) sum for the current partition
            # Example: For partition [(1,), (2, 3)], it calculates MI for (1,) and (2, 3)
            current_sum = sum(self.nodes[tuple(sorted(subset))].get_mi for subset in partition)

            # If the current partition has a higher MI sum, update the best partition
            if current_sum > max_sum:
                max_sum = current_sum
                best_partition = [self.nodes[tuple(sorted(subset))] for subset in partition]

        return best_partition  

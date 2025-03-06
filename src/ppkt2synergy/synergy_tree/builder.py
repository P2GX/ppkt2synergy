import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from .tree_node import TreeNode,LeafNode,InternalNode
from .mi_calculator import MutualInformationCalculator
from .partition import PartitionGenerator

class SynergyTreeBuilder:
    """
    Builds synergy trees from feature data and target variable.
    """
    def __init__(
            self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            max_k: Optional[int] = None
            ):
        """
        X: Feature matrix (DataFrame with column names)
        y: Target variable (binary Series)
        max_k: Maximum feature combination order to build
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("Inputs must be pandas DataFrame and Series")

        self.feature_names = X.columns.tolist()
        self.mi_calculator = MutualInformationCalculator(X.values.astype(int), 
                                             y.values.astype(int))
        self.nodes: Dict[Tuple[int], TreeNode] = {}
        self.roots: Dict[Tuple[int], TreeNode] = {}
        self.max_k = max_k or X.shape[1]
        self.partition_gen = PartitionGenerator(self.nodes)  # Inject node cache

    def build(self) -> List[TreeNode]:
        """
        Builds and returns synergy trees starting from the root nodes.
        """
        self._init_leaf_nodes()
        for k in range(2, self.max_k + 1):
            self._build_layer(k)
        return list(self.roots.values())

    def _init_leaf_nodes(self) -> None:
        """
        Initializes leaf nodes for individual features.
        """
        for i in range(self.mi_calculator.feature_matrix.shape[1]):
            features = (i,)
            mi = self.mi_calculator.compute_mi(features)
            self.nodes[features] = LeafNode(features, mi, self.feature_names[i])

    def _build_layer(self, k: int) -> None:
        """
        Builds nodes for feature combinations of order k.
        """
        new_nodes = {}
        all_features = list(range(self.mi_calculator.feature_matrix.shape[1]))

        for feature_comb in combinations(all_features, k):
            features = tuple(sorted(feature_comb))
            if features in self.nodes:
                continue
            
            joint_mi = self.mi_calculator.compute_mi(features)
            best_partition = self.partition_gen.find_max_partition(list(features))
            if not best_partition:
                continue

            synergy = joint_mi - sum(child.mi for child in best_partition)
            if synergy <= 0:
                continue

            children =  best_partition
            new_node = InternalNode(features, joint_mi, synergy, children)
            new_nodes[features] = new_node
            self._update_roots(new_node, children)

        self.nodes.update(new_nodes)

    
    def _update_roots(
            self, 
            new_node: InternalNode, 
            children: List[TreeNode]
            ) -> None:
        """
        Updates root nodes by removing old children and adding the new node.
        """
        for child in children:
            if child.features in self.roots:
                del self.roots[child.features]
        self.roots[new_node.features] = new_node
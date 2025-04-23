import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from .tree_node import TreeNode,LeafNode,InternalNode
from .mi_calculator import MutualInformationCalculator
from .partition import PartitionGenerator
import math

class SynergyTreeBuilder:
    """
    Builds synergy trees from feature data and a binary target variable.

    This class constructs a hierarchy of feature combinations based on their mutual 
    information (MI) with the target variable. It builds a tree structure where each 
    node represents a combination of features, and the edges capture their synergy.
    
    Attributes:
        feature_names (List[str]): List of feature names.
        mi_calculator (MutualInformationCalculator): Object to compute mutual information.
        nodes (Dict[Tuple[int], TreeNode]): Cache storing computed nodes.
        roots (Dict[Tuple[int], TreeNode]): Root nodes of the constructed synergy trees.
        max_k (int): Maximum combination size (tree depth).
        partition_gen (PartitionGenerator): Helper to generate feature partitions.
    
    Example:
            >>> X = pd.DataFrame({'A': [0,1,1], 'B': [1,0,1], 'C': [0,1,0]})
            >>> y = pd.Series([1, 0, 1])
            >>> builder = SynergyTreeBuilder(X, y, max_k=3)
            >>> trees = builder.build()
            >>> for tree in trees:
            ...     print(tree)
    """
    def __init__(
            self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            max_k: Optional[int] = None
            ):
        """
        Initializes the SynergyTreeBuilder.

        Args:
            X (pd.DataFrame): Feature matrix of shape `(n_samples, n_features)`, 
                            where each column represents a feature.
            y (pd.Series): Binary target variable of shape `(n_samples,)`, 
                        containing class labels (0 or 1).
            max_k (Optional[int]): Maximum feature combination order. If provided, 
                                it will be `min(max_k, n_features)`. If `None`, 
                                it defaults to `n_features`.

        Raises:
            TypeError: If X is not a DataFrame or y is not a Series.
        """
        # Validate input
        self._validate_input(X, y, max_k) 

        self.feature_names = X.columns.tolist()
        self.mi_calculator = MutualInformationCalculator(X.values.astype(int), 
                                             y.values.astype(int))
        self.nodes: Dict[Tuple[int], TreeNode] = {}
        self.roots: Dict[Tuple[int], TreeNode] = {}
        self.max_k = min(math.ceil(max_k), X.shape[1]) if max_k > 0 else X.shape[1]
        self.partition_gen = PartitionGenerator(self.nodes)

    def _validate_input(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        max_k: Optional[int] =None
    ) -> None:
        """
        Validates the input data for SynergyTreeBuilder.

        Args:
            X (pd.DataFrame): Feature matrix of shape `(n_samples, n_features)`.
            y (pd.Series): Binary target variable of shape `(n_samples,)`.
            max_k (Optional[int]): Maximum feature combination order.

        Raises:
            ValueError: If X or y is None.
            TypeError: If X is not a DataFrame or y is not a Series.
            ValueError: If the number of features is less than 2.
            ValueError: If max_k is invalid.
        """
        # Check if X or y is None
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        # Check if X is a DataFrame and y is a Series
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be pandas DataFrame")
        
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("If y is a DataFrame, it must have exactly one column")
            y = y.iloc[:, 0]  # 转为 Series

        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series or a single-column DataFrame")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of rows in X and y must match")

        # Check if the number of features is at least 2
        if X.shape[1] < 2:
            raise ValueError("At least 2 features are required to build a synergy tree")
        
        if X.shape[0] < 20:
            raise ValueError("At least 20 samples are required")
        
        if X.isnull().values.any() or y.isnull().values.any():
            raise ValueError("X and y must not contain NaN values")

        # Check if max_k is valid
        if max_k is not None and max_k <= 0:
            raise ValueError(f"max_k must be between 2 and {X.shape[1]}")
          
    def build(
            self
            ) -> List[TreeNode]:
        """
        Builds synergy trees from individual feature nodes up to max_k order.

        Returns:
            List[TreeNode]: A list of root nodes representing different trees.
        """
        self._init_leaf_nodes()
        for k in range(2, self.max_k + 1):
            self._build_layer(k)
        return list(self.roots.values())

    def _init_leaf_nodes(
            self
            ) -> None:
        """
        Initializes tree leaf nodes for individual features.

        This method computes the mutual information (MI) of each feature with `y`
        and stores it as a `LeafNode`.

        Example:
            Given feature matrix X with 3 features, it initializes:
            {
                (0,): LeafNode(feature=(0,), mi=0.5),
                (1,): LeafNode(feature=(1,), mi=0.3),
                (2,): LeafNode(feature=(2,), mi=0.4)
            }
        """
        for i in range(self.mi_calculator.feature_matrix.shape[1]):
            features = (i,) # Single-feature tuple
            mi = self.mi_calculator.compute_mi(features)
            self.nodes[features] = LeafNode(features, mi, self.feature_names[i])

    def _build_layer(
            self, 
            k: int
            ) -> None:
        """
        Constructs feature combinations of order `k` and adds them to the tree.

        This method generates all possible feature subsets of size `k`, calculates 
        their joint mutual information, and determines the best partition using 
        `PartitionGenerator`.

        Args:
            k (int): The feature combination size.

        Example:
            When `k=2`, this method generates all feature pairs:
            - (0, 1), (0, 2), (1, 2)
            - Computes their joint MI
            - Determines if they provide synergistic information
        """
        new_nodes = {}
        all_features = list(range(len(self.feature_names)))

        # Generate feature subsets of size k
        for feature_comb in combinations(all_features, k):
            features = tuple(sorted(feature_comb))
            if features in self.nodes:
                continue
            
            joint_mi = self.mi_calculator.compute_mi(features)
            best_partition = self.partition_gen.find_max_partition(features)
            if not best_partition:
                continue

            # Calculate synergy (joint MI - sum of MI from partitions)
            synergy = joint_mi - sum(child.get_mi for child in best_partition)
            if synergy <= 0:
                continue

            # Create an InternalNode and update the structure
            children =  best_partition
            new_node = InternalNode(features, joint_mi, synergy, children)
            new_nodes[features] = new_node
            self._update_roots(new_node, children)
        # Add new nodes to the cache
        self.nodes.update(new_nodes)

    
    def _update_roots(
            self, 
            new_node: InternalNode, 
            children: List[TreeNode]
            ) -> None:
        """
        Updates root nodes by removing children and adding the new node.

        Args:
            new_node (InternalNode): Newly created feature combination node.
            children (List[TreeNode]): The child nodes forming the new node.

        Example:
            Suppose we initially have root nodes:
            roots = {
                (0,1): InternalNode(0,1),
                (1,2): InternalNode(1,2),
                (0,2): InternalNode(0,2)
            }
            If a new node (0, 1, 2) with children (0,1),(2) is created, we remove (0,1) from `roots`
            and add (0,1,2).

            Updated roots:
            roots = {
                (0, 1, 2): InternalNode(0,1,2),
                (1, 2): InternalNode(1,2),
                (0, 2): InternalNode(0,2)
            }
        """
        for child in children:
            if child.get_feature_indices in self.roots:
                del self.roots[child.get_feature_indices]
        self.roots[new_node.get_feature_indices] = new_node

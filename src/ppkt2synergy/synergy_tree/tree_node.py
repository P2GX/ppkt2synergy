
from typing import  List, Tuple
from abc import ABCMeta, abstractmethod

class TreeNode(metaclass=ABCMeta):
    """
    Abstract base class for all tree nodes
    """

    @property
    @abstractmethod
    def get_feature_indices(self) -> Tuple[int]:
        """Returns the feature indices associated with the node."""
        pass

    @property
    @abstractmethod
    def get_mi(self) -> float:
        """Returns the mutual information score for this node."""
        pass

    @abstractmethod
    def is_leaf(self) -> bool:
        """Returns whether the node is a leaf."""
        pass

class LeafNode(TreeNode):
    """
    Represents a leaf node in the tree.
    """
    def __init__(self, feature_indices: Tuple[int], mi: float, label: str):
        """
        Initializes the leaf node with features, MI score, and label.
        """
        self._feature_indices = feature_indices
        self._mi = mi
        self.label = label

    @property
    def get_feature_indices(self) -> Tuple[int]:
        return self._feature_indices

    @property
    def get_mi(self) -> float:
        return self._mi

    def is_leaf(self) -> bool:
        return True

class InternalNode(TreeNode):
    """
    Represents an internal node in the tree.
    """
    def __init__(self, feature_indices: Tuple[int], mi: float, synergy: float, children: List[TreeNode]):
        """
        Initializes the internal node with features, MI score, synergy score, and children.
        """
        self._feature_indices = feature_indices
        self._mi = mi
        self.synergy = synergy
        self.children = children

    @property
    def get_feature_indices(self) -> Tuple[int]:
        return self._feature_indices

    @property
    def get_mi(self) -> float:
        return self._mi
    
    @property
    def get_synergy(self) -> float:
        return self.synergy
        
    @property
    def get_children(self) -> List[TreeNode]:
        return self.children

    def is_leaf(self) -> bool:
        return False


from typing import  List, Tuple
from abc import ABCMeta, abstractmethod

class TreeNode(metaclass=ABCMeta):
    """
    Abstract base class for all tree nodes
    """

    @property
    @abstractmethod
    def features(self) -> Tuple[int]:
        """Returns the feature indices associated with the node."""
        pass

    @property
    @abstractmethod
    def mi(self) -> float:
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
    def __init__(self, features: Tuple[int], mi: float, label: str):
        """
        Initializes the leaf node with features, MI score, and label.
        """
        self._features = features
        self._mi = mi
        self.label = label

    @property
    def features(self) -> Tuple[int]:
        return self._features

    @property
    def mi(self) -> float:
        return self._mi

    def is_leaf(self) -> bool:
        return True

class InternalNode(TreeNode):
    """
    Represents an internal node in the tree.
    """
    def __init__(self, features: Tuple[int], mi: float, synergy: float, children: List[TreeNode]):
        """
        Initializes the internal node with features, MI score, synergy score, and children.
        """
        self._features = features
        self._mi = mi
        self.synergy = synergy
        self.children = children

    @property
    def features(self) -> Tuple[int]:
        return self._features

    @property
    def mi(self) -> float:
        return self._mi

    def is_leaf(self) -> bool:
        return False

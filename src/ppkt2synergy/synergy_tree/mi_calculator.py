from functools import lru_cache
import numpy as np
from sklearn.metrics import mutual_info_score
from typing import Tuple

class MutualInformationCalculator:
    """Computes mutual information (MI) between features and a binary target."""
    
    def __init__(self, feature_matrix: np.ndarray, target: np.ndarray):
        """Initializes the calculator with validated data.
        
        Args:
            feature_matrix: Binary matrix (n_samples x n_features) with 0/1 values.
            target: Binary target array (n_samples,) with 0/1 values.
            
        Raises:
            ValueError: If input dimensions mismatch.
            ValueError: If any value is not 0 or 1.
            ConstantTargetError: If target has no variation.
        """
        self.feature_matrix = feature_matrix
        self.target = target
        self._validate_input()


    def _validate_input(self):
        """Validates input data for binary values and consistency."""

        if not (np.all(np.isin(self.feature_matrix, [0, 1])) and 
                np.all(np.isin(self.target, [0, 1]))):
            raise ValueError("Features and target must contain only 0 and 1")
     
        if np.all(self.target == 0) or np.all(self.target == 1):
            raise ValueError("Target has no variation (all values identical)")
     
        if self.feature_matrix.shape[0] != self.target.shape[0]:
            raise ValueError(f"Feature matrix rows ({self.feature_matrix.shape[0]}) "
                             f"must match target length ({len(self.target)})")


    @lru_cache(maxsize=None)
    def compute_mi(self, feature_indices: Tuple[int]) -> float:
        """Computes mutual information for a feature combination.
        
        Args:
            feature_indices: Indices of features to combine (e.g., (0, 2)).
            
        Returns:
            Mutual information score in bits.
            
        Raises:
            IndexError: If any index is out of bounds.
        """
        if not all(0 <= idx < self.feature_matrix.shape[1] for idx in feature_indices):
            raise ValueError("Feature indices must be within the valid range of the feature matrix columns")

        joint_state = self.feature_matrix[:, list(feature_indices)]
        state_str = np.apply_along_axis(lambda x: ''.join(x.astype(str)), 1, joint_state)
        return mutual_info_score(state_str, self.target)


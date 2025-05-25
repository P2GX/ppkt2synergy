from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle 
from typing import Tuple, Union, Optional

class PairwiseSynergyAnalyzer:
    """
    Analyzes the pairwise synergy between features using mutual information and permutation testing.
    
    Example:
        from ppkt2synergy import CohortDataLoader, PhenopacketMatrixProcessor,PairwiseSynergyAnalyzer
        >>> phenopackets = CohortDataLoader.from_ppkt_store("FBN1")
        >>> hpo_matrix, disease_status = PhenopacketMatrixProcessor.prepare_hpo_data(
        ...     phenopackets, threshold=0.5, mode="leaf", use_label=True)
        >>> analyzer = PairwiseSynergyAnalyzer(hpo_matrix, target, n_permutations=100)
        >>> synergy, pvalues = analyzer.compute_pairwise_synergy_matrix()
        >>> analyzer.plot_synergy_heatmap(significance_threshold=0.05)
    """
    def __init__(
            self, 
            hpo_data: Tuple[pd.DataFrame,Optional[pd.DataFrame]], 
            target: Union[pd.Series, pd.DataFrame],
            n_permutations: int = 100,
            min_individuals_for_synergy_caculation: int = 40, 
            random_state: int = 42,
            relationship_mask: Union[pd.DataFrame, np.ndarray] = None
        ):
        """
        Initialize the analyzer with feature data and target variable.

        Agrs:
        hpo_data(Tuple[pd.DataFrame,Optional[pd.DataFrame]]):
            - Feature matrix of shape (n_samples, n_features): 
                Non-NaN values must be 0 or 1. DataFrame inputs will be converted to a NumPy array.
            - relationship_mask (n_features, n_features):
                Optional 2D array (n_features x n_features) indicating valid feature pairs to evaluate.
                Can be used to skip predefined pairs (e.g. based on HPO hierarchy or previous results).
                If provided, it will be converted to a NumPy array and used to initialize the synergy matrix.
        target(Union[pd.Series, pd.DataFrame]):
            Target vector of shape (n_samples,). Series/DataFrame inputs will be converted to a 1D NumPy array.
        n_permutations(int): (default: 100)
            Number of permutations for calculating p-values.
        min_individuals_for_synergy_caculation(int): (default: 40)
            Minimum number of samples required to calculate synergy.
        random_state(int): (default: 42)
            Seed for reproducible results.

        Raises:
        ValueError:
            - If hpo_matrix is not a 2D array.
            - If target's length does not match hpo_matrix's row count.
            - If hpo_matrix contains values other than 0, 1, or NaN.
            - If mask has an incompatible shape.
            - If min_individuals_for_synergy_caculation is less than 40.
        """
        if isinstance(hpo_data, tuple):
            hpo_matrix, relationship_mask = hpo_data
        else:
            raise TypeError("hpo_data must be a tuple of (hpo_matrix, relationship_mask)")
        if isinstance(hpo_matrix, pd.DataFrame):
            self.hpo_terms = hpo_matrix.columns 
            hpo_matrix = hpo_matrix.to_numpy()
        else:
            raise TypeError("hpo_matrix must be a pandas DataFrame")
        
        if not np.all(np.isin(hpo_matrix[~np.isnan(hpo_matrix)], [0, 1])):
            raise ValueError("Non-NaN values in HPO Matrix must be either 0 or 1")
        
        if isinstance(target, (pd.Series, pd.DataFrame)):
            target = target.to_numpy().ravel()
        
        if len(target) != hpo_matrix.shape[0]:
            raise ValueError("The number of samples in Target must match the number of samples in HPO Matrix")
        if not np.all(np.isin(target[~np.isnan(target)], [0, 1])):
            raise ValueError("Target must contain only 0, 1, or NaN")
        

        self.X = hpo_matrix.astype(float)
        self.y = target
        self.n_features = hpo_matrix.shape[1]
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_state)

        if relationship_mask is not None:
            if isinstance(relationship_mask, pd.DataFrame):
                relationship_mask_numpy = relationship_mask.to_numpy() 
            elif isinstance(relationship_mask, np.ndarray):
                relationship_mask_numpy = relationship_mask
            else:
                raise ValueError("mask must be a pd.DataFrame or np.ndarray")
            if relationship_mask_numpy.shape[0] != relationship_mask_numpy.shape[1] or \
                relationship_mask_numpy.shape[0] != hpo_matrix.shape[1]:
                raise ValueError("mask must have the same number of rows and columns as hpo_matrix has features")
            if not np.all(np.isin(relationship_mask_numpy[~np.isnan(relationship_mask_numpy)], [0])):
                raise ValueError("relationship_mask must contain only 0 or NaN")
            self.synergy_matrix = relationship_mask_numpy.astype(float)
        else:
            self.synergy_matrix = np.zeros((self.n_features, self.n_features))

        self.pvalue_matrix = np.ones((self.n_features, self.n_features))

        if min_individuals_for_synergy_caculation < 40:
            raise ValueError("min_individuals_for_synergy_caculation must be greater than 40.")
        self.min_individuals_for_synergy_caculation = min_individuals_for_synergy_caculation

    @staticmethod
    def _encode_joint_binary_index( 
            xi:np.ndarray, 
            xj:np.ndarray
        ) -> np.ndarray:
        """
        Encodes two binary features into a unique integer index via bitwise operations.

        Args:
        xi(ndarray): 
            Feature i's values (int type, 0/1, no NaN).
        xj(ndarray): 
            Feature j's values (int type, 0/1, no NaN).

        Returns:
            ndarray:
            Combined index calculated as: 
            \( \text{joint\_index} = 2 \times \text{xi} + \text{xj} \)
            Possible values: 0 (0b00), 1 (0b01), 2 (0b10), 3 (0b11).

        Example:
        >>> xi = np.array([0, 1, 0, 1], dtype=int)
        >>> xj = np.array([0, 0, 1, 1], dtype=int)
        >>> _encode_joint_binary_index(xi, xj)
        array([0, 2, 1, 3])  # Corresponding to 0b00, 0b10, 0b01, 0b11
        """
        return (xi.astype(int) << 1) | xj.astype(int)

    def evaluate_pair_synergy(
            self, 
            i:int,
            j:int
        ) -> Tuple[int, int, float, float]: 
        """
        Compute synergy and permutation-based p-value for feature pair (i, j).

        Synergy is calculated as:
            synergy = I(X_i, X_j; Y) - [I(X_i; Y) + I(X_j; Y)]

        where I(a; b) denotes the mutual information (in bits) between a and b.

        Args:
        i (int): 
            Index of the first feature.
        j (int): 
            Index of the second feature.

        Returns:
            Tuple[int, int, float, float]:
                - i (int): Index of the first feature.
                - j (int): Index of the second feature.
                - corrected_synergy (float): Corrected synergy score.
                - p_value (float): Empirical p-value from permutation test.
        """
        mask = (
            ~np.isnan(self.X[:, i]) 
            & ~np.isnan(self.X[:, j]) 
            & ~np.isnan(self.y)
        )
        xi = self.X[mask, i]
        xj = self.X[mask, j]
        y = self.y[mask]
        if len(y) < self.min_individuals_for_synergy_caculation: 
            return i, j, np.nan, 1.0

        xi = xi.astype(int)
        xj = xj.astype(int)

        if np.all(xi == xi[0]) or np.all(xj == xj[0]) or np.all(y == y[0]):
            return i, j, np.nan, 1.0
        
        if np.array_equal(xi, xj):
            return i, j, 0.0, 1.0

        mi_i = mutual_info_score(xi, y) / np.log(2)
        mi_j = mutual_info_score(xj, y) / np.log(2)

        joint_index = self._encode_joint_binary_index(xi, xj)
        mi_ij = mutual_info_score(joint_index, y) / np.log(2)

        observed_synergy = mi_ij - (mi_i + mi_j)

        # Permutation testing for p-value calculation
        perm_synergies = np.zeros(self.n_permutations)
        for k in range(self.n_permutations):
            y_perm = self.rng.permutation(y)  # Shuffle the target values
            mi_i_perm = mutual_info_score(xi, y_perm) / np.log(2)
            mi_j_perm = mutual_info_score(xj, y_perm) / np.log(2)
            mi_ij_perm = mutual_info_score(joint_index, y_perm) / np.log(2)
            perm_synergies[k] = mi_ij_perm - (mi_i_perm + mi_j_perm)

        # Calculate p-value as the proportion of permuted synergies greater than or equal to the observed synergy
        p_value = (np.abs(perm_synergies) >= np.abs(observed_synergy)).mean()
        
        # Correct the observed synergy by subtracting the mean of the permuted synergies
        corrected_synergy = observed_synergy - perm_synergies.mean()

        return i, j, corrected_synergy, p_value

    def compute_synergy_matrix(
            self, 
            n_jobs=-1
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes the pairwise synergy scores and permutation-based p-values for all feature pairs.

        Only feature pairs not masked by the synergy matrix (i.e., not NaN) will be evaluated.
        Results are stored in symmetric matrices and converted to pandas DataFrames with feature names.

        Args:
            n_jobs (int, optional): (default: -1)
                Number of parallel jobs to run. Set to -1 to use all available CPU cores.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - synergy_matrix (pd.DataFrame): 
                    Symmetric matrix (n_features x n_features) containing synergy scores between feature pairs.
                - pvalue_matrix (pd.DataFrame): 
                    Symmetric matrix of empirical p-values for each synergy score.
        """
        combinations_list = [    
            (i, j) for i in range(self.n_features) for j in range(i + 1, self.n_features)
            if not np.isnan(self.synergy_matrix[i, j])
        ]

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.evaluate_pair_synergy)(i, j) for i, j in combinations_list
        )
        for i, j, synergy, pval in results:
            self.synergy_matrix[i, j] = self.synergy_matrix[j, i] = synergy
            self.pvalue_matrix[i, j] = self.pvalue_matrix[j, i] = pval

        valid_mask = ~(
            (np.isnan(self.synergy_matrix).all(axis=0)) | 
            (np.nan_to_num(self.synergy_matrix, nan=0).sum(axis=0) == 0)
        )
        valid_hpo_terms = self.hpo_terms[valid_mask]

        np.fill_diagonal(self.synergy_matrix, np.nan)
        np.fill_diagonal(self.pvalue_matrix, np.nan)

        self.synergy_matrix = pd.DataFrame(
            self.synergy_matrix[np.ix_(valid_mask, valid_mask)],
            index=valid_hpo_terms,
            columns=valid_hpo_terms
        )

        self.pvalue_matrix = pd.DataFrame(
            self.pvalue_matrix[np.ix_(valid_mask, valid_mask)],
            index=valid_hpo_terms,
            columns=valid_hpo_terms
        )

        return self.synergy_matrix, self.pvalue_matrix
    
    def filter_weak_synergy(
            self, 
            lower_bound: float = 0.1, 
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter out feature pairs with weak synergy based on given threshold.

        Args:
            lower_bound (float): (default: 0.1)
                The minimum synergy value to keep.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - Synergy score matrix with weak synergy pairs removed (set as NaN).
                - Corresponding p-value matrix with the same filtering applied.
        """
        synergy_matrix, p_value = self.synergy_matrix.copy(), self.pvalue_matrix.copy()

        # Mask weak synergy values
        mask = synergy_matrix < lower_bound
        synergy_matrix[mask] = np.nan
        p_value[mask] = np.nan

        # Remove rows/columns that are completely NaN
        mask_rows = synergy_matrix.isna().all(axis=1)
        mask_cols = synergy_matrix.isna().all(axis=0)
        synergy_matrix_cleaned = synergy_matrix.loc[~mask_rows, ~mask_cols]
        p_value_cleaned = p_value.loc[~mask_rows, ~mask_cols]

        return synergy_matrix_cleaned, p_value_cleaned    


    def plot_synergy_heatmap(
            self, 
            lower_bound: float = 0.1,
            significance_threshold : float=0.05,
            target_name: str = ""
        ) -> None:
        """
        Plot a heatmap of synergy scores with red boxes around significant pairs.

        Args:
        lower_bound (float): (default: 0.1)
            Minimum synergy value to keep in the heatmap.
        significance_threshold (float): (default: 0.05)
            Significance threshold for p-values.
        target_name (str): (default: "")
            Name of the target variable for the plot title.
        """
        synergy_matrix, pvalue_matrix = self.filter_weak_synergy(lower_bound=lower_bound)

        if synergy_matrix.empty or np.isnan(synergy_matrix.values).all():
            raise ValueError("No sufficient synergy pairs to plot. Try adjusting the lower_bound parameter.")

        num_rows, num_cols = synergy_matrix.shape
        cell_size = 0.6
        min_figsize = (4, 3)

        figsize = (max(cell_size * num_cols, min_figsize[0])*1.1, 
                max(cell_size * num_rows, min_figsize[1]))

        cell_width = figsize[0] / max(num_cols, 1)
        cell_height = figsize[1] / max(num_rows, 1)
        base_fontsize = min(cell_width, cell_height) * 18
        annot_fontsize = base_fontsize * 0.8

        nan_mask = synergy_matrix.isna()
        cmap = plt.get_cmap("managua").copy()
        cmap.set_bad(color='#F5F5F5')

        plt.figure(figsize=figsize, dpi=150)
        ax = sns.heatmap(
            synergy_matrix,
            annot=True, 
            fmt=".2f", 
            cmap=cmap, 
            center=0, 
            square=True,
            linewidths=0.5,
            linecolor="gray", 
            cbar_kws={"shrink": 0.8,"label": "Corrected Synergy"},
            mask=nan_mask,
            annot_kws={"size": annot_fontsize}
        )

        ax.set_xticks(np.arange(synergy_matrix.shape[1]) + 0.5)
        ax.set_yticks(np.arange(synergy_matrix.shape[0]) + 0.5)
        ax.set_xticklabels(synergy_matrix.columns, rotation=90, fontsize=base_fontsize)
        ax.set_yticklabels(synergy_matrix.index, rotation=0, fontsize=base_fontsize)

        # Add red boxes for significant pairs (p < alpha)
        for i in range(num_rows):
            for j in range(num_rows):
                if i < j and pvalue_matrix.iloc[i, j] < significance_threshold:
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))
                    ax.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor='red', lw=2))

        plt.title(f"Pairwise Synergy Heatmap based on {target_name} \n(Significant values p < {significance_threshold} highlighted)")
        plt.tight_layout()
        plt.show()
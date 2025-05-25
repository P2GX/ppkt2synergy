import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, List,Optional
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import scipy.stats


class HPOStatisticsAnalyzer:

    """
    Analyze pairwise statistical relationships between HPO terms and disease-related targets in a cohort.

    This class provides tools to compute and visualize correlations between HPO terms and disease status 
    (or variant effect matrices), using a variety of statistical tests such as Spearman, Kendall, or Phi coefficient.
    
    It supports filtering weak correlations and highlighting statistically significant relationships in a heatmap.

    Example:
        from ppkt2synergy import CohortDataLoader, HPOStatisticsAnalyzer
        >>> phenopackets = CohortDataLoader.from_ppkt_store('FBN1')
        >>> hpo_matrix, target_matrix = PhenopacketMatrixProcessor.prepare_hpo_data(
                phenopackets, threshold=0.5, mode='leaf', use_label=True)
        >>> analyzer =HPOStatisticsAnalyzer(hpo_matrix, target_matrix, min_individuals_for_correlation_test=40)
        >>> coef_matrix, pval_matrix = analyzer.compute_correlation_matrices("Spearman")
        >>> analyzer.plot_correlation_heatmap_with_significance("Spearman")
    
    Notes:
        - Requires at least 30 valid data points per pairwise comparison.
        - Assumes binary input matrices (0/1 presence/absence format).
    """
    def __init__(
            self,  
            hpo_data: Tuple[pd.DataFrame,Optional[pd.DataFrame]], 
            target_matrix: Union[pd.Series, pd.DataFrame],
            min_individuals_for_correlation_test: int = 30,
        ):
            """
            Initialize the HPOStatisticsAnalyzer.

            Args:
            hpo_data(Tuple[pd.DataFrame,Optional[pd.DataFrame]]):
            - Feature matrix of shape (n_samples, n_features): 
                Non-NaN values must be 0 or 1. DataFrame inputs will be converted to a NumPy array.
            - relationship_mask (n_features, n_features):
                Optional 2D array (n_features x n_features) indicating valid feature pairs to evaluate.
                Can be used to skip predefined pairs (e.g. based on HPO hierarchy or previous results).
                If provided, it will be converted to a NumPy array and used to initialize the synergy matrix.
            target_matrix(pd.DataFrame):
                Target vector of shape (n_samples, n_targets)
            min_individuals_for_correlation_test(int): (default: 30)
                Minimum number of valid individuals required to perform correlation tests.

            Raises:
                ValueError: If min_individuals_for_correlation_test is less than 30.
            """
            if isinstance(hpo_data, tuple):
                hpo_matrix, relationship_mask = hpo_data
            else:
                raise TypeError("hpo_data must be a tuple of (hpo_matrix, relationship_mask)")
            if isinstance(hpo_matrix, pd.DataFrame):
                self.hpo_terms = hpo_matrix.columns 
                self.n_features = hpo_matrix.shape[1]
            else:
                raise TypeError("hpo_matrix must be a pandas DataFrame")
            
            if not np.all(np.isin(hpo_matrix.to_numpy()[~np.isnan(hpo_matrix.to_numpy())], [0, 1])):
                raise ValueError("Non-NaN values in HPO Matrix must be either 0 or 1")
            
            if not isinstance(target_matrix, pd.DataFrame):
                raise TypeError("target_matrix must be a pandas DataFrame")
            
            if target_matrix.shape[0] != hpo_matrix.shape[0]:
                raise ValueError("The number of samples in Target Matrix must match the number of samples in HPO Matrix")
            if not np.all(np.isin(target_matrix.to_numpy()[~np.isnan(target_matrix.to_numpy())], [0, 1])):
                raise ValueError("Target Matrix must contain only 0, 1, or NaN")
            
            self.relationship_mask = None
            if relationship_mask is not None:
                if isinstance(relationship_mask, pd.DataFrame):
                    self.relationship_mask = relationship_mask.to_numpy() 
                elif isinstance(relationship_mask, np.ndarray):
                    self.relationship_mask = relationship_mask
                else:
                    raise ValueError("mask must be a pd.DataFrame or np.ndarray")
                    
                if relationship_mask.shape[0] != relationship_mask.shape[1] or \
                    relationship_mask.shape[0] != hpo_matrix.shape[1]:
                    raise ValueError("mask must have the same number of rows and columns as hpo_matrix has features")
                if not np.all(np.isin(self.relationship_mask[~np.isnan(self.relationship_mask)], [0])):
                    raise ValueError("relationship_mask must contain only 0 or NaN")
            
            self.combined_matrix = pd.concat([hpo_matrix,target_matrix], axis=1).reindex(hpo_matrix.index)
            
            if min_individuals_for_correlation_test < 30:
                raise ValueError("min_individuals_for_correlation_test must not be less than 30.")
            self.min_individuals_for_correlation_test = min_individuals_for_correlation_test
            self._supported_stats = {"spearman", "kendall", "phi"}

    def _calculate_pairwise_stats( 
            self,
            observed_status_A: np.ndarray, 
            observed_status_B: np.ndarray,
            stats_name: str = "spearman"
        ) -> Dict[str, Union[float, str]]:
            """
            Calculate selected statistical metric (spearman, kendall, or phi) and its p-value
            for two binary (0/1) observed status vectors.

            Args:
                observed_status_A(np.ndarray): 
                    Binary values (0/1) for the first variable.
                observed_status_B(np.ndarray): 
                    Binary values (0/1) for the second variable.
                stats_name(str): (default: "spearman")
                    One of "spearman", "kendall", or "phi".

            Returns:
                Dict[str, Union[float, str]]: 
                    A dictionary with the selected statistic and its p-value.

            Raises:
                ValueError: If the provided stats_name is not supported.
            """
            results = {}

            if stats_name == "spearman":
                coef, pval = scipy.stats.spearmanr(observed_status_A, observed_status_B)
                results["spearman"] = coef
                results["spearman_p_value"] = pval

            elif stats_name == "kendall":
                coef, pval = scipy.stats.kendalltau(observed_status_A, observed_status_B)
                results["kendall"] = coef
                results["kendall_p_value"] = pval

            elif stats_name == "phi":
                confusion_matrix = pd.crosstab(observed_status_A, observed_status_B)
                try:
                    chi2, p, _, _ = scipy.stats.chi2_contingency(confusion_matrix)
                    n = confusion_matrix.sum().sum()
                    phi = np.sqrt(chi2 / n)
                    results["phi"] = phi
                    results["phi_p"] = p
                except ValueError:
                    results["phi"] = np.nan
                    results["phi_p"] = np.nan
            else:
                raise ValueError(f"Unsupported stats_name '{stats_name}'. Choose from 'spearman', 'kendall', 'phi'.")

            return results


    def _calculate_pairwise_correlation(
            self,
            col_A: int,
            col_B: int, 
            stats_name: str = "spearman"
        ) -> Dict[str, Union[float, str]]:
            """
            Perform correlation tests between two columns (HPO terms, diseases).

            Args:
                col_A(int): 
                    The first column to correlate.
                col_B(int): 
                    The second column to correlate.
                stats_name(str): (default: "spearman")
                    One of "spearman", "kendall", or "phi".

            Returns:
                Dict[str, Union[float, str]]: 
                    A dictionary containing correlation coefficients and p-values for each test.

            Raises:
                ValueError: If insufficient data for correlation test or invalid columns (all 0 or 1).
            """
            
            matrix = self.combined_matrix.values
            mask = (~np.isnan(matrix[:, col_A])) & (~np.isnan(matrix[:, col_B]))
            col_A_values = matrix[mask, col_A]
            col_B_values = matrix[mask, col_B]
                        
            if len(col_A_values) < self.min_individuals_for_correlation_test:
                return None

            if np.array_equal(col_A_values, col_B_values):
                return {stats_name: 1.0, f"{stats_name}_p_value": 0.0}

            if (np.unique(col_A_values).size == 1) or (np.unique(col_B_values).size == 1):
                return None

            try:
                correlation = self._calculate_pairwise_stats(col_A_values, col_B_values, stats_name=stats_name)
            except Exception as e:
                return None

            return correlation 

    def compute_correlation_matrix(
            self, 
            stats_name: str = "spearman", 
            n_jobs: int = -1
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute pairwise correlation and p-value matrices.

        Args:
            stats_name(str): (default: "spearman") 
                One of "spearman", "kendall", or "phi".
            n_jobs(int): (default: -1)
                Number of parallel jobs to use (-1 uses all cores).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                correlation matrix, p-value matrix
        """
        if stats_name not in self._supported_stats:
            raise ValueError(f"Unsupported stats_name '{stats_name}'. Choose from {self._supported_stats}.")
        
        columns = self.combined_matrix.columns
        n_cols = len(columns)

        def compute_pair(i, j):
            if self.relationship_mask is not None:
                if i < self.n_features and j < self.n_features:
                    if pd.isna(self.relationship_mask[i, j]):
                        return None
            try:
                result = self._calculate_pairwise_correlation(i, j,stats_name=stats_name)
                if result is None:
                    return None
                if stats_name in result and f"{stats_name}_p_value" in result:
                    return (i, j, result[stats_name], result[f"{stats_name}_p_value"])
                else:
                    return None
            except ValueError:
                return None

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_pair)(i, j)
            for i in range(n_cols) for j in range(i + 1, n_cols)
        )

        matrix = np.full((n_cols, n_cols), np.nan)
        pvalue_matrix = np.full((n_cols, n_cols), np.nan)

        for r in results:
            if r is not None:
                i, j, coef, pval = r
                matrix[i, j] = coef
                matrix[j, i] = coef
                pvalue_matrix[i, j] = pval
                pvalue_matrix[j, i] = pval
                
        np.fill_diagonal(matrix, np.nan)
        np.fill_diagonal(pvalue_matrix, np.nan)

        invalid_mask = (np.isnan(matrix).all(axis=0)) | (np.nan_to_num(matrix, nan=0).sum(axis=0) == 0)

        valid_mask = ~invalid_mask

        filtered_columns = self.combined_matrix.columns[valid_mask]

        coef_df = pd.DataFrame(matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)
        pval_df = pd.DataFrame(pvalue_matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)

        return coef_df, pval_df

    
    def filter_weak_correlations(
            self, 
            stats_name: str = "spearman", 
            lower_bound: float=-0.55, 
            upper_bound: float=0.55
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove weak correlations from the correlation matrix based on the given threshold.

        Args:
            stats_name(str): (default: "spearman") 
                The name of the statistic to calculate (e.g., "spearman", "kendall", "phi").
            lower_bound(float): (default: -0.55)
                The lower bound for filtering weak correlations.
            upper_bound(float): (default: 0.55)
                The upper bound for filtering weak correlations.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - A DataFrame of cleaned correlation coefficients.
                - A DataFrame of cleaned p-values.
        """
        coef_matrix,p_value = self.compute_correlation_matrix(stats_name)

        mask = (coef_matrix > lower_bound) & (coef_matrix < upper_bound)
        coef_matrix[mask] = np.nan
        
        p_value[mask] = np.nan
    
        mask_rows = coef_matrix.isna().all(axis=1)
        mask_cols = coef_matrix.isna().all(axis=0)
        coef_matrix_cleaned = coef_matrix.loc[~mask_rows, ~mask_cols]
        p_value_cleaned = p_value.loc[~mask_rows, ~mask_cols]

        return coef_matrix_cleaned, p_value_cleaned
    
    def plot_correlation_heatmap_with_significance(
            self,
            stats_name: str = "spearman",
            lower_bound: float = -0.55,
            upper_bound: float = 0.55,
            significance_threshold: float = 0.05,
            title_name: str = ""
        ) -> None:
            """
            Plot a heatmap of the filtered correlation matrix, with red boxes around statistically significant correlations.

            Args:
                stats_name(str): (default: "spearman")
                    ame of the correlation statistic.
                lower_bound(float): (default: -0.55)
                    ower bound for weak correlation filtering.
                upper_bound(float): (default: 0.55)
                    Upper bound for weak correlation filtering.
                significance_threshold(float): (default: 0.05)
                    Significance threshold for p-values.
                
            The function generates a heatmap visualization of the correlation matrix and highlights statistically significant correlations (where p-value < significance_threshold) with red boxes.
            """

            coef_matrix, pval_matrix = self.filter_weak_correlations(
                stats_name=stats_name,
                lower_bound=lower_bound,
                upper_bound=upper_bound
            )
            if coef_matrix.empty or np.isnan(coef_matrix.values).all():
                raise ValueError("Coefficient matrix is empty or all values are NaN. Nothing to plot.")

            num_rows, num_cols = coef_matrix.shape
            cell_size = 0.6
            min_figsize = (3, 2)

            figsize = (max(cell_size * num_cols, min_figsize[0])*1.1, 
                    max(cell_size * num_rows, min_figsize[1]))

            cell_width = figsize[0] / max(num_cols, 1)
            cell_height = figsize[1] / max(num_rows, 1)
            base_fontsize = min(cell_width, cell_height) * 18
            annot_fontsize = base_fontsize * 0.8

            nan_mask = coef_matrix.isna()
            cmap = plt.get_cmap("managua").copy()
            cmap.set_bad(color='#F5F5F5')

            plt.figure(figsize=figsize, dpi=200)
            ax = sns.heatmap(
                coef_matrix,
                cmap=cmap,
                center=0,
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                linecolor="gray",
                cbar_kws={"shrink": 0.8,"label": f"{stats_name} correlation"},
                mask=nan_mask,
                annot_kws={"size": annot_fontsize}
            )

            ax.set_xticks(np.arange(coef_matrix.shape[1]) + 0.5)
            ax.set_yticks(np.arange(coef_matrix.shape[0]) + 0.5)
            ax.set_xticklabels(coef_matrix.columns, rotation=90, fontsize=base_fontsize)
            ax.set_yticklabels(coef_matrix.index, rotation=0, fontsize=base_fontsize)


            # Add red rectangles where p-value < alpha
            for i in range(coef_matrix.shape[0]):
                for j in range(coef_matrix.shape[1]):
                    if i >= j:
                        continue  # only upper triangle
                    if not np.isnan(pval_matrix.iloc[i, j]) and pval_matrix.iloc[i, j] < significance_threshold:
                        ax.add_patch(plt.Rectangle(
                            (j, i), 1, 1,
                            fill=False,
                            edgecolor='red',
                            linewidth=2
                        ))
                        ax.add_patch(plt.Rectangle(
                            (i, j), 1, 1,
                            fill=False,
                            edgecolor='red',
                            linewidth=2
                        ))

            plt.title(f"{stats_name} Correlation Heatmap for {title_name}\n(Significant values p < {significance_threshold} highlighted)")
            plt.tight_layout()
            plt.show()
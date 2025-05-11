import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, List
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
        >>> hpo_matrix, target_matrix = HPOMatrixProcessor.prepare_hpo_data(
            phenopackets, threshold=0.5, mode='leaf', use_label=True)
        >>> analyzer = HPOStatisticsAnalyzer(hpo_matrix, target_matrix, min_individuals_for_correlation_test=40)
        >>> coef_matrix, pval_matrix = analyzer.compute_correlation_matrices("Spearman")
        >>> analyzer.plot_correlation_heatmap_with_significance("Spearman")
    
    Notes:
        - Requires at least 30 valid data points per pairwise comparison.
        - Assumes binary input matrices (0/1 presence/absence format).
    """
    def __init__(
            self,  
            matrices: List[pd.DataFrame],
            min_individuals_for_correlation_test: int = 30
        ):
            """
            Initialize the HPOStatisticsAnalyzer.

            Args:
                matrices: List of DataFrames containing HPO terms and disease status/variant effect matrices.
                    Each DataFrame should have the same number of rows (individuals).
                min_individuals_for_correlation_test: Minimum number of valid individuals required to perform correlation tests (default: 40).

            Raises:
                ValueError: If min_individuals_for_correlation_test is less than 30.
            """
            if not all(isinstance(df, pd.DataFrame) for df in matrices):
                raise TypeError("All elements in 'matrices' must be pandas DataFrames.")
            self.combined_matrix = pd.concat(matrices, axis=1).reindex(matrices[0].index)
            
            if min_individuals_for_correlation_test < 30:
                raise ValueError("min_individuals_for_correlation_test must not be less than 30.")
            self.min_individuals_for_correlation_test = min_individuals_for_correlation_test
            self._supported_stats = {"Spearman", "Kendall", "phi"}

    def _calculate_pairwise_stats( 
            self,
            observed_status_A: pd.Series, 
            observed_status_B: pd.Series,
            stats_name: str
        ) -> Dict[str, Union[float, str]]:
            """
            Calculate selected statistical metric (Spearman, Kendall, or Phi) and its p-value
            for two binary (0/1) observed status vectors.

            Args:
                observed_status_A: Binary values (0/1) for the first variable.
                observed_status_B: Binary values (0/1) for the second variable.
                stats_name: One of "Spearman", "Kendall", or "phi".

            Returns:
                dict: A dictionary with the selected statistic and its p-value.

            Raises:
                ValueError: If the provided stats_name is not supported.
            """
            results = {}

            if stats_name == "Spearman":
                coef, pval = scipy.stats.spearmanr(observed_status_A, observed_status_B)
                results["Spearman"] = coef
                results["Spearman_p_value"] = pval

            elif stats_name == "Kendall":
                coef, pval = scipy.stats.kendalltau(observed_status_A, observed_status_B)
                results["Kendall"] = coef
                results["Kendall_p_value"] = pval

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
                raise ValueError(f"Unsupported stats_name '{stats_name}'. Choose from 'Spearman', 'Kendall', 'phi'.")

            return results


    def _calculate_pairwise_correlation(
            self,
            col_A: str,
            col_B: str, 
            stats_name: str = "Spearman"
        ) -> Dict[str, Union[float, str]]:
            """
            Perform correlation tests between two columns (HPO terms, diseases).

            Args:
                col_A: The first column to correlate.
                col_B: The second column to correlate.
                stats_name: One of "Spearman", "Kendall", or "phi".

            Returns:
                dict: A dictionary containing correlation coefficients and p-values for each test.

            Raises:
                ValueError: If insufficient data for correlation test or invalid columns (all 0 or 1).
            """

            filtered_matrix = self.combined_matrix[[col_A, col_B]].dropna()

            if filtered_matrix.shape[0] <= self.min_individuals_for_correlation_test:
                raise ValueError(f"Insufficient data (less than {self.min_individuals_for_correlation_test} valid entries) to perform the analysis.")

            # Check if any columns are entirely 0 or 1
            invalid_columns = filtered_matrix.columns[
                (filtered_matrix == 0).all() | (filtered_matrix == 1).all()
            ]
            if not invalid_columns.empty:
                raise ValueError(f"The following columns are entirely 0 or 1 and are invalid for analysis: {list(invalid_columns)}")

            correlation = self._calculate_pairwise_stats(filtered_matrix[col_A],filtered_matrix[col_B],stats_name = stats_name)
            
            return correlation  

    def compute_correlation_matrix(
            self, 
            stats_name: str, 
            n_jobs: int = -1
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute pairwise correlation and p-value matrices.

        Args:
            stats_name: One of "Spearman", "Kendall", or "phi".
            n_jobs: Number of parallel jobs to use (-1 uses all cores).

        Returns:
            Tuple of (correlation matrix, p-value matrix) as DataFrames.
        """
        if stats_name not in self._supported_stats:
            raise ValueError(f"Unsupported stats_name '{stats_name}'. Choose from {self._supported_stats}.")
        
        columns = self.combined_matrix.columns
        n_cols = len(columns)

        def compute_pair(i, j):
            col_A, col_B = columns[i], columns[j]
            try:
                result = self._calculate_pairwise_correlation(col_A, col_B,stats_name=stats_name)
                if result and stats_name in result and f"{stats_name}_p_value" in result:
                    return (i, j, result[stats_name], result[f"{stats_name}_p_value"])
            except ValueError:
                return None

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_pair)(i, j)
            for i in range(n_cols) for j in range(i + 1, n_cols)
        )

        matrix = np.full((n_cols, n_cols), np.nan)
        pvalue_matrix = np.full((n_cols, n_cols), np.nan)

        for r in results:
            if r:
                i, j, coef, pval = r
                matrix[i, j] = coef
                matrix[j, i] = coef
                pvalue_matrix[i, j] = pval
                pvalue_matrix[j, i] = pval

        coef_df = pd.DataFrame(matrix, index=columns, columns=columns)
        pval_df = pd.DataFrame(pvalue_matrix, index=columns, columns=columns)

        return coef_df, pval_df

    
    
    def filter_weak_correlations(
            self, 
            stats_name: str, 
            lower_bound=-0.55, 
            upper_bound=0.55
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove weak correlations from the correlation matrix based on the given threshold.

        Args:
            stats_name: The name of the statistic to calculate (e.g., "Spearman", "Kendall", "Phi").
            lower_bound: The lower bound for filtering weak correlations (default: -0.55).
            upper_bound: The upper bound for filtering weak correlations (default: 0.55).

        Returns:
            tuple: A tuple containing two cleaned DataFrames:
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
            stats_name: str,
            lower_bound: float = -0.55,
            upper_bound: float = 0.55,
            significance_threshold: float = 0.05,
            title_name: str = ""
        ):
            """
            Plot a heatmap of the filtered correlation matrix, with red boxes around statistically significant correlations.

            Args:
                stats_name: Name of the correlation statistic (e.g., 'Spearman').
                lower_bound: Lower bound for weak correlation filtering.
                upper_bound: Upper bound for weak correlation filtering.
                significance_threshold: Significance threshold for p-values (default: 0.05).
                
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
            cell_size = 0.4 
            min_figsize = (4, 2)

            figsize = (max(cell_size * num_cols, min_figsize[0])*1.1, 
                    max(cell_size * num_rows, min_figsize[1]))

            cell_width = figsize[0] / max(num_cols, 1)
            cell_height = figsize[1] / max(num_rows, 1)
            base_fontsize = min(cell_width, cell_height) * 15
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
import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple,Optional
from joblib import Parallel, delayed
import scipy.stats
import plotly.graph_objs as go
import logging
logger = logging.getLogger(__name__)

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
        >>> analyzer =HPOStatisticsAnalyzer(hpo_matrix, min_individuals_for_correlation_test=40)
        >>> coef_matrix, pval_matrix = analyzer.compute_correlation_matrices("Spearman")
        >>> analyzer.plot_correlation_heatmap_with_significance("Spearman")
    
    Notes:
        - Requires at least 30 valid data points per pairwise comparison.
        - Assumes binary input matrices (0/1 presence/absence format).
    """
    def __init__(
            self,  
            hpo_data: Tuple[pd.DataFrame,Optional[pd.DataFrame]], 
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
                self.hpo_matrix = hpo_matrix
                self.hpo_terms = hpo_matrix.columns
                self.n_features = hpo_matrix.shape[1]
            else:
                raise TypeError("hpo_matrix must be a pandas DataFrame")
            
            if not np.all(np.isin(hpo_matrix.to_numpy()[~np.isnan(hpo_matrix.to_numpy())], [0, 1])):
                raise ValueError("Non-NaN values in HPO Matrix must be either 0 or 1")
            
            self.relationship_mask = None
            if relationship_mask is not None:
                if isinstance(relationship_mask, pd.DataFrame):
                    self.relationship_mask = relationship_mask.to_numpy() 
                else:
                    raise ValueError("relationship_mask must be a pd.DataFrame")
                    
                if relationship_mask.shape[0] != relationship_mask.shape[1] or \
                    relationship_mask.shape[0] != hpo_matrix.shape[1]:
                    raise ValueError("relationship_mask must have the same number of rows and columns as hpo_matrix has features")
                
                if not np.all(np.isin(self.relationship_mask[~np.isnan(self.relationship_mask)], [0])):
                    raise ValueError("relationship_mask must contain only 0 or NaN")
            
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
            if stats_name == "spearman":
                coef, pval = scipy.stats.spearmanr(observed_status_A, observed_status_B)
                return coef, pval

            elif stats_name == "kendall":
                coef, pval = scipy.stats.kendalltau(observed_status_A, observed_status_B)
                return coef, pval

            elif stats_name == "phi":
                confusion_matrix = confusion_matrix = pd.crosstab(observed_status_A, observed_status_B, dropna=False)
                try:
                    chi2, p, _, _ = scipy.stats.chi2_contingency(confusion_matrix)
                    n = confusion_matrix.sum().sum()
                    phi = np.sqrt(chi2 / n)
                    return phi, p
                except ValueError:
                    return np.nan, np.nan
            else:
                raise ValueError(f"Unsupported stats_name '{stats_name}'. Choose from 'spearman', 'kendall', 'phi'.")

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
                Optional[Dict[str, Union[float, str]]]:
                    Dictionary with correlation results, or None if invalid or insufficient data.


            Raises:
                ValueError: If insufficient data for correlation test or invalid columns (all 0 or 1).
            """
            
            matrix = self.hpo_matrix.values
            mask = (~np.isnan(matrix[:, col_A])) & (~np.isnan(matrix[:, col_B]))
            col_A_values = matrix[mask, col_A]
            col_B_values = matrix[mask, col_B]
                        
            if len(col_A_values) < self.min_individuals_for_correlation_test:
                return (col_A, col_B, np.nan, np.nan)
            
            if np.all(col_A_values == col_A_values[0]) or np.all(col_B_values == col_B_values[0]):
                return (col_A, col_B, np.nan, np.nan)

            try:
                coef, p_val = self._calculate_pairwise_stats(col_A_values, col_B_values, stats_name=stats_name)
                return (col_A, col_B, coef, p_val)
            except Exception as e:
                return (col_A, col_B, np.nan, np.nan)

    def compute_correlation_matrix(
            self, 
            stats_name: str = "spearman", 
            n_jobs: int = -1,
            output_file: Optional[str] = None
        ) -> None:
        """
        Compute pairwise correlation and p-value matrices.

        Args:
            stats_name(str): (default: "spearman") 
                One of "spearman", "kendall", or "phi".
            n_jobs(int): (default: -1)
                Number of parallel jobs to use (-1 uses all cores).
            output_file (Optional[str]): (default: None)
                If provided, saves the result to Excel.
        Returns:
            None: Sets `self.coef_df` and `self.pval_df` with results.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                correlation matrix, p-value matrix
        """
        if stats_name not in self._supported_stats:
            raise ValueError(f"Unsupported stats_name '{stats_name}'. Choose from {self._supported_stats}.")
        
        columns = self.hpo_terms
        n_cols = len(columns)

        def compute_pair(i, j):
            if self.relationship_mask is not None:
                if i < self.n_features and j < self.n_features:
                    if pd.isna(self.relationship_mask[i, j]):
                        return (i,j, np.nan, np.nan)
            try:
                i, j, coef, pval = self._calculate_pairwise_correlation(i, j, stats_name=stats_name)
                return (i, j, coef, pval)
            except Exception:
                return (i, j, np.nan, np.nan)

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_pair)(i, j)
            for i in range(n_cols) for j in range(i + 1, n_cols)
        )

        matrix = np.full((n_cols, n_cols), np.nan)
        pvalue_matrix = np.full((n_cols, n_cols), np.nan)

        for r in results:
            i, j, coef, pval = r
            matrix[i, j] = coef
            matrix[j, i] = coef
            pvalue_matrix[i, j] = pval
            pvalue_matrix[j, i] = pval
   

        valid_mask = ~(np.isnan(matrix).all(axis=0)) | (np.nan_to_num(matrix, nan=0).sum(axis=0) == 0)
        if len(valid_mask) == 0:
            logger.warning("Warning: No valid correlation between HPO terms. Correlation matrix will be empty.")
        
        filtered_columns = self.hpo_terms[valid_mask]
        self.coef_df = pd.DataFrame(matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)
        self.pval_df = pd.DataFrame(pvalue_matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)

        if output_file is not None:
            rows = []
            for i, f1 in enumerate(self.coef_df.index):
                for j, f2 in enumerate(self.coef_df.columns):
                    if j > i:  # only upper triangle
                        coef_val = self.coef_df.iloc[i, j]
                        pval_val = self.pval_df.iloc[i, j]
                        if not np.isnan(coef_val):
                            rows.append({
                                "Feature1": f1,
                                "Feature2": f2,
                                "Coefficient": coef_val,
                                "P-value": pval_val
                            })

            df_long = pd.DataFrame(rows)
            if output_file.endswith(".csv"):
                df_long.to_csv(output_file, index=False)
            else:
                df_long.to_excel(output_file, index=False)

    
    def filter_weak_correlations(
            self, 
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
        if not hasattr(self, 'coef_df') or not hasattr(self, 'pval_df'):
            raise RuntimeError("Correlation matrix not found. Please run `compute_correlation_matrix()` first.")

        coef_matrix,p_value = self.coef_df.copy(), self.pval_df.copy()

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
            title_name: str = "",
            output_file=None
        ) -> None:
            """
            Plot an interactive Plotly heatmap showing correlation coefficients between features,
            with red boxes indicating statistically significant correlations (based on p-values).

            Parameters:
                stats_name (str): 
                    Type of correlation coefficient to use ("spearman", "pearson", etc.).
                lower_bound (float): 
                    Lower threshold to filter out weak correlations.
                upper_bound (float): 
                    Upper threshold to filter out weak correlations.
                title_name (str): 
                    Optional subtitle to display under the main title.
                output_file (str or None): 
                    If provided, saves the plot to an HTML file.
            """
            # --- Compute correlation and filter weak correlations ---
            coef_matrix, pval_matrix = self.filter_weak_correlations(
                lower_bound=lower_bound,
                upper_bound=upper_bound
            )

            if coef_matrix.empty or np.isnan(coef_matrix.values).all():
                raise ValueError("Coefficient matrix is empty. Try adjusting the lower_bound parameter.")

            # --- Dynamic layout scaling based on matrix size ---
            n_rows, n_cols = coef_matrix.shape
            cell_size = 60  # Base pixel size per cell

            max_dim = max(n_rows, n_cols)
            fig_size = min(1200, max_dim * cell_size)  # Cap total figure size to avoid excessive width

            title_fontsize = max(14 + max_dim // 2, 28)
            label_fontsize = max(8, 12 - max_dim // 8)
            annot_fontsize = max(6, 12 - max_dim // 8)

            # --- Prepare matrix and annotations ---
            display_matrix = coef_matrix.fillna(0)
            text_matrix = np.where(
                np.isnan(coef_matrix.values),
                "",
                coef_matrix.round(2).astype(str)
            )

            # --- Generate custom hover text per cell ---
            valid_mask = ~np.isnan(coef_matrix.values)
            hover_text = np.empty_like(coef_matrix, dtype=object)
            hover_text[valid_mask] = [
                f"<b>X</b>: {col}<br><b>Y</b>: {row}<br>"
                f"<b>Corr</b>: {coef:.2f}<br><b>p-val</b>: {pval:.6f}"
                for row, col, coef, pval in zip(
                    np.repeat(coef_matrix.index, n_cols)[valid_mask.ravel()],
                    np.tile(coef_matrix.columns, n_rows)[valid_mask.ravel()],
                    coef_matrix.values[valid_mask],
                    pval_matrix.values[valid_mask]
                )
            ]
            hover_text[~valid_mask] = ""

            # --- Create heatmap figure ---
            fig = go.Figure(
                go.Heatmap(
                    z=display_matrix.values,
                    x=coef_matrix.columns,
                    y=coef_matrix.index,
                    colorscale="RdBu",
                    zmid=0,
                    text=text_matrix,
                    texttemplate=f"<span style='font-size:{annot_fontsize}px'>%{{text}}</span>",
                    hovertext=hover_text,
                    hoverinfo="text",
                    colorbar=dict(title="Corr.", len=0.8, thickness=title_fontsize),
                    zmin=-1,
                    zmax=1,
                    xgap=1,
                    ygap=1,
                )
            )

            # --- Adjust layout ---
            max_ylabel_len = max(len(str(lbl)) for lbl in coef_matrix.index)
            left_margin = 60 + max_ylabel_len * label_fontsize

            fig.update_layout(
                title=dict(
                    text=f"<b>{stats_name.capitalize()} Correlation</b><br>"
                        f"<span style='font-size:0.8em'>{title_name}</span>",
                    x=0.5,
                    xanchor="center",
                    yanchor="top",
                    font=dict(
                        size=min(title_fontsize, 24),
                        family="Arial"
                    )
                ),
                xaxis=dict(
                    tickangle=90,
                    tickfont=dict(size=label_fontsize),
                ),
                yaxis=dict(
                    tickfont=dict(size=label_fontsize),
                    scaleanchor="x",
                    scaleratio=1
                ),
                width=fig_size + left_margin,
                height=fig_size + left_margin,
                plot_bgcolor="rgba(240,240,240,0.1)"
            )

            # --- Save or show plot ---
            if output_file:
                fig.write_html(output_file)

            fig.show()
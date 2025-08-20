import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple,Optional
from joblib import Parallel, delayed
import scipy.stats
from .correlation_type import CorrelationType
import plotly.graph_objs as go
import logging
from os import path
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
            min_cooccurrence_count = 1
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
            min_cooccurrence_count (int, default=1):
                Minimum number of co-occurrences (both features present, 1/1) 
                **and** co-exclusions (both features absent, 0/0) required 
                for a feature pair to be considered valid for correlation testing.
                This ensures that both positive and negative concordance are observed 
                more than once, avoiding spurious correlations.

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
            self.min_coccurrence_count = min_cooccurrence_count

    def _calculate_pairwise_stats( 
            self,
            observed_status_A: np.ndarray, 
            observed_status_B: np.ndarray,
            correlation_type: CorrelationType = CorrelationType.SPEARMAN,
        ) -> Dict[str, Union[float, str]]:
            """
            Calculate selected statistical metric (spearman, kendall, or phi) and its p-value
            for two binary (0/1) observed status vectors.

            Args:
                observed_status_A(np.ndarray): 
                    Binary values (0/1) for the first variable.
                observed_status_B(np.ndarray): 
                    Binary values (0/1) for the second variable.
                correlation_type (CorrelationType): (default: CorrelationType.SPEARMAN)
                    Correlation metric to compute. One of:
                    - CorrelationType.SPEARMAN
                    - CorrelationType.KENDALL
                    - CorrelationType.PHI

            Returns:
                Dict[str, Union[float, str]]: 
                    A dictionary with the selected statistic and its p-value.

            Raises:
                ValueError: If the provided correlation_name is not supported.
            """
            if correlation_type == CorrelationType.SPEARMAN:
                coef, pval = scipy.stats.spearmanr(observed_status_A, observed_status_B)
                return coef, pval

            elif correlation_type == CorrelationType.KENDALL:
                coef, pval = scipy.stats.kendalltau(observed_status_A, observed_status_B)
                return coef, pval

            elif correlation_type == CorrelationType.PHI:
                confusion_matrix = pd.crosstab(observed_status_A, observed_status_B, dropna=False)
                try:
                    chi2, p, _, _ = scipy.stats.chi2_contingency(confusion_matrix)
                    n = confusion_matrix.sum().sum()
                    phi = np.sqrt(chi2 / n)
                    return phi, p
                except ValueError:
                    return np.nan, np.nan

            else:
                raise ValueError(f"Unsupported CorrelationType '{stats_type}'.")

    def _calculate_pairwise_correlation(
            self,
            col_A: int,
            col_B: int, 
            correlation_type: CorrelationType = CorrelationType.SPEARMAN,
        ) -> Dict[str, Union[float, str]]:
            """
            Perform correlation tests between two columns (HPO terms, diseases).

            Args:
                col_A(int): 
                    The first column to correlate.
                col_B(int): 
                    The second column to correlate.
                correlation_type (CorrelationType): (default: CorrelationType.SPEARMAN)
                    Correlation metric to compute. One of:
                    - CorrelationType.SPEARMAN
                    - CorrelationType.KENDALL
                    - CorrelationType.PHI

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

            count_11 = np.sum((col_A_values == 1) & (col_B_values == 1))
            count_10 = np.sum((col_A_values == 1) & (col_B_values == 0))
            count_01 = np.sum((col_A_values == 0) & (col_B_values == 1))
            count_00 = np.sum((col_A_values == 0) & (col_B_values == 0))
            total = len(col_A_values)
                        
            if len(col_A_values) < self.min_individuals_for_correlation_test:
                return (col_A, col_B, np.nan, np.nan, {"00":0,"01":0,"10":0,"11":0,"N":0})
            
            if np.all(col_A_values == col_A_values[0]) or np.all(col_B_values == col_B_values[0]):
                return (col_A, col_B, np.nan, np.nan, {"00":0,"01":0,"10":0,"11":0,"N":0})
            
            # --- Count co-occurrence ---
            observed_observed = np.sum((col_A_values == 1) & (col_B_values == 1))
            excluded_excluded = np.sum((col_A_values == 0) & (col_B_values == 0))

            if observed_observed <= self.min_coccurrence_count or excluded_excluded <= self.min_coccurrence_count:
                return (col_A, col_B, np.nan, np.nan, {"00":0,"01":0,"10":0,"11":0,"N":0})

            try:
                coef, p_val = self._calculate_pairwise_stats(col_A_values, col_B_values, correlation_type=correlation_type)
                return (col_A, col_B, coef, p_val, {"00":count_00,"01":count_01,"10":count_10,"11":count_11,"N":total})
            except Exception as e:
                return (col_A, col_B, np.nan, np.nan, {"00":0,"01":0,"10":0,"11":0,"N":0})

    def compute_correlation_matrix(
            self, 
            correlation_type: CorrelationType = CorrelationType.SPEARMAN, 
            n_jobs: int = -1,
        ) -> None:
        """
        Compute pairwise correlation and p-value matrices.

        Args:
            correlation_type (CorrelationType): (default: CorrelationType.SPEARMAN)
                Correlation metric to compute. One of:
                - CorrelationType.SPEARMAN
                - CorrelationType.KENDALL
                - CorrelationType.PHI
            n_jobs(int): (default: -1)
                Number of parallel jobs to use (-1 uses all cores).
        Returns:
            pd.DataFrame:
                DataFrame with correlation coefficients and p-values
                - Each row of the exported table contains:
                    * Feature1 (str): HPO term or feature name.
                    * Feature2 (str): HPO term or feature name.
                    * Coefficient (float): correlation coefficient.
                    * P-value (float): corresponding p-value.
        """
        if not isinstance(correlation_type, CorrelationType):
            raise ValueError(f"stats_type must be a CorrelationType, got {type(correlation_type)}")
        
        columns = self.hpo_terms
        n_cols = len(columns)

        def compute_pair(i, j):
            if self.relationship_mask is not None:
                if i < self.n_features and j < self.n_features:
                    if pd.isna(self.relationship_mask[i, j]):
                        return (i,j, np.nan, np.nan, {"00":0,"01":0,"10":0,"11":0,"N":0})
            try:
                return self._calculate_pairwise_correlation(i, j, correlation_type=correlation_type)
            except Exception:
                return (i, j, np.nan, np.nan, {"00":0,"01":0,"10":0,"11":0,"N":0})

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_pair)(i, j)
            for i in range(n_cols) for j in range(i + 1, n_cols)
        )

        matrix = np.full((n_cols, n_cols), np.nan)
        pvalue_matrix = np.full((n_cols, n_cols), np.nan)

        rows = []
        for r in results:
            i, j, coef, pval, counts = r
            matrix[i, j] = coef
            matrix[j, i] = coef
            pvalue_matrix[i, j] = pval
            pvalue_matrix[j, i] = pval
            f1, f2 = self.hpo_matrix.columns[i], self.hpo_matrix.columns[j]
            if j > i:  # only upper triangle
                if not np.isnan(coef):
                    rows.append({
                        "Feature1": f1,
                        "Feature2": f2,
                        "Coefficient": coef,
                        "P_value": pval,
                        "Count_00": counts["00"],
                        "Count_01": counts["01"],
                        "Count_10": counts["10"],
                        "Count_11": counts["11"],
                        "Total": counts["N"]
                    }) 
   
        self.correlation_results = pd.DataFrame(rows)

        valid_mask = ~(np.isnan(matrix).all(axis=0)) | (np.nan_to_num(matrix, nan=0).sum(axis=0) == 0)
        if len(valid_mask) == 0:
            logger.warning("Warning: No valid correlation between HPO terms. Correlation matrix will be empty.")
        
        filtered_columns = self.hpo_terms[valid_mask]
        self.coef_df = pd.DataFrame(matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)
        self.pval_df = pd.DataFrame(pvalue_matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)

        return self.correlation_results
    
    def save_correlation_results(
            self, 
            output_file: str
        ) -> None:
        """
        Export the computed correlation results to a file.

        The correlation matrices (`self.coef_df` and `self.pval_df`) must have 
        been computed previously by calling `compute_correlation_matrix`.

        Args:
            output_file (str):
                Path to the output file. Supported formats:
                - ".csv": saves as a CSV file.
                - ".xlsx" or other extensions: saves as an Excel file.

        Raises:
            ValueError:
                If `self.correlation_results` has not been initialized
                (i.e., `compute_correlation_matrix` has not been run).

        Example:
            >>> analyzer.compute_correlation_matrix()
            >>> analyzer.save_correlation_results("correlations.csv")
            >>> analyzer.save_correlation_results("correlations.xlsx")
        """
        
        if not hasattr(self, "correlation_results"):
            raise ValueError("Correlation results not computed. Run compute_correlation_matrix() first.")
        
        ext = path.splitext(output_file)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            raise ValueError(f"Unsupported file format: {ext}. Use '.csv' or '.xlsx'.")

        
        if output_file.endswith(".csv"):
            self.correlation_results.to_csv(output_file, index=False)
        else:
            self.correlation_results.to_excel(output_file, index=False)    

    
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
        ) -> go.Figure:
        """
        Create an interactive Plotly heatmap showing correlation coefficients between features,
        with hover information for p-values.

        Parameters:
            stats_name (str): 
                Type of correlation coefficient to use ("spearman", "kendall", "phi").
            lower_bound (float): 
                Lower threshold to filter out weak correlations.
            upper_bound (float): 
                Upper threshold to filter out weak correlations.
            title_name (str): 
                Optional subtitle to display under the main title.

        Returns:
            plotly.graph_objects.Figure:
                A Plotly Figure object for the correlation heatmap.

        Example:
            >>> # Compute correlations first
            >>> analyzer.compute_correlation_matrix()
            >>> # Generate heatmap (returns a Plotly Figure)
            >>> fig = analyzer.plot_correlation_heatmap_with_significance(
            ...     stats_name="spearman",
            ...     lower_bound=-0.5,
            ...     upper_bound=0.5,
            ...     title_name="Cohort A"
            ... )
            >>> # Show in Jupyter or browser
            >>> fig.show()
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
        hover_text = np.empty_like(coef_matrix, dtype=object)
        counts_lookup = {}
        for row in self.correlation_results.itertuples():
            # forward (original counts)
            counts_lookup[(row.Feature1, row.Feature2)] = {
                "Coefficient": row.Coefficient,
                "P_value": row.P_value,
                "Count_00": row.Count_00,
                "Count_01": row.Count_01,
                "Count_10": row.Count_10,
                "Count_11": row.Count_11,
                "Total": row.Total,
            }
            # backward (exchange Count_01 和 Count_10)
            counts_lookup[(row.Feature2, row.Feature1)] = {
                "Coefficient": row.Coefficient,
                "P_value": row.P_value,
                "Count_00": row.Count_00,
                "Count_01": row.Count_10,  # swapped
                "Count_10": row.Count_01,  # swapped
                "Count_11": row.Count_11,
                "Total": row.Total,
            }


        hover_text = []
        for i, row in enumerate(coef_matrix.index):
            hover_row = []
            for j, col in enumerate(coef_matrix.columns):
                coef = coef_matrix.iloc[i, j]
                pval = pval_matrix.iloc[i, j]
                if np.isnan(coef):
                    hover_row.append("")
                else:
                    counts = counts_lookup.get((row, col), {})
                    hover_row.append(
                        f"<b>X</b>: {col}<br><b>Y</b>: {row}<br>"
                        f"<b>Corr</b>: {coef:.2f}<br><b>p-val</b>: {pval:.6f}<br>"
                        f"<b>Counts</b>: 00={counts.get('Count_00', 0)}, "
                        f"01={counts.get('Count_01', 0)}, "
                        f"10={counts.get('Count_10', 0)}, "
                        f"11={counts.get('Count_11', 0)}<br>"
                        f"<b>Total</b>: {counts.get('Total', 0)}"
                    )
            hover_text.append(hover_row)
          

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
        return fig
    

    def save_correlation_heatmap(
            self, 
            fig: go.Figure, 
            output_file: str
        ) -> None:
        """
        Save a correlation heatmap figure to an HTML file.

        Args:
            fig (plotly.graph_objects.Figure): 
                The heatmap figure generated by `plot_correlation_heatmap_with_significance`.
            output_file (str): 
                Path to the HTML file where the figure should be saved. Must end with '.html'.

        Raises:
            ValueError:
                If the output_file extension is not '.html'.

        Example:
            >>> fig = analyzer.plot_correlation_heatmap_with_significance()
            >>> analyzer.save_correlation_heatmap(fig, "correlation_heatmap.html")
        """
        if not output_file.endswith(".html"):
            raise ValueError("output_file must have a '.html' extension")
        fig.write_html(output_file)
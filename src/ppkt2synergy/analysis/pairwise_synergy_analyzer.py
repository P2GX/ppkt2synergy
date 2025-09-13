from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
import plotly.graph_objs as go
import logging
import os
from scipy.sparse import coo_matrix, triu

logger = logging.getLogger(__name__)

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
        

        self.X = hpo_matrix
        self.y = target
        self.n_features = hpo_matrix.shape[1]
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_state)

        if relationship_mask is not None:
            if isinstance(relationship_mask, pd.DataFrame):
                relationship_mask_numpy = relationship_mask.to_numpy() 
            else:
                raise ValueError("mask must be a pd.DataFrame.")
            
            if relationship_mask_numpy.shape[0] != relationship_mask_numpy.shape[1] or \
                relationship_mask_numpy.shape[0] != hpo_matrix.shape[1]:
                raise ValueError("mask must have the same number of rows and columns as hpo_matrix has features")
            
            if not np.all(np.isin(relationship_mask_numpy[~np.isnan(relationship_mask_numpy)], [0])):
                raise ValueError("relationship_mask must contain only 0 or NaN")
            
            self.synergy_matrix = relationship_mask_numpy
        else:
            self.synergy_matrix = np.full((self.n_features, self.n_features), np.nan)

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
        mask = (~np.isnan(self.X[:, i]) & ~np.isnan(self.X[:, j]) & ~np.isnan(self.y))
        xi = self.X[mask, i]
        xj = self.X[mask, j]
        y = self.y[mask]
        total = len(xi)

        if np.all(xi == xi[0]) or np.all(xj == xj[0]) or np.all(y == y[0]):
            return i, j, np.nan, np.nan, {}
        
        if np.array_equal(xi, xj):
            return i, j, np.nan, np.nan, {}
        
        counts = {
        # y=0 
        "00|y=0": np.sum((xi == 0) & (xj == 0) & (y == 0)),
        "01|y=0": np.sum((xi == 0) & (xj == 1) & (y == 0)),
        "10|y=0": np.sum((xi == 1) & (xj == 0) & (y == 0)),
        "11|y=0": np.sum((xi == 1) & (xj == 1) & (y == 0)),
        "N_y0": np.sum(y == 0),

        # y=1 
        "00|y=1": np.sum((xi == 0) & (xj == 0) & (y == 1)),
        "01|y=1": np.sum((xi == 0) & (xj == 1) & (y == 1)),
        "10|y=1": np.sum((xi == 1) & (xj == 0) & (y == 1)),
        "11|y=1": np.sum((xi == 1) & (xj == 1) & (y == 1)),
        "N_y1": np.sum(y == 1),

        "N": total,
        }

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

        return i, j, corrected_synergy, p_value, counts


    def compute_synergy_matrix(
            self, 
            n_jobs=-1,
        ) -> None:
        """
        Compute the pairwise synergy scores and permutation-based p-values for all valid feature pairs.

        The synergy score is evaluated for each pair of HPO terms/features that:
        - Have at least `self.min_individuals_for_synergy_caculation` valid samples.
        - Are not masked out in the existing `self.synergy_matrix` (i.e., not NaN).

        Results are stored in two symmetric matrices (`self.synergy_matrix`, `self.pvalue_matrix`)
        and an exported long-format table (`self.synergy_results`).

        Args:
            n_jobs (int, optional): (default: -1)
                Number of parallel jobs to run. 
                Set to -1 to use all available CPU cores.

        Returns:
            pd.DataFrame:
                DataFrame with one row per valid feature pair.
                Each row contains:
                    * Feature1 (str): first HPO term or feature name.
                    * Feature2 (str): second HPO term or feature name.
                    * Synergy (float): synergy score between the pair.
                    * P_value (float): permutation-based p-value.
                    * Count_00_y0 (int): count of samples with (0,0) under label y=0.
                    * Count_01_y0 (int): count of samples with (0,1) under label y=0.
                    * Count_10_y0 (int): count of samples with (1,0) under label y=0.
                    * Count_11_y0 (int): count of samples with (1,1) under label y=0.
                    * N_y0 (int): total number of valid samples under label y=0.
                    * Count_00_y1 (int): count of samples with (0,0) under label y=1.
                    * Count_01_y1 (int): count of samples with (0,1) under label y=1.
                    * Count_10_y1 (int): count of samples with (1,0) under label y=1.
                    * Count_11_y1 (int): count of samples with (1,1) under label y=1.
                    * N_y1 (int): total number of valid samples under label y=1.
                    * Total (int): total number of valid samples (N_y0 + N_y1).
        """
        mask = ~np.isnan(self.X)  # True = non-NaN
        # Compute valid counts for each pair
        valid_counts = mask.T.astype(int) @ mask.astype(int)
        valid_counts_sparse = triu(coo_matrix(valid_counts), k=1)
        rows, cols, counts = valid_counts_sparse.row, valid_counts_sparse.col, valid_counts_sparse.data
        # Filter pairs based on existing synergy_matrix and minimum counts
        ontology_values = self.synergy_matrix[rows, cols]
        ontology_candidate = ~np.isnan(ontology_values)
        candidate_idx = np.where(ontology_candidate & (counts >= self.min_individuals_for_synergy_caculation))[0]
        

        rows_cand, cols_cand = rows[candidate_idx], cols[candidate_idx]

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.evaluate_pair_synergy)(i, j) for i, j in zip(rows_cand, cols_cand)
        )
        pvalue_matrix = np.full((self.n_features, self.n_features), np.nan)

        rows = []
        for i, j, synergy, pval, counts in results:
            self.synergy_matrix[i, j] = self.synergy_matrix[j, i] = synergy
            pvalue_matrix[i, j] = pvalue_matrix[j, i] = pval
            f1, f2 = self.hpo_terms[i], self.hpo_terms[j]
            if j > i:  # only upper triangle
                if not np.isnan(synergy):
                    rows.append({
                        "Feature1": f1,
                        "Feature2": f2,
                        "Synergy": synergy,
                        "P_value": pval,
                        "Count_00_y0": counts["00|y=0"],
                        "Count_01_y0": counts["01|y=0"],
                        "Count_10_y0": counts["10|y=0"],
                        "Count_11_y0": counts["11|y=0"],
                        "N_y0": counts["N_y0"],
                        "Count_00_y1": counts["00|y=1"],
                        "Count_01_y1": counts["01|y=1"],
                        "Count_10_y1": counts["10|y=1"],
                        "Count_11_y1": counts["11|y=1"],
                        "N_y1": counts["N_y1"],
                        "Total": counts["N"]
                    })

        valid_mask = ~((np.isnan(self.synergy_matrix).all(axis=0)) | (np.nan_to_num(self.synergy_matrix, nan=0).sum(axis=0) == 0))
        valid_hpo_terms = self.hpo_terms[valid_mask]
        if len(valid_hpo_terms) == 0:
            logger.warning("Warning: No valid synergy between HPO terms. Synergy matrix will be empty.")

        self.synergy_matrix = pd.DataFrame(
            self.synergy_matrix[np.ix_(valid_mask, valid_mask)],
            index=valid_hpo_terms,
            columns=valid_hpo_terms
        )

        self.pvalue_matrix = pd.DataFrame(
            pvalue_matrix[np.ix_(valid_mask, valid_mask)],
            index=valid_hpo_terms,
            columns=valid_hpo_terms
        )
        self.synergy_results = pd.DataFrame(rows)
        return self.synergy_results
    

    def save_synergy_results(
            self, 
            output_file: str
        ) -> None:
        """
        Export synergy scores and p-values to a file (CSV or Excel).

        The synergy results (`self.synergy_matrix` and `self.pvalue_matrix`) 
        must be computed first by calling `compute_synergy_matrix`. 
        Only the upper triangle of the symmetric matrices is exported to avoid duplication.

        Args:
            output_file (str):
                Path to the output file. Supported formats:
                - ".csv": saves as a CSV file.
                - ".xlsx": saves as an Excel file.

        Raises:
            ValueError:
                - If synergy results are not computed yet.
                - If file extension is not supported.
            OSError:
                If the output path is invalid or not writable. 

        Example:
            >>> analyzer.compute_synergy_matrix()
            >>> analyzer.save_synergy_results("synergy_results.csv")
            >>> analyzer.save_synergy_results("synergy_results.xlsx")
        """
        if not hasattr(self, "synergy_results"):
            raise ValueError("Synergy results not computed. Run compute_synergy_matrix() first.")

        ext = os.path.splitext(output_file)[1].lower()
        if ext not in [".csv", ".xlsx"]:
            raise ValueError(f"Unsupported file format: {ext}. Use '.csv' or '.xlsx'.")

        
        if output_file.endswith(".csv"):
            self.synergy_results.to_csv(output_file, index=False)
        else:
            self.synergy_results.to_excel(output_file, index=False)
   
    
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
        if not hasattr(self, 'pvalue_matrix'):
            raise RuntimeError("Synergy matrix not found. Please run `compute_synergy_matrix()` first.")
        
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
            target_name: str = "",
        ) -> go.Figure:
        """
        Generate a Plotly heatmap figure of pairwise synergy scores for HPO features.

        This function computes a heatmap of synergy scores, annotates significant pairs,
        and prepares hover text with correlation and p-values.

        Args:
            lower_bound (float, optional): Minimum synergy value to include in the heatmap.
                                        Defaults to 0.1.
            target_name (str, optional): Name of the target variable for the plot title.
                                        Defaults to "".

        Returns:
            go.Figure: A Plotly heatmap figure object ready for display or saving.

        Raises:
            ValueError: If no sufficient synergy pairs exist after filtering.

        Example:
            >>> fig = analyzer.create_synergy_heatmap(lower_bound=0.2, target_name="Disease Status")
            >>> fig.show()
        """

        synergy_matrix, pvalue_matrix = self.filter_weak_synergy(lower_bound=lower_bound)

        if synergy_matrix.empty or np.isnan(synergy_matrix.values).all():
            raise ValueError("No sufficient synergy pairs to plot. Try adjusting the lower_bound parameter.")

        n_rows, n_cols = synergy_matrix.shape
        cell_size = 60  # Base pixel size per cell

        max_dim = max(n_rows, n_cols)
        fig_size = min(1200, max_dim * cell_size)  # Cap total figure size to avoid excessive width

        title_fontsize = max(14 + max_dim // 2, 28)
        label_fontsize = max(8, 12 - max_dim // 8)
        annot_fontsize = max(6, 12 - max_dim // 8)

        # --- Prepare matrix and annotations ---
        display_matrix = synergy_matrix.fillna(0)
        text_matrix = np.where(
            np.isnan(synergy_matrix.values),
            "",
            synergy_matrix.round(2).astype(str)
        )

        # --- Generate custom hover text per cell ---
        hover_text = np.empty_like(synergy_matrix, dtype=object)
        counts_lookup = {}
        for row in self.synergy_results.itertuples():
            # forward (original counts)
            counts_lookup[(row.Feature1, row.Feature2)] = {
            "Synergy": row.Synergy,
            "P_value": row.P_value,  
            "Count_00|y=0": row.Count_00_y0,
            "Count_01|y=0": row.Count_01_y0,
            "Count_10|y=0": row.Count_10_y0,
            "Count_11|y=0": row.Count_11_y0,
            "N_y0": row.N_y0,
            "Count_00|y=1": row.Count_00_y1,
            "Count_01|y=1": row.Count_01_y1,
            "Count_10|y=1": row.Count_10_y1,
            "Count_11|y=1": row.Count_11_y1,
            "N_y1": row.N_y1,
            "Total": row.Total
            }
        
            # backward (swap 01 和 10)
            counts_lookup[(row.Feature2, row.Feature1)] = {
            "Synergy": row.Synergy,
            "P_value": row.P_value, 
            "Count_00|y=0": row.Count_00_y0,
            "Count_01|y=0": row.Count_10_y0,  # swap 01 和 10
            "Count_10|y=0": row.Count_01_y0,  # swap 01 和 10
            "Count_11|y=0": row.Count_11_y0,
            "N_y0": row.N_y0,
            "Count_00|y=1": row.Count_00_y1,
            "Count_01|y=1": row.Count_10_y1,  # swap 01 和 10
            "Count_10|y=1": row.Count_01_y1,  # swap 01 和 10
            "Count_11|y=1": row.Count_11_y1,
            "N_y1": row.N_y1,
            "Total": row.Total
            }
        
        hover_text = []
        for i, row in enumerate(pvalue_matrix.index):
            hover_row = []
            for j, col in enumerate(pvalue_matrix.columns):
                synergy = synergy_matrix.iloc[i, j]
                pval = pvalue_matrix.iloc[i, j]
                if np.isnan(synergy):

                    hover_row.append("")
                else:
                    counts = counts_lookup.get((row, col), {})
                    hover_row.append(
                        f"<b>X1</b>: {col}<br><b>X2</b>: {row}<br>"
                        f"<b>Synergy</b>: {synergy:.2f}<br><b>p-val</b>: {pval:.6f}<br>"
                        f"<b>Counts (y=0)</b>: 00={counts.get('Count_00|y=0', 0)}, "
                        f"01={counts.get('Count_01|y=0', 0)}, "
                        f"10={counts.get('Count_10|y=0', 0)}, "
                        f"11={counts.get('Count_11|y=0', 0)} (N={counts.get('N_y0', 0)})<br>"
                        f"<b>Counts (y=1)</b>: 00={counts.get('Count_00|y=1', 0)}, "
                        f"01={counts.get('Count_01|y=1', 0)}, "
                        f"10={counts.get('Count_10|y=1', 0)}, "
                        f"11={counts.get('Count_11|y=1', 0)} (N={counts.get('N_y1', 0)})<br>"
                        f"<b>Total</b>: {counts.get('Total', 0)}"
                    )
            hover_text.append(hover_row)
       
     
        # --- Create heatmap figure ---
        fig = go.Figure(
            go.Heatmap(
                z=display_matrix.values,
                x=synergy_matrix.columns,
                y=synergy_matrix.index,
                colorscale='Blues',
                zmid=0,
                text=text_matrix,
                texttemplate=f"<span style='font-size:{annot_fontsize}px'>%{{text}}</span>",
                hovertext=hover_text,
                hoverinfo="text",
                colorbar=dict(title="Synergy", len=0.8, thickness=title_fontsize),
                zmin=0,
                zmax=np.nanmax(display_matrix.values),
                xgap=1,
                ygap=1,
                )
            )

        # --- Adjust layout ---
        max_ylabel_len = max(len(str(lbl)) for lbl in synergy_matrix.index)
        left_margin = 60 + max_ylabel_len * label_fontsize

        fig.update_layout(
            title=dict(
                text=f"<b>Pairwise Synergy Heatmap of HPO Features</b><br>"
                    f"<span style='font-size:0.8em'>With respect to {target_name}</span>",
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
    
    def save_synergy_heatmap(
            self, 
            fig: go.Figure, 
            output_file: str
        ) -> None:
        """
        Save a pairwise synergy heatmap figure to an HTML file.

        Args:
            fig (plotly.graph_objects.Figure): 
                The heatmap figure generated by `create_synergy_heatmap` or `plot_synergy_heatmap`.
            output_file (str): 
                Path to the HTML file where the figure should be saved. Must end with '.html'.

        Raises:
            ValueError:
                If the output_file extension is not '.html'.

        Example:
            >>> fig = analyzer.create_synergy_heatmap(lower_bound=0.2, target_name="Disease Status")
            >>> analyzer.save_synergy_heatmap(fig, "synergy_heatmap.html")
        """
        if not output_file.endswith(".html"):
            raise ValueError("output_file must have a '.html' extension")
        fig.write_html(output_file)

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.sparse import coo_matrix, triu

from ..core import PhenotypeDataset

logger = logging.getLogger(__name__)


class SynergyResult:
    """
    Data class to hold synergy analysis results for pairs of HPO terms with respect to a target.
    """
    def __init__(
        self, 
        synergy_results: pd.DataFrame, 
        synergy_matrix: pd.DataFrame, 
        pvalue_matrix: pd.DataFrame,
        label_mapping: dict, 
        condition_name: str
    ) -> None:
        self.synergy_results = synergy_results
        self.synergy_matrix = synergy_matrix
        self.pvalue_matrix = pvalue_matrix
        self.label_mapping = label_mapping
        self.condition_name = condition_name if condition_name else "Target Condition"
        self.fig: go.Figure | None = None

    @property
    def results_table(self) -> pd.DataFrame:  
        """Get a safe copy of the synergy results table."""
        return self.synergy_results.copy()

    def save_synergy_results(
        self, 
        synergy_threshold: float = 0.01,
        adj_pval_threshold: float = 0.05,
        output_file: str = "synergy_results.csv"
    ) -> None:
        """
        Save synergy results to a CSV or Excel file.

        Parameters
        ----------
        synergy_threshold : float, default=0.01
            Minimum synergy value to retain.

        adj_pval_threshold : float, default=0.3
            Maximum adjusted p-value to retain.

        output_file : str, default="synergy_results.csv"
            Output file path. Supported formats are ``.csv``.
        """
        if self.synergy_results.empty:
            logger.warning("Synergy results table is empty. Saving empty file.")
            df = self.synergy_results.copy()
        else:
            df = self.synergy_results.copy()
            if synergy_threshold < 0.0:
                raise ValueError("synergy_threshold must be non-negative.")
            df = df[df["synergy"].abs() >= synergy_threshold]

            if not 0.0 <= adj_pval_threshold <= 1.0:
                raise ValueError("adj_pval_threshold must be between 0.0 and 1.0.")
            df = df[df["adj_p_value"] < adj_pval_threshold]

     
        df.to_csv(output_file, index=False)
    
    def filter_weak_synergy(
        self, 
        synergy_threshold: float = 0.01, 
        adj_pval_threshold: float = 0.05
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter the synergy and p-value matrices by effect size and significance.
        """
        synergy_matrix = self.synergy_matrix.copy()
        p_value = self.pvalue_matrix.copy()

        if synergy_threshold < 0.0:
            raise ValueError("synergy_threshold must be non-negative.")
        mask = synergy_matrix.abs() < synergy_threshold
        synergy_matrix[mask] = np.nan
        p_value[mask] = np.nan
        
        if adj_pval_threshold < 0.0 or adj_pval_threshold > 1.0:
            raise ValueError("corrected_alpha must be between 0.0 and 1.0.")
        
        if not self.synergy_results.empty:
            non_signif = self.synergy_results.loc[
            (self.synergy_results["adj_p_value"] >= adj_pval_threshold), ["HPO_A", "HPO_B"]
            ]
            for _, row in non_signif.iterrows():
                hpo_a, hpo_b = row["HPO_A"], row["HPO_B"]
                if hpo_a in synergy_matrix.index and hpo_b in synergy_matrix.columns:
                    synergy_matrix.loc[hpo_a, hpo_b] = np.nan
                    synergy_matrix.loc[hpo_b, hpo_a] = np.nan
                    p_value.loc[hpo_a, hpo_b] = np.nan
                    p_value.loc[hpo_b, hpo_a] = np.nan

        # Remove rows/columns that are completely NaN
        mask_rows = synergy_matrix.isna().all(axis=1)
        mask_cols = synergy_matrix.isna().all(axis=0)
        synergy_matrix_cleaned = synergy_matrix.loc[~mask_rows, ~mask_cols]
        p_value_cleaned = p_value.loc[~mask_rows, ~mask_cols]

        return synergy_matrix_cleaned, p_value_cleaned    

    @staticmethod
    def _format_hpo_pair(hpo_id: str, label: str | None) -> str:
        """Format an HPO term for display."""
        if label:
            return f"{label} ({hpo_id})"
        return hpo_id
    
    @staticmethod
    def _format_pmids_for_tooltip(pmids: str | list[str] | None, max_pmids: int = 5) -> str:
        """Format PMID values for hover text."""
        if pmids is None or pmids == "":
            return "None"

        if isinstance(pmids, str):
            pmid_list = [p.strip() for p in pmids.split(";") if p.strip()]
        else:
            pmid_list = [str(p).strip() for p in pmids if str(p).strip()]

        if not pmid_list:
            return "None"

        if len(pmid_list) <= max_pmids:
            return ", ".join(pmid_list)

        shown = ", ".join(pmid_list[:max_pmids])
        remaining = len(pmid_list) - max_pmids
        return f"{shown} ... (+{remaining} more)"

    def plot_synergy_heatmap(
            self, 
            synergy_threshold: float = 0.01,
            adj_pval_threshold: float = 0.05,
        ) -> go.Figure:
        """
        Plot an interactive heatmap of pairwise synergy values.
        """
        raw_synergy, pvalue_matrix = self.filter_weak_synergy(synergy_threshold=synergy_threshold, adj_pval_threshold=adj_pval_threshold)

        if raw_synergy.empty or np.isnan(raw_synergy.values).all():
            raise ValueError(
                "No sufficient synergy pairs remain after filtering. "
                "Try adjusting `synergy_threshold` or `adj_pval_threshold`."
            )

        synergy_matrix = raw_synergy.copy()

        n_rows, n_cols = raw_synergy.shape
        cell_size = 60  # Base pixel size per cell

        max_dim = max(n_rows, n_cols)
        fig_size = min(1200, max_dim * cell_size)  # Cap total figure size to avoid excessive width

        title_fontsize = max(14 + max_dim // 2, 28)
        label_fontsize = max(8, 12 - max_dim // 8)
        annot_fontsize = max(6, 12 - max_dim // 8)

        triangle_mask = pd.DataFrame(
            np.tril(np.ones(synergy_matrix.shape, dtype=bool), k=0),
            index=synergy_matrix.index,
            columns=synergy_matrix.columns
        )
        synergy_matrix = synergy_matrix.where(triangle_mask)
        pvalue_matrix = pvalue_matrix.where(triangle_mask)
        display_matrix = synergy_matrix.where(triangle_mask)
        nan_bg = pd.DataFrame(np.nan, index=synergy_matrix.index, columns=synergy_matrix.columns)
        nan_bg[triangle_mask & synergy_matrix.isna()] = 2

        text_matrix = np.where(
            np.isnan(synergy_matrix.values),
            "",
            synergy_matrix.round(2).astype(str)
        )

        counts_lookup = {}
        for _, row in self.synergy_results.iterrows():
            forward = {
                "Synergy": row["synergy"],
                "P_value": row["p_value"],
                "P_value_corrected": row.get("adj_p_value", None),
                "Count_00|y=0": row["n(A:E/B:E)_y0"],
                "Count_01|y=0": row["n(A:E/B:O)_y0"],
                "Count_10|y=0": row["n(A:O/B:E)_y0"],
                "Count_11|y=0": row["n(A:O/B:O)_y0"],
                "N_y0": row["N_y0"],
                "Count_00|y=1": row["n(A:E/B:E)_y1"],
                "Count_01|y=1": row["n(A:E/B:O)_y1"],
                "Count_10|y=1": row["n(A:O/B:E)_y1"],
                "Count_11|y=1": row["n(A:O/B:O)_y1"],
                "N_y1": row["N_y1"],
                "n_individuals": row["n_individuals"],
            }

            backward = {
                "Synergy": row["synergy"],
                "P_value": row["p_value"],
                "P_value_corrected": row.get("adj_p_value", None),
                "Count_00|y=0": row["n(A:E/B:E)_y0"],
                "Count_01|y=0": row["n(A:O/B:E)_y0"],
                "Count_10|y=0": row["n(A:E/B:O)_y0"],
                "Count_11|y=0": row["n(A:O/B:O)_y0"],
                "N_y0": row["N_y0"],
                "Count_00|y=1": row["n(A:E/B:E)_y1"],
                "Count_01|y=1": row["n(A:O/B:E)_y1"],
                "Count_10|y=1": row["n(A:E/B:O)_y1"],
                "Count_11|y=1": row["n(A:O/B:O)_y1"],
                "N_y1": row["N_y1"],
                "n_individuals": row["n_individuals"],
            }

            if "n_pmids" in row.index:
                forward["n_pmids"] = row["n_pmids"]
                forward["pmids"] = row.get("pmids", "")
                backward["n_pmids"] = row["n_pmids"]
                backward["pmids"] = row.get("pmids", "")

            counts_lookup[(row["HPO_A"], row["HPO_B"])] = forward
            counts_lookup[(row["HPO_B"], row["HPO_A"])] = backward

        
        hover_text = []
        for i, row in enumerate(pvalue_matrix.index):
            hover_row = []
            for j, col in enumerate(pvalue_matrix.columns):
                synergy = synergy_matrix.iloc[i, j]
                pval = pvalue_matrix.iloc[i, j]

                display_row = self._format_hpo_pair(row, self.label_mapping.get(row))
                display_col = self._format_hpo_pair(col, self.label_mapping.get(col))

                if not triangle_mask.iloc[i, j] or np.isnan(synergy):
                    hover_row.append("")
                else:
                    counts = counts_lookup.get((row, col), {})
                    pmid_block = ""
                    if "n_pmids" in counts:
                        pmids_raw = counts.get("pmids", "")
                        pmid_text = self._format_pmids_for_tooltip(pmids_raw, max_pmids=4)
                        pmid_block = (
                            f"<b>N_PMIDs</b>: {int(counts.get('n_pmids', 0))}<br>"
                            f"<b>PMIDs</b>: {pmid_text}"
                        )
                    hover_row.append(
                        f"<b>HPO_A</b>: {display_col}<br><b>HPO_B</b>: {display_row}<br>"
                        f"<b>Synergy</b>: {synergy:.2f}<br><b>p-val</b>: {pval:.6f}<br>"
                        f"<b>adj_p_val</b>: {counts.get('P_value_corrected', np.nan):.6f}<br>"
                        f"<b>Counts (y=0)</b><br>"
                        f"&nbsp;&nbsp;E/E: {counts.get('Count_00|y=0', 0)}, "
                        f"E/O: {counts.get('Count_01|y=0', 0)}, "
                        f"O/E: {counts.get('Count_10|y=0', 0)}, "
                        f"O/O: {counts.get('Count_11|y=0', 0)} "
                        f" (<i>N={counts.get('N_y0', 0)}</i>)<br>"
                        f"<b>Counts (y=1)</b><br>"
                        f"&nbsp;&nbsp;E/E: {counts.get('Count_00|y=1', 0)}, "
                        f"E/O: {counts.get('Count_01|y=1', 0)}, "
                        f"O/E: {counts.get('Count_10|y=1', 0)}, "
                        f"O/O: {counts.get('Count_11|y=1', 0)} "
                        f" (<i>N={counts.get('N_y1', 0)}</i>)<br>"
                        f"<b>Total_individuals</b>: {counts.get('n_individuals', 0)}<br>"
                        f"{pmid_block}"
                    )
            hover_text.append(hover_row)
       
        synergy_matrix.rename(columns=self.label_mapping, index=self.label_mapping, inplace=True)

        # --- Create heatmap figure ---
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=nan_bg.values,
            x=synergy_matrix.columns,
            y=synergy_matrix.index,
            colorscale = [
                [0, "#f8fbfe"],
                [1, "#f8fbfe"],
            ],
            showscale=False,
            hoverinfo="skip",
            xgap=1,
            ygap=1,
        ))
        values = display_matrix.to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]

        neutral_color = "#F7F5F1"

        if finite_values.size == 0:
            # All values are missing
            min_value = -1.0
            max_value = 1.0

            colorscale = [
                [0.0, "#24476B"],
                [0.5, neutral_color],
                [1.0, "#5A2E7A"],
            ]

        else:
            min_value = float(np.min(finite_values))
            max_value = float(np.max(finite_values))

            if min_value < 0 < max_value:

                zero_position = (
                    -min_value
                    / (max_value - min_value)
                )

                colorscale = [
                    [0.0, "#24476B"],
                    [zero_position / 2, "#AFC3D9"],
                    [zero_position, neutral_color],
                    [
                        zero_position
                        + (1 - zero_position) / 2,
                        "#C8A9CF",
                    ],
                    [1.0, "#5A2E7A"],
                ]

            elif min_value >= 0 and max_value > 0:

                # Include zero in the displayed range
                min_value = 0.0

                colorscale = [
                    [0.0, neutral_color],
                    [0.5, "#C8A9CF"],
                    [1.0, "#5A2E7A"],
                ]

            elif max_value <= 0 and min_value < 0:

                # Include zero in the displayed range
                max_value = 0.0

                colorscale = [
                    [0.0, "#24476B"],
                    [0.5, "#AFC3D9"],
                    [1.0, neutral_color],
                ]

            else:
                # All finite values are exactly zero
                min_value = -1.0
                max_value = 1.0

                colorscale = [
                    [0.0, "#24476B"],
                    [0.5, neutral_color],
                    [1.0, "#5A2E7A"],
                ]
        fig.add_trace(go.Heatmap(
                z=display_matrix.values,
                x=synergy_matrix.columns,
                y=synergy_matrix.index,
                colorscale=colorscale,
                zmin=min_value,
                zmax=max_value,
                text=text_matrix,
                texttemplate=f"<span style='font-size:{annot_fontsize}px'>%{{text}}</span>",
                hovertext=hover_text,
                hoverinfo="text",
                colorbar=dict(title="Synergy", len=0.8, thickness=title_fontsize),
                xgap=1,
                ygap=1,
                )
            )

        max_ylabel_len = max(len(str(lbl)) for lbl in synergy_matrix.index)
        left_margin = 60 + max_ylabel_len * label_fontsize

        fig.update_layout(
            title=dict(
                text=f"<b>Pairwise Synergy Heatmap of HPO Features</b><br>"
                    f"<span style='font-size:0.8em'>With respect to {self.condition_name}</span>",
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
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        fig.update_yaxes(autorange="reversed")
        self.fig = fig
        return fig
    
    def save_synergy_heatmap(
        self,
        output_file: str = "synergy_heatmap.html",
        *,
        width: int | None = None,
        height: int | None = None,
        scale: float = 2.0,
    ) -> None:
        """
        Save the synergy heatmap as HTML or a static image.

        Parameters
        ----------
        output_file : str
            Output path. Supported extensions are ``.html``, ``.png``,
            ``.svg``, ``.pdf``, and ``.jpg``.

        width : int, optional
            Export width in pixels for static image formats.

        height : int, optional
            Export height in pixels for static image formats.

        scale : float, default=2.0
            Resolution multiplier for raster images such as PNG.
            This has no meaningful effect on vector formats such as SVG or PDF.

        Raises
        ------
        RuntimeError
            If no heatmap has been generated.

        ValueError
            If the output format is unsupported.
        """
        if self.fig is None:
            raise RuntimeError(
                "No heatmap figure found. "
                "Please run `plot_synergy_heatmap()` first."
            )

        output_path = Path(output_file)
        extension = output_path.suffix.lower()

        supported_extensions = {
            ".html",
            ".png",
            ".svg",
            ".pdf",
            ".jpg",
        }

        if extension not in supported_extensions:
            raise ValueError(
                "Unsupported output format. "
                "Use one of: .html, .png, .svg, .pdf, .jpg"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if extension == ".html":
            self.fig.write_html(str(output_path))
        else:
            self.fig.write_image(
                str(output_path),
                width=width,
                height=height,
                scale=scale,
            )


class SynergyAnalyzer:
    """
    Analyze pairwise synergy between HPO terms with respect to a target.

    This class computes pairwise feature synergy using mutual information
    and permutation testing. Targets can be retrieved from pre-built target
    matrices or generated from metadata.
    """

    def __init__(
        self, 
        dataset: PhenotypeDataset, 
        min_individuals_for_synergy_calculation: int = 30, 
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        dataset : PhenotypeDataset
            Analysis-ready dataset containing HPO features, targets, and metadata.

        n_permutations : int, default=1000
            Number of permutations used to estimate p-values.

        min_individuals_for_synergy_calculation : int, default=30
            Minimum number of valid individuals required to evaluate a feature pair.

        random_state : int, default=42
            Random seed for reproducible permutation testing.
        """
        if not isinstance(dataset, PhenotypeDataset):
            raise TypeError("dataset must be an instance of PhenotypeDataset.")
        
        self.dataset = dataset
        hpo_df =self.dataset.hpo_data.matrix
        self.X = hpo_df.to_numpy(copy=True)
        self.hpo_terms = hpo_df.columns
        self.n_features = hpo_df.shape[1]
        self.individual_ids = hpo_df.index
    
        self.label_mapping = self.dataset.hpo_data.label_mapping
        self.rng = np.random.default_rng(random_state)

        relationship_mask_df = self.dataset.hpo_data.relationship_mask
        if relationship_mask_df is not None:
            self.relationship_mask = relationship_mask_df.to_numpy(copy=True)
        else:
            logger.warning("No relationship_mask provided. All feature pairs will be evaluated for synergy.")
            self.relationship_mask = np.zeros((self.n_features, self.n_features))
            np.fill_diagonal(self.relationship_mask, np.nan)

        self.min_individuals_for_synergy_calculation = min_individuals_for_synergy_calculation

    @staticmethod
    def _encode_joint_binary_index( 
        xi:np.ndarray, 
        xj:np.ndarray
    ) -> np.ndarray:
        """Encode two binary features into a joint integer representation in {0, 1, 2, 3}."""
        return (xi.astype(int) << 1) | xj.astype(int)

    def evaluate_pair_synergy(
        self, 
        i:int,
        j:int,
        n_perms: int = 5000,
        include_pmids: bool = True
    ) -> tuple[int, int, float, float, dict[str, Any]]: 
        """
        Compute synergy and a permutation-based p-value for one feature pair.

        Parameters
        ----------
        i : int
            Index of the first feature.

        j : int
            Index of the second feature.

        n_perms : int, default=5000
            Number of permutations used to estimate p-values.

        include_pmids : bool, default=True
            If ``True``, aggregate PMIDs from contributing individuals.

        Returns
        -------
        tuple[int, int, float, float, dict]
            Feature indices, corrected synergy, p-value, and count summary.
        """
        mask = (~np.isnan(self.X[:, i]) & ~np.isnan(self.X[:, j]) & ~np.isnan(self.y))
        xi = self.X[mask, i]
        xj = self.X[mask, j]
        y = self.y[mask]
        total = len(xi)

        empty_counts: dict[str, Any] = {
            "00|y=0": 0,
            "01|y=0": 0,
            "10|y=0": 0,
            "11|y=0": 0,
            "N_y0": 0,
            "00|y=1": 0,
            "01|y=1": 0,
            "10|y=1": 0,
            "11|y=1": 0,
            "N_y1": 0,
            "N": 0,
            "n_pmid": np.nan,
            "pmids": [],
        }

        if total == 0 or np.all(xi == xi[0]) or np.all(xj == xj[0]) or np.all(y == y[0]):
            return i, j, np.nan, np.nan, empty_counts
        
        if np.array_equal(xi, xj):
            return i, j, np.nan, np.nan, empty_counts

        if include_pmids:
            individual_ids = self.individual_ids[mask]
            all_pmids_series = self.dataset.get_pmids()
            pmids_list = all_pmids_series.loc[individual_ids].to_numpy()

            all_pmids = sorted(
                {
                    str(pmid)
                    for pmids in pmids_list
                    if pmids is not None
                    for pmid in pmids
                    if pd.notna(pmid)
                }
            )
            n_pmids = len(all_pmids)
        else:
            all_pmids = []
            n_pmids = np.nan
        
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
        "n_pmid": n_pmids,
        "pmids": all_pmids,
        }

        mi_i = mutual_info_score(xi, y) / np.log(2)
        mi_j = mutual_info_score(xj, y) / np.log(2)

        joint_index = self._encode_joint_binary_index(xi, xj)
        mi_ij = mutual_info_score(joint_index, y) / np.log(2)
        observed_synergy = mi_ij - (mi_i + mi_j)
        abs_observed = np.abs(observed_synergy)

  
        extreme_count = 0
        
        for _ in range(n_perms):
            y_perm = self.rng.permutation(y)

            mi_i_p = mutual_info_score(xi, y_perm) / np.log(2)
            mi_j_p = mutual_info_score(xj, y_perm) / np.log(2)
            mi_ij_p = mutual_info_score(joint_index, y_perm) / np.log(2)
            
            sim_synergy = mi_ij_p - (mi_i_p + mi_j_p)
            
            if np.abs(sim_synergy) >= abs_observed:
                extreme_count += 1
                

        p_value = (extreme_count + 1) / (n_perms + 1)

        return i, j, float(observed_synergy), float(p_value), counts

    def compute_synergy_matrix(
        self, 
        condition: pd.Series,
        n_jobs=-1,
        include_pmids: bool = True,
        n_perms: int = 5000,
    ) -> SynergyResult:
        """
        Compute pairwise synergy scores for all valid HPO term pairs.

        Parameters
        ----------
        condition : pd.Series
            Boolean condition to filter the dataset.

        n_jobs : int, default=-1
            Number of parallel jobs. ``-1`` uses all available CPUs.

        include_pmids : bool, default=True
            If ``True``, aggregate PMIDs from contributing individuals and
            include them in the result table.

        n_perms : int, default=5000
            Number of Monte Carlo target-label permutations performed for each
            phenotype pair when exhaustive enumeration is not used. The resulting
            empirical p-value has a minimum attainable value of
            1 / (n_perms + 1). Larger values provide finer p-value resolution
            and lower Monte Carlo uncertainty at greater computational cost.

        Returns
        -------
        SynergyResult
            An object encapsulating the long-format synergy statistics, symmetric 
            score/p-value matrices, and helper plotting methods.
        """
        self.y = condition.to_numpy(copy=True)
        cond_name = condition.name if condition.name else "unnamed_condition"
        
        has_y_one = np.any(self.y == 1)
        has_y_zero = np.any(self.y == 0)

        if not has_y_one or not has_y_zero:

            if not has_y_one and not has_y_zero:
                reason = (
                    "The target contains no positive (1) or "
                    "negative (0) values."
                )
            elif not has_y_one:
                reason = "The target contains no positive values (1)."
            else:
                reason = "The target contains no negative values (0)."

            raise ValueError(
                "[Phenotype-pair synergy: invalid target]\n"
                f"{reason}\n"
                "At least one value from each target group is required."
            )

        has_x_one = np.any(self.X == 1)
        has_x_zero = np.any(self.X == 0)

        if not has_x_one or not has_x_zero:

            if not has_x_one and not has_x_zero:
                reason = (
                    "The HPO matrix contains no observed (1) or "
                    "excluded (0) annotations."
                )
            elif not has_x_one:
                reason = (
                    "The HPO matrix contains no observed annotations (1)."
                )
            else:
                reason = (
                    "The HPO matrix contains no excluded annotations (0)."
                )

            raise ValueError(
                "[Phenotype-pair synergy: invalid HPO matrix]\n"
                f"{reason}\n"
                "At least one annotation of each type is required."
            )

        mask_X = ~np.isnan(self.X)  # X valid
        mask_y = ~np.isnan(self.y)  # y valid
        mask_combined = mask_X & mask_y[:, None]  # shape (n_samples, n_features)

        valid_counts = mask_combined.T.astype(int) @ mask_combined.astype(int)
        valid_counts_sparse = triu(coo_matrix(valid_counts), k=1)
        rows, cols, counts = valid_counts_sparse.row, valid_counts_sparse.col, valid_counts_sparse.data

        ontology_values = self.relationship_mask[rows, cols]
        ontology_candidate = ~np.isnan(ontology_values)

        n_pairs_after_filter = np.sum(ontology_candidate)

        candidate_idx = np.where(ontology_candidate & (counts >= self.min_individuals_for_synergy_calculation))[0]
        rows_cand, cols_cand = rows[candidate_idx], cols[candidate_idx]
        pairs = list(zip(rows_cand, cols_cand))

        if len(pairs) == 0:
            logger.warning(
                "[Phenotype-pair synergy: no eligible pairs]\n"
                f"Pairs after ontology filtering: "
                f"{n_pairs_after_filter}\n"
                f"Pairs meeting effective N >= "
                f"{self.min_individuals_for_synergy_calculation}: "
                f"{len(pairs)}\n"
                "No pairs were evaluated."
            )
            empty_df = pd.DataFrame(columns=["HPO_A", "HPO_B", "synergy", "p_value", "adj_p_value"])
            empty_matrix = pd.DataFrame(index=self.hpo_terms, columns=self.hpo_terms, dtype=float)
            return SynergyResult(empty_df, empty_matrix, empty_matrix, self.label_mapping, cond_name)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.evaluate_pair_synergy)(
                i, j, n_perms=n_perms,
                include_pmids=include_pmids) for i, j in tqdm(pairs, desc="Calculating pairwise synergy")
        )
        synergy_matrix = np.full((self.n_features, self.n_features), np.nan)
        pvalue_matrix = np.full((self.n_features, self.n_features), np.nan)

        rows = []
        for i, j, synergy, pval, counts in results:
            synergy_matrix[i, j] = synergy_matrix[j, i] = synergy
            pvalue_matrix[i, j] = pvalue_matrix[j, i] = pval
            hpo_a, hpo_b = self.hpo_terms[i], self.hpo_terms[j]
            if j > i:  # only upper triangle
                if not np.isnan(synergy):
                    row_data = {
                        "HPO_A": hpo_a,
                        **({"HPO_A_label": self.label_mapping.get(hpo_a)} if self.label_mapping.get(hpo_a) else {}),
                        "HPO_B": hpo_b,
                        **({"HPO_B_label": self.label_mapping.get(hpo_b)} if self.label_mapping.get(hpo_b) else {}),
                        "synergy": synergy,
                        "p_value": pval,
                        "n(A:E/B:E)_y0": counts["00|y=0"],
                        "n(A:E/B:O)_y0": counts["01|y=0"],
                        "n(A:O/B:E)_y0": counts["10|y=0"],
                        "n(A:O/B:O)_y0": counts["11|y=0"],
                        "N_y0": counts["N_y0"],
                        "n(A:E/B:E)_y1": counts["00|y=1"],
                        "n(A:E/B:O)_y1": counts["01|y=1"],
                        "n(A:O/B:E)_y1": counts["10|y=1"],
                        "n(A:O/B:O)_y1": counts["11|y=1"],
                        "N_y1": counts["N_y1"],
                        "n_individuals": counts["N"],
                    }
                    if include_pmids:
                        row_data["n_pmids"] = counts["n_pmid"]
                        row_data["pmids"] = ";".join(counts.get("pmids", []))

                    rows.append(row_data)

        valid_mask = ~(np.isnan(synergy_matrix).all(axis=0)) 
        valid_hpo_terms = self.hpo_terms[valid_mask]
        if len(valid_hpo_terms) == 0:
            logger.warning(
                "[Phenotype-pair synergy: no valid results]\n"
                f"Pairs evaluated: {len(pairs)}\n"
                "Pairs producing a valid synergy score: 0\n"
                "For every evaluated pair, at least one phenotype or "
                "the target had no variation within the effective sample, "
                "or the two phenotype profiles were identical."
            )
            empty_df = pd.DataFrame(columns=["HPO_A", "HPO_B", "synergy", "p_value", "adj_p_value"])
            empty_matrix = pd.DataFrame(index=self.hpo_terms, columns=self.hpo_terms, dtype=float)
            return SynergyResult(empty_df, empty_matrix, empty_matrix, self.label_mapping, cond_name)


        self.synergy_matrix = pd.DataFrame(
            synergy_matrix[np.ix_(valid_mask, valid_mask)],
            index=valid_hpo_terms,
            columns=valid_hpo_terms
        )

        self.pvalue_matrix = pd.DataFrame(
            pvalue_matrix[np.ix_(valid_mask, valid_mask)],
            index=valid_hpo_terms,
            columns=valid_hpo_terms
        )
        self.synergy_results = pd.DataFrame(rows)
        if not self.synergy_results.empty:
            pvals = self.synergy_results["p_value"].values
            _, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
            loc = int(self.synergy_results.columns.get_loc("p_value"))
            self.synergy_results.insert(
                loc + 1,
                "adj_p_value",
                pvals_corrected
            )
            self.synergy_results.sort_values(by="adj_p_value", ascending=True, inplace=True)
        else:
            self.synergy_results["adj_p_value"] = pd.Series(dtype=float)
        
        return SynergyResult(
            synergy_results=self.synergy_results,
            synergy_matrix=self.synergy_matrix,
            pvalue_matrix=self.pvalue_matrix,
            label_mapping=self.label_mapping,
            condition_name=cond_name
        )
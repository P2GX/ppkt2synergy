from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.stats
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix, triu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ..core import PhenotypeDataset

logger = logging.getLogger(__name__)


class CorrelationResult:
    """
    A class to store, manage, and visualize HPO pairwise correlation results.
    """

    def __init__(
        self, 
        correlation_results:pd.DataFrame, 
        coef_matrix: pd.DataFrame, 
        pval_matrix: pd.DataFrame, 
        label_mapping: dict[str, str]
    ) -> None:
        self.correlation_results = correlation_results      
        self.coef_matrix = coef_matrix           
        self.pval_matrix = pval_matrix         
        self.label_mapping = label_mapping
        self.fig: go.Figure | None = None

    @property
    def results_table(self) -> pd.DataFrame:  
        """Get a safe copy of the correlation results table."""
        return self.correlation_results.copy()

    def save_correlation_results(
        self, 
        corr_threshold: float = 0.1,
        adj_pval_threshold: float = 0.05,
        output_file: str="correlation_results.csv"
    ) -> None:
        """
        Save correlation results to a CSV or Excel file.

        Parameters
        ----------
        corr_threshold : float, default=0.0
            Minimum correlation coefficient to retain.

        adj_pval_threshold : float, default=0.05
            Maximum adjusted p-value to retain.

        output_file : str, default="correlation_results.csv"
            Output file path. Supported formats are ``.csv``.

        Raises
        ------
        ValueError
            If correlation results have not been computed or if thresholds
            are invalid.
        """
        if self.correlation_results.empty:
            logger.warning("Correlation results table is empty. Saving empty file.")
            df = self.correlation_results.copy()
        else:
            df = self.correlation_results.copy()
            if not 0.0 <= corr_threshold <= 1.0:
                raise ValueError("corr_threshold must be between 0.0 and 1.0")
            df = df[df["correlation"].abs() >= corr_threshold]

            if not 0.0 <= adj_pval_threshold <= 1.0:
                raise ValueError("adj_pval_threshold must be between 0.0 and 1.0")
            df = df[df["adj_p_value"] < adj_pval_threshold]

        df.to_csv(output_file, index=False)  
  
    def filter_weak_correlations(
        self, 
        corr_threshold: float = 0.1,
        adj_pval_threshold: float = 0.05
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter the correlation and p-value matrices by effect size and significance.

        Parameters
        ----------
        corr_threshold : float, default=0.1
            Minimum correlation coefficient to retain.

        adj_pval_threshold : float, default=0.05
            Maximum adjusted p-value to retain.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Filtered correlation matrix and filtered p-value matrix.
        """
        coef_matrix = self.coef_matrix.copy()
        p_value = self.pval_matrix.copy()

        if not 0.0 <= corr_threshold <= 1.0:
            raise ValueError("corr_threshold must be between 0.0 and 1.0")
        mask = coef_matrix.abs() < corr_threshold
        coef_matrix[mask] = np.nan
        p_value[mask] = np.nan

        if not 0.0 <= adj_pval_threshold <= 1.0:
            raise ValueError("adj_pval_threshold must be between 0.0 and 1.0")
        
        if not self.correlation_results.empty:
            non_signif = self.correlation_results.loc[
                (self.correlation_results["adj_p_value"] >= adj_pval_threshold),
                ["HPO_A", "HPO_B"]
            ]
            for _, row in non_signif.iterrows():
                hpo_a, hpo_b = row["HPO_A"], row["HPO_B"]
                if hpo_a in coef_matrix.index and hpo_b in coef_matrix.columns:
                    coef_matrix.loc[hpo_a, hpo_b] = np.nan
                    coef_matrix.loc[hpo_b, hpo_a] = np.nan
                    p_value.loc[hpo_a, hpo_b] = np.nan
                    p_value.loc[hpo_b, hpo_a] = np.nan
    
        mask_rows = coef_matrix.isna().all(axis=1)
        mask_cols = coef_matrix.isna().all(axis=0)
        coef_matrix_cleaned = coef_matrix.loc[~mask_rows, ~mask_cols]
        p_value_cleaned = p_value.loc[~mask_rows, ~mask_cols]

        return coef_matrix_cleaned, p_value_cleaned
    
    @staticmethod
    def _format_hpo_pair( 
        hpo_id: str, 
        label: str | None
    ) -> str:
        """Format an HPO term for display."""
        if label:
            return f"{label} ({hpo_id})"
        return hpo_id
    
    @staticmethod
    def _format_pmids_for_tooltip(
        pmids: str | list[str] | None,
        max_pmids: int = 5,
    ) -> str:
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

    def plot_correlation_heatmap_with_significance(
        self,
        corr_threshold: float = 0.1,
        adj_pval_threshold: float = 0.05,
        title_name: str | None = None,
    ) -> go.Figure:
        """
        Plot an interactive correlation heatmap with statistical filtering.
        """
        raw_coef, pval_matrix = self.filter_weak_correlations(
            corr_threshold=corr_threshold,
            adj_pval_threshold=adj_pval_threshold
        )

        if raw_coef.empty or np.isnan(raw_coef.values).all():
            raise ValueError(
                "The coefficient matrix is empty after filtering. "
                "Try adjusting `corr_threshold` or `adj_pval_threshold`."
            )
        
        coef_matrix = raw_coef.copy()

        n_rows, n_cols = coef_matrix.shape
        cell_size = 60  # Base pixel size per cell
        max_dim = max(n_rows, n_cols)
        fig_size = min(1200, max_dim * cell_size)  # Cap total figure size to avoid excessive width

        title_fontsize = max(14 + max_dim // 2, 28)
        label_fontsize = max(8, 12 - max_dim // 8)
        annot_fontsize = max(6, 12 - max_dim // 8)

        triangle_mask = pd.DataFrame(
            np.tril(np.ones(coef_matrix.shape, dtype=bool), k=0),
            index=coef_matrix.index,
            columns=coef_matrix.columns
        )
        coef_matrix = coef_matrix.where(triangle_mask)
        pval_matrix = pval_matrix.where(triangle_mask)
        display_matrix = coef_matrix.where(triangle_mask)

        nan_bg = pd.DataFrame(np.nan, index=coef_matrix.index, columns=coef_matrix.columns)
        nan_bg[triangle_mask & coef_matrix.isna()] = 2

        text_matrix = np.where(
            np.isnan(coef_matrix.values),
            "",
            coef_matrix.round(2).astype(str)
        )

        counts_lookup = {}
        for _, row in self.correlation_results.iterrows():
            forward = {
                "Coefficient": row["correlation"],
                "P_value": row["p_value"],
                "P_value_corrected": row.get("adj_p_value", None), 
                "Count_00": row["n(A:E/B:E)"],
                "Count_01": row["n(A:E/B:O)"],
                "Count_10": row["n(A:O/B:E)"],
                "Count_11": row["n(A:O/B:O)"],
                "n_individuals": row["n_individuals"],
            }

            backward = {
                "Coefficient": row["correlation"],
                "P_value": row["p_value"],
                "P_value_corrected": row.get("adj_p_value", None),
                "Count_00": row["n(A:E/B:E)"],
                "Count_01": row["n(A:O/B:E)"],  # swapped
                "Count_10": row["n(A:E/B:O)"],  # swapped
                "Count_11": row["n(A:O/B:O)"],
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
        for i, row in enumerate(coef_matrix.index):
            hover_row = []
            for j, col in enumerate(coef_matrix.columns):
                coef = coef_matrix.iloc[i, j]
                pval = pval_matrix.iloc[i, j]

                display_row = self._format_hpo_pair(row, self.label_mapping.get(row))
                display_col = self._format_hpo_pair(col, self.label_mapping.get(col))

                if not triangle_mask.iloc[i, j] or np.isnan(coef):
                    hover_row.append("")
                else:
                    counts = counts_lookup.get((row, col), {})
                    pmid_block = ""
                    if "n_pmids" in counts:
                        pmid_text = self._format_pmids_for_tooltip(
                            counts.get("pmids", ""),
                            max_pmids=4,
                        )
                        pmid_block = (
                            f"<b>N_PMIDs</b>: {int(counts.get('n_pmids', 0))}<br>"
                            f"<b>PMIDs</b>: {pmid_text}"
                        )
                    hover_row.append(
                        f"<b>HPO_A</b>: {display_col}<br><b>HPO_B</b>: {display_row}<br>"
                        f"<b>Corr</b>: {coef:.2f}<br><b>p-val</b>: {pval:.6f}<br>"
                        f"<b>adj_p_val</b>: {counts.get('P_value_corrected', np.nan):.6f}<br>"
                        f"<b>Counts(A/B): E/E</b>: {counts.get('Count_00', 0)}, "
                        f"<b>E/O</b>: {counts.get('Count_01', 0)}, "
                        f"<b>O/E</b>: {counts.get('Count_10', 0)}, "
                        f"<b>O/O</b>: {counts.get('Count_11', 0)}<br>"
                        f"<b>Total_individuals</b>: {counts.get('n_individuals', 0)}<br>"
                        f"{pmid_block}"
                    )
            hover_text.append(hover_row)
        
        coef_matrix.rename(index=self.label_mapping, columns=self.label_mapping, inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=nan_bg.values,
            x=coef_matrix.columns,
            y=coef_matrix.index,
            colorscale=[[0, "#eef4fb"], [1, "#eef4fb"]],
            showscale=False,
            hoverinfo="skip",
            xgap=1,
            ygap=1,
        ))
        fig.add_trace(go.Heatmap(
                z=display_matrix.values,
                x=coef_matrix.columns,
                y=coef_matrix.index,
                colorscale=[ 
                [0.00, "#203864"],   # navy
                [0.50, "#F7F4ED"],   # ivory
                [1.00, "#7A1F3D"]    # wine
                ],
                zmin=-1,
                zmax=1,
                zmid=0,
                text=text_matrix,
                texttemplate=f"<span style='font-size:{annot_fontsize}px'>%{{text}}</span>",
                hovertext=hover_text,
                hoverinfo="text",
                colorbar=dict(title="Corr.", len=0.8, thickness=title_fontsize),
                xgap=1,
                ygap=1,
            ))
        
        max_ylabel_len = max(len(str(lbl)) for lbl in coef_matrix.index) if not coef_matrix.empty else 10
        left_margin = 60 + max_ylabel_len * label_fontsize

        clean_subtitle = title_name.strip() if title_name and title_name.strip() else ""

        main_title = "<b>Phi Coefficient Matrix for HPO Pairwise Associations</b>"
        
        full_title = f"{main_title}<br><span style='font-size:0.8em'>{clean_subtitle}</span>" if clean_subtitle else main_title

        fig.update_layout(
            title=dict(
                text=full_title,
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
    
    def save_correlation_heatmap(
        self, 
        output_file: str = "correlation_heatmap.html",
        *,
        width: int | None = None,
        height: int | None = None,
        scale: float = 2.0,
        ) -> None:
        """
        Save a correlation heatmap as an HTML or a static image.

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
            raise RuntimeError("No heatmap figure found. Please run `plot_correlation_heatmap_with_significance()` first.")
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

class HPOCorrelationAnalyzer:
    """
    Analyze pairwise correlations between HPO terms using the Phi coefficient and Fisher's exact test.
    """

    def __init__(
        self,  
        dataset: PhenotypeDataset, 
        min_individuals_for_correlation_test: int = 20,
    ) -> None:
        """
        Parameters
        ----------
        dataset : PhenotypeDataset
            Dataset containing HPO feature data and metadata.

        min_individuals_for_correlation_test : int, default=20
            Minimum number of valid individuals required to evaluate a
            pairwise correlation.
        """

        if not isinstance(dataset, PhenotypeDataset):
            raise TypeError("`dataset` must be a `PhenotypeDataset` instance.")
        self.dataset= dataset
        self.hpo_matrix = self.dataset.hpo_data.matrix
        self.hpo_terms = self.hpo_matrix.columns
        self.n_features = self.hpo_matrix.shape[1]
        self.label_mapping = self.dataset.hpo_data.label_mapping
        self.individual_ids = self.hpo_matrix.index 

        relationship_mask_df = self.dataset.hpo_data.relationship_mask
        if relationship_mask_df is not None:
            self.relationship_mask = relationship_mask_df.to_numpy(copy=True)
        else:
            logger.warning("No relationship_mask provided. All feature pairs will be evaluated for correlation.")
            self.relationship_mask = np.zeros((self.n_features, self.n_features))
            np.fill_diagonal(self.relationship_mask, np.nan)    
        
        self.min_individuals_for_correlation_test = min_individuals_for_correlation_test

    @staticmethod
    def _calculate_stats( observed_status_A: np.ndarray, observed_status_B: np.ndarray) -> tuple[float, float]:
        """Compute the Phi correlation coefficient and Fisher's Exact test p-value."""
        confusion_matrix = pd.crosstab(observed_status_A, observed_status_B, dropna=False)
        if confusion_matrix.shape == (2, 2):
            a = confusion_matrix.iloc[0, 0]
            b = confusion_matrix.iloc[0, 1]
            c = confusion_matrix.iloc[1, 0]
            d = confusion_matrix.iloc[1, 1]
            
            numerator = (a * d) - (b * c)
            denominator = np.sqrt(int(a + b) * int(c + d) * int(a + c) * int(b + d))
            phi = numerator / denominator if denominator != 0 else np.nan
        else:
            phi = np.nan

        try:
            _, pval = scipy.stats.fisher_exact(confusion_matrix)
        except:
            pval = np.nan

        return phi, pval     

    def _calculate_pairwise_correlation(
        self,
        col_a: int,
        col_b: int, 
        include_pmids: bool = True
    ) -> tuple[int, int, float, float, dict[str,Any]]:
        """Compute the correlation between two specific HPO term columns."""
        matrix = self.hpo_matrix.values
        mask = (~np.isnan(matrix[:, col_a])) & (~np.isnan(matrix[:, col_b]))
        col_a_values = matrix[mask, col_a]
        col_b_values = matrix[mask, col_b]

        count_11 = np.sum((col_a_values == 1) & (col_b_values == 1))
        count_10 = np.sum((col_a_values == 1) & (col_b_values == 0))
        count_01 = np.sum((col_a_values == 0) & (col_b_values == 1))
        count_00 = np.sum((col_a_values == 0) & (col_b_values == 0))
        total = len(col_a_values)

        empty_counts: dict[str, Any] = {
            "00": 0,
            "01": 0,
            "10": 0,
            "11": 0,
            "N": 0,
            "n_pmid": np.nan,
            "pmids": [],
        }
        
        if total == 0 or np.all(col_a_values == col_a_values[0]) or np.all(col_b_values == col_b_values[0]):
            return (col_a, col_b, np.nan, np.nan, empty_counts)
        
        try:
            coef, p_val = self._calculate_stats(col_a_values, col_b_values)
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

            return (col_a, col_b, coef, p_val, {
                "00": count_00,
                "01": count_01,
                "10": count_10,
                "11": count_11,
                "N": total,
                "n_pmid": n_pmids,
                "pmids": all_pmids,
            })
        except Exception as e:
            logger.error(
                "Error calculating correlation for columns %d and %d: %s",
                col_a,
                col_b,
                e,
            )
            return col_a, col_b, np.nan, np.nan, empty_counts
        
    def compute_correlation_matrix(
        self,  
        n_jobs: int = -1,
        include_pmids: bool = True
    ) -> CorrelationResult:
        """
        Compute pairwise correlations between HPO terms.

        Parameters
        ----------
        correlation_type : str | CorrelationType, default="spearman"
            Correlation metric to compute.
            Supported values:
            - "spearman"
            - "phi"

        n_jobs : int, default=-1
            Number of parallel jobs. ``-1`` uses all available CPUs.

        include_pmids : bool, default=True
            If ``True``, aggregate PMIDs from contributing individuals.

        Returns
        -------
        CorrelationResult
            An object encapsulating the long-format correlationnstatistics, symmetric 
                score/p-value matrices, and helper plotting methods.
        """
        x = self.hpo_matrix.to_numpy()

        has_one = np.any(x == 1)
        has_zero = np.any(x == 0)

        if not has_one or not has_zero:

            if not has_one and not has_zero:
                reason = (
                    "The HPO matrix contains no observed (1) "
                    "or excluded (0) annotations."
                )
            elif not has_one:
                reason = (
                    "The HPO matrix contains no observed "
                    "annotations (1)."
                )
            else:
                reason = (
                    "The HPO matrix contains no excluded "
                    "annotations (0)."
                )

            raise ValueError(
                "[Pairwise association: invalid HPO matrix]\n"
                f"{reason}\n"
                "At least one observed annotation and one excluded "
                "annotation are required."
            )
        mask = ~np.isnan(x)  
        valid_counts = mask.T.astype(int) @ mask.astype(int)
        valid_counts_sparse = triu(coo_matrix(valid_counts), k=1)
        rows, cols, counts = (
            valid_counts_sparse.row, 
            valid_counts_sparse.col, 
            valid_counts_sparse.data
        )

        ontology_values = self.relationship_mask[rows, cols]
        ontology_candidate = ~np.isnan(ontology_values)

        n_pairs_after_filter = np.sum(ontology_candidate)

        candidate_idx = np.where(ontology_candidate & (counts >= self.min_individuals_for_correlation_test))[0]
        rows_cand, cols_cand = rows[candidate_idx], cols[candidate_idx]
        pairs = list(zip(rows_cand, cols_cand))

        if len(pairs) == 0:
            logger.warning(
                "[Pairwise association: no eligible pairs]\n"
                f"Pairs after ontology filtering: {n_pairs_after_filter}\n"
                f"Pairs meeting effective N >= "
                f"{self.min_individuals_for_correlation_test}: {len(pairs)}\n"
                "No pairs were tested."
            )
            empty_df = pd.DataFrame(columns=["HPO_A", "HPO_B", "correlation", "p_value", "adj_p_value"])
            empty_matrix = pd.DataFrame(index=self.hpo_terms, columns=self.hpo_terms, dtype=float)
            return CorrelationResult(empty_df, empty_matrix, empty_matrix, self.label_mapping)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._calculate_pairwise_correlation)(i, j, include_pmids=include_pmids)
            for i, j in tqdm(pairs, desc="Calculating pairwise correlation")
        )
        
        coef_matrix = np.full((self.n_features, self.n_features), np.nan)
        pvalue_matrix = np.full((self.n_features, self.n_features), np.nan)

        rows = []
        for r in results:
            i, j, coef, pval, counts = r
            coef_matrix[i, j] = coef
            coef_matrix[j, i] = coef
            pvalue_matrix[i, j] = pval
            pvalue_matrix[j, i] = pval

            hpo_a, hpo_b = self.hpo_terms[i], self.hpo_terms[j]
            if j > i:  
                if not np.isnan(coef):
                    row_data = {
                    "HPO_A": hpo_a,
                    **({"HPO_A_label": self.label_mapping.get(hpo_a)} if self.label_mapping.get(hpo_a) else {}),
                    "HPO_B": hpo_b,
                    **({"HPO_B_label": self.label_mapping.get(hpo_b)} if self.label_mapping.get(hpo_b) else {}),
                    "correlation": coef,
                    "p_value": pval,
                    "n(A:E/B:E)": counts["00"],
                    "n(A:E/B:O)": counts["01"],
                    "n(A:O/B:E)": counts["10"],
                    "n(A:O/B:O)": counts["11"],
                    "n_individuals": counts["N"],
                    }
                    if include_pmids:
                        row_data["n_pmids"] = counts["n_pmid"]
                        row_data["pmids"] = ";".join(counts.get("pmids", []))
                    rows.append(row_data)

        valid_mask = ~(np.isnan(coef_matrix).all(axis=0)) 

        if not np.any(valid_mask):
            logger.warning(
                "[Pairwise association: no valid results]\n"
                f"Pairs evaluated: {len(pairs)}\n"
                "Pairs producing a valid phi coefficient: 0\n"
                "For every evaluated pair, at least one phenotype had "
                "no variation within the effective sample, or the "
                "calculation failed."
            )
            empty_df = pd.DataFrame(columns=["HPO_A", "HPO_B", "correlation", "p_value", "adj_p_value"])
            empty_matrix = pd.DataFrame(index=self.hpo_terms, columns=self.hpo_terms, dtype=float)
            return CorrelationResult(empty_df, empty_matrix, empty_matrix, self.label_mapping)

        filtered_columns = self.hpo_terms[valid_mask]

        self.coef_df = pd.DataFrame(coef_matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)
        self.pval_df = pd.DataFrame(pvalue_matrix[np.ix_(valid_mask, valid_mask)], index=filtered_columns, columns=filtered_columns)
        self.correlation_results = pd.DataFrame(rows)

        if not self.correlation_results.empty:
            pvals = self.correlation_results["p_value"].values
            _, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
            loc = int(self.correlation_results.columns.get_loc("p_value"))
            self.correlation_results.insert(loc + 1, "adj_p_value", pvals_corrected)
            self.correlation_results.sort_values(by="adj_p_value", ascending=True, inplace=True)
        else:
            self.correlation_results["adj_p_value"] = pd.Series(dtype=float)

        return CorrelationResult(
            correlation_results = self.correlation_results,
            coef_matrix = self.coef_df,
            pval_matrix = self.pval_df,
            label_mapping = self.label_mapping
        )
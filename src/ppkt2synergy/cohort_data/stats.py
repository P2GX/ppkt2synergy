import numpy as np
from ppkt2synergy import CohortMatrixGenerator
import pandas as pd
import hpotk
from typing import List, Dict, Set, Union, IO,Tuple
import scipy.stats
import phenopackets as ppkt
from sklearn.metrics import matthews_corrcoef
import networkx as nx
import matplotlib.pyplot as plt


# Do not test terms pairs with less than 30 individuals
MIN_INDIVIDUALS_FOR_CORRELATION_TEST = 30

class CohortAnalyzer:

    """
    A class to analyze statistical relationships between HPO terms in a cohort.
    """
    def __init__(
        self,  
        cohort_matrix_generator: CohortMatrixGenerator,
        file: Union[IO, str] = None
    ):
        """
        Initialize the CohortAnalyzer class with a DataPreprocessor instance.
        """
        self.hpo = self._load_hpo(file)
        self.cohort_matrix_generator = cohort_matrix_generator
        self.combined_matrix = self._combine_matrices()

    def _load_hpo(
        self, 
        file: Union[IO, str]
        ) -> hpotk.MinimalOntology:
        """
        Load the HPO ontology from a file or the latest version.

        Args:
            file: A file path or file-like object to load the HPO ontology. If None, the latest HPO is loaded.

        Returns:
            A minimal HPO ontology object.
        """
        if file is None:
            store = hpotk.configure_ontology_store()
            return store.load_minimal_hpo()
        else:
            return hpotk.load_minimal_ontology(file)   

    
    def _combine_matrices(self) -> pd.DataFrame:
        """
        Combine HPO term matrix, sex matrix, and disease matrix into a single DataFrame.

        Returns:
            A combined DataFrame with all columns.
        """
        hpo_matrix = self.cohort_matrix_generator.hpo_term_observation_matrix
        sex_matrix = self.cohort_matrix_generator.sex_matrix
        disease_matrix = self.cohort_matrix_generator.disease_matrix

        # Combine matrices
        combined_matrix = hpo_matrix.join(sex_matrix).join(disease_matrix)
        return combined_matrix


    def calculate_stats(
        self, 
        observed_status_A: pd.Series, 
        observed_status_B: pd.Series
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate statistical metrics including Spearman correlation, Kendall's Tau, and their p-values 
        for two binary (0/1) observed status vectors.

        Args:
            observed_status_A (pd.Series): Binary values (0/1) for the first variable. Must have at least 30 valid entries.
            observed_status_B (pd.Series): Binary values (0/1) for the second variable. Must have at least 30 valid entries.

        Returns:
            dict: A dictionary containing the following metrics:
                - Spearman: Spearman rank correlation coefficient.
                - Spearman_p_value: P-value for Spearman rank correlation test.
                - Kendall's Tau: Kendall's Tau correlation coefficient.
                - Kendall_p_value: P-value for Kendall's Tau correlation test.
        """

        results = {}  
        if len(observed_status_A) >= MIN_INDIVIDUALS_FOR_CORRELATION_TEST:
            # Calculate Spearman correlation
            spearman_corr, spearman_p = scipy.stats.spearmanr(observed_status_A, observed_status_B)
            results["Spearman"] = spearman_corr
            results["Spearman_p_value"] = spearman_p

            # Calculate Kendall's Tau
            tau, p_value = scipy.stats.kendalltau(observed_status_A, observed_status_B)
            results["Kendall"] = tau
            results["Kendall_p_value"] = p_value

            # Calculate Phi coefficient
            confusion_matrix = pd.crosstab(observed_status_A, observed_status_B)
            n11 = confusion_matrix.loc[1, 1] if (1 in confusion_matrix.index and 1 in confusion_matrix.columns) else 0
            n00 = confusion_matrix.loc[0, 0] if (0 in confusion_matrix.index and 0 in confusion_matrix.columns) else 0
            n01 = confusion_matrix.loc[0, 1] if (0 in confusion_matrix.index and 1 in confusion_matrix.columns) else 0
            n10 = confusion_matrix.loc[1, 0] if (1 in confusion_matrix.index and 0 in confusion_matrix.columns) else 0

            n1_dot = n11 + n10  # Total where A = 1
            n0_dot = n00 + n01  # Total where A = 0
            n_dot1 = n11 + n01  # Total where B = 1
            n_dot0 = n10 + n00  # Total where B = 0
            n = n11 + n10 + n01 + n00  # Total number of observations

            if n1_dot == 0 or n0_dot == 0 or n_dot1 == 0 or n_dot0 == 0:
                results["Phi"] = np.nan
                results["Phi_p_value"] = np.nan
            else:
                denominator = (n1_dot * n0_dot * n_dot1 * n_dot0) ** 0.5
                phi = (n11 * n00 - n01 * n10) / denominator
                results["Phi"] = phi

                # Chi-Square Test for P-value
                chi2, p_chi2, _, _ = scipy.stats.chi2_contingency(confusion_matrix)
                results["Phi_p_value"] = p_chi2

        else:
            raise ValueError(f"Insufficient data (less than {MIN_INDIVIDUALS_FOR_CORRELATION_TEST} valid entries) to perform the analysis.")

        return results

    
    def test_columns(
        self,
        col_A: str,
        col_B: str
    ) -> Dict[str, Union[float, str]]:
        """
        Perform correlation tests between two columns (HPO terms, sex, diseases).

        Args:
            col_A: The first column to correlate.
            col_B: The second column to correlate.

        Returns:
            dict: A dictionary containing correlation coefficients and p-values for each test.
        """
        if col_A == col_B:
            raise ValueError(f"The two columns must be different. Both columns are the same: {col_A}")

        if col_A not in self.combined_matrix.columns or col_B not in self.combined_matrix.columns:
            raise ValueError(f"One or both of the columns {col_A}, {col_B} are not present in the combined matrix.")

        # Check for HPO term relationships if both columns are HPO terms
        if col_A.startswith("HP:") and col_B.startswith("HP:"):
            if self.hpo.graph.is_ancestor_of(col_A, col_B):
                raise ValueError(f"{col_A} is an ancestor of {col_B}")
            elif self.hpo.graph.is_descendant_of(col_A, col_B):
                raise ValueError(f"{col_A} is a descendant of {col_B}")

        filtered_matrix = self.combined_matrix[[col_A, col_B]].dropna()

        # Check if any columns are entirely 0 or 1
        invalid_columns = filtered_matrix.columns[
            (filtered_matrix == 0).all() | (filtered_matrix == 1).all()
        ]
        if not invalid_columns.empty:
            raise ValueError(f"The following columns are entirely 0 or 1 and are invalid for analysis: {list(invalid_columns)}")

        try:
            stats = self.calculate_stats(filtered_matrix[col_A],filtered_matrix[col_B])
        except ValueError as e:
            raise

        return stats

    
    def calculate_correlation_matrix(self, stats_name: str):
        """
        Calculate a correlation matrix for all columns in the combined matrix.

        Args:
            stats_name: The name of the statistic to calculate (e.g., "Spearman", "Kendall", "Phi").

        Returns:
            A tuple containing:
                - A DataFrame of correlation coefficients.
                - A DataFrame of p-values.
        """
        columns = self.combined_matrix.columns
        n_cols = len(columns)

        matrix = np.full((n_cols, n_cols), np.nan)
        pvalue_matrix = np.full((n_cols, n_cols), np.nan)

        for i, col_A in enumerate(columns):
            for j in range(i + 1, n_cols):
                col_B = columns[j]
                try:
                    result = self.test_columns(col_A, col_B)
                    if result is not None:
                        matrix[i, j] = result[stats_name]
                        matrix[j, i] = result[stats_name]
                        pvalue_matrix[i, j] = result[f"{stats_name}_p_value"]
                        pvalue_matrix[j, i] = result[f"{stats_name}_p_value"]
                except ValueError:
                    continue

        # Convert to DataFrames
        df = pd.DataFrame(matrix, index=columns, columns=columns)
        pvalue_df = pd.DataFrame(pvalue_matrix, index=columns, columns=columns)

        # Rename rows and columns using HPO labels and disease labels if available
        if hasattr(self.cohort_matrix_generator, "hpo_labels") and hasattr(self.cohort_matrix_generator, "disease_labels"):
            hpo_labels = self.cohort_matrix_generator.hpo_labels
            disease_labels = self.cohort_matrix_generator.disease_labels

            # Create a combined label dictionary
            combined_labels = {**hpo_labels, 'Sex': 'sex',**disease_labels}

            # Rename rows and columns
            df_rename = df.rename(index=combined_labels, columns=combined_labels)
            pvalue_df_rename =pvalue_df.rename(index=combined_labels, columns=combined_labels)

        return df_rename,pvalue_df_rename
    
    def clean_matrices(self, stats_name: str, lower_bound=-0.45, upper_bound=0.45):
        coef_matrix,p_value = self.calculate_correlation_matrix(stats_name)

        mask = (coef_matrix > lower_bound) & (coef_matrix < upper_bound)
        coef_matrix[mask] = np.nan
        
        p_value[mask] = np.nan
    
        rows_to_drop = coef_matrix.index[coef_matrix.isna().all(axis=1)]
        cols_to_drop = coef_matrix.columns[coef_matrix.isna().all(axis=0)]
        
        coef_matrix_cleaned = coef_matrix.drop(index=rows_to_drop, columns=cols_to_drop)
        p_value_cleaned = p_value.drop(index=rows_to_drop, columns=cols_to_drop)

        return coef_matrix_cleaned, p_value_cleaned
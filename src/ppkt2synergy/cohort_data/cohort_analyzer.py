import numpy as np
from .matrix_generator import CohortMatrixGenerator
import pandas as pd
from typing import Dict, Union, IO, Tuple
from .hpo_utils import load_hpo, calculate_pairwise_stats
import seaborn as sns
import matplotlib.pyplot as plt


# Do not tests terms pairs with less than 40 individuals
MIN_INDIVIDUALS_FOR_CORRELATION_TEST = 40

class HPOCorrelationAnalyzer:

    """
    A class to analyze statistical relationships between HPO terms and disease terms in a cohort.
    Example:
        from ppkt2synergy import CohortDataLoader, CohortMatrixGenerator, HPOCorrelationAnalyzer
        # Load phenopackets and create matrix generator
        >>> phenopackets = CohortDataLoader.from_ppkt_store('FBN1')
        >>> matrix_gen = CohortMatrixGenerator(phenopackets)
        >>> correlation_matrix, pvalue_matrix = HPOCorrelationAnalyzer(cohort_matrix_generator=matrix_gen).generate_correlation_matrix("Spearman)
    """
    def __init__(
        self,  
        cohort_matrix_generator: CohortMatrixGenerator,
        file: Union[IO, str] = None
    ):
        """
        Initialize the HPOCorrelationAnalyzer.

        Args:
            cohort_matrix_generator: The object that generates HPO and disease matrices.
            file: Path to the HPO ontology file.
        """
        self.hpo = load_hpo(file)
        self.cohort_matrix_generator = cohort_matrix_generator
        self.combined_matrix = self._combine_matrices()

    
    def _combine_matrices(
            self
            ) -> pd.DataFrame:
        """
        Combine HPO term matrix and disease matrix into a single DataFrame.

        Returns:
            A combined DataFrame with all columns.
        """
        hpo_matrix = self.cohort_matrix_generator.hpo_term_observation_matrix
        disease_matrix = self.cohort_matrix_generator.target_matrix

        # Combine matrices
        combined_matrix = hpo_matrix.join(disease_matrix)
        return combined_matrix
    

    def _validate_hpo_terms(
        self,
        term1: str, 
        term2: str, 
    ) -> None:
        """
        Validate if two HPO terms can be compared.

        Args:
            term1: First HPO term ID (e.g., "HP:0000123").
            term2: Second HPO term ID.
            hpo_ontology: Loaded HPO ontology.

        Raises:
            ValueError: If terms are identical or have ancestor-descendant relationship.
        """
        if term1 == term2:
            raise ValueError(f"Cannot compare term with itself: {term1}")

        if self.hpo.graph.is_ancestor_of(term1, term2):
            raise ValueError(f"{term1} is an ancestor of {term2}")
        elif self.hpo.graph.is_descendant_of(term1, term2):
            raise ValueError(f"{term1} is a descendant of {term2}")
    

    def _calculate_pairwise_correlation(
        self,
        col_A: str,
        col_B: str
    ) -> Dict[str, Union[float, str]]:
        """
        Perform correlation tests between two columns (HPO terms, diseases).

        Args:
            col_A: The first column to correlate.
            col_B: The second column to correlate.

        Returns:
            dict: A dictionary containing correlation coefficients and p-values for each tests.
        """
        # Validation
        if col_A.startswith("HP:") and col_B.startswith("HP:"):
            try:
                self._validate_hpo_terms(col_A, col_B)
            except ValueError:  # If there's an ancestor-descendant relationship or identical terms
                return None

        filtered_matrix = self.combined_matrix[[col_A, col_B]].dropna()

        if filtered_matrix.shape[0] <= MIN_INDIVIDUALS_FOR_CORRELATION_TEST:
            raise ValueError(f"Insufficient data (less than {MIN_INDIVIDUALS_FOR_CORRELATION_TEST} valid entries) to perform the analysis.")

        # Check if any columns are entirely 0 or 1
        invalid_columns = filtered_matrix.columns[
            (filtered_matrix == 0).all() | (filtered_matrix == 1).all()
        ]
        if not invalid_columns.empty:
            raise ValueError(f"The following columns are entirely 0 or 1 and are invalid for analysis: {list(invalid_columns)}")

        correlation = calculate_pairwise_stats(filtered_matrix[col_A],filtered_matrix[col_B])
        
        return correlation

    
    def generate_correlation_matrix(
            self, 
            stats_name: str,
            use_labels: bool = True
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate a correlation matrix for all columns in the combined matrix.

        Args:
            stats_name: The name of the statistic to calculate (e.g., "Spearman", "Kendall", "Phi").
            use_labels: Whether to use HPO and disease labels instead of term IDs (default: True).

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
                    result = self._calculate_pairwise_correlation(col_A, col_B)
                    if result is not None:
                        matrix[i, j] = result[stats_name]
                        matrix[j, i] = result[stats_name]
                        pvalue_matrix[i, j] = result[f"{stats_name}_p_value"]
                        pvalue_matrix[j, i] = result[f"{stats_name}_p_value"]
                except ValueError:
                    continue

        # Convert to DataFrames
        correlation_matrix = pd.DataFrame(matrix, index=columns, columns=columns)
        pvalue_matrix = pd.DataFrame(pvalue_matrix, index=columns, columns=columns)

        # Rename rows and columns using HPO labels and disease labels
        if use_labels:
            hpo_labels = self.cohort_matrix_generator.hpo_labels
            disease_labels = self.cohort_matrix_generator.target_labels

            # Create a combined label dictionary
            combined_labels = {**hpo_labels, **disease_labels}

            correlation_matrix = correlation_matrix.rename(index=combined_labels, columns=combined_labels)
            pvalue_matrix = pvalue_matrix.rename(index=combined_labels, columns=combined_labels)
        
        return correlation_matrix, pvalue_matrix
    
    
    def clean_matrices(
            self, 
            stats_name: str, 
            lower_bound=-0.55, 
            upper_bound=0.55
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove weak correlations from the correlation matrix based on the given threshold.

        Args:
            stats_name: The name of the statistic to calculate (e.g., "Spearman", "Kendall", "Phi").
            lower_bound: The lower bound for filtering weak correlations (default: -0.45).
            upper_bound: The upper bound for filtering weak correlations (default: 0.45).

        Returns:
            A tuple containing:
                - A DataFrame of the cleaned correlation coefficients.
                - A DataFrame of the cleaned p-values.
        """
        coef_matrix,p_value = self.generate_correlation_matrix(stats_name)

        mask = (coef_matrix > lower_bound) & (coef_matrix < upper_bound)
        coef_matrix[mask] = np.nan
        
        p_value[mask] = np.nan
    
        rows_to_drop = coef_matrix.index[coef_matrix.isna().all(axis=1)]
        cols_to_drop = coef_matrix.columns[coef_matrix.isna().all(axis=0)]
        
        coef_matrix_cleaned = coef_matrix.drop(index=rows_to_drop, columns=cols_to_drop)
        p_value_cleaned = p_value.drop(index=rows_to_drop, columns=cols_to_drop)

        return coef_matrix_cleaned, p_value_cleaned